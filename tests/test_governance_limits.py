"""预算守卫 + trace 在整个图上的行为：多步工具链、死循环、步数/token/耗时、被中止后转人工。"""

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from carepilot.api.app import create_app
from carepilot.auth import Principal, Role
from carepilot.config import Settings
from carepilot.gateway.guard import BudgetExceeded, RunBudget, RunGuard
from carepilot.graph.build import build_graph
from carepilot.graph.nodes import Models
from carepilot.graph.state import ToolPlan
from carepilot.observability.hooks import record_llm_call
from carepilot.observability.tracer import RunTracer, TraceStore
from carepilot.retrieval.base import Chunk, Hit
from carepilot.sandbox import tools as sbx
from carepilot.sandbox.data import build_world
from carepilot.sandbox.logistics import LogisticsClient
from carepilot.store import TicketStore, seed_demo_users
from carepilot.tools import TicketSystem
from carepilot.toolbox import Toolbox, principal_to_dict

from fakes import FakeRetriever, ScriptedInvoker, make_draft, make_triage

C1 = Principal("c1", Role.CUSTOMER, frozenset({"S1", "S2"}))
H = lambda uid: {"X-User-Id": uid}  # noqa: E731
DATA_EXC = lambda **kw: make_triage(category="data_exception", severity="medium", fault_description="查订单", **kw)  # noqa: E731


class Cycle:
    """永远返回同一个结果的假模型：模拟一个陷入死循环的规划模型。"""

    def __init__(self, result):
        self.result, self.calls = result, 0

    def invoke(self, messages):
        self.calls += 1
        return self.result


class Counter:
    """每次返回不同参数的假模型：模拟一个不断换参数乱试的规划模型。"""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return ToolPlan(tool="query_order", order_id=f"X{self.calls:08d}")


def build_env(tmp_path, planner, *, max_tool_calls=4, draft=None):
    world = build_world(7)
    store = TicketStore(tmp_path / "t.sqlite")
    seed_demo_users(store)
    tb = Toolbox(world=world, store=store, logistics=LogisticsClient(lambda oid: sbx.query_shipment(world, oid), timeout_s=0.3, retries=1, sleep=lambda s: None),
                 db_path=tmp_path / "g.sqlite")
    models = Models(triage=ScriptedInvoker(DATA_EXC()), draft=draft or ScriptedInvoker(make_draft(cited_ids=["1", "2"])), planner=planner)
    graph = build_graph(Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1"), FakeRetriever(), TicketSystem(),
                        InMemorySaver(), models=models, toolbox=tb, max_tool_calls=max_tool_calls)
    return graph, store, tb, world


def run(graph, store, message, *, guard=None, tracer=None):
    ticket = store.create_ticket(C1, subject="x", shop_id="S1", thread_id="t")
    thread = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread, **({"guard": guard} if guard else {}), **({"tracer": tracer} if tracer else {})}}
    graph.invoke({"ticket_id": ticket.ticket_id, "run_id": thread, "customer_message": message, "principal": principal_to_dict(C1)}, cfg)
    return ticket, cfg


def orders(world):
    s1 = [o for o in world.orders.values() if o.shop_id == "S1"]
    shipped = next(o for o in s1 if o.order_id in world.shipments)
    other = next(o for o in s1 if o.order_id != shipped.order_id)
    return shipped, other


def test_planner_can_chain_two_read_tools_and_the_reply_cites_both(tmp_path):
    world = build_world(7)
    shipped, other = orders(world)
    plans = ScriptedInvoker([ToolPlan(tool="query_order", order_id=other.order_id),
                             ToolPlan(tool="query_shipment", order_id=shipped.order_id), ToolPlan(tool="none")])
    graph, store, tb, _ = build_env(tmp_path, plans)
    ticket, cfg = run(graph, store, f"订单 {other.order_id} 什么状态，另外 {shipped.order_id} 物流到哪了")
    pending = [i.value for t in graph.get_state(cfg).tasks for i in t.interrupts][0]
    assert [c["tool"] for c in pending["tool_run"]["calls"]] == ["query_order", "query_shipment"]
    assert len(pending["citations"]) == 2 and plans.call_count == 3
    assert "已执行的查询" in str(plans.calls[2][1].content)  # 后续规划看到的是执行结果的状态，不是数据本身


def test_duplicate_proposals_are_not_executed_twice(tmp_path):
    world = build_world(7)
    shipped, _ = orders(world)
    same = ToolPlan(tool="query_order", order_id=shipped.order_id)
    graph, store, tb, _ = build_env(tmp_path, ScriptedInvoker([same, same, ToolPlan(tool="none")]))
    ticket, cfg = run(graph, store, f"查订单 {shipped.order_id}")
    executed = [e for e in tb.gateway.audit(ticket_id=ticket.ticket_id) if e["tool"] == "query_order" and e["decision"] == "executed"]
    assert len(executed) == 1


def test_distinct_shop_names_are_not_guarded_as_duplicate_calls(tmp_path):
    world = build_world(7)
    shops = [world.shops[shop_id] for shop_id in ("S1", "S2", "S3")]
    plans = ScriptedInvoker([*(ToolPlan(tool="query_shop_auth", shop_name=shop.name) for shop in shops), ToolPlan(tool="none")])
    graph, store, _, _ = build_env(tmp_path, plans)
    message = "查询授权状态：" + "、".join(shop.name for shop in shops)
    _, cfg = run(graph, store, message, guard=RunGuard(RunBudget(max_steps=100)))

    values = graph.get_state(cfg).values

    assert [call["outcome"] for call in values["tool_run"]["calls"]] == ["ok", "ok", "not_found"]


def test_endless_identical_proposals_are_cut_off_as_a_loop(tmp_path):
    world = build_world(7)
    shipped, _ = orders(world)
    looping = Cycle(ToolPlan(tool="query_order", order_id=shipped.order_id))
    graph, store, tb, _ = build_env(tmp_path, looping, max_tool_calls=50)
    guard = RunGuard(RunBudget(max_steps=20))
    with pytest.raises(BudgetExceeded) as exc:
        run(graph, store, f"查订单 {shipped.order_id}", guard=guard)
    assert exc.value.reason == "loop" and looping.calls <= 4 and guard.steps <= 20  # 20 步之内被打断


def test_endless_new_arguments_hit_the_step_limit_within_twenty_steps(tmp_path):
    counter = Counter()
    graph, store, tb, _ = build_env(tmp_path, counter, max_tool_calls=100)
    guard = RunGuard(RunBudget(max_steps=20))
    with pytest.raises(BudgetExceeded) as exc:
        run(graph, store, "随便查点什么", guard=guard)
    assert exc.value.reason == "steps" and counter.calls <= 20


def test_token_budget_counts_tokens_recorded_inside_a_node(tmp_path):
    class Hungry:
        calls = 0

        def invoke(self, messages):
            Hungry.calls += 1
            record_llm_call(SimpleNamespace(usage_metadata={"input_tokens": 30000, "output_tokens": 5000}),
                            purpose="planner", model="m", latency_ms=1.0)
            return ToolPlan(tool="query_order", order_id=f"X{Hungry.calls:08d}")

    graph, store, tb, _ = build_env(tmp_path, Hungry(), max_tool_calls=50)
    with pytest.raises(BudgetExceeded) as exc:
        run(graph, store, "查订单", guard=RunGuard(RunBudget(max_tokens=50_000)))
    assert exc.value.reason == "tokens" and Hungry.calls == 2  # 第二次调用后累计 70000 > 50000


def test_time_budget_uses_an_injected_clock(tmp_path):
    now = {"t": 0.0}

    class Slow:
        def invoke(self, messages):
            now["t"] += 100.0
            return ToolPlan(tool="query_order", order_id=f"X{int(now['t']):08d}")

    graph, store, tb, _ = build_env(tmp_path, Slow(), max_tool_calls=50)
    with pytest.raises(BudgetExceeded) as exc:
        run(graph, store, "查订单", guard=RunGuard(RunBudget(max_seconds=120), clock=lambda: now["t"]))
    assert exc.value.reason == "time"


def test_every_node_gets_a_span_and_llm_usage_lands_on_the_right_span(tmp_path):
    class Chain:
        def __init__(self, parsed):
            self.parsed = parsed

        def invoke(self, messages):
            return {"raw": SimpleNamespace(usage_metadata={"input_tokens": 120, "output_tokens": 30}, content="x"), "parsed": self.parsed, "parsing_error": None}

    class FakeLLM:
        def __init__(self, parsed):
            self.parsed = parsed

        def with_structured_output(self, schema, method, include_raw):
            return Chain(self.parsed)

    from carepilot.llm import StructuredLLM
    from carepilot.graph.state import DraftResult, Triage

    triage = StructuredLLM(FakeLLM(make_triage(category="feature_howto", feature_name="库存同步")), Triage, purpose="triage", model="fake-model")
    draft = StructuredLLM(FakeLLM(DraftResult(answerable=True, reply="请重新授权", cited_ids=["1"])), DraftResult, purpose="draft", model="fake-model")
    hit = Hit(Chunk(id="企业版/库存#0", title="库存", text="库存 > 失败\n重新授权", doc_id="企业版/库存", meta={"top_dir": "企业版", "title_key": "库存"}), 1.0)
    graph = build_graph(Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1"), FakeRetriever([hit]), TicketSystem(), InMemorySaver(),
                        models=Models(triage=triage, draft=draft))
    tracer = RunTracer("run-1", ticket_id="T-000001")
    graph.invoke({"ticket_id": "T-000001", "customer_message": "库存同步失败怎么办"},
                 {"configurable": {"thread_id": "run-1", "tracer": tracer, "guard": RunGuard(RunBudget())}})
    trace = tracer.snapshot()
    names = [s.name for s in trace.spans]
    assert names[:3] == ["classify", "retrieve", "draft_reply"] and names[-1] == "human_review"
    by = {s.name: s for s in trace.spans}
    assert [(c.purpose, c.model, c.prompt_tokens, c.completion_tokens) for c in by["classify"].llm_calls] == [("triage", "fake-model", 120, 30)]
    assert [c.purpose for c in by["draft_reply"].llm_calls] == ["draft"] and by["retrieve"].llm_calls == []
    assert by["human_review"].output_summary == "等待人工审核" and by["human_review"].error is None  # 等审核不是错误
    assert "category=feature_howto" in by["classify"].output_summary and trace.total_tokens == (240, 60)


# ---------------------------------------------------------------- 通过 HTTP：trace 落盘 + 中止后转人工
def make_app(tmp_path, planner, *, max_steps=20):
    world = build_world(7)
    store = TicketStore(tmp_path / "t.sqlite")
    seed_demo_users(store)
    tb = Toolbox(world=world, store=store, logistics=LogisticsClient(lambda oid: sbx.query_shipment(world, oid), timeout_s=0.3, retries=1, sleep=lambda s: None),
                 db_path=tmp_path / "g.sqlite")
    chunk = Chunk(id="企业版/库存同步#0", title="库存同步", text="库存同步 > 失败\n重新授权", doc_id="企业版/库存同步",
                  meta={"top_dir": "企业版", "title_key": "库存同步", "source_url": "https://wiki.example/x"})
    app = create_app(
        settings=Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1", trace_dir=tmp_path / "traces", run_max_steps=max_steps),
        models=Models(triage=ScriptedInvoker(DATA_EXC()), draft=ScriptedInvoker(make_draft(cited_ids=["1"])), planner=planner),
        retriever=FakeRetriever([Hit(chunk, 1.0)]), checkpointer=InMemorySaver(), governed=True, store=store, toolbox=tb, chunks=[chunk])
    return app, world, store, tb


def test_http_run_writes_one_cumulative_trace_json_that_survives_the_review_phase(tmp_path):
    world = build_world(7)
    shipped, _ = orders(world)
    app, _, store, tb = make_app(tmp_path, ScriptedInvoker([ToolPlan(tool="query_order", order_id=shipped.order_id), ToolPlan(tool="none")]))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": f"查订单 {shipped.order_id}"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        files = list((tmp_path / "traces").glob("*.json"))
        assert len(files) == 1
        mid = json.loads(files[0].read_text(encoding="utf-8"))
        assert mid["status"] == "awaiting_review" and mid["category"] == "data_exception" and mid["ticket_id"] == tid
        assert [s["name"] for s in mid["spans"]][:3] == ["classify", "plan_tools", "draft_reply"]
        c.post(f"/tickets/{tid}/review", json={"action": "approve", "confirm": True}, headers=H("a1"))
        final = json.loads(files[0].read_text(encoding="utf-8"))
        assert final["status"] == "resolved" and len(final["spans"]) > len(mid["spans"])  # 审核阶段的节点续写在同一份里
        assert final["spans"][-1]["name"] == "execute_actions" and final["ended_at"] >= final["started_at"]
        assert len(list((tmp_path / "traces").glob("*.json"))) == 1  # 一次 run 就是一份文件
        assert TraceStore(tmp_path / "traces").load(final["run_id"]).run_id == final["run_id"]


def test_a_runaway_planner_is_aborted_to_a_human_with_a_note_and_no_state_change(tmp_path):
    world = build_world(7)
    shipped, _ = orders(world)
    app, _, store, tb = make_app(tmp_path, Cycle(ToolPlan(tool="query_order", order_id=shipped.order_id)))
    with TestClient(app) as c:
        r = c.post("/tickets", json={"message": f"查订单 {shipped.order_id}"}, headers=H("c1"))
        assert r.status_code == 200
        tid = r.json()["ticket"]["ticket_id"]
        staff = c.get(f"/tickets/{tid}", headers=H("a1")).json()
        assert staff["status"] == "needs_human" and staff["abort_reason"] == "budget:loop"
        assert staff["ticket"]["status"] == "new"  # 守卫只中止，不替人改工单状态
        notes = [m["body"] for m in staff["messages"] if m["kind"] == "note"]
        assert any("预算守卫中止" in n and "loop" in n for n in notes)
        customer = c.get(f"/tickets/{tid}", headers=H("c1")).json()
        assert customer["status"] == "needs_human" and "abort_reason" not in customer
        assert all(m["visibility"] == "public" for m in customer["messages"])  # 内部备注客户看不到
        trace = json.loads(next((tmp_path / "traces").glob("*.json")).read_text(encoding="utf-8"))
        assert trace["status"] == "budget_exceeded"
        assert not any(e["tool"] in ("send_reply", "set_ticket_status", "escalate_ticket") for e in tb.gateway.audit(ticket_id=tid))


def test_a_human_can_still_take_over_an_aborted_ticket(tmp_path):
    world = build_world(7)
    shipped, _ = orders(world)
    app, _, store, tb = make_app(tmp_path, Cycle(ToolPlan(tool="query_order", order_id=shipped.order_id)))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": f"查订单 {shipped.order_id}"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        assert c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a1")).status_code == 409  # 没有待审核的草稿
        a1 = store.get_principal("a1")
        store.claim(a1, tid)
        store.transition(a1, tid, "open", reason="人工接手")
        assert store.get_ticket(a1, tid).owner_id == "a1"


def test_tokens_spent_by_an_ordinary_node_are_charged_when_the_node_finishes(tmp_path):
    class HungryTriage:
        def invoke(self, messages):
            record_llm_call(SimpleNamespace(usage_metadata={"input_tokens": 60000, "output_tokens": 100}),
                            purpose="triage", model="m", latency_ms=1.0)
            return make_triage(category="feature_howto", feature_name="库存同步")

    graph = build_graph(Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1"), FakeRetriever(), TicketSystem(), InMemorySaver(),
                        models=Models(triage=HungryTriage(), draft=ScriptedInvoker(make_draft())))
    guard = RunGuard(RunBudget(max_tokens=50_000))
    with pytest.raises(BudgetExceeded) as exc:
        graph.invoke({"ticket_id": "T-1", "customer_message": "库存同步怎么设置"}, {"configurable": {"thread_id": "tk", "guard": guard}})
    assert exc.value.reason == "tokens" and guard.steps == 1  # classify 一结束就被打断，没有继续检索和生成
