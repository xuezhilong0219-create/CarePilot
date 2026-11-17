"""治理模式的 HTTP 接口：真实的工单存储、网关、沙箱；只有模型是脚本化的假实现。"""

import json
import sqlite3

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from carepilot.api.app import create_app
from carepilot.config import Settings
from carepilot.graph.nodes import Models
from carepilot.graph.state import ToolPlan
from carepilot.retrieval.base import Chunk, Hit
from carepilot.sandbox import tools as sbx
from carepilot.sandbox.data import build_world
from carepilot.sandbox.logistics import LogisticsClient
from carepilot.store import TicketStore, seed_demo_users
from carepilot.toolbox import Toolbox

from fakes import FakeRetriever, ScriptedInvoker, make_draft, make_triage

H = lambda uid: {"X-User-Id": uid}  # noqa: E731
DATA_EXC = lambda **kw: make_triage(category="data_exception", severity="high", fault_description="查订单", **kw)  # noqa: E731
CHUNK = Chunk(id="企业版/库存同步#0", title="库存同步", text="库存同步 > 常见问题 > 失败\n重新授权后再同步。", doc_id="企业版/库存同步",
              meta={"top_dir": "企业版", "title_key": "库存同步", "source_url": "https://wiki.example/inventory"})


def build(tmp_path, triage, draft=None, planner=None, *, hits=None, models_extra=None, settings_extra=None):
    world = build_world(7)
    store = TicketStore(tmp_path / "t.sqlite")
    seed_demo_users(store)
    logistics = LogisticsClient(lambda oid: sbx.query_shipment(world, oid), timeout_s=0.3, retries=1, sleep=lambda s: None)
    tb = Toolbox(world=world, store=store, logistics=logistics, db_path=tmp_path / "g.sqlite")
    t_inv, d_inv, p_inv = (x if isinstance(x, ScriptedInvoker) else ScriptedInvoker(x) for x in (triage, draft, planner))
    app = create_app(
        settings=Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1", trace_dir=tmp_path / "traces",
                          **(settings_extra or {})),
        models=Models(triage=t_inv, draft=d_inv, planner=p_inv),
        retriever=FakeRetriever(hits if hits is not None else [Hit(CHUNK, 1.0)]),
        checkpointer=InMemorySaver(), governed=True, store=store, toolbox=tb, chunks=[CHUNK],
    )
    return app, world, store, tb, t_inv


def own_order(world, shop="S1"):
    return next(o for o in world.orders.values() if o.shop_id == shop)


def test_identity_is_required_and_role_comes_from_the_user_table(tmp_path):
    app, *_ = build(tmp_path, make_triage(), make_draft())
    with TestClient(app) as c:
        assert c.post("/tickets", json={"message": "hi"}).status_code == 401
        assert c.post("/tickets", json={"message": "hi"}, headers=H("nobody")).status_code == 401
        spoof = c.get("/tickets/T-000001/audit", headers={"X-User-Id": "c1", "X-Role": "admin"})  # 自报角色无效
        assert spoof.status_code == 403
        assert c.get("/health").status_code == 200  # 健康检查不要求身份


def test_customer_creates_ticket_sees_only_customer_fields_and_staff_see_internal_ones(tmp_path):
    app, world, store, *_ = build(tmp_path, DATA_EXC(), make_draft(cited_ids=["1"]), ToolPlan(tool="none"))
    with TestClient(app) as c:
        r = c.post("/tickets", json={"message": "账单金额不对，帮我核对"}, headers=H("c1"))
        assert r.status_code == 200
        v = r.json()
        tid = v["ticket"]["ticket_id"]
        assert v["ticket"]["category"] == "data_exception" and v["ticket"]["priority"] == "high"  # 分诊结果回写了工单
        assert "owner_id" not in v["ticket"] and "pending_review" not in v and "actions" not in v and "triage" not in v
        assert [m["body"] for m in v["messages"]] == ["账单金额不对，帮我核对"]
        staff = c.get(f"/tickets/{tid}", headers=H("a1")).json()
        assert staff["status"] == "awaiting_review" and staff["pending_review"]["resolution"] == "escalate"
        assert "sla_remaining_minutes" in staff["ticket"] and staff["triage"]["category"] == "data_exception"


def test_customers_cannot_see_each_others_tickets_and_unknown_is_404(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        assert c.get(f"/tickets/{tid}", headers=H("c2")).status_code == 403
        assert c.post(f"/tickets/{tid}/messages", json={"message": "插一句"}, headers=H("c2")).status_code == 403
        assert c.get("/tickets/T-999999", headers=H("c1")).status_code == 404
        assert [t["ticket_id"] for t in c.get("/tickets", headers=H("c2")).json()] == []
        assert [t["ticket_id"] for t in c.get("/tickets", headers=H("a1")).json()] == [tid]


def test_customer_cannot_use_staff_endpoints(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        for method, path in (("post", f"/tickets/{tid}/review"), ("post", f"/tickets/{tid}/confirm"),
                             ("get", f"/tickets/{tid}/handoff"), ("get", f"/tickets/{tid}/audit"), ("get", "/citations/x")):
            kw = {"json": {"action": "approve"}} if path.endswith("review") else {}
            assert getattr(c, method)(path, headers=H("c1"), **kw).status_code == 403, path


def test_full_flow_query_review_two_step_confirm_and_idempotent_send(tmp_path):
    world = build_world(7)
    mine = own_order(world)
    app, world, store, tb, _ = build(tmp_path, DATA_EXC(), make_draft(reply="订单已发出。", cited_ids=["1"]),
                                     [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])
    with TestClient(app) as c:
        v = c.post("/tickets", json={"message": f"帮我查订单 {mine.order_id}"}, headers=H("c1")).json()
        tid = v["ticket"]["ticket_id"]
        assert v["status"] == "awaiting_review"
        card = c.get(f"/tickets/{tid}", headers=H("a1")).json()["pending_review"]
        assert card["citation_details"][0]["kind"] == "tool" and card["tool_run"]["outcome"] == "ok"
        # 审核：坐席自动认领；批准后进入"待二次确认"，客户收不到任何东西
        r = c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a1")).json()
        assert r["status"] == "awaiting_confirmation" and r["ticket"]["owner_id"] == "a1" and r["pending_confirmation"]["tool"] == "send_reply"
        assert [m["body"] for m in c.get(f"/tickets/{tid}", headers=H("c1")).json()["messages"]] == [f"帮我查订单 {mine.order_id}"]
        for _ in range(3):  # 重复确认：每次都 200，客户只会收到一条
            done = c.post(f"/tickets/{tid}/confirm", headers=H("a1"))
            assert done.status_code == 200 and done.json()["ticket"]["status"] == "resolved"
        bodies = [m["body"] for m in c.get(f"/tickets/{tid}", headers=H("c1")).json()["messages"]]
        assert bodies == [f"帮我查订单 {mine.order_id}", "订单已发出。"]
        assert not any(m["kind"] == "draft" for m in c.get(f"/tickets/{tid}", headers=H("c1")).json()["messages"])  # 草稿不可见
        audit = c.get(f"/tickets/{tid}/audit", headers=H("a1")).json()
        assert {"query_order", "add_note", "send_reply", "set_ticket_status"} <= {a["tool"] for a in audit}
        assert any(a["tool"] == "send_reply" and a["decision"] == "replayed" for a in audit)  # 重复确认在审计里是回放


def test_other_agents_cannot_review_or_confirm_a_ticket_owned_by_someone_else(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        assert c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a1")).status_code == 200
        assert c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a2")).status_code == 409  # 已不在待审核
        assert c.post(f"/tickets/{tid}/confirm", headers=H("a2")).status_code in (403, 409)


def test_agent_two_cannot_review_while_agent_one_owns_the_ticket(tmp_path):
    app, world, store, tb, _ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        store.claim(store.get_principal("a1"), tid)
        assert c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a2")).status_code == 403


def test_escalation_writes_handoff_visible_to_the_next_agent_only(tmp_path):
    app, world, store, tb, _ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "抖店结算账单少了86块"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        r = c.post(f"/tickets/{tid}/review", json={"action": "approve", "note": "需要查结算明细"}, headers=H("a1")).json()
        assert r["status"] == "escalated"
        store.release(store.get_principal("a1"), tid)  # 坐席一下班，释放
        assert c.post(f"/tickets/{tid}/review", json={"action": "approve"}, headers=H("a2")).status_code == 409
        h = c.get(f"/tickets/{tid}/handoff", headers=H("a2")).json()
        assert "需要查结算明细" in h["escalation_reason"] and h["last_customer_message"] == "抖店结算账单少了86块"
        assert any(a.startswith("escalate_ticket：") and "data_support_team" in a for a in h["actions_done"])
        assert "转交 data_support_team" in h["escalation_reason"]
        assert c.get(f"/tickets/{tid}/handoff", headers=H("c1")).status_code == 403
        customer_view = c.get(f"/tickets/{tid}", headers=H("c1")).json()
        assert all(m["visibility"] == "public" for m in customer_view["messages"])  # 内部备注一条都不可见


def test_three_turns_on_one_ticket_keep_the_slots_and_pass_history(tmp_path):
    turn = lambda **kw: make_triage(category="platform_integration", **kw)  # noqa: E731
    app, world, store, tb, t_inv = build(
        tmp_path, [turn(platform_name="京东", shop_name="小熊数码"), turn(feature_name="库存同步"), turn()],
        [make_draft(cited_ids=["1"])] * 3)
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "京东店铺小熊数码授权过期了"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        c.post(f"/tickets/{tid}/messages", json={"message": "它的库存同步也失败了"}, headers=H("c1"))
        c.post(f"/tickets/{tid}/messages", json={"message": "还是不行"}, headers=H("c1"))
        slots = store.get_context(tid)["slots"]
        assert slots["platform_name"] == "京东" and slots["shop_name"] == "小熊数码" and slots["feature_name"] == "库存同步"
        last_prompt = str(t_inv.calls[2][1].content)
        assert "【对话历史】" in last_prompt and "京东店铺小熊数码授权过期了" in last_prompt
        assert last_prompt.count("还是不行") == 1  # 本句只出现在"本句"里，不会同时混进历史


def test_citation_endpoint_returns_the_original_snippet_to_staff(tmp_path):
    app, *_ = build(tmp_path, make_triage(category="feature_howto", feature_name="库存同步", product_edition="企业版"),
                    make_draft(cited_ids=["1"]))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "库存同步失败怎么办"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        card = c.get(f"/tickets/{tid}", headers=H("a1")).json()["pending_review"]
        d = card["citation_details"][0]
        assert d["doc_title"] == "库存同步" and d["section"] == "常见问题 > 失败" and d["edition"] == "企业版"
        got = c.get(d["link"], headers=H("a1"))
        assert got.status_code == 200 and "重新授权" in got.json()["text"] and got.json()["source_url"] == "https://wiki.example/inventory"
        assert c.get("/citations/不存在#9", headers=H("a1")).status_code == 404
        customer_view = json.dumps(c.get(f"/tickets/{tid}", headers=H("c1")).json(), ensure_ascii=False)
        assert "wiki.example" not in customer_view and "citation" not in customer_view  # 内部链接不给客户


def test_reject_escalates_and_stream_endpoint_emits_nodes_then_result(tmp_path):
    app, *_ = build(tmp_path, make_triage(category="feature_howto", feature_name="库存同步"), make_draft(cited_ids=["1"]))
    with TestClient(app) as c:
        events = []
        with c.stream("POST", "/tickets?stream=true", json={"message": "库存同步失败怎么办"}, headers=H("c1")) as r:
            ev = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    ev = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and ev:
                    events.append((ev, json.loads(line[5:])))
        names = [e for e, _ in events]
        assert names[-1] == "result" and names[0] == "node" and "error" not in names
        assert [d["node"] for e, d in events if e == "node"][:2] == ["classify", "retrieve"]
        tid = events[-1][1]["ticket"]["ticket_id"]
        r = c.post(f"/tickets/{tid}/review", json={"action": "reject", "note": "转人工"}, headers=H("a1")).json()
        assert r["status"] == "escalated" and r["ticket"]["status"] == "escalated"


def test_message_validation_and_closed_tickets(tmp_path):
    app, world, store, tb, _ = build(tmp_path, make_triage(), make_draft())
    with TestClient(app) as c:
        assert c.post("/tickets", json={"message": ""}, headers=H("c1")).status_code == 422
        assert c.post("/tickets", json={"message": "x" * 2001}, headers=H("c1")).status_code == 422
        assert c.post("/tickets", json={"message": "hi", "shop_id": "S3"}, headers=H("c1")).status_code == 403  # 不是他的店铺
        assert c.post("/tickets", json={"message": "hi"}, headers=H("a1")).status_code == 403  # 只有客户能开单


def leave_orphan(tb, *, tool="query_order", key="orphan-key", args=None):
    """模拟「上次执行者崩了」：账本里一行 started，没有租约。"""
    now = tb.gateway.clock()
    args_json = json.dumps(args or {"order_id": "X123456"})
    principal = json.dumps({"user_id": "a1", "role": "agent", "shop_ids": []})
    with tb.gateway.ledger.transaction() as db:
        db.execute("INSERT INTO action_ledger (idem_key,run_id,tool,status,created_at,updated_at)"
                   " VALUES (?,?,?,'started',?,?)", (key, "run-orphan", tool, now, now))
        db.execute("INSERT INTO action_inputs (idem_key,principal_json,ticket_id,args_json,approver_id,approval_id,recovery_status)"
                   " VALUES (?,?,?,?,NULL,NULL,NULL)", (key, principal, "T-000001", args_json))


def ledger_status(tb) -> str:
    with sqlite3.connect(tb.gateway.ledger.path) as db:
        return db.execute("SELECT status FROM action_ledger").fetchone()[0]


def test_reconcile_endpoint_is_admin_only(tmp_path):
    app, *_ = build(tmp_path, make_triage(), make_draft())
    with TestClient(app) as c:
        assert c.post("/admin/reconcile", headers=H("c1")).status_code == 403
        assert c.post("/admin/reconcile", headers=H("a1")).status_code == 403
        assert c.post("/admin/reconcile").status_code == 401
        assert c.get("/admin/inflight", headers=H("a1")).status_code == 403
        assert c.post("/admin/reconcile", headers=H("admin1")).status_code == 200


def test_dry_run_lists_the_orphan_without_touching_it(tmp_path):
    app, world, store, tb, _ = build(tmp_path, make_triage(), make_draft())
    leave_orphan(tb)
    with TestClient(app) as c:
        body = c.post("/admin/reconcile", params={"dry_run": "true"}, headers=H("admin1")).json()
        assert body["dry_run"] is True
        assert [o["status"] for o in body["outcomes"]] == ["would_reconcile"]
        assert ledger_status(tb) == "started"
        assert tb.gateway.audit(ticket_id="T-000001") == []
        body = c.post("/admin/reconcile", headers=H("admin1")).json()
        assert [o["status"] for o in body["outcomes"]] == ["needs_manual"]
        assert ledger_status(tb) == "started"  # 无 verify 的工具只能交给人
        assert [e["code"] for e in tb.gateway.audit(ticket_id="T-000001")] == ["reconciled_needs_manual"]


def test_inflight_endpoint_reports_whether_the_lease_is_held(tmp_path):
    app, world, store, tb, _ = build(tmp_path, make_triage(), make_draft())
    leave_orphan(tb)
    with TestClient(app) as c:
        rows = c.get("/admin/inflight", headers=H("admin1")).json()["inflight"]
        assert [(r["tool"], r["live"]) for r in rows] == [("query_order", False)]
        tb.gateway.leases.acquire("orphan-key")
        try:
            rows = c.get("/admin/inflight", headers=H("admin1")).json()["inflight"]
            assert rows[0]["live"] is True
            body = c.post("/admin/reconcile", headers=H("admin1")).json()
            assert [o["status"] for o in body["outcomes"]] == ["skipped_inflight"]
        finally:
            tb.gateway.leases.release("orphan-key")

def test_sweeper_only_runs_when_an_interval_is_configured(tmp_path):
    app, *_ = build(tmp_path, make_triage(), make_draft(), settings_extra={"reconcile_interval_s": 0.05})
    with TestClient(app):
        sweeper = app.state.sweeper
        assert sweeper is not None and sweeper._thread is not None and sweeper._thread.is_alive()
        assert "carepilot-reconcile" == sweeper._thread.name
    assert sweeper._thread is None  # 关停时线程被收起，不留野线程


def test_sweeper_stays_off_by_default(tmp_path):
    app, *_ = build(tmp_path, make_triage(), make_draft())
    with TestClient(app):
        assert getattr(app.state, "sweeper", None) is None

def test_trace_is_staff_only(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(cited_ids=["1"]), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单金额不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        assert c.get(f"/tickets/{tid}/trace").status_code == 401
        assert c.get(f"/tickets/{tid}/trace", headers=H("c1")).status_code == 403
        assert c.get(f"/tickets/{tid}/trace", headers=H("a1")).status_code == 200


def test_trace_lists_each_node_with_its_time_and_tokens(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(cited_ids=["1"]), ToolPlan(tool="none"))
    with TestClient(app) as c:
        tid = c.post("/tickets", json={"message": "账单金额不对"}, headers=H("c1")).json()["ticket"]["ticket_id"]
        t = c.get(f"/tickets/{tid}/trace", headers=H("a1")).json()
        names = [n["node"] for n in t["nodes"]]
        assert "classify" in names and "human_review" in names  # 跑过的节点都留了记录
        assert all(isinstance(n["duration_ms"], (int, float)) and n["duration_ms"] >= 0 for n in t["nodes"])
        assert t["run_id"] and t["status"] == "awaiting_review"
        assert t["tokens"]["total"] == t["tokens"]["prompt"] + t["tokens"]["completion"]
        assert isinstance(t["tokens"]["usage_missing"], bool)
        # 停在等待审核的轮次落盘的 ended_at 是 0.0，接口要给一个能看的耗时
        assert t["open"] is True
        assert t["duration_ms"] >= 0  # 假模型瞬间返回，算力时间可能就是 0
        assert t["wall_ms"] >= t["duration_ms"]  # 全程不会比算力短
        assert t["ended_at"] >= t["started_at"]
        assert all(isinstance(n["llm"], list) for n in t["nodes"])  # 假模型不产生 usage，但结构要在


def test_trace_is_404_for_a_ticket_that_never_ran(tmp_path):
    app, *_ = build(tmp_path, DATA_EXC(), make_draft(), ToolPlan(tool="none"))
    with TestClient(app) as c:
        assert c.get("/tickets/T-000404/trace", headers=H("a1")).status_code == 404
