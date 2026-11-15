"""接入工具箱后的图：规划→查询只读工具、草稿存内部备注、写操作经网关两步确认、多轮槽位继承、产品线过滤。
全部使用真实的工单存储、沙箱、网关；只有模型是脚本化的假实现。"""

import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from carepilot.auth import Principal, Role
from carepilot.config import Settings
from carepilot.graph.build import build_graph
from carepilot.graph.nodes import (
    Models, build_edition_index, build_query, citation_details, filter_by_edition, merge_slots, normalize_edition,
)
from carepilot.graph.state import ToolPlan
from carepilot.retrieval.base import Chunk, Hit
from carepilot.sandbox import tools as sbx
from carepilot.sandbox.data import build_world
from carepilot.sandbox.logistics import LogisticsClient
from carepilot.store import TicketStore, seed_demo_users
from carepilot.tools import TicketSystem
from carepilot.toolbox import Toolbox, principal_to_dict

from fakes import FakeRetriever, ScriptedInvoker, make_draft, make_hit, make_triage

C1 = Principal("c1", Role.CUSTOMER, frozenset({"S1", "S2"}))
A1 = Principal("a1", Role.AGENT)
A2 = Principal("a2", Role.AGENT)
ADMIN = Principal("admin1", Role.ADMIN)


@dataclass
class Env:
    graph: Any
    tb: Toolbox
    store: TicketStore
    world: Any
    retriever: FakeRetriever
    triage: ScriptedInvoker
    draft: ScriptedInvoker
    planner: ScriptedInvoker

    def start(self, message: str, *, customer=C1, history=None, slots=None):
        ticket = self.store.create_ticket(customer, subject=message[:20], shop_id="S1", thread_id="t")
        thread = uuid.uuid4().hex
        state = {"ticket_id": ticket.ticket_id, "run_id": thread, "customer_message": message,
                 "principal": principal_to_dict(customer), "history": history or [], "slots": slots or {}}
        cfg = {"configurable": {"thread_id": thread}}
        self.graph.invoke(state, cfg)
        return ticket, cfg

    def snap(self, cfg):
        return self.graph.get_state(cfg)

    def pending(self, cfg):
        return [i.value for t in self.snap(cfg).tasks for i in t.interrupts]

    def review(self, cfg, **decision):
        self.graph.invoke(Command(resume=decision), cfg)
        return self.snap(cfg).values


def order_of(world, shop, *, shipped=None):
    return next(o for o in world.orders.values()
                if o.shop_id == shop and (shipped is None or (o.order_id in world.shipments) == shipped))


@pytest.fixture
def make_env(tmp_path):
    def _make(triage, draft=None, planner=None, hits=None, transport=None):
        world = build_world(7)
        store = TicketStore(tmp_path / "t.sqlite")
        seed_demo_users(store)
        logistics = LogisticsClient(transport or (lambda oid: sbx.query_shipment(world, oid)), timeout_s=0.3, retries=1, sleep=lambda s: None)
        tb = Toolbox(world=world, store=store, logistics=logistics, db_path=tmp_path / "gw.sqlite")
        t_inv, d_inv, p_inv = (x if isinstance(x, ScriptedInvoker) else ScriptedInvoker(x) for x in (triage, draft, planner))
        retriever = FakeRetriever(hits or [])
        settings = Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1")
        graph = build_graph(settings, retriever, TicketSystem(), InMemorySaver(),
                            models=Models(triage=t_inv, draft=d_inv, planner=p_inv), toolbox=tb)
        return Env(graph, tb, store, world, retriever, t_inv, d_inv, p_inv)

    return _make


DATA_EXC = lambda **kw: make_triage(category="data_exception", severity="medium", fault_description="查订单", **kw)  # noqa: E731


def test_data_exception_queries_own_order_then_drafts_from_facts_and_saves_internal_draft(make_env):
    world = build_world(7)
    mine = order_of(world, "S1")
    env = make_env(DATA_EXC(), make_draft(reply="您的订单已发出。", cited_ids=["1"]),
                   [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])
    ticket, cfg = env.start(f"帮我查下订单 {mine.order_id} 的状态")

    pending = env.pending(cfg)[0]
    assert pending["resolution"] == "auto_reply" and pending["tool_run"]["outcome"] == "ok"
    assert pending["citations"] == [f"tool:query_order:{mine.order_id}"]
    assert pending["citation_details"][0]["kind"] == "tool" and mine.order_id in pending["citation_details"][0]["snippet"]
    assert env.retriever.calls == []  # 查客户自己的数据，不检索知识库
    assert mine.order_id in str(env.draft.calls[0][1].content)  # 回复模型拿到的是查询结果事实
    drafts = [m for m in env.store.list_messages(A1, ticket.ticket_id) if m.kind == "draft"]
    assert len(drafts) == 1 and drafts[0].visibility == "internal" and drafts[0].body == "您的订单已发出。"
    assert env.store.list_messages(C1, ticket.ticket_id) == []  # 草稿对客户不可见


def test_foreign_order_and_nonexistent_order_get_the_same_followup(make_env):
    world = build_world(7)
    foreign = order_of(world, "S3")
    replies = []
    for oid in (foreign.order_id, "9" * 12):
        env = make_env(DATA_EXC(), None, [ToolPlan(tool="query_order", order_id=oid), ToolPlan(tool="none")])
        ticket, cfg = env.start(f"订单 {oid} 到哪了")
        v = env.snap(cfg).values
        assert v["status"] == "need_info" and env.draft.call_count == 0
        replies.append(v["followup_question"])
        assert [m.body for m in env.store.list_messages(C1, ticket.ticket_id)] == [v["followup_question"]]
    assert replies[0] == replies[1]  # 无权和不存在不可区分


def test_identifier_the_model_made_up_is_never_executed(make_env):
    world = build_world(7)
    mine = order_of(world, "S1")
    env = make_env(DATA_EXC(), None, [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])  # 客户消息里根本没有
    ticket, cfg = env.start("我的订单到哪了")
    v = env.snap(cfg).values
    assert v["status"] == "need_info" and "订单号" in v["followup_question"]
    assert env.tb.gateway.audit(ticket_id=ticket.ticket_id, limit=50) and all(e["tool"] != "query_order" for e in env.tb.gateway.audit(ticket_id=ticket.ticket_id))


def test_no_applicable_tool_escalates_and_reviewer_executes_through_the_gateway(make_env):
    env = make_env(DATA_EXC(), None, ToolPlan(tool="none"))
    ticket, cfg = env.start("抖店9月结算账单金额怎么少了86块？")
    p = env.pending(cfg)[0]
    assert p["resolution"] == "escalate" and p["route_to"] == ["data_support_team"] and "没有合适的查询工具" in p["escalate_reason"]
    v = env.review(cfg, action="approve", reviewer=principal_to_dict(ADMIN), note="需要查账单")
    assert v["status"] == "escalated" and v["actions"][0]["tool"] == "escalate_ticket" and v["actions"][0]["ok"]
    assert env.store.get_ticket(ADMIN, ticket.ticket_id).status == "escalated"
    msgs = env.store.list_messages(A1, ticket.ticket_id)
    assert any(m.kind == "system" and "升级原因" in m.body and "需要查账单" in m.body for m in msgs)
    assert any(m.kind == "note" and "已完成动作" in m.body and "工具" in m.body for m in msgs)
    assert env.store.list_messages(C1, ticket.ticket_id) == []
    assert [e["decision"] for e in env.tb.gateway.audit(ticket_id=ticket.ticket_id) if e["tool"] == "escalate_ticket"][-1] == "executed"


def test_logistics_outage_degrades_to_a_fixed_reply_without_calling_the_draft_model(make_env):
    world = build_world(7)
    shipped = order_of(world, "S1", shipped=True)

    def down(oid):
        raise TimeoutError("carrier down")

    env = make_env(DATA_EXC(), None, [ToolPlan(tool="query_shipment", order_id=shipped.order_id), ToolPlan(tool="none")], transport=down)
    ticket, cfg = env.start(f"订单 {shipped.order_id} 物流到哪了")
    p = env.pending(cfg)[0]
    assert p["resolution"] == "auto_reply" and "暂时" in p["draft"] and env.draft.call_count == 0
    assert p["tool_run"]["outcome"] == "unavailable"


def test_approve_is_two_step_nothing_reaches_the_customer_until_confirmed(make_env):
    world = build_world(7)
    mine = order_of(world, "S1")
    env = make_env(DATA_EXC(), make_draft(reply="订单正在配送。", cited_ids=["1"]), [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])
    ticket, cfg = env.start(f"查订单 {mine.order_id}")
    env.store.claim(A1, ticket.ticket_id)
    v = env.review(cfg, action="approve", reviewer=principal_to_dict(A1))
    assert v["status"] == "awaiting_confirmation" and v["pending_confirmation"]["tool"] == "send_reply"
    assert env.store.list_messages(C1, ticket.ticket_id) == []  # 已批准但未确认：客户什么都没收到
    aid = v["pending_confirmation"]["approval_id"]
    assert env.tb.gateway.confirm(aid, A1).result["ticket_id"] == ticket.ticket_id
    again = [env.tb.gateway.confirm(aid, A1) for _ in range(3)]
    assert all(o.replayed for o in again)
    assert [m.body for m in env.store.list_messages(C1, ticket.ticket_id)] == ["订单正在配送。"]  # 恰好一条


def test_confirm_flag_sends_once_and_resolves_the_ticket_and_edit_uses_reviewer_text(make_env):
    world = build_world(7)
    mine = order_of(world, "S1")
    env = make_env(DATA_EXC(), make_draft(reply="草稿原文", cited_ids=["1"]), [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])
    ticket, cfg = env.start(f"查订单 {mine.order_id}")
    env.store.claim(A1, ticket.ticket_id)
    v = env.review(cfg, action="edit", reply="审核人改写后的回复", reviewer=principal_to_dict(A1), confirm=True)
    assert v["status"] == "resolved" and [a["tool"] for a in v["actions"]] == ["send_reply", "set_ticket_status"]
    assert [m.body for m in env.store.list_messages(C1, ticket.ticket_id)] == ["审核人改写后的回复"]
    assert env.store.get_ticket(A1, ticket.ticket_id).status == "resolved"


def test_reviewer_who_does_not_own_the_ticket_cannot_send(make_env):
    world = build_world(7)
    mine = order_of(world, "S1")
    env = make_env(DATA_EXC(), make_draft(cited_ids=["1"]), [ToolPlan(tool="query_order", order_id=mine.order_id), ToolPlan(tool="none")])
    ticket, cfg = env.start(f"查订单 {mine.order_id}")
    env.store.claim(A1, ticket.ticket_id)
    v = env.review(cfg, action="approve", reviewer=principal_to_dict(A2), confirm=True)  # a2 不是负责人
    assert v["status"] == "action_failed" and v["actions"][0]["code"] == "not_owner"
    assert env.store.list_messages(C1, ticket.ticket_id) == []


def test_review_without_reviewer_is_refused(make_env):
    env = make_env(DATA_EXC(), None, ToolPlan(tool="none"))
    ticket, cfg = env.start("账单对不上")
    with pytest.raises(ValueError, match="reviewer"):
        env.review(cfg, action="approve")


def test_reject_escalates_with_the_reviewers_note(make_env):
    env = make_env(make_triage(category="feature_howto", feature_name="库存同步"), make_draft(cited_ids=["1"]), hits=[make_hit()])
    ticket, cfg = env.start("库存同步怎么设置")
    v = env.review(cfg, action="reject", reviewer=principal_to_dict(ADMIN), note="客户情绪激动，转人工")
    assert v["status"] == "escalated"
    assert any("客户情绪激动" in m.body for m in env.store.list_messages(A1, ticket.ticket_id) if m.kind == "system")


def test_every_audit_row_for_a_run_is_traceable_to_actor_and_approver(make_env):
    env = make_env(DATA_EXC(), None, ToolPlan(tool="none"))
    ticket, cfg = env.start("账单金额不对")
    env.review(cfg, action="approve", reviewer=principal_to_dict(ADMIN))
    rows = env.tb.gateway.audit(ticket_id=ticket.ticket_id)
    assert any(r["decision"] == "executed" and r["approver_id"] == "admin1" and r["actor_id"] == "admin1" for r in rows)


# ---------------------------------------------------------------- 多轮：槽位继承 / 指代 / 检索词
def test_merge_slots_keeps_earlier_values_and_new_values_win():
    prev = {"platform_name": "京东", "shop_name": "小熊数码", "feature_name": None, "partner_name": None, "product_edition": None}
    new = {"platform_name": None, "shop_name": None, "feature_name": "库存同步", "partner_name": None, "product_edition": "极速版"}
    assert merge_slots(prev, new) == {"platform_name": "京东", "shop_name": "小熊数码", "feature_name": "库存同步",
                                      "partner_name": None, "product_edition": "极速版"}
    assert merge_slots(None, {"platform_name": "淘宝"})["platform_name"] == "淘宝"
    third = merge_slots(merge_slots(prev, new), {"platform_name": "抖店"})
    assert third["platform_name"] == "抖店" and third["shop_name"] == "小熊数码" and third["feature_name"] == "库存同步"


def test_three_turns_same_ticket_slots_do_not_get_lost(make_env):
    turn = lambda **kw: make_triage(category="platform_integration", **kw)  # noqa: E731
    env = make_env([turn(platform_name="京东", shop_name="小熊数码"), turn(feature_name="库存同步"), turn()],
                   [make_draft(cited_ids=["1"])] * 3, hits=[make_hit()])
    slots: dict = {}
    history: list[dict] = []
    for text in ("京东店铺小熊数码授权过期了", "它的库存同步也失败了", "还是不行"):
        ticket, cfg = env.start(text, history=history, slots=slots)
        slots = env.snap(cfg).values["slots"]
        history.append({"role": "customer", "text": text})
    assert slots["platform_name"] == "京东" and slots["shop_name"] == "小熊数码" and slots["feature_name"] == "库存同步"
    third_prompt = str(env.triage.calls[2][1].content)
    assert "【对话历史】" in third_prompt and "库存同步也失败了" in third_prompt  # 历史确实传给了分诊


def test_standalone_question_replaces_the_pronoun_in_the_retrieval_query(make_env):
    env = make_env(make_triage(category="feature_howto", feature_name="库存同步", standalone_question="库存同步失败后怎么重新同步"),
                   make_draft(cited_ids=["1"]), hits=[make_hit()])
    env.start("它失败了怎么办", history=[{"role": "customer", "text": "库存同步在哪设置"}])
    assert "库存同步失败后怎么重新同步" in env.retriever.calls[0][0] and "它失败了怎么办" not in env.retriever.calls[0][0]
    assert build_query({"standalone_question": None, "feature_name": None}, "原话") == "原话"


# ---------------------------------------------------------------- 产品线过滤 / 结构化引用
def _chunk(doc, top, title_key, text="x"):
    return Hit(Chunk(id=f"{doc}#0", title=title_key, text=f"{title_key}\n{text}", doc_id=doc,
                     meta={"top_dir": top, "title_key": title_key}), 1.0)


def test_normalize_edition():
    assert normalize_edition("我们用的极速版") == "极速版2.0" and normalize_edition("企业版") == "企业版"
    assert normalize_edition("旗舰版") is None and normalize_edition(None) is None


def test_edition_filter_drops_only_the_wrong_siblings():
    hits = [_chunk("企业版/订单", "企业版", "订单"), _chunk("极速版2.0/订单", "极速版2.0", "订单"),
            _chunk("企业版/只有企业版有", "企业版", "只有企业版有"), _chunk("物流对接/顺丰", "物流对接", "顺丰")]
    index = build_edition_index([h.chunk for h in hits])
    rows = [{"id": h.chunk.id, "top_dir": h.chunk.meta["top_dir"], "title_key": h.chunk.meta["title_key"]} for h in hits]
    kept = [r["id"] for r in filter_by_edition(rows, "极速版2.0", index)]
    assert "企业版/订单#0" not in kept  # 极速版里有同名文档：丢掉企业版那份
    assert "企业版/只有企业版有#0" in kept  # 极速版里没有对应文档：保留，不让客户拿不到答案
    assert "物流对接/顺丰#0" in kept and "极速版2.0/订单#0" in kept
    assert [r["id"] for r in filter_by_edition(rows, None, index)] == [r["id"] for r in rows]


def test_retrieve_widens_the_search_and_filters_when_the_customer_names_an_edition(make_env, tmp_path):
    hits = [_chunk("企业版/订单", "企业版", "订单"), _chunk("极速版2.0/订单", "极速版2.0", "订单")]
    retriever = FakeRetriever(hits)
    graph = build_graph(Settings(llm_api_key="t", llm_base_url="http://localhost:0/v1"), retriever, TicketSystem(), InMemorySaver(),
                        models=Models(ScriptedInvoker(make_triage(category="feature_howto", feature_name="订单", product_edition="极速版")),
                                      ScriptedInvoker(make_draft(cited_ids=["1"]))),
                        edition_index=build_edition_index([h.chunk for h in hits]))
    graph.invoke({"ticket_id": "T-1", "customer_message": "我们用极速版，订单怎么合并"}, {"configurable": {"thread_id": "e"}})
    assert retriever.calls[0][1] == 12  # 默认 4，客户明说产品线时放大 3 倍再过滤
    v = graph.get_state({"configurable": {"thread_id": "e"}}).values
    assert [h["id"] for h in v["hits"]] == ["极速版2.0/订单#0"]


def test_citation_details_have_document_section_edition_and_snippet():
    hit = {"id": "企业版/订单#2", "kind": "kb", "doc_id": "企业版/订单", "top_dir": "企业版",
           "text": "订单 > 合并 > 手动合并\n选中订单后点击合并。", "title": "x"}
    d = citation_details(["企业版/订单#2", "不存在"], [hit])
    assert len(d) == 1 and d[0]["doc_title"] == "订单" and d[0]["section"] == "合并 > 手动合并"
    assert d[0]["edition"] == "企业版" and d[0]["snippet"].startswith("选中订单") and d[0]["doc_id"] == "企业版/订单"


def test_vague_message_gets_the_generic_followup_even_when_a_slot_is_also_missing(make_env):
    # 回归（真实服务端到端发现）：「我的店铺有问题」被分成 platform_integration 且缺 platform_name，
    # 旧逻辑只追问"电商平台名称"，客户答了平台之后系统仍然不知道问题是什么
    env = make_env(make_triage(category="platform_integration", is_vague=True), make_draft(), hits=[])
    ticket, cfg = env.start("我的店铺有问题，帮我看看")
    v = env.snap(cfg).values
    assert v["missing_fields"] == ["platform_name"]  # 槽位确实缺
    assert "具体遇到的问题" in v["followup_question"] and "电商平台名称" not in v["followup_question"]
    assert env.retriever.calls == [] and env.draft.call_count == 0


def test_not_vague_but_unanswerable_still_asks_for_the_missing_slot(make_env):
    env = make_env(make_triage(category="platform_integration", is_vague=False), make_draft(answerable=False), hits=[make_hit()])
    ticket, cfg = env.start("授权过期了怎么办")
    assert "电商平台名称" in env.snap(cfg).values["followup_question"]
