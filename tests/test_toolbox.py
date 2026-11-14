"""工具箱 + 网关 + 工单存储 + 沙箱的集成测试：全部用真实实现，不用 mock。"""

import pytest

from carepilot.auth import Principal, Role
from carepilot.gateway.spec import Executed, Gate, NeedsApproval, Rejected
from carepilot.graph.state import ToolPlan
from carepilot.sandbox.data import build_world
from carepilot.sandbox.logistics import LogisticsClient
from carepilot.sandbox import tools as sbx
from carepilot.store import TicketStore, build_handoff, seed_demo_users
from carepilot.toolbox import KNOWN_TEAMS, SYSTEM_PRINCIPAL, Toolbox

C1 = Principal("c1", Role.CUSTOMER, frozenset({"S1", "S2"}))
C2 = Principal("c2", Role.CUSTOMER, frozenset({"S3"}))
A1 = Principal("a1", Role.AGENT)
A2 = Principal("a2", Role.AGENT)
ADMIN = Principal("admin1", Role.ADMIN)


class Env:
    def __init__(self, tmp_path, transport=None):
        self.world = build_world(7)
        self.store = TicketStore(tmp_path / "tickets.sqlite")
        seed_demo_users(self.store)
        self.transport = transport or (lambda oid: sbx.query_shipment(self.world, oid))
        self.logistics = LogisticsClient(lambda oid: self.transport(oid), timeout_s=0.3, retries=1, sleep=lambda s: None)
        self.tb = Toolbox(world=self.world, store=self.store, logistics=self.logistics, db_path=tmp_path / "gw.sqlite")

    def order_of(self, shop_id, *, shipped=None):
        for o in self.world.orders.values():
            if o.shop_id == shop_id and (shipped is None or (o.order_id in self.world.shipments) == shipped):
                return o
        raise AssertionError("no such order")

    def ticket(self, customer=C1):
        return self.store.create_ticket(customer, subject="库存同步失败", shop_id="S1", thread_id="th-1")


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def plan(tool, **kw):
    return ToolPlan(tool=tool, **kw)


def test_customer_reads_own_order_and_other_shops_order_looks_nonexistent(env):
    mine, theirs = env.order_of("S1"), env.order_of("S3")
    ok = env.tb.run_plan(C1, "r1", None, plan("query_order", order_id=mine.order_id), source_text=f"帮我查订单 {mine.order_id}")
    assert ok.outcome == "ok" and mine.order_id in ok.facts_text and ok.source_id == f"tool:query_order:{mine.order_id}"
    other = env.tb.run_plan(C1, "r1", None, plan("query_order", order_id=theirs.order_id), source_text=theirs.order_id)
    ghost = env.tb.run_plan(C1, "r1", None, plan("query_order", order_id="9" * 12), source_text="9" * 12)
    assert other.outcome == ghost.outcome == "not_found" and other.data == ghost.data  # 不泄露订单是否存在


def test_facts_never_contain_remark_or_plain_phone(env):
    for o in list(env.world.orders.values()):
        r = env.tb.run_plan(ADMIN, "r2", None, plan("query_order", order_id=o.order_id), source_text=o.order_id)
        assert o.buyer_phone not in r.facts_text and (len(o.remark) < 6 or o.remark not in r.facts_text)
        assert o.buyer_phone not in str(r.data)


def test_provenance_check_rejects_identifier_not_in_customer_message(env):
    mine = env.order_of("S1")
    r = env.tb.run_plan(C1, "r3", None, plan("query_order", order_id=mine.order_id), source_text="我的订单到哪了")
    assert r.outcome == "invalid" and r.detail == "order_id"  # 模型凭空给出的订单号不执行
    assert env.tb.gateway.audit(run_id="r3") == []  # 根本没有走到网关
    none = env.tb.run_plan(C1, "r3", None, plan("query_order"), source_text="订单没了")
    assert none.outcome == "invalid"
    assert env.tb.run_plan(C1, "r3", None, plan("none"), source_text="x").outcome == "none"


def test_malformed_identifier_is_rejected_by_the_schema_gate(env):
    r = env.tb.run_plan(C1, "r4", None, plan("query_order", order_id="12 34;DROP"), source_text="12 34;DROP")
    assert r.outcome == "invalid" and r.detail == "format"
    assert [e["gate"] for e in env.tb.gateway.audit(run_id="r4")] == ["schema"]


def test_shipment_degrades_instead_of_raising_when_carrier_is_down(tmp_path):
    def down(order_id):
        raise TimeoutError("carrier api timeout")

    e = Env(tmp_path, transport=down)
    o = e.order_of("S1", shipped=True)
    r = e.tb.run_plan(C1, "r5", None, plan("query_shipment", order_id=o.order_id), source_text=o.order_id)
    assert r.outcome == "unavailable" and "暂时" in r.facts_text and r.source_id.endswith(":unavailable")


def test_shipment_ok_unshipped_and_foreign_orders(env):
    shipped, unshipped = env.order_of("S1", shipped=True), env.order_of("S1", shipped=False)
    ok = env.tb.run_plan(C1, "r6", None, plan("query_shipment", order_id=shipped.order_id), source_text=shipped.order_id)
    assert ok.outcome == "ok" and "承运商" in ok.facts_text
    assert env.tb.run_plan(C1, "r6", None, plan("query_shipment", order_id=unshipped.order_id), source_text=unshipped.order_id).outcome == "not_found"
    foreign = env.order_of("S3", shipped=True)
    assert env.tb.run_plan(C1, "r6", None, plan("query_shipment", order_id=foreign.order_id), source_text=foreign.order_id).outcome == "not_found"


def test_shop_auth_is_scoped_to_the_callers_shops(env):
    assert env.tb.run_plan(C1, "r7", None, plan("query_shop_auth", shop_id="S1"), source_text="S1 授权").outcome == "ok"
    assert env.tb.run_plan(C1, "r7", None, plan("query_shop_auth", shop_id="S3"), source_text="S3 授权").outcome == "not_found"
    assert env.tb.run_plan(A1, "r7-staff", None, plan("query_shop_auth", shop_id="S3"), source_text="S3").outcome == "ok"


def test_query_ticket_customer_sees_less_and_cannot_read_others(env):
    t = env.ticket(C1)
    mine = env.tb.run_plan(C1, "r8", None, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    assert mine.outcome == "ok" and "SLA" not in mine.facts_text and "owner_id" not in mine.data
    staff = env.tb.run_plan(A1, "r8-staff", None, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    assert staff.outcome == "ok" and "SLA 剩余" in staff.facts_text and "owner_id" in staff.data
    denied = env.tb.run_plan(C2, "r8-c2", None, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    assert denied.outcome == "denied" and denied.detail == "not_owner"  # B1：越权访问 403 口径


def test_customer_cannot_use_write_tools_and_non_owner_agent_cannot_send(env):
    t = env.ticket(C1)
    out = env.tb.gateway.call(__ctx(C1, "r9", t), "send_reply", {"ticket_id": t.ticket_id, "text": "我自己回复"})
    assert isinstance(out, Rejected) and out.gate is Gate.PERMISSION and out.code == "role_denied"
    env.store.claim(A1, t.ticket_id)
    out = env.tb.gateway.call(__ctx(A2, "r9", t), "send_reply", {"ticket_id": t.ticket_id, "text": "抢答"})
    assert isinstance(out, Rejected) and out.code == "not_owner"
    assert env.store.list_messages(C1, t.ticket_id) == []


def __ctx(principal, run_id, ticket):
    from carepilot.gateway.spec import ToolContext

    return ToolContext(principal, run_id, ticket.ticket_id)


def test_status_change_needs_approval_and_illegal_transitions_are_blocked(env):
    t = env.ticket(C1)
    env.store.claim(A1, t.ticket_id)
    env.store.transition(A1, t.ticket_id, "open")
    out = env.tb.gateway.call(__ctx(A1, "r10", t), "set_ticket_status", {"ticket_id": t.ticket_id, "to_status": "on_hold"})
    assert isinstance(out, NeedsApproval) and env.store.get_ticket(A1, t.ticket_id).status == "open"  # 批准前不执行
    done = env.tb.gateway.approve(out.approval_id, A1)
    assert isinstance(done, Executed) and env.store.get_ticket(A1, t.ticket_id).status == "on_hold"
    bad = env.tb.gateway.call(__ctx(A1, "r10", t), "set_ticket_status", {"ticket_id": t.ticket_id, "to_status": "closed"})
    assert isinstance(bad, Rejected) and bad.gate is Gate.RULE and bad.code == "illegal_transition"


def test_send_reply_requires_approval_then_confirmation_and_is_idempotent(env):
    t = env.ticket(C1)
    env.store.claim(A1, t.ticket_id)
    out = env.tb.propose_and_approve(A1, A1, "r11", t.ticket_id, "send_reply", {"ticket_id": t.ticket_id, "text": "已为您处理"})
    assert isinstance(out, NeedsApproval) and out.stage == "confirm"
    assert env.store.list_messages(C1, t.ticket_id) == []  # 只批准、未确认：客户什么都收不到
    first = env.tb.gateway.confirm(out.approval_id, A1)
    assert isinstance(first, Executed) and not first.replayed
    for _ in range(3):  # 重复确认
        again = env.tb.gateway.confirm(out.approval_id, A1)
        assert isinstance(again, Executed) and again.replayed
    assert [m.body for m in env.store.list_messages(C1, t.ticket_id)] == ["已为您处理"]  # 只有一条消息
    assert isinstance(env.tb.gateway.confirm(out.approval_id, C1), Rejected)  # 客户不能确认


def test_escalate_writes_reason_and_actions_for_the_next_agent(env):
    t = env.ticket(C1)
    env.store.claim(A1, t.ticket_id)
    out = env.tb.propose_and_approve(A1, A1, "r12", t.ticket_id, "escalate_ticket",
                                     {"ticket_id": t.ticket_id, "team": "data_support_team",
                                      "reason": "需要查询客户的具体数据", "actions_summary": "已查询订单：未找到"})
    assert isinstance(out, Executed) and env.store.get_ticket(A1, t.ticket_id).status == "escalated"
    env.store.release(A1, t.ticket_id)
    env.store.claim(A2, t.ticket_id)  # 下一个坐席接续
    msgs = env.store.list_messages(A2, t.ticket_id)
    h = build_handoff(env.store.get_ticket(A2, t.ticket_id), msgs, [])
    assert "需要查询客户的具体数据" in h.escalation_reason and h.owner_id == "a2"
    assert any("已查询订单：未找到" in m.body for m in msgs)
    assert all(m.visibility == "internal" for m in msgs)
    assert env.store.list_messages(C1, t.ticket_id) == []  # 客户看不到任何内部内容


def test_unknown_team_is_rejected_by_the_rule_gate(env):
    t = env.ticket(C1)
    env.store.claim(A1, t.ticket_id)
    out = env.tb.gateway.call(__ctx(A1, "r13", t), "escalate_ticket",
                              {"ticket_id": t.ticket_id, "team": "made_up_team", "reason": "x"})
    assert isinstance(out, Rejected) and out.gate is Gate.RULE and out.code == "unknown_team"


def test_draft_note_is_internal_and_idempotent_per_run(env):
    t = env.ticket(C1)
    args = {"ticket_id": t.ticket_id, "body": "草稿：请重新授权", "kind": "draft"}
    a = env.tb.gateway.call(__ctx(SYSTEM_PRINCIPAL, "r14", t), "add_note", args)
    b = env.tb.gateway.call(__ctx(SYSTEM_PRINCIPAL, "r14", t), "add_note", args)
    assert isinstance(a, Executed) and isinstance(b, Executed) and b.replayed
    assert env.store.list_messages(C1, t.ticket_id) == []
    assert len([m for m in env.store.list_messages(A1, t.ticket_id) if m.kind == "draft"]) == 1


def test_audit_records_every_attempt_including_rejections(env):
    t = env.ticket(C1)
    env.tb.gateway.call(__ctx(C1, "r15", t), "send_reply", {"ticket_id": t.ticket_id, "text": "x"})
    env.tb.run_plan(C1, "r15", t.ticket_id, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    rows = env.tb.gateway.audit(run_id="r15")
    assert [(r["tool"], r["decision"]) for r in rows] == [("send_reply", "rejected"), ("query_ticket", "approved"), ("query_ticket", "executed")]


def test_every_team_used_by_the_graph_is_a_known_team():
    from carepilot.graph.nodes import TEAM_BY_CATEGORY

    assert set(TEAM_BY_CATEGORY.values()) | {"oncall"} <= KNOWN_TEAMS


def test_same_run_cannot_replay_another_principals_cached_result(env):
    # 回归：同一 run 里坐席曾拿到客户那次查询的缓存结果
    t = env.ticket(C1)
    env.tb.run_plan(C1, "shared", None, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    leaked = env.tb.run_plan(A1, "shared", None, plan("query_ticket", ticket_id=t.ticket_id), source_text=t.ticket_id)
    assert leaked.outcome == "error" and leaked.detail == "principal_mismatch" and leaked.data == {}


def test_shop_auth_by_name_resolves_inside_the_callers_scope_only(env):
    mine = next(s for s in env.world.shops.values() if s.shop_id == "S1")
    foreign = next(s for s in env.world.shops.values() if s.shop_id == "S3")
    ok = env.tb.run_plan(C1, "n1", None, plan("query_shop_auth", shop_name=mine.name), source_text=f"帮我查下 {mine.name} 的授权")
    assert ok.outcome == "ok" and mine.name in ok.facts_text
    other = env.tb.run_plan(C1, "n2", None, plan("query_shop_auth", shop_name=foreign.name), source_text=foreign.name)
    ghost = env.tb.run_plan(C1, "n3", None, plan("query_shop_auth", shop_name="根本没有的店"), source_text="根本没有的店")
    assert other.outcome == ghost.outcome == "not_found" and other.data == ghost.data  # 别家店铺和不存在不可区分
    made_up = env.tb.run_plan(C1, "n4", None, plan("query_shop_auth", shop_name=mine.name), source_text="我的店授权怎么样")
    assert made_up.outcome == "invalid"  # 名称不是客户说过的：不执行
    assert env.tb.run_plan(A1, "n5", None, plan("query_shop_auth", shop_name=foreign.name), source_text=foreign.name).outcome == "ok"
    assert env.tb.resolve_shop(C1, foreign.name) is None and env.tb.resolve_shop(A1, foreign.name) == "S3"
