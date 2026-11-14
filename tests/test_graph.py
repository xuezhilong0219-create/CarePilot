import os
import sqlite3
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from carepilot.config import ROOT
from fakes import RecordingTicketSystem, make_draft, make_hit, make_triage

CFG = {"configurable": {"thread_id": "t1"}}
INITIAL = {"ticket_id": "T-1", "customer_message": "京东店铺授权失效了，订单下不来"}


def _state(env, cfg=CFG):
    return env.graph.get_state(cfg)


def _interrupts(snap):
    return [i.value for task in snap.tasks for i in task.interrupts]


def _resume(env, decision, cfg=CFG):
    return env.graph.invoke(Command(resume=decision), cfg)


def test_1_missing_required_field_asks_without_retrieval(make_env):
    env = make_env(
        make_triage(category="platform_integration", severity="high"),
        make_draft(),
        hits=[make_hit()],
    )
    env.graph.invoke(INITIAL, CFG)

    snap = _state(env)
    assert snap.values["status"] == "need_info"
    assert snap.values["resolution"] == "need_info"
    assert "电商平台名称" in snap.values["followup_question"]
    assert env.retriever.calls == []
    assert env.draft.call_count == 0


def test_2_approve_sends_draft_and_resolves(make_env):
    draft = make_draft(reply="请按以下步骤重新授权：……")
    env = make_env(make_triage(feature_name="店铺授权"), draft, hits=[make_hit()])
    env.graph.invoke(INITIAL, CFG)
    assert _interrupts(_state(env)), "应在人工审核处暂停"

    _resume(env, {"action": "approve"})

    snap = _state(env)
    assert snap.values["status"] == "resolved"
    assert len(env.retriever.calls) == 1
    assert env.draft.call_count == 1
    assert env.system.replies == [{"ticket_id": "T-1", "text": draft.reply}]


def test_3_edit_sends_reviewer_text_not_draft(make_env):
    draft = make_draft(reply="模型草稿")
    env = make_env(make_triage(feature_name="店铺授权"), draft, hits=[make_hit()])
    env.graph.invoke(INITIAL, CFG)

    _resume(env, {"action": "edit", "reply": "审核人改写后的回复"})

    assert _state(env).values["status"] == "resolved"
    assert env.system.replies == [{"ticket_id": "T-1", "text": "审核人改写后的回复"}]
    assert env.system.replies[0]["text"] != draft.reply


def test_4_reject_escalates_without_reply(make_env):
    env = make_env(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    env.graph.invoke(INITIAL, CFG)

    _resume(env, {"action": "reject", "note": "转人工排查"})

    snap = _state(env)
    assert snap.values["status"] == "escalated"
    assert env.system.replies == []
    assert [e["team"] for e in env.system.escalations] == ["product_support_team"]
    assert env.system.escalations[0]["note"] == "转人工排查"


def test_5_empty_hits_escalates_without_calling_draft(make_env):
    env = make_env(make_triage(feature_name="店铺授权"), make_draft(), hits=[])
    env.graph.invoke(INITIAL, CFG)

    snap = _state(env)
    assert snap.values["resolution"] == "escalate"
    assert snap.values["escalate_reason"] == "知识库无相关资料"
    assert len(env.retriever.calls) == 1
    assert env.draft.call_count == 0

    _resume(env, {"action": "approve"})
    assert _state(env).values["status"] == "escalated"


def test_6_unanswerable_escalates(make_env):
    env = make_env(make_triage(feature_name="店铺授权"), make_draft(answerable=False), hits=[make_hit()])
    env.graph.invoke(INITIAL, CFG)

    snap = _state(env)
    assert snap.values["resolution"] == "escalate"
    assert snap.values["escalate_reason"] == "资料不足以可靠回答"
    assert env.draft.call_count == 1

    _resume(env, {"action": "approve"})
    snap = _state(env)
    assert snap.values["status"] == "escalated"
    assert env.system.replies == []


def test_7_citation_validation(make_env):
    # 引用全是编造的编号 → 拒绝放行
    bad = make_env(
        make_triage(feature_name="店铺授权"),
        make_draft(cited_ids=["made-up#9"]),
        hits=[make_hit("doc#0"), make_hit("doc#1")],
    )
    bad.graph.invoke(INITIAL, CFG)
    snap = _state(bad)
    assert snap.values["resolution"] == "escalate"
    assert snap.values["escalate_reason"] == "回复没有可校验的引用，拒绝放行"
    assert "draft" not in snap.values

    # 部分存在 → 只保留真实存在的编号
    partial = make_env(
        make_triage(feature_name="店铺授权"),
        make_draft(cited_ids=["doc#0", "made-up#9"]),
        hits=[make_hit("doc#0"), make_hit("doc#1")],
    )
    partial.graph.invoke(INITIAL, CFG)
    snap = _state(partial)
    assert snap.values["resolution"] == "auto_reply"
    assert snap.values["citations"] == ["doc#0"]
    assert _interrupts(snap)


def test_8_critical_adds_oncall_to_routing(make_env):
    env = make_env(
        make_triage(category="platform_integration", severity="critical", platform_name="淘宝", shop_name="小熊数码"),
        make_draft(answerable=False),
        hits=[make_hit()],
    )
    env.graph.invoke(INITIAL, CFG)

    snap = _state(env)
    assert snap.values["route_to"] == ["platform_integration_team", "oncall"]

    _resume(env, {"action": "approve"})
    snap = _state(env)
    assert snap.values["status"] == "escalated"
    assert [e["team"] for e in env.system.escalations] == ["platform_integration_team", "oncall"]


def test_9_shop_auth_needs_both_names(make_env):
    full = make_env(
        make_triage(category="platform_integration", platform_name="淘宝", shop_name="小熊数码"),
        make_draft(),
        hits=[make_hit()],
    )
    full.graph.invoke(INITIAL, CFG)
    snap = _state(full)
    assert snap.values["shop_status"] == {"platform_name": "淘宝", "shop_name": "小熊数码", "authorized": True}
    assert full.system.auth_checks == [("淘宝", "小熊数码")]

    # 缺 shop_name（不是必填字段）→ 流程继续，但不能查授权
    partial = make_env(
        make_triage(category="platform_integration", platform_name="淘宝"),
        make_draft(),
        hits=[make_hit()],
    )
    partial.graph.invoke(INITIAL, CFG)
    snap = _state(partial)
    assert "shop_status" not in snap.values
    assert partial.system.auth_checks == []
    assert _interrupts(snap)


def test_10_sqlite_checkpoint_resumes_in_new_graph(make_env):
    fd, name = tempfile.mkstemp(dir=ROOT, prefix=".tmp-carepilot-checkpoint-", suffix=".sqlite")
    os.close(fd)
    db = Path(name)
    cfg = {"configurable": {"thread_id": "persist"}}
    draft = make_draft(reply="跨实例恢复后发送的回复")

    conn1 = sqlite3.connect(db, check_same_thread=False)
    try:
        saver1 = SqliteSaver(conn1)
        saver1.setup()
        env1 = make_env(make_triage(feature_name="店铺授权"), draft, hits=[make_hit()], checkpointer=saver1)
        env1.graph.invoke(INITIAL, cfg)
        assert _interrupts(env1.graph.get_state(cfg))
    finally:
        conn1.close()

    # 同一个 sqlite 文件、全新的连接 / 图实例 / 工单系统
    conn2 = sqlite3.connect(db, check_same_thread=False)
    try:
        env2 = make_env(
            make_triage(feature_name="店铺授权"),
            draft,
            hits=[make_hit()],
            checkpointer=SqliteSaver(conn2),
            system=RecordingTicketSystem(),
        )
        env2.graph.invoke(Command(resume={"action": "approve"}), cfg)
        snap = env2.graph.get_state(cfg)
        assert snap.values["status"] == "resolved"
        assert env2.system.replies == [{"ticket_id": "T-1", "text": draft.reply}]
        # 恢复走的是 checkpoint，没有重新分类 / 检索
        assert env2.triage.call_count == 0
        assert env2.retriever.calls == []
    finally:
        conn2.close()
        db.unlink(missing_ok=True)


def test_11_resume_after_resolved_is_silent_noop(make_env):
    """固定当前行为：已 resolved 的 thread 再 resume 不会抛错、不重放节点、状态不变。"""
    env = make_env(make_triage(feature_name="店铺授权"), make_draft(), hits=[make_hit()])
    env.graph.invoke(INITIAL, CFG)
    _resume(env, {"action": "approve"})
    before = _state(env).values
    assert before["status"] == "resolved"
    replies_before = list(env.system.replies)

    result = _resume(env, {"action": "reject", "note": "迟到的驳回"})

    after = _state(env)
    assert result["status"] == "resolved"
    assert after.values["status"] == "resolved"
    assert after.values["resolution"] == "auto_reply"
    assert after.next == ()
    assert env.system.replies == replies_before
    assert env.system.escalations == []
