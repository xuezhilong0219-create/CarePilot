"""上下文预算：确定性压缩的行为，以及它真的接进了三个节点。"""

from __future__ import annotations

from carepilot.graph.context import (
    DEFAULT_BUDGET_TOKENS,
    TURN_CHARS,
    compress_history,
    estimate_tokens,
)
from fakes import make_draft, make_hit, make_triage

CFG = {"configurable": {"thread_id": "ctx"}}


def turns(*texts: str, role: str = "customer") -> list[dict]:
    return [{"role": role, "text": t} for t in texts]


def test_estimate_tokens_counts_cjk_by_char_and_ascii_by_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("订单") == 2  # 一个汉字一个
    assert estimate_tokens("abcd") == 1  # 四个 ASCII 一个
    assert estimate_tokens("ab") == 1  # 向上取整，不返回 0.5


def test_empty_history_gives_empty_text_and_empty_report():
    view = compress_history([])

    assert view.text == ""
    assert view.report["turns"] == 0
    assert view.report["tokens_before"] == 0


def test_short_conversation_is_kept_verbatim():
    view = compress_history(turns("授权过期了", "哪个平台", "京东"))

    assert view.text.splitlines() == ["客户：授权过期了", "客户：哪个平台", "客户：京东"]
    assert view.report["verbatim"] == 3 and view.report["digested"] == 0


def test_long_turn_is_clipped_at_turn_chars_not_three_hundred():
    """回归：原来每条硬截 300 字，贴报错的工单后半段（往往才是错误码）直接被切掉。"""
    text = "导出订单时报错：" + "堆栈" * 300 + " 错误码 WDT-9981"
    view = compress_history([{"role": "customer", "text": text}])

    line = view.text
    assert line.endswith("…")
    assert len(line) <= len("客户：") + TURN_CHARS + 1
    assert TURN_CHARS > 300  # 这条断言就是这次改动的意义


def test_fitting_conversation_is_not_touched():
    """放得下就一个字都不动：压缩是有代价的，不该为短会话付这份代价。"""
    history = turns(*["客户描述了一整段现象，" * 10 for _ in range(6)])  # 约 1100 token，预算内

    view = compress_history(history)

    assert view.report["digested"] == 0 and view.report["dropped"] == 0
    assert len(view.text.splitlines()) == 6


def test_older_turns_become_one_line_digests_and_keep_order():
    # 每轮 500 字、7 轮，超预算 → 才会压
    history = turns(*[f"第{i}个问题。麻烦看下这个报错。" + "补充说明" * 120 for i in range(1, 8)])

    view = compress_history(history)
    lines = view.text.splitlines()

    assert len(lines) == 7
    assert view.report["verbatim"] == 3 and view.report["digested"] == 4
    assert lines[0] == "客户：第1个问题"  # 摘要只留第一句，但顺序不变
    assert lines[-1].startswith("客户：第7个问题。麻烦看下这个报错。")  # 最近的逐字，不做句子切分


def test_digests_never_invent_words():
    """压缩不能引入原文没有的措辞——每行摘要的正文都必须是某条原文的前缀。"""
    history = turns(*["京东店铺授权失效，订单下不来。已经重新授权过一次了" + "补充" * 20 for _ in range(8)])
    originals = [" ".join(h["text"].split()) for h in history]

    view = compress_history(history, budget_tokens=300)

    for line in view.text.splitlines():
        payload = line.split("：", 1)[1].rstrip("…")
        assert any(o.startswith(payload) for o in originals), line


def test_tight_budget_drops_oldest_but_keeps_recent_verbatim():
    history = turns(*[f"问题{i}：这是一段比较长的描述，占了预算" * 3 for i in range(1, 11)])

    view = compress_history(history, budget_tokens=40)

    lines = view.text.splitlines()
    assert view.report["dropped"] > 0
    assert len(lines) == 3  # 只剩最近三轮逐字
    assert lines[-1].startswith("客户：问题10")  # 最近一轮永远在
    assert view.report["over_budget"] is True  # 逐字保留是硬约束，宁可超预算


def test_customer_only_drops_agent_turns():
    history = [
        {"role": "customer", "text": "订单号 A-1 查一下"},
        {"role": "agent", "text": "好的，正在查"},
        {"role": "customer", "text": "还是没结果"},
    ]

    view = compress_history(history, customer_only=True)

    assert "正在查" not in view.text
    assert view.report["turns"] == 2


def test_long_conversation_actually_costs_less():
    """30 轮、每轮 500 字：压缩后应该只剩零头，而不是把 15000 字原样塞进提示词。"""
    history = turns(*[("这是一段很长的描述，客户把整个操作过程都贴上来了。" * 25) for _ in range(30)])

    view = compress_history(history)

    before, after = view.report["tokens_before"], view.report["tokens_after"]
    assert before > 10_000
    assert after < before * 0.2
    assert after <= DEFAULT_BUDGET_TOKENS * 1.2  # 允许最近三轮逐字带来的超出


# ---------------------------------------------------------------- 接进节点了吗


def _long_error() -> str:
    """总长不到 600 字，但错误码落在第 450 字之后——旧实现的 300 字截断刚好切掉它。"""
    return "导出订单时报错：" + "Traceback 堆栈信息 " * 30 + "错误码 WDT-9981"


def test_classify_sees_the_whole_error_dump(make_env):
    """分诊看得到 600 字那段里的错误码，说明 300 字硬截断确实被换掉了。"""
    env = make_env(make_triage(category="feature_howto"), make_draft(cited_ids=["1"]), hits=[])
    history = turns(_long_error())

    env.graph.invoke({"ticket_id": "T-1", "customer_message": "这个报错怎么办", "history": history}, CFG)

    prompt = env.triage.calls[0][-1].content
    assert "WDT-9981" in prompt
    assert "【对话历史】" in prompt


def test_draft_reply_receives_history_so_pronouns_resolve(make_env):
    """回复节点原来完全看不到历史，「这个」只能靠猜。"""
    env = make_env(make_triage(category="feature_howto"), make_draft(cited_ids=["1"]), hits=[make_hit()])
    history = turns("极速版导出订单报错 E1024", "还是不行")

    env.graph.invoke({"ticket_id": "T-1", "customer_message": "这个要怎么设置", "history": history}, CFG)

    prompt = env.draft.calls[0][-1].content
    assert "【对话历史】" in prompt
    assert "E1024" in prompt
    assert prompt.index("【资料】") < prompt.index("【对话历史】")  # 依据在前，历史在后


def test_single_turn_draft_prompt_is_untouched(make_env):
    """单轮对话（历史为空）的提示词必须和改动前一个字节都不差——评估基线就建在它上面。"""
    from carepilot.graph.nodes import DRAFT_PROMPT

    env = make_env(make_triage(category="feature_howto"), make_draft(cited_ids=["1"]), hits=[make_hit()])

    env.graph.invoke({"ticket_id": "T-1", "customer_message": "极速版怎么导出订单"}, CFG)

    system, human = (m.content for m in env.draft.calls[0])
    assert system == DRAFT_PROMPT
    assert "【对话历史】" not in human


def test_history_rule_appears_only_with_history(make_env):
    """历史非空时才追加那一条说明；分成独立常量，注入实验的「去掉一行防护」才仍然只去掉一行。"""
    from carepilot.graph.nodes import DRAFT_HISTORY_RULE, DRAFT_PROMPT

    env = make_env(make_triage(category="feature_howto"), make_draft(cited_ids=["1"]), hits=[make_hit()])

    env.graph.invoke(
        {"ticket_id": "T-1", "customer_message": "这个要怎么设置", "history": turns("导出报错 E1024")}, CFG
    )

    assert env.draft.calls[0][0].content == DRAFT_PROMPT + DRAFT_HISTORY_RULE
