"""对话历史的上下文预算：确定性压缩，不调用模型。

之前是固定「最近 6 轮、每条截 300 字」——两个问题：
1. 300 字截断对贴报错的工单太短，报错栈往往在第 400 个字才出现；
2. 6 轮是拍的，30 轮的会话和 6 轮的会话付一样的成本，而前者根本装不下。

为什么不用 LLM 摘要：
- 这条链路每一步的成本都被真实 usage 统计过，多一次调用就多一次不可复现的失败点；
- 摘要会引入原文没有的措辞，而本项目的引用校验、审计、交接摘要都靠「能逐字回溯」；
- 确定性压缩可以用单测钉死：输出的每一行都必须是某条原文的前缀。

预算用估算 token（CJK 一个字算一个，ASCII 四个字符算一个）。不引 tokenizer 依赖，
误差约 10%，够用来决定「留几轮」，不够用来报成本——成本另有真实 usage 统计。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

log = logging.getLogger(__name__)

#: 最近几轮必须逐字保留：客户的指代（「它」「这个」「还是不行」）都挂在最近这几轮上。
VERBATIM_RECENT = 3
#: 逐字轮的单轮上限。比原来的 300 字宽，因为报错原文经常在后半段。
TURN_CHARS = 600
#: 更早的轮次压成一行摘要，只留第一句。
DIGEST_CHARS = 60
#: 默认预算（估算 token）。三个逐字轮各 600 字约 1800，再留余量给更早轮次的摘要；
#: 预算比「最近几轮逐字」还小时，摘要会全部让位，report 里标 over_budget。
DEFAULT_BUDGET_TOKENS = 3000

_CJK = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
_SENTENCE_END = re.compile(r"[。！？!?\n]")


def estimate_tokens(text: str) -> int:
    """够用就好的 token 估算：中文按字，ASCII 按 4 字符。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def _who(role: str) -> str:
    return "客户" if role == "customer" else "客服"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _first_sentence(text: str) -> str:
    """摘要只取第一句：客户第一句通常就是问题本身，后面是补充和客套。"""
    flat = " ".join(text.split())
    m = _SENTENCE_END.search(flat)
    return (flat[: m.start()] if m else flat).strip() or flat


@dataclass(frozen=True)
class HistoryView:
    """压好的历史 + 一份可写进日志的账。"""

    text: str
    report: dict = field(default_factory=dict)


def compress_history(
    history: Sequence[Mapping[str, str]],
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    verbatim_recent: int = VERBATIM_RECENT,
    customer_only: bool = False,
) -> HistoryView:
    """把历史压进预算：能原样放下就原样放，放不下才压。

    压缩是有代价的（更早的轮次只剩一行摘要），所以只在真的超预算时才动手：
    短会话一个字都不动；长会话里最近几轮仍逐字保留（「它」「这个」都挂在最近这几轮上），
    更早的每轮压成一行摘要，还装不下就从最旧的开始丢，report 里记 ``dropped``。
    最近几轮是硬保留的——即使它们自己就超了预算，也不丢，只在 report 里标 ``over_budget``。
    """
    rows = [(str(h.get("role") or ""), str(h.get("text") or "")) for h in history]
    if customer_only:
        rows = [r for r in rows if r[0] == "customer"]
    if not rows:
        return HistoryView("", {"turns": 0, "verbatim": 0, "digested": 0, "dropped": 0,
                                "tokens_before": 0, "tokens_after": 0, "over_budget": False})

    before = sum(estimate_tokens(t) for _, t in rows)
    full = [f"{_who(role)}：{_clip(text, TURN_CHARS)}" for role, text in rows]
    if estimate_tokens("\n".join(full)) <= budget_tokens:
        text = "\n".join(full)
        return HistoryView(text, {"turns": len(rows), "verbatim": len(rows), "digested": 0, "dropped": 0,
                                  "tokens_before": before, "tokens_after": estimate_tokens(text),
                                  "over_budget": False})

    recent, older = rows[-verbatim_recent:], rows[:-verbatim_recent]
    lines = [f"{_who(role)}：{_clip(text, TURN_CHARS)}" for role, text in recent]
    budget = budget_tokens - sum(estimate_tokens(line) for line in lines)

    digests: list[str] = []
    dropped = 0
    for role, text in reversed(older):  # 从新往旧装，装不下的自然是最旧的
        line = f"{_who(role)}：{_clip(_first_sentence(text), DIGEST_CHARS)}"
        cost = estimate_tokens(line)
        if cost > budget:
            dropped += 1
            continue
        budget -= cost
        digests.append(line)
    digests.reverse()

    text = "\n".join(digests + lines)
    after = estimate_tokens(text)
    return HistoryView(text, {
        "turns": len(rows),
        "verbatim": len(recent),
        "digested": len(digests),
        "dropped": dropped,
        "tokens_before": before,
        "tokens_after": after,
        "over_budget": after > budget_tokens,
    })


def log_history_view(view: HistoryView, *, node: str, customer_only: bool = False) -> None:
    """一行日志，方便从服务端日志里核对「这轮到底塞了多少上下文」。"""
    r = view.report
    if r.get("turns"):
        log.info("context[%s%s]: 轮次=%d 逐字=%d 摘要=%d 丢弃=%d token %d→%d%s",
                 node, "/仅客户" if customer_only else "", r["turns"], r["verbatim"], r["digested"],
                 r["dropped"], r["tokens_before"], r["tokens_after"], "（超预算）" if r["over_budget"] else "")
