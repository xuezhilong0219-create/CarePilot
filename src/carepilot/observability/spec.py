"""节点级 trace 的数据结构（冻结接口）。一次 run = 一份 JSON，可在 HTML 查看器里按节点展开。

约定：
- 摘要（input_summary / output_summary）是**截断后的文本**（最多 300 字符），不放完整正文，更不放敏感信息明文；
  调用方负责在写入前脱敏，这里只做长度截断。
- token 数只用网关响应里的 usage，拿不到就记 0 并在 LlmCall.usage_missing=True 标出，**不估算**。
- 单价不在这里：成本是 trace 之外、按用户配置的单价算出来的（见 observability/cost.py）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SUMMARY_MAX = 300


def clip(text: str | None, limit: int = SUMMARY_MAX) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class LlmCall:
    purpose: str  # triage / draft / rewrite / agent / judge ...
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    usage_missing: bool = False


@dataclass
class Span:
    name: str  # 图的节点名，如 classify / retrieve / draft_reply
    started_at: float
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    llm_calls: list[LlmCall] = field(default_factory=list)
    retries: int = 0
    error: str | None = None


@dataclass
class RunTrace:
    run_id: str
    ticket_id: str | None = None
    category: str | None = None  # 分诊意图，成本按它分组
    started_at: float = 0.0
    ended_at: float = 0.0
    status: str = "running"  # running / resolved / escalated / need_info / error / budget_exceeded
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunTrace":
        spans = [
            Span(**{**s, "llm_calls": [LlmCall(**c) for c in s.get("llm_calls", [])]}) for s in data.get("spans", [])
        ]
        return cls(**{**data, "spans": spans})

    @property
    def llm_calls(self) -> list[LlmCall]:
        return [c for s in self.spans for c in s.llm_calls]

    @property
    def total_tokens(self) -> tuple[int, int]:
        calls = self.llm_calls
        return sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls)
