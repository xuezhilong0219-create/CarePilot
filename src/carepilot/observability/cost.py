"""成本统计（D2）：token 来自 trace，单价只能来自用户配置，缺单价就只报 token。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .spec import RunTrace

_MTOKEN = 1_000_000
_UNPRICED = "未配置单价：仅统计 token"


class PricingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtoken: float | None = None
    output_per_mtoken: float | None = None

    @property
    def priced(self) -> bool:
        # 只配了一半也算没配：缺一半就算不出完整成本，不能拿 0 顶替。
        return self.input_per_mtoken is not None and self.output_per_mtoken is not None


@dataclass(frozen=True)
class Pricing:
    models: dict[str, ModelPrice] = field(default_factory=dict)
    currency: str = ""
    source: str = ""
    as_of: str = ""


def _opt_float(value: Any, model: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PricingError(f"{model} 的单价不是数字: {value!r}") from exc
    if number < 0:
        raise PricingError(f"{model} 的单价不能为负: {number}")
    return number


def load_pricing(path: Path) -> Pricing:
    """文件不存在视为全部未配置；任何模型给了单价就必须有 source 和 as_of。"""
    file = Path(path)
    if not file.exists():
        return Pricing()
    data = json.loads(file.read_text(encoding="utf-8"))
    source = str(data.get("source") or "")
    as_of = str(data.get("as_of") or "")
    models: dict[str, ModelPrice] = {}
    for model, cfg in (data.get("models") or {}).items():
        cfg = cfg or {}
        inp = _opt_float(cfg.get("input_per_mtoken"), model)
        out = _opt_float(cfg.get("output_per_mtoken"), model)
        if (inp is not None or out is not None) and (not source or not as_of):
            raise PricingError(f"模型 {model} 给了单价，但缺少 source 或 as_of")
        models[model] = ModelPrice(inp, out)
    return Pricing(models=models, currency=str(data.get("currency") or ""), source=source, as_of=as_of)


def model_cost(pricing: Pricing, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = pricing.models.get(model)
    if price is None or not price.priced:
        return None
    return (prompt_tokens * price.input_per_mtoken + completion_tokens * price.output_per_mtoken) / _MTOKEN


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _cost_text(cost: float | None, currency: str) -> str:
    if cost is None:
        return _UNPRICED
    return f"{cost:.4f} {currency}".strip()


@dataclass
class CostReport:
    runs: int
    calls_by_purpose: dict[str, int]
    prompt_tokens: int
    completion_tokens: int
    usage_missing_calls: int
    per_ticket: list[dict[str, Any]]
    per_category: list[dict[str, Any]]
    per_model: list[dict[str, Any]]
    currency: str = ""
    source: str = ""
    as_of: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        from ..evaluation.report import markdown_table

        total_calls = sum(self.calls_by_purpose.values())
        lines = ["## 成本统计", ""]
        lines.append(
            f"运行 {self.runs} 次，LLM 调用 {total_calls} 次；"
            f"prompt {self.prompt_tokens} tokens，completion {self.completion_tokens} tokens。"
        )
        if self.source and self.as_of:
            lines.append(f"单价来源：{self.source}（{self.as_of}）")
        else:
            lines.append(_UNPRICED)
        if self.usage_missing_calls:
            lines.append(
                f"> ⚠ token 统计不完整：{self.usage_missing_calls} 次调用缺少 usage，按 0 计入（未估算）。"
            )
        lines.append("")

        lines.append("### 按 purpose")
        lines.append(markdown_table(["purpose", "调用次数"], sorted(self.calls_by_purpose.items())))
        lines.append("")

        lines.append("### 按工单")
        lines.append(
            markdown_table(
                ["run_id", "ticket_id", "category", "prompt", "completion", "成本"],
                [
                    [
                        row["run_id"],
                        row.get("ticket_id") or "-",
                        row.get("category") or "unknown",
                        row["prompt_tokens"],
                        row["completion_tokens"],
                        _cost_text(row["cost"], self.currency),
                    ]
                    for row in self.per_ticket
                ],
            )
        )
        lines.append("")

        lines.append("### 按意图")
        lines.append(
            markdown_table(
                ["category", "run 数", "平均 token", "平均成本"],
                [
                    [
                        row["category"],
                        row["runs"],
                        round(row["avg_total_tokens"], 2),
                        _cost_text(row["avg_cost"], self.currency),
                    ]
                    for row in self.per_category
                ],
            )
        )
        lines.append("")

        lines.append("### 按模型")
        lines.append(
            markdown_table(
                ["model", "调用次数", "prompt", "completion"],
                [
                    [row["model"], row["calls"], row["prompt_tokens"], row["completion_tokens"]]
                    for row in self.per_model
                ],
            )
        )
        return "\n".join(lines)


def summarize_cost(traces: Sequence[RunTrace], pricing: Pricing) -> CostReport:
    traces = list(traces)
    calls_by_purpose: Counter[str] = Counter()
    prompt_total = completion_total = usage_missing = 0
    per_ticket: list[dict[str, Any]] = []
    per_model: dict[str, dict[str, Any]] = {}
    cat_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "prompt": 0, "completion": 0, "costs": []}
    )

    for trace in traces:
        calls = trace.llm_calls
        trace_prompt = sum(c.prompt_tokens for c in calls)
        trace_completion = sum(c.completion_tokens for c in calls)
        prompt_total += trace_prompt
        completion_total += trace_completion
        usage_missing += sum(int(c.usage_missing) for c in calls)

        call_costs: list[float | None] = []
        for call in calls:
            calls_by_purpose[call.purpose] += 1
            cost = model_cost(pricing, call.model, call.prompt_tokens, call.completion_tokens)
            call_costs.append(cost)
            model_row = per_model.setdefault(
                call.model, {"model": call.model, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            model_row["calls"] += 1
            model_row["prompt_tokens"] += call.prompt_tokens
            model_row["completion_tokens"] += call.completion_tokens

        # 任一调用缺单价就整单不给成本，避免用 0 假装便宜。
        ticket_cost = None if any(c is None for c in call_costs) else sum(call_costs)  # type: ignore[arg-type]
        total_tokens = trace_prompt + trace_completion
        category = trace.category or "unknown"
        per_ticket.append(
            {
                "run_id": trace.run_id,
                "ticket_id": trace.ticket_id,
                "category": category,
                "prompt_tokens": trace_prompt,
                "completion_tokens": trace_completion,
                "total_tokens": total_tokens,
                "cost": ticket_cost,
            }
        )

        acc = cat_acc[category]
        acc["runs"] += 1
        acc["prompt"] += trace_prompt
        acc["completion"] += trace_completion
        acc["costs"].append(ticket_cost)

    per_category: list[dict[str, Any]] = []
    for category, acc in cat_acc.items():
        runs = acc["runs"]
        priced = [c for c in acc["costs"] if c is not None]
        per_category.append(
            {
                "category": category,
                "runs": runs,
                "avg_prompt_tokens": _ratio(acc["prompt"], runs),
                "avg_completion_tokens": _ratio(acc["completion"], runs),
                "avg_total_tokens": _ratio(acc["prompt"] + acc["completion"], runs),
                "avg_cost": _ratio(sum(priced), len(priced)) if priced else None,
            }
        )
    per_category.sort(key=lambda row: row["category"])

    return CostReport(
        runs=len(traces),
        calls_by_purpose=dict(sorted(calls_by_purpose.items())),
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        usage_missing_calls=usage_missing,
        per_ticket=per_ticket,
        per_category=per_category,
        per_model=sorted(per_model.values(), key=lambda row: row["model"]),
        currency=pricing.currency,
        source=pricing.source,
        as_of=pricing.as_of,
    )
