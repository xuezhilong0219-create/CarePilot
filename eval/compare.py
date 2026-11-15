"""并排比较多份评估报告，相对第一份报告显示差值。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from carepilot.evaluation.report import load_report, markdown_table

# 两种报告的核心指标，按展示顺序排列；缺失的指标显示 "-"
CORE_KEYS = [
    "n",
    "hit@k_strict",
    "hit@k_lenient",
    "mrr_strict",
    "mrr_lenient",
    "recall@k_strict",
    "recall@k_lenient",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p90",
    "errors",
    "error_rate",
    "triage.category_accuracy",
    "triage.severity_accuracy",
    "triage.severity_within_one_accuracy",
    "triage.entity_accuracy.platform_name",
    "triage.entity_accuracy.feature_name",
    "triage.entity_accuracy.partner_name",
    "triage.error_text_accuracy",
    "reply.answer_rate",
    "reply.citation_hit_rate",
    "reply.correct_refusal_rate",
    "reply.wrong_auto_rate",
    "latency_ms.classify_mean",
    "latency_ms.retrieve_mean",
    "latency_ms.draft_mean",
    "latency_ms.total_mean",
    "latency_ms.total_p90",
]


def _get(summary: dict[str, Any], dotted: str) -> Any:
    value: Any = summary
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _cell(value: Any, baseline: Any, first: bool) -> str:
    if value is None:
        return "-"
    if first or baseline is None or not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return _fmt(value)
    delta = value - baseline
    sign = "+" if delta >= 0 else ""
    return f"{_fmt(value)} ({sign}{delta:.3f})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并排比较评估报告，差值相对第一份。")
    parser.add_argument("reports", type=Path, nargs="+", help="报告 json 文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [load_report(path) for path in args.reports]
    names = [report.get("name") or Path(path).stem for report, path in zip(reports, args.reports, strict=True)]
    summaries = [report.get("summary") or {} for report in reports]

    keys = [key for key in CORE_KEYS if any(_get(summary, key) is not None for summary in summaries)]
    rows: list[list[Any]] = []
    for key in keys:
        values = [_get(summary, key) for summary in summaries]
        baseline = values[0]
        rows.append([key] + [_cell(value, baseline, i == 0) for i, value in enumerate(values)])

    print(markdown_table(["metric"] + names, rows))
    print(f"\n差值基准: {names[0]}  ({args.reports[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
