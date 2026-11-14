"""完整流水线离线评估：分诊 → 检索 → 起草，不走人工审核 interrupt。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from carepilot.config import ROOT, get_settings
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.runners import run_pipeline, summarize_pipeline
from carepilot.evaluation.schema import load_dataset

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在评估集上跑完整工单流水线。")
    parser.add_argument("--dataset", type=Path, required=True, help="评估集 jsonl")
    parser.add_argument("--kb-dir", type=Path, required=True, help="知识库目录")
    parser.add_argument("--retriever", default="bm25", help="检索器种类，见 retrieval.factory")
    parser.add_argument("--index-dir", type=Path, default=None, help="向量索引目录")
    parser.add_argument("--exclude-dirs", default="", help="按 top_dir 排除，逗号分隔")
    parser.add_argument("--exclude-flags", default="", help="按 manifest flags 排除，逗号分隔")
    parser.add_argument("--triage-model", default=None, help="覆盖 llm_fast_model")
    parser.add_argument("--draft-model", default=None, help="覆盖 llm_main_model")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--name", default=None, help="实验名，写进报告文件名")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def _build_retriever(args: argparse.Namespace, settings):
    # 延迟 import：factory 由别的 worker 实现，--help 不依赖它
    from carepilot.retrieval.factory import build_retriever

    return build_retriever(
        args.retriever,
        args.kb_dir,
        index_dir=args.index_dir,
        exclude_dirs=_split(args.exclude_dirs),
        exclude_flags=_split(args.exclude_flags),
        settings=settings,
    )


def _print_summary(summary: dict[str, Any]) -> None:
    triage, reply = summary["triage"], summary["reply"]
    rows = [
        ["category accuracy", triage["category_accuracy"]],
        ["severity accuracy", triage["severity_accuracy"]],
        ["severity within-1", triage["severity_within_one_accuracy"]],
        ["entity platform_name", triage["entity_accuracy"]["platform_name"]],
        ["entity feature_name", triage["entity_accuracy"]["feature_name"]],
        ["entity partner_name", triage["entity_accuracy"]["partner_name"]],
        ["error_text accuracy", triage["error_text_accuracy"]],
        ["answer rate", reply["answer_rate"]],
        ["citation hit rate", reply["citation_hit_rate"]],
        ["correct refusal rate", reply["correct_refusal_rate"]],
        ["wrong auto rate", reply["wrong_auto_rate"]],
        ["latency total mean (ms)", summary["latency_ms"]["total_mean"]],
        ["errors", f"{summary['errors']} / {summary['n']}"],
    ]
    print(markdown_table(["metric", "value"], rows))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    settings = get_settings()
    if args.triage_model:
        settings.llm_fast_model = args.triage_model
    if args.draft_model:
        settings.llm_main_model = args.draft_model
    retriever = _build_retriever(args, settings)

    records = run_pipeline(items, settings, retriever, concurrency=args.concurrency)
    summary = summarize_pipeline(items, records)
    name = args.name or f"pipeline_{args.retriever}"
    config = {
        "dataset": str(args.dataset),
        "kb_dir": str(args.kb_dir),
        "retriever": args.retriever,
        "index_dir": str(args.index_dir) if args.index_dir else None,
        "exclude_dirs": _split(args.exclude_dirs),
        "exclude_flags": _split(args.exclude_flags),
        "triage_model": settings.llm_fast_model,
        "draft_model": settings.llm_main_model,
        "limit": args.limit,
        "concurrency": args.concurrency,
    }
    json_path, items_path = save_report(args.report_dir, name, summary, records, config)

    print(f"\n## {name}  ({len(items)} 条样本)")
    _print_summary(summary)
    print(f"\nreport: {json_path}\nitems:  {items_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
