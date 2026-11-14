"""检索离线评估：同一份数据集上对比不同检索器 / 检索词模式。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from carepilot.config import ROOT
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.runners import run_retrieval, summarize_retrieval
from carepilot.evaluation.schema import load_dataset

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在评估集上跑检索并输出 hit@k / MRR / recall。")
    parser.add_argument("--dataset", type=Path, required=True, help="评估集 jsonl")
    parser.add_argument("--kb-dir", type=Path, required=True, help="知识库目录")
    parser.add_argument("--retriever", default="bm25", help="检索器种类，见 retrieval.factory")
    parser.add_argument("--index-dir", type=Path, default=None, help="向量索引目录")
    parser.add_argument("--exclude-dirs", default="", help="按 top_dir 排除，逗号分隔")
    parser.add_argument("--exclude-flags", default="", help="按 manifest flags 排除，逗号分隔")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-mode", choices=["raw", "triaged", "both"], default="raw")
    parser.add_argument("--name", default=None, help="实验名，写进报告文件名")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def _build_retriever(args: argparse.Namespace):
    # 延迟 import：factory 由别的 worker 实现，单测和 --help 不依赖它
    from carepilot.retrieval.factory import build_retriever

    return build_retriever(
        args.retriever,
        args.kb_dir,
        index_dir=args.index_dir,
        exclude_dirs=_split(args.exclude_dirs),
        exclude_flags=_split(args.exclude_flags),
    )


def _print_summary(summary: dict[str, Any]) -> None:
    cutoffs = sorted({int(k.split("@")[1].split("_")[0]) for k in summary if k.startswith("hit@") and k[4].isdigit()})
    rows = []
    for mode in ("strict", "lenient"):
        rows.append([f"hit@N ({mode})", *[summary.get(f"hit@{n}_{mode}", "-") for n in cutoffs]])
    chunk = sorted({int(k.split("@")[1].split("_")[0]) for k in summary if k.startswith("chunk_hit@")})
    print(markdown_table(["metric", *[f"N={n} 篇" for n in cutoffs]], rows))
    print()
    rows = [[f"chunk_hit@{n} ({mode})", summary.get(f"chunk_hit@{n}_{mode}", "-")] for n in chunk for mode in ("strict", "lenient")]
    rows += [
        ["MRR (strict)", summary["mrr_strict"]],
        ["MRR (lenient)", summary["mrr_lenient"]],
        ["latency p50 (ms)", summary["latency_ms_p50"]],
        ["latency p90 (ms)", summary["latency_ms_p90"]],
        ["samples", summary["n"]],
    ]
    print(markdown_table(["metric", "value"], rows))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    retriever = _build_retriever(args)

    modes = ["raw", "triaged"] if args.query_mode == "both" else [args.query_mode]
    for mode in modes:
        records = run_retrieval(items, retriever, k=args.k, query_mode=mode)
        summary = summarize_retrieval(records, args.k)
        # both 模式下两份报告必须有不同的名字，否则同一秒写入时后一份会覆盖前一份
        name = f"{args.name}_{mode}" if args.name and len(modes) > 1 else (args.name or f"retrieval_{args.retriever}_{mode}")
        config = {
            "dataset": str(args.dataset),
            "kb_dir": str(args.kb_dir),
            "retriever": args.retriever,
            "index_dir": str(args.index_dir) if args.index_dir else None,
            "exclude_dirs": _split(args.exclude_dirs),
            "exclude_flags": _split(args.exclude_flags),
            "k": args.k,
            "query_mode": mode,
            "limit": args.limit,
        }
        json_path, items_path = save_report(args.report_dir, name, summary, records, config)
        print(f"\n## {name}  ({len(items)} 条样本, mode={mode})")
        _print_summary(summary)
        print(f"\nreport: {json_path}\nitems:  {items_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
