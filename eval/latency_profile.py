"""延迟账：一次真实工单的时间到底花在哪儿。

为什么单独一个脚本：README 里写过一句「真实链路的一秒级延迟来自模型，不在网关」，
但网关那半句量过（`eval/bench_gateway.py`），模型那半句一直是推断。这里把它变成数字。

做法：走和线上同一条流水线（`evaluation.runners.run_pipeline`：分诊 → 检索 → 起草），
计的是节点自己的墙钟，不是估算。**串行跑**（默认 `--concurrency 1`）：并发会让模型侧
排队，测出来的延迟会被自己污染——要测的是「一次要多久」，不是「并行能压多少」。

口径：只跑可答样本（`answerable`），因为不可答的样本根本不会走到起草节点，
混进来会让「起草占多少」这个比例失真。取法是等距抽样，可复现，不用随机数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from carepilot.config import ROOT, get_settings
from carepilot.evaluation.metrics import percentile
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.runners import run_pipeline
from carepilot.evaluation.schema import EvalItem, load_dataset

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"
DEFAULT_DATASET = ROOT / "eval" / "real" / "dataset_full.jsonl"
DEFAULT_KB = ROOT / "data" / "kb_real"

# 节点名 → 表里显示的名字。顺序就是流水线的顺序，表也跟着这个顺序读。
NODES: dict[str, str] = {
    "classify_ms": "分诊 classify",
    "retrieve_ms": "检索 retrieve",
    "draft_ms": "起草 draft_reply",
    "ask_for_info_ms": "追问 ask_for_info",
}
TOTAL = "total_ms"
# 这些节点的耗时里包含模型调用，剩下的是本地计算。
LLM_NODES = ("classify_ms", "draft_ms", "ask_for_info_ms")


def pick(items: list[EvalItem], limit: int) -> list[EvalItem]:
    """只在可答样本里等距抽 limit 条。等距而不是随机会让报告可复现。"""
    pool = [it for it in items if it.answerable]
    if limit <= 0 or limit >= len(pool):
        return pool
    step = len(pool) / limit
    return [pool[int(i * step)] for i in range(limit)]


def _bucket(label: str, values: list[float]) -> dict[str, Any]:
    return {
        "label": label,
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按节点汇总：p50 / p95 / 均值 / 最大，样本数各自算（不是每条都走所有节点）。"""
    per_node: dict[str, Any] = {}
    for key, label in NODES.items():
        vals = [r["timings_ms"][key] for r in rows if key in (r.get("timings_ms") or {})]
        if vals:
            per_node[key] = {
                "label": label,
                "n": len(vals),
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "mean": sum(vals) / len(vals),
                "max": max(vals),
            }
    totals = [r["timings_ms"][TOTAL] for r in rows if TOTAL in (r.get("timings_ms") or {})]
    per_node[TOTAL] = {
        "label": "整单 total",
        "n": len(totals),
        "p50": percentile(totals, 50),
        "p95": percentile(totals, 95),
        "mean": sum(totals) / len(totals) if totals else 0.0,
        "max": max(totals) if totals else 0.0,
    }
    # 模型 vs 本地：报绝对毫秒，不报百分比。百分比在两头都是「100.0%」或「0.0%」，
    # 看着像算错了，也说不出「差多少」——这里要回答的正是差多少。
    llm_ms, local_ms = [], []
    for row in rows:
        t = row.get("timings_ms") or {}
        if not t.get(TOTAL):
            continue
        llm = sum(t.get(k, 0.0) for k in LLM_NODES)
        llm_ms.append(llm)
        local_ms.append(max(0.0, t[TOTAL] - llm))  # 实际不可能为负，防浮点抖动
    per_node["llm_ms"] = _bucket("模型调用合计", llm_ms)
    per_node["local_ms"] = _bucket("本地计算（检索/渲染/记账）", local_ms)
    return per_node


def _ms(value: float) -> str:
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def render(summary: dict[str, Any], n_items: int) -> str:
    rows: list[list[Any]] = []
    for key in [*NODES, TOTAL]:
        if key not in summary:
            continue
        s = summary[key]
        rows.append([s["label"], s["n"], _ms(s["p50"]), _ms(s["p95"]), _ms(s["mean"]), _ms(s["max"])])
    for key in ("llm_ms", "local_ms"):
        if key not in summary:
            continue
        s = summary[key]
        rows.append([s["label"], s["n"], _ms(s["p50"]), _ms(s["p95"]), _ms(s["mean"]), _ms(s["max"])])
    return markdown_table(["节点", "样本", "p50", "p95", "均值", "最大"], rows) + f"\n\n共 {n_items} 条工单。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按节点拆一次真实工单的耗时。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB)
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--limit", type=int, default=40, help="可答样本里等距抽多少条（0 = 全部）")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="默认 1：并发会让模型侧排队，测出来的就不是单次延迟了",
    )
    parser.add_argument(
        "--from-items",
        type=Path,
        default=None,
        help="不跑模型，直接从上次的 .items.jsonl 重新汇总（换了统计口径时用这个）",
    )
    parser.add_argument("--name", default="latency")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.from_items:
        rows = [json.loads(line) for line in args.from_items.read_text(encoding="utf-8").splitlines() if line.strip()]
        config = {"from_items": str(args.from_items)}
    else:
        items = pick(load_dataset(args.dataset), args.limit)
        if not items:
            print("数据集里没有可答样本。")
            return 2

        from carepilot.retrieval.factory import build_retriever

        retriever = build_retriever(args.retriever, args.kb_dir, settings=settings)
        rows = run_pipeline(items, settings, retriever, concurrency=max(1, args.concurrency))
        config = {
            "dataset": str(args.dataset),
            "kb_dir": str(args.kb_dir),
            "retriever": args.retriever,
            "concurrency": max(1, args.concurrency),
            "model_fast": settings.llm_fast_model,
            "model_main": settings.llm_main_model,
        }

    summary = summarize(rows)
    print(render(summary, len(rows)))

    json_path, items_path = save_report(
        args.report_dir,
        args.name,
        {"nodes": summary, "n_items": len(rows)},
        rows,
        config,
    )
    print(f"\n报告：{json_path.name} / {items_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
