"""首字延迟：流式起草到底能救回多少白屏时间。

延迟账（`eval/latency_profile.py`）量到起草这一步 p50 3.01 s、整单 p50 6.01 s，这段时间里
客户在 SSE 上什么也看不到。直觉是流式输出，但**结构化输出拿不到增量**：
`with_structured_output(DraftResult)` 必须等整个 JSON 解析完才能给出对象，实测 `.stream()`
只有一个 chunk、6 秒后才到。所以流式的前提是把起草契约改成「自由文本 + 代码里解析 [n] 标记」。

这个脚本先给两个数字，再决定要不要动契约：
  1. 自由文本流式的首字 vs 整段（prompt 由 `nodes.draft_messages` 拼，和线上同一份）；
  2. 流出来的自由文本里能不能解析出合法引用——改成自由文本后，`answerable` 只能由它判断。

不写结论：数字不够就说明别改。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

from carepilot.config import ROOT, get_settings
from carepilot.evaluation.metrics import percentile
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.runners import _query_for  # noqa: PLC2701  同一个口径，别另抄一份
from carepilot.evaluation.sampling import even_sample
from carepilot.evaluation.schema import EvalItem, load_dataset
from carepilot.graph.nodes import draft_messages, make_nodes, resolve_citations
from carepilot.graph.routing import after_classify
from carepilot.llm import get_llm
from carepilot.tools import TicketSystem

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"
DEFAULT_DATASET = ROOT / "eval" / "real" / "dataset_full.jsonl"
DEFAULT_KB = ROOT / "data" / "kb_real"

# 模型在正文里的引用写法就是 [1]~[N]，和 DRAFT_PROMPT 要求的一致。
_REF = re.compile(r"\[(\d{1,2})\]")


def pick(items: list[EvalItem], limit: int) -> list[EvalItem]:
    """只在可答样本里抽。不可答的样本走不到起草节点。"""
    return even_sample([it for it in items if it.answerable], limit)


def citations_in(text: str, hits: list[dict]) -> list[str]:
    """从自由文本里抠出 [n]，再用和线上同一个函数校验能不能还原成真实 chunk。"""
    refs = [f"[{n}]" for n in (int(m) for m in _REF.findall(text))]
    return resolve_citations(refs, [h["id"] for h in hits])


def _ms(value: float) -> str:
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    total = [r["total_ms"] for r in rows if r.get("total_ms") is not None]
    with_cites = [r for r in rows if r.get("citations")]
    return {
        "n": len(rows),
        "ttft": {"n": len(ttft), "p50": percentile(ttft, 50), "p95": percentile(ttft, 95),
                 "max": max(ttft) if ttft else 0.0},
        "total": {"n": len(total), "p50": percentile(total, 50), "p95": percentile(total, 95),
                  "max": max(total) if total else 0.0},
        "with_citations": len(with_cites),
        "parseable_rate": len(with_cites) / len(rows) if rows else 0.0,
    }


def render(summary: dict[str, Any]) -> str:
    ttft, total = summary["ttft"], summary["total"]
    rows = [
        ["首字（第一个非空 chunk）", ttft["n"], _ms(ttft["p50"]), _ms(ttft["p95"]), _ms(ttft["max"])],
        ["整段（流结束）", total["n"], _ms(total["p50"]), _ms(total["p95"]), _ms(total["max"])],
    ]
    head = markdown_table(["量的是哪一段", "样本", "p50", "p95", "最大"], rows)
    rate = f"{summary['parseable_rate']:.3f}（{summary['with_citations']}/{summary['n']}）"
    return head + f"\n\n自由文本里能解析出**合法引用**的比例：{rate}。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="量起草这一步的首字延迟，判断流式值不值。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB)
    parser.add_argument("--limit", type=int, default=8, help="可答样本里等距抽多少条（0 = 全部）")
    parser.add_argument("--max-tokens", type=int, default=512, help="自由文本没有 schema 约束，给个上限防跑飞")
    parser.add_argument("--name", default="ttft")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    items = pick(load_dataset(args.dataset), args.limit)
    if not items:
        print("数据集里没有可答样本。")
        return 2

    from carepilot.retrieval.factory import build_retriever

    retriever = build_retriever("bm25", args.kb_dir, settings=settings)
    nodes = make_nodes(settings, retriever, TicketSystem())
    model = get_llm(settings, "main")

    rows: list[dict[str, Any]] = []
    for item in items:
        state: dict[str, Any] = {"ticket_id": item.qid, "customer_message": item.question}
        state.update(nodes["classify"](state))
        if after_classify(state) != "retrieve":
            continue
        state.update(nodes["retrieve"](state))
        hits = state.get("hits") or []
        if not hits:
            continue

        messages = draft_messages(hits, item.question)
        started = time.perf_counter()
        first_at = None
        parts: list[str] = []
        for chunk in model.stream(messages, max_tokens=args.max_tokens):
            text = chunk.content
            if first_at is None and text:
                first_at = (time.perf_counter() - started) * 1000
            parts.append(text if isinstance(text, str) else str(text))
        total_ms = (time.perf_counter() - started) * 1000
        body = "".join(parts)
        cited = citations_in(body, hits)
        rows.append({
            "qid": item.qid,
            "question": item.question,
            "query": _query_for(item, "triaged"),
            "ttft_ms": first_at,
            "total_ms": total_ms,
            "chars": len(body),
            "citations": cited,
            "reply": body[:800],
        })
        print(f"{item.qid} 首字 {_ms(first_at or 0.0)} / 整段 {_ms(total_ms)} / 引用 {len(cited)}")

    if not rows:
        print("这一批没有走到起草节点，没法量首字。")
        return 2

    summary = summarize(rows)
    print()
    print(render(summary))
    json_path, items_path = save_report(
        args.report_dir,
        args.name,
        summary,
        rows,
        {"dataset": str(args.dataset), "retriever": "bm25", "limit": args.limit,
         "model_main": settings.llm_main_model, "max_tokens": args.max_tokens},
    )
    print(f"\n报告：{json_path.name} / {items_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
