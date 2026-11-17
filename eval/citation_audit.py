"""引用抽样核对：把「回复里的具体信息，能在引用到的原文里找到吗」摆成一张表。

为什么单独一个脚本：`docs/status.md` 的 A1 一直是 ◐，因为「引用编号能还原成真实
chunk」是代码保证的（`resolve_citations` 只认 1..N），但「引用到的内容真的支撑了这句
回复」只能一条条看。这个脚本不判分——它把材料一次摆好：问题、回复、每个引用的标题/
章节/原文片段，人（或另一个判分器）照着读就行。

口径：只跑可答样本里等距抽的 N 条；只列 `resolution=auto_reply` 的（升级/追问没有引用
可核）。抽样可复现，不用随机数。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from carepilot.config import ROOT, get_settings
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.runners import run_pipeline
from carepilot.evaluation.sampling import even_sample
from carepilot.evaluation.schema import EvalItem, load_dataset
from carepilot.retrieval.base import Chunk
from carepilot.retrieval.chunking import load_kb

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"
DEFAULT_DATASET = ROOT / "eval" / "real" / "dataset_full.jsonl"
DEFAULT_KB = ROOT / "data" / "kb_real"
SNIPPET = 320


def split_breadcrumb(chunk: Chunk) -> tuple[str, str]:
    """chunk 首行是「文档 > 章节 > 小节」的面包屑；返回 (标题, 章节)。"""
    first, _, rest = chunk.text.partition("\n")
    parts = [p.strip() for p in first.split(">") if p.strip()]
    title = parts[0] if parts else chunk.title
    section = parts[-1] if len(parts) > 1 else ""
    return title, section


def pick(items: list[EvalItem], limit: int) -> list[EvalItem]:
    return even_sample([it for it in items if it.answerable], limit)


def collect(rows: list[dict[str, Any]], chunks: dict[str, Chunk]) -> list[dict[str, Any]]:
    """把每条自动回复连同它的引用原文摆出来；没有引用的（升级/追问）跳过。"""
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("error") or row.get("resolution") != "auto_reply":
            continue
        citations = list(row.get("citations") or [])
        if not citations:
            continue
        detail: list[dict[str, Any]] = []
        for cid in citations:
            chunk = chunks.get(cid)
            if chunk is None:  # 引用了知识库里不存在的 id，本身就是问题
                detail.append({"id": cid, "title": "", "section": "", "snippet": "", "missing": True})
                continue
            title, section = split_breadcrumb(chunk)
            body = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else chunk.text
            detail.append({"id": cid, "title": title, "section": section,
                           "snippet": body[:SNIPPET].strip(), "missing": False})
        out.append({
            "qid": row["qid"],
            "question": row["question"],
            # state["draft"] 存的是回复正文（不是 DraftResult 对象）
            "reply": (row.get("draft") or "").strip(),
            "cited_ids": citations,
            "citations": detail,
            "gold_doc_ids": row.get("gold_doc_ids") or [],
            "cited_doc_ids": row.get("cited_doc_ids") or [],
        })
    return out


def render(audit: list[dict[str, Any]]) -> str:
    """一条回复一块：读的人不需要来回翻 JSON。"""
    blocks: list[str] = []
    for i, item in enumerate(audit, 1):
        rows = [[c["id"], c["title"] or "(知识库里没有这个 id)", c["section"] or "—", c["snippet"]] for c in item["citations"]]
        blocks.append(
            f"### {i}. {item['qid']}\n\n"
            f"**问**：{item['question']}\n\n"
            f"**答**：{item['reply']}\n\n"
            + markdown_table(["引用", "文档", "章节", "原文片段"], rows)
        )
    head = f"共 {len(audit)} 条自动回复。判断标准：回复里的每个具体信息（数字、按钮名、路径、限制），是否能在它引用的片段里找到原文。"
    return head + "\n\n" + "\n\n".join(blocks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抽样核对回复里的引用是否支撑结论。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB)
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--limit", type=int, default=10, help="可答样本里等距抽多少条（0 = 全部）")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--name", default="citation_audit")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=None, help="把渲染好的 markdown 也写一份到这里")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    items = pick(load_dataset(args.dataset), args.limit)
    if not items:
        print("数据集里没有可答样本。")
        return 2

    from carepilot.retrieval.factory import build_retriever

    retriever = build_retriever(args.retriever, args.kb_dir, settings=settings)
    rows = run_pipeline(items, settings, retriever, concurrency=max(1, args.concurrency))
    chunks = {chunk.id: chunk for chunk in load_kb(args.kb_dir)}
    audit = collect(rows, chunks)
    body = render(audit)
    print(body)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")

    config = {"dataset": str(args.dataset), "kb_dir": str(args.kb_dir), "retriever": args.retriever,
              "model_main": settings.llm_main_model, "n_items": len(rows), "n_auto_reply": len(audit)}
    json_path, items_path = save_report(args.report_dir, args.name, {"audit": audit, "n": len(audit)}, rows, config)
    print(f"\n报告：{json_path}\n明细：{items_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
