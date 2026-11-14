"""产品线过滤评测（A2「旧版本陷阱」）：客户明说自己用的版本时，回复的资料里会不会混进另一个产品线的同名文档。

不让模型出题：LLM 生成陷阱题的成功率太低（实测约一半输出为空或回显提示词），14 对同名文档里能生成的只有 3 对。
这里用模板问句，直接测"检索 + 产品线过滤"这一步——这才是命中旧文档的发生位置，也不引入生成噪声。

对每个企业版/极速版的同名配对、每个方向各一条问句（"我们用的是{版本}，{标题}怎么用"），分别在
  A. 不过滤（基线）  B. 开启产品线过滤（线上行为：客户消息经分诊后带 product_edition）
下检索，统计前 K 个 chunk 里出现"另一个产品线的同名文档"的比例（命中旧文档率）。

局限（都影响怎么引用这些数字）：
  - 问句是模板。第一版用"我们用的是{版本}，{标题}怎么用"，套话（"怎么用"、版本名都是高频词）把标题词稀释掉了，
    正确的同版本文档常排不进前 4（28 条里只有 12 条），所以"正确文档命中率"反映的是模板差，不是检索差。
    现在提供两种问句（--style title / chatty）并排报告。
  - "过滤后命中旧文档率 0"在构造上基本是保证的（判定和过滤是同一个条件），只能说明过滤器按定义执行了；
    有信息量的是"不过滤时的比例"（说明问题真实存在）和"过滤有没有把正确文档挤出去"。
  - 14 对是这个知识库里全部能配对的同名文档（已剔除版本日志和太短的），样本量本身就小。
  - 过滤的前提是"客户说出了版本"，没说版本时不过滤（也不应该猜）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carepilot.config import get_settings
from carepilot.evaluation.edition_traps import EDITION_LABEL, find_edition_pairs
from carepilot.evaluation.gen_core import load_manifest
from carepilot.graph.nodes import build_edition_index, filter_by_edition, hit_row, normalize_edition
from carepilot.retrieval.chunking import load_kb
from carepilot.retrieval.factory import build_retriever


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-dir", type=Path, default=Path("data/kb_real"))
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--k", type=int, default=4, help="回复节点实际看到的 chunk 数")
    ap.add_argument("--style", choices=["title", "chatty", "both"], default="both")
    ap.add_argument("--out", type=Path, default=Path("eval/real/reports/edition_filter.json"))
    args = ap.parse_args()

    entries = load_manifest(args.kb_dir)
    pairs = find_edition_pairs(entries)
    chunks = load_kb(args.kb_dir)
    index = build_edition_index(chunks)
    retriever = build_retriever(args.retriever, args.kb_dir, settings=get_settings() if args.retriever.endswith("llm") else None)

    styles = {"title": lambda label, title: f"{label} {title}", "chatty": lambda label, title: f"我们用的是{label}，{title}怎么用"}
    report = {}
    for style in (["title", "chatty"] if args.style == "both" else [args.style]):
        rows = []
        for a, b in pairs:
            for entry, other in ((a, b), (b, a)):
                edition_dir = normalize_edition(EDITION_LABEL[entry.top_dir])
                query = styles[style](EDITION_LABEL[entry.top_dir], entry.title)
                raw = [hit_row(h) for h in retriever.search(query, k=args.k * 3)]
                before = raw[: args.k]
                after = filter_by_edition(raw, edition_dir, index)[: args.k]

                def stale(hits: list[dict]) -> bool:  # 命中另一个产品线的同名文档
                    return any(h["top_dir"] == other.top_dir and h["title_key"] == entry.title_key for h in hits)

                def gold(hits: list[dict]) -> bool:
                    return any(h["doc_id"] == entry.doc_id for h in hits)

                rows.append({"title": entry.title, "edition": entry.top_dir, "stale_before": stale(before), "stale_after": stale(after),
                             "gold_before": gold(before), "gold_after": gold(after)})
        n = len(rows)
        f = lambda key: sum(r[key] for r in rows)  # noqa: E731
        lost = [r for r in rows if r["gold_before"] and not r["gold_after"]]
        print(f"\n[{style}] 同名配对 {len(pairs)} 对 → {n} 条问句，检索器 {args.retriever}，K={args.k}")
        print(f"  命中旧文档率   不过滤 {f('stale_before')}/{n} = {f('stale_before') / n:.1%}   过滤后 {f('stale_after')}/{n} = {f('stale_after') / n:.1%}")
        print(f"  本版本正确文档在前 {args.k}：不过滤 {f('gold_before')}/{n}，过滤后 {f('gold_after')}/{n}；过滤挤出正确文档 {len(lost)} 条")
        report[style] = rows
    rows = report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
