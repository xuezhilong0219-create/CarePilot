"""把 run_pipeline 的逐条结果按「消息类型」汇总最终去向。总体平均数会把不同类型的好坏互相抵消，分组才看得出问题在哪。

四组及各自的期望：
  可答       有标准答案            期望 auto_reply，且引用命中标准答案文档
  需查数据   data_exception 负样本   期望 escalate（知识库答不了，不能自动回复）
  无关       闲聊/产品之外          期望 escalate 或 need_info，不能自动回复
  含糊       手写的笼统消息 v*/w*    期望 need_info（追问）

用法：
    uv run python eval/outcomes.py --dataset eval/real/dataset_full.jsonl eval/real/reports/*_ds_before.items.jsonl eval/real/reports/*_ds_after.items.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from carepilot.evaluation.schema import EvalItem, load_dataset

ORDER = ["可答", "需查数据", "无关", "含糊"]
OUTCOMES = ["auto_reply", "need_info", "escalate", "ERROR"]


def group_of(item: EvalItem) -> str:
    if item.qid[0] in "vw":
        return "含糊"
    if item.answerable:
        return "可答"
    return "需查数据" if item.category == "data_exception" else "无关"


def outcome_of(row: dict) -> str:
    return "ERROR" if row.get("error") else row.get("resolution") or "ERROR"


def citation_hit(row: dict, item: EvalItem) -> bool:
    return bool(set(row.get("cited_doc_ids") or []) & set(item.gold_doc_ids)) or bool(
        set(row.get("cited_title_keys") or []) & set(item.gold_title_keys)
    )


def summarize(path: Path, items: dict[str, EvalItem]) -> dict:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    out: dict = {}
    for g in ORDER:
        grp = [(r, items[r["qid"]]) for r in rows if r["qid"] in items and group_of(items[r["qid"]]) == g]
        c = Counter(outcome_of(r) for r, _ in grp)
        entry = {"n": len(grp), **{k: c.get(k, 0) for k in OUTCOMES}}
        if g == "可答":
            entry["solved"] = sum(1 for r, i in grp if outcome_of(r) == "auto_reply" and citation_hit(r, i))
        out[g] = entry
    out["_latency_ms_mean"] = sum(r["timings_ms"].get("total_ms", 0) for r in rows) / max(1, len(rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("reports", nargs="+", type=Path)
    args = ap.parse_args()
    items = {i.qid: i for i in load_dataset(args.dataset)}

    for path in args.reports:
        s = summarize(path, items)
        print(f"\n### {path.name.replace('.items.jsonl', '')}   平均单条耗时 {s['_latency_ms_mean'] / 1000:.1f}s")
        print("| 类型 | 条数 | auto_reply | need_info | escalate | ERROR | 期望的去向占比 |")
        print("|---|---|---|---|---|---|---|")
        for g in ORDER:
            e = s[g]
            n = max(1, e["n"])
            if g == "可答":
                good = f"解决(放行且引用命中) {e['solved']}/{e['n']} = {e['solved'] / n:.0%}"
            elif g == "需查数据":
                good = f"升级 {e['escalate']}/{e['n']} = {e['escalate'] / n:.0%}"
            elif g == "无关":
                ok = e["escalate"] + e["need_info"]
                good = f"未自动回复 {ok}/{e['n']} = {ok / n:.0%}"
            else:
                good = f"追问 {e['need_info']}/{e['n']} = {e['need_info'] / n:.0%}"
            print(f"| {g} | {e['n']} | {e['auto_reply']} | {e['need_info']} | {e['escalate']} | {e['ERROR']} | {good} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
