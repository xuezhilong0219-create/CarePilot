"""只跑分诊节点（flash 模型，便宜、快），看三件事：
  1. is_vague：对「可答样本」的误判率、对「手写含糊样本」的召回率；
  2. data_exception：对「需查客户数据的负样本」的召回、对「可答样本」的误伤；
  3. category 准确率（仅供参考：标签是按文档所在目录机械打的，噪声大，不要据此调提示词）。

dev/test 按 qid 哈希对半拆（和 tune_hybrid.py 同一个拆法）：改提示词只看 dev 的失败样本，在 test 上报告。
    uv run python eval/triage_only.py --name v1
"""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from carepilot.config import get_settings
from carepilot.evaluation.schema import EvalItem, load_dataset
from carepilot.graph import nodes
from carepilot.graph.state import Triage
from carepilot.llm import StructuredLLM, get_llm


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def pct(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.1%}" if den else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("eval/real/dataset.jsonl"))
    ap.add_argument("--vague", type=Path, default=Path("eval/real/dataset_vague.jsonl"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    settings = get_settings().model_copy(update={"llm_timeout": 120})
    llm = StructuredLLM(get_llm(settings, "fast"), Triage)
    main_items = load_dataset(args.dataset)
    vague_ids = set()
    items: list[EvalItem] = list(main_items)
    if args.vague.exists():
        vague = load_dataset(args.vague)
        vague_ids = {v.qid for v in vague}
        items += vague

    def run(item: EvalItem) -> dict:
        try:
            t = llm.invoke([SystemMessage(nodes.TRIAGE_PROMPT), HumanMessage(item.question)])
            return {"qid": item.qid, "triage": t.model_dump(), "error": None}
        except Exception as exc:
            return {"qid": item.qid, "triage": None, "error": type(exc).__name__}

    with ThreadPoolExecutor(args.concurrency) as ex:
        results = {r["qid"]: r for r in ex.map(run, items)}
    by_id = {i.qid: i for i in items}

    def view(split: str | None):
        def pick(pred):
            return [
                q for q, r in results.items()
                if r["triage"] and pred(by_id[q]) and (split is None or split_of(q) == split)
            ]
        ans = pick(lambda i: i.answerable and i.qid not in vague_ids)
        neg_data = pick(lambda i: not i.answerable and i.category == "data_exception")
        neg_other = pick(lambda i: not i.answerable and i.category == "other" and i.qid not in vague_ids)
        vag = pick(lambda i: i.qid in vague_ids)
        cat = lambda q: results[q]["triage"]["category"]
        vg = lambda q: results[q]["triage"]["is_vague"]
        return {
            "n": {"answerable": len(ans), "neg_data_exception": len(neg_data), "neg_other": len(neg_other), "vague": len(vag)},
            "vague_false_positive_on_answerable": pct(sum(vg(q) for q in ans), len(ans)),
            "vague_recall_on_handwritten": pct(sum(vg(q) for q in vag), len(vag)),
            "data_exception_recall_on_negatives": pct(sum(cat(q) == "data_exception" for q in neg_data), len(neg_data)),
            "data_exception_false_positive_on_answerable": pct(sum(cat(q) == "data_exception" for q in ans), len(ans)),
            "category_accuracy_noisy_labels": pct(sum(cat(q) == by_id[q].category for q in ans), len(ans)),
        }

    report = {"name": args.name, "errors": sum(1 for r in results.values() if r["error"]),
              "all": view(None), "dev": view("dev"), "test": view("test")}
    for part in ("all", "dev", "test"):
        print(f"\n[{part}] {report[part]['n']}")
        for k, v in report[part].items():
            if k != "n":
                print(f"   {k:46s} {v}")
    print(f"\n错误数: {report['errors']}")

    # dev 里的失败样本：这是允许据此改提示词的唯一依据
    fails = []
    for q, r in results.items():
        if not r["triage"] or split_of(q) != "dev":
            continue
        i, t = by_id[q], r["triage"]
        if i.answerable and q not in vague_ids and t["category"] == "data_exception":
            fails.append(("可答却判 data_exception", q, i.question))
        if i.answerable and q not in vague_ids and t["is_vague"]:
            fails.append(("可答却判笼统", q, i.question))
        if not i.answerable and i.category == "data_exception" and t["category"] != "data_exception":
            fails.append(("需查数据却没判 data_exception", q, i.question + f"  → 模型判 {t['category']}"))
        if q in vague_ids and not t["is_vague"]:
            fails.append(("含糊却没判笼统", q, i.question))
    print("\n--- dev 失败样本（只看这些来改提示词）：")
    for kind, q, text in fails:
        print(f"  [{kind}] {q}: {text[:90]}")

    out = Path("eval/real/reports") / f"triage_{args.name}.json"
    out.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
