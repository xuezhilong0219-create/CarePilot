"""混合检索的融合参数调优，dev/test 对半拆：只在 dev 上选参数，在 test 上报告，避免在同一批数据上调完再报数字。

拆分按 qid 的哈希，稳定、与样本顺序无关。目标函数是 chunk_hit@4（回复节点实际能看到的前 4 个 chunk），
并列时看 MRR。用法：
    uv run python eval/tune_hybrid.py --dataset eval/real/dataset.jsonl --kb-dir data/kb_real
"""

import argparse
import hashlib
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

from carepilot.evaluation.metrics import hit_curve, retrieval_metrics
from carepilot.evaluation.runners import _query_for
from carepilot.evaluation.schema import EvalItem, load_dataset
from carepilot.retrieval.base import Hit, Retriever
from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.chunking import load_kb
from carepilot.retrieval.embedding import FastEmbedEmbedder
from carepilot.retrieval.hybrid import HybridRetriever
from carepilot.retrieval.vector import VectorIndex, VectorRetriever

VECTOR_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
RRF_KS = (10, 30, 60)


class Memo:
    """缓存子检索器的结果：同一个 (query, k) 只检索一次，12 组配置共用。"""

    def __init__(self, inner: Retriever):
        self.inner = inner
        self.cache: dict[tuple[str, int], list[Hit]] = {}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        key = (query, k)
        if key not in self.cache:
            self.cache[key] = self.inner.search(query, k)
        return self.cache[key]


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def score(items: Sequence[EvalItem], retriever: Retriever, mode: str) -> dict[str, float]:
    rows = []
    for item in items:
        hits = retriever.search(_query_for(item, mode), k=10)
        m = retrieval_metrics(hits, item.gold_doc_ids, item.gold_title_keys, 10)
        m["curve"] = hit_curve(hits, item.gold_doc_ids, item.gold_title_keys)
        rows.append(m)
    n = len(rows)
    return {
        "chunk_hit@4": sum(r["curve"]["chunk_hit@4_strict"] for r in rows) / n,
        "hit@1": sum(r["curve"]["hit@1_strict"] for r in rows) / n,
        "hit@10": sum(r["curve"]["hit@10_strict"] for r in rows) / n,
        "mrr": sum(r["mrr_strict"] for r in rows) / n,
        "n": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--kb-dir", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/real/reports/tune_hybrid.json"))
    args = ap.parse_args()

    # 不可答样本没有标准答案，不参与检索指标
    items = [i for i in load_dataset(args.dataset) if i.gold_doc_ids or i.gold_title_keys]
    dev = [i for i in items if split_of(i.qid) == "dev"]
    test = [i for i in items if split_of(i.qid) == "test"]
    print(f"可答样本 {len(items)}：dev {len(dev)} / test {len(test)}")

    chunks = load_kb(args.kb_dir)
    embedder = FastEmbedEmbedder()
    index_dir = args.index_dir or Path("data/index") / args.kb_dir.name
    bm25 = Memo(BM25Retriever(chunks))
    vector = Memo(VectorRetriever(chunks, embedder, VectorIndex.load(index_dir, chunks, embedder.name)))

    results = []
    for wv, rk in itertools.product(VECTOR_WEIGHTS, RRF_KS):
        hybrid = HybridRetriever([bm25, vector], weights=[1.0, wv], rrf_k=rk)
        results.append({"vector_weight": wv, "rrf_k": rk, "dev_raw": score(dev, hybrid, "raw")})

    results.sort(key=lambda r: (-r["dev_raw"]["chunk_hit@4"], -r["dev_raw"]["mrr"]))
    best = results[0]
    default = next(r for r in results if r["vector_weight"] == 1.0 and r["rrf_k"] == 60)

    print("\n在 dev 上的排名（raw 检索词，目标 chunk_hit@4，并列看 MRR）：")
    print("vector_w  rrf_k  chunk_hit@4   MRR   hit@1  hit@10")
    for r in results:
        d = r["dev_raw"]
        mark = "  <- 选中" if r is best else ("  <- 默认" if r is default else "")
        print(f"  {r['vector_weight']:<6}  {r['rrf_k']:<5}  {d['chunk_hit@4']:.3f}      {d['mrr']:.3f}  {d['hit@1']:.3f}  {d['hit@10']:.3f}{mark}")

    bm25_only = score(test, bm25, "raw"), score(test, bm25, "triaged")
    print("\n在 test 上（调参时没见过这批样本）：")
    print("配置                          模式      chunk_hit@4   MRR   hit@1  hit@10")
    report = {"n_dev": len(dev), "n_test": len(test), "grid_dev": results, "test": {}}
    configs = {
        "bm25 单独": None,
        f"hybrid 默认(w=1,k=60)": (default["vector_weight"], default["rrf_k"]),
        f"hybrid 选中(w={best['vector_weight']},k={best['rrf_k']})": (best["vector_weight"], best["rrf_k"]),
    }
    for name, cfg in configs.items():
        for mi, mode in enumerate(("raw", "triaged")):
            if cfg is None:
                s = bm25_only[mi]
            else:
                s = score(test, HybridRetriever([bm25, vector], weights=[1.0, cfg[0]], rrf_k=cfg[1]), mode)
            report["test"][f"{name}/{mode}"] = s
            print(f"{name:<30}{mode:<9} {s['chunk_hit@4']:.3f}      {s['mrr']:.3f}  {s['hit@1']:.3f}  {s['hit@10']:.3f}")

    report["selected"] = {"vector_weight": best["vector_weight"], "rrf_k": best["rrf_k"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
