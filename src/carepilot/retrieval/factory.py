from collections.abc import Sequence
from pathlib import Path

from carepilot.config import Settings

from .base import Retriever
from .bm25 import BM25Retriever
from .chunking import load_kb
from .embedding import FastEmbedEmbedder
from .hybrid import HybridRetriever
from .rerank import CrossEncoderReranker, LLMReranker, RerankedRetriever, make_llm_score_fn
from .vector import VectorIndex, VectorRetriever

# 向量路的 RRF 权重（BM25 为 1.0）。在 203 条可答样本的 dev 半边上网格搜索选出，test 半边验证（eval/tune_hybrid.py）：
# 等权重时 MRR / hit@1 明显低于 BM25 单独（test hit@1 0.554 对 0.674），降到 0.25 后补回，且 chunk_hit@4 不低于 BM25。
# 注意 test 只有 92 条，chunk_hit@4 的提升（+4.4 个点）在统计上不显著，不要把它当成"混合检索更好"的证据。
HYBRID_VECTOR_WEIGHT = 0.25

RETRIEVER_KINDS = ("bm25", "vector", "hybrid", "hybrid_rerank_ce", "hybrid_rerank_llm")


def build_retriever(
    kind: str,
    kb_dir: Path,
    *,
    index_dir: Path | None = None,
    exclude_dirs: Sequence[str] = (),
    exclude_flags: Sequence[str] = (),
    settings: Settings | None = None,
) -> Retriever:
    if kind not in RETRIEVER_KINDS:
        raise ValueError(f"Unknown retriever kind {kind!r}; choose from {RETRIEVER_KINDS}")
    if kind == "hybrid_rerank_llm" and settings is None:
        raise ValueError("settings is required for hybrid_rerank_llm")
    chunks = load_kb(kb_dir, exclude_dirs, exclude_flags)
    if kind == "bm25":
        return BM25Retriever(chunks)
    embedder = FastEmbedEmbedder()
    index_dir = Path(index_dir) if index_dir is not None else Path("data/index") / Path(kb_dir).name
    if (index_dir / "embeddings.npy").exists() or (index_dir / "chunk_ids.json").exists():
        index = VectorIndex.load(index_dir, chunks, embedder.name)
    else:
        index = VectorIndex.build(chunks, embedder)
        index.save(index_dir)
    vector = VectorRetriever(chunks, embedder, index)
    if kind == "vector":
        return vector
    hybrid = HybridRetriever([BM25Retriever(chunks), vector], weights=[1.0, HYBRID_VECTOR_WEIGHT])
    if kind == "hybrid":
        return hybrid
    if kind == "hybrid_rerank_ce":
        return RerankedRetriever(hybrid, CrossEncoderReranker())
    return RerankedRetriever(hybrid, LLMReranker(make_llm_score_fn(settings)))
