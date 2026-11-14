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
    hybrid = HybridRetriever([BM25Retriever(chunks), vector])
    if kind == "hybrid":
        return hybrid
    if kind == "hybrid_rerank_ce":
        return RerankedRetriever(hybrid, CrossEncoderReranker())
    return RerankedRetriever(hybrid, LLMReranker(make_llm_score_fn(settings)))
