import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from .base import Chunk, Hit, Retriever


class HybridRetriever:
    def __init__(
        self, retrievers: Sequence[Retriever], weights: Sequence[float] | None = None,
        rrf_k: float = 60, max_chunks_per_doc: int = 2,
    ) -> None:
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers) or any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("weights must be finite, nonnegative, and match retrievers")
        if not math.isfinite(rrf_k) or rrf_k < 0 or max_chunks_per_doc <= 0:
            raise ValueError("Require rrf_k >= 0 and max_chunks_per_doc > 0")
        self.rrf_k = rrf_k
        self.max_chunks_per_doc = max_chunks_per_doc

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        fetch_k = max(20, k * self.max_chunks_per_doc)
        active = [(r, w) for r, w in zip(self.retrievers, self.weights, strict=True) if w > 0]
        while True:
            scores: dict[str, float] = defaultdict(float)
            chunks: dict[str, Chunk] = {}
            exhausted = True
            for retriever, weight in active:
                hits = retriever.search(query, fetch_k)
                exhausted = exhausted and len(hits) < fetch_k
                seen: set[str] = set()
                for rank, hit in enumerate(hits, 1):
                    chunk_id = hit.chunk.id
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    chunks.setdefault(chunk_id, hit.chunk)
                    scores[chunk_id] += weight / (self.rrf_k + rank)
            result: list[Hit] = []
            per_doc: Counter[str] = Counter()
            for chunk_id in sorted(scores, key=lambda cid: -scores[cid]):
                chunk = chunks[chunk_id]
                doc_id = chunk.doc_id or chunk.id.split("#", 1)[0]
                if per_doc[doc_id] >= self.max_chunks_per_doc:
                    continue
                result.append(Hit(chunk, scores[chunk_id]))
                per_doc[doc_id] += 1
                if len(result) == k:
                    return result
            if exhausted:
                return result
            fetch_k *= 2
