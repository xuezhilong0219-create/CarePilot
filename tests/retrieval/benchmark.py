"""Run with uv run python -W ignore tests/retrieval/benchmark.py."""
import json
import random
from time import perf_counter

import numpy as np

from carepilot.retrieval.base import Chunk
from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.vector import VectorIndex, VectorRetriever


class BenchmarkEmbedder:
    name = "benchmark/precomputed-query"

    def __init__(self, vector: np.ndarray):
        self.vector = vector

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(self.vector, (len(texts), 1))


def main() -> None:
    rng = random.Random(20261114)
    vocabulary = [f"term{i:04d}" for i in range(3000)]
    domains = ["库存同步", "店铺授权", "电子面单", "物流追踪", "货品匹配", "仓储对接"]
    chunks = [Chunk(
        f"doc{i // 4}#{i % 4}", domains[i % len(domains)],
        domains[i % len(domains)] + "\n" + " ".join(rng.choices(vocabulary, k=55)) + " " + "，".join(rng.choices(domains, k=8)),
        f"doc{i // 4}",
    ) for i in range(30_000)]
    started = perf_counter()
    bm25 = BM25Retriever(chunks)
    build_s = perf_counter() - started
    queries = ["库存同步 电子面单", "仓储对接"] + [" ".join(rng.sample(vocabulary, 3)) for _ in range(98)]
    for query in queries[:5]:
        bm25.search(query)
    lexical_ms = []
    for query in queries:
        started = perf_counter()
        bm25.search(query)
        lexical_ms.append((perf_counter() - started) * 1000)
    generator = np.random.default_rng(20261114)
    matrix = generator.normal(size=(len(chunks), 512)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    embedder = BenchmarkEmbedder(matrix[0])
    vector = VectorRetriever(chunks, embedder, VectorIndex(matrix, [c.id for c in chunks], embedder.name))
    for _ in range(5):
        vector.search("warmup")
    vector_ms = []
    for i in range(100):
        embedder.vector = matrix[i]
        started = perf_counter()
        vector.search("synthetic query")
        vector_ms.append((perf_counter() - started) * 1000)
    print(json.dumps(dict(
        chunks=len(chunks), dimension=512, seed=20261114, tokenizer="jieba",
        bm25_build_s=build_s, bm25_query_p50_ms=float(np.percentile(lexical_ms, 50)),
        bm25_query_p95_ms=float(np.percentile(lexical_ms, 95)), bm25_query_max_ms=max(lexical_ms),
        vector_query_p50_ms=float(np.percentile(vector_ms, 50)),
        vector_query_p95_ms=float(np.percentile(vector_ms, 95)),
        vector_note="Exact NumPy cosine search; precomputed query embedding excludes model inference",
    ), indent=2))


if __name__ == "__main__":
    main()
