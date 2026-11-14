import json
import os
from time import perf_counter

import pytest

from carepilot.config import ROOT
from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.chunking import load_kb
from carepilot.retrieval.embedding import FastEmbedEmbedder
from carepilot.retrieval.vector import VectorIndex, VectorRetriever


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 to download and run the real model")
def test_real_chinese_model_recovers_synonym_query(tmp_path):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    started = perf_counter()
    cache_dir = ROOT / "data/models/integration-bge-small-zh-v1.5"
    embedder = FastEmbedEmbedder(cache_dir=cache_dir)
    loaded = perf_counter()
    chunks = load_kb(ROOT / "data/kb")
    corpus = chunks * 32
    encoded_start = perf_counter()
    embeddings = embedder.encode([chunk.text for chunk in corpus])
    encoded = perf_counter()
    index = VectorIndex.build(chunks, embedder)
    index.save(tmp_path)
    retriever = VectorRetriever(chunks, embedder, VectorIndex.load(tmp_path, chunks, embedder.name))
    bm25 = BM25Retriever(chunks)
    query = "库存没同步上去"
    assert retriever.search(query, 1)[0].chunk.title == "库存同步失败"
    synonym = "仓内数量没传给线上"
    assert bm25.search(synonym, 4) == []
    semantic = retriever.search(synonym, 1)
    assert semantic[0].chunk.title == "库存同步失败"
    sizes = [path.stat().st_size for path in cache_dir.rglob("*") if path.is_file() and not path.name.endswith(".lock")]
    print(json.dumps(dict(
        model=embedder.name, model_load_s=loaded-started, embedding_chunks=len(corpus),
        embedding_s=encoded-encoded_start, chunks_per_second=len(corpus)/(encoded-encoded_start),
        cache_bytes=sum(sizes), shape=embeddings.shape,
        synonym_query=synonym, bm25_hits=0, vector_top1=semantic[0].chunk.title,
    ), ensure_ascii=False))
