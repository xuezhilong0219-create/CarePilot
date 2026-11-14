import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from carepilot.retrieval.base import Chunk
from carepilot.retrieval.vector import VectorIndex, VectorRetriever


def test_vector_round_trip_and_exact_match(tmp_path, chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, chunks, embedder.name)
    assert loaded.embeddings.dtype == np.float32
    assert loaded.chunk_ids == [c.id for c in chunks]
    metadata = json.loads((tmp_path / "chunk_ids.json").read_text(encoding="utf-8"))
    assert metadata["embedder_name"] == embedder.name
    assert metadata["chunk_count"] == len(chunks)
    retriever = VectorRetriever(chunks, embedder, loaded)
    hits = retriever.search(chunks[2].text, 20)
    assert hits[0].chunk == chunks[2]
    assert hits[0].score == pytest.approx(1)
    assert len(hits) == len(chunks)
    assert retriever.search("  ") == []
    assert retriever.search("query", 0) == []


@pytest.mark.parametrize("change", ["count", "ids", "content", "order", "model"])
def test_index_rejects_stale_corpus_or_model(tmp_path, chunks, embedder, change):
    VectorIndex.build(chunks, embedder).save(tmp_path)
    name = embedder.name
    if change == "count":
        chunks = chunks[:-1]
    elif change == "ids":
        chunks[0] = replace(chunks[0], id="new")
    elif change == "content":
        chunks[0] = replace(chunks[0], text="changed")
    elif change == "order":
        chunks = chunks[::-1]
    else:
        name = "different-model"
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        VectorIndex.load(tmp_path, chunks, name)


def test_vector_top_k_matches_full_sort_including_ties(chunks, embedder):
    matrix = np.array([[1, 0], [0, 1], [1, 0], [-1, 0], [0, -1]], dtype=np.float32)
    index = VectorIndex(matrix, [c.id for c in chunks], embedder.name)
    assert [i for i, _ in index.search(np.array([1, 0]), 2)] == [0, 2]
    assert [i for i, _ in index.search(np.array([1, 0]), 4)] == [0, 2, 1, 4]
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(3), 2)


def test_corrupt_and_incomplete_indices_are_clear_errors(tmp_path, chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    index.save(tmp_path)
    np.save(tmp_path / "embeddings.npy", np.ones((2, 3)))
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        VectorIndex.load(tmp_path, chunks, embedder.name)
    (tmp_path / "embeddings.npy").unlink()
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        VectorIndex.load(tmp_path, chunks, embedder.name)


def test_empty_index_and_duplicate_ids(tmp_path, embedder):
    index = VectorIndex.build([], embedder)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, [], embedder.name)
    assert VectorRetriever([], embedder, loaded).search("query") == []
    with pytest.raises(ValueError, match="unique"):
        VectorIndex.build([Chunk("a", "t", "x"), Chunk("a", "t", "y")], embedder)


@pytest.mark.parametrize("matrix", [np.array([[np.nan, 1]]), np.array([[0, 0]]), np.array([[2, 0]])])
def test_invalid_embeddings_are_rejected(matrix):
    with pytest.raises(ValueError):
        VectorIndex(matrix, ["a"], "fake")
