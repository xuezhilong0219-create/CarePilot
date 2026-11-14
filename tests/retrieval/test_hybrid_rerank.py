import pytest

from carepilot.retrieval.base import Chunk, Hit
from carepilot.retrieval.hybrid import HybridRetriever
from carepilot.retrieval.rerank import LLMReranker, RerankedRetriever


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k=4):
        self.calls.append(k)
        return self.hits[:k]


def test_weighted_rrf_deduplicates_chunk_and_caps_document():
    a, b, c = [Chunk(x, x, x, x) for x in "abc"]
    one = FixedRetriever([Hit(a, 100), Hit(b, 80), Hit(c, 20)])
    two = FixedRetriever([Hit(b, .1), Hit(c, .09), Hit(a, .08)])
    hits = HybridRetriever([one, two], weights=[1, 2]).search("q", 3)
    assert [h.chunk.id for h in hits] == ["b", "a", "c"]
    assert hits[0].score == pytest.approx(1 / 62 + 2 / 61)


def test_document_cap_fetches_enough_candidates():
    hits = [Hit(Chunk(f"long#{i}", "long", str(i), "long"), 100-i) for i in range(40)]
    hits += [Hit(Chunk(f"other{i}#0", "other", "text", f"other{i}"), 1) for i in range(3)]
    source = FixedRetriever(hits)
    result = HybridRetriever([source], max_chunks_per_doc=2).search("q", 5)
    assert len(result) == 5
    assert sum(hit.chunk.doc_id == "long" for hit in result) == 2
    assert len(source.calls) > 1


def test_rrf_counts_each_source_once_and_handles_empty_sources():
    hit = Hit(Chunk("a", "a", "a"), 1)
    result = HybridRetriever([FixedRetriever([hit, hit])]).search("q")
    assert result[0].score == pytest.approx(1 / 61)
    assert HybridRetriever([]).search("q") == []
    with pytest.raises(ValueError):
        HybridRetriever([FixedRetriever([])], weights=[-1])


def test_llm_reranker_scores_all_passages_in_one_call(chunks):
    calls = []
    def score(query, passages):
        calls.append((query, passages))
        return [1, 8, 5, 8, 0]
    result = LLMReranker(score).rerank("q", [Hit(c, 99) for c in chunks], 2)
    assert [h.chunk.id for h in result] == [chunks[1].id, chunks[3].id]
    assert len(calls) == 1
    assert calls[0] == ("q", [c.text for c in chunks])


@pytest.mark.parametrize("scores", [[1], [1, float("nan")], [-1, 5], [1, 11]])
def test_llm_reranker_rejects_invalid_scores(scores):
    hits = [Hit(Chunk(str(i), "t", "x"), 0) for i in range(2)]
    with pytest.raises(ValueError, match="score"):
        LLMReranker(lambda q, p: scores).rerank("q", hits, 2)


def test_reranked_retriever_fetches_candidates_and_respects_k(chunks):
    source = FixedRetriever([Hit(c, i) for i, c in enumerate(chunks)])
    reranker = LLMReranker(lambda q, p: list(range(len(p))))
    retriever = RerankedRetriever(source, reranker, fetch_k=3)
    assert retriever.search("q", 1)[0].chunk == chunks[2]
    assert source.calls == [3]
    assert len(retriever.search("q", 5)) == 5
    assert retriever.search("q", 0) == []
