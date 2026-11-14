import math
from collections import Counter

import pytest

from carepilot.config import ROOT
from carepilot.retrieval.base import Chunk
from carepilot.retrieval.bm25 import BM25Retriever, tokenize
from carepilot.retrieval.chunking import load_kb


def legacy_search(chunks: list[Chunk], query: str, tokenizer: str) -> list[tuple[str, float]]:
    docs = [Counter(tokenize(c.text, tokenizer)) for c in chunks]
    lengths = [sum(doc.values()) for doc in docs]
    avg = sum(lengths) / len(lengths)
    df = Counter(token for doc in docs for token in doc)
    idf = {t: math.log(1 + (len(chunks) - n + .5) / (n + .5)) for t, n in df.items()}
    scored = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        score = 0.0
        for token in tokenize(query, tokenizer):
            tf = doc.get(token, 0)
            if tf:
                score += idf[token] * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / avg))
        if score > 0:
            scored.append((chunk.id, score))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


@pytest.mark.parametrize("tokenizer", ["jieba", "bigram"])
@pytest.mark.parametrize("query", ["京东店铺授权已失效怎么处理", "库存同步失败", "仓库接口超时", "授权 授权", "zzzqqq"])
def test_inverted_matches_legacy_ranking_without_title_boost(tokenizer: str, query: str):
    chunks = load_kb(ROOT / "data/kb")
    expected = legacy_search(chunks, query, tokenizer)
    hits = BM25Retriever(chunks, tokenizer=tokenizer, title_weight=1).search(query, k=100)
    assert [h.chunk.id for h in hits] == [row[0] for row in expected]
    assert [h.score for h in hits] == pytest.approx([row[1] for row in expected])


@pytest.mark.parametrize("mode", ["jieba", "bigram"])
def test_error_codes_and_models_remain_whole(mode: str):
    tokens = tokenize("报错 ERR_AUTH-401 v1.5 ZX_123 API404", mode)
    assert {"err_auth-401", "v1.5", "zx_123", "api404"}.issubset(tokens)


def test_bigram_and_domain_dictionary():
    assert tokenize("库存同步", "bigram") == ["库存", "存同", "同步"]
    assert "旺店通" in tokenize("旺店通店铺授权")


def test_title_boost_changes_ranking_and_ties_are_stable():
    chunks = [Chunk("body", "日志", "日志\n库存同步"), Chunk("title", "库存同步", "库存同步\n日志")]
    assert BM25Retriever(chunks, title_weight=1).search("库存同步")[0].chunk.id == "body"
    assert BM25Retriever(chunks).search("库存同步")[0].chunk.id == "title"


def test_empty_corpus_query_and_nonpositive_k():
    assert BM25Retriever([]).search("库存") == []
    retriever = BM25Retriever([Chunk("x", "标题", "库存同步")])
    assert retriever.search("") == []
    assert retriever.search("库存", 0) == []
    assert retriever.search("库存", -1) == []
    with pytest.raises(ValueError, match="tokenizer"):
        BM25Retriever([], tokenizer="unknown")
