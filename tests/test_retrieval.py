from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.chunking import load_kb, split_markdown
from carepilot.config import ROOT


def test_split_markdown_keeps_title_in_chunk():
    chunks = split_markdown("doc", "# 标题A\n内容一\n\n# 标题B\n内容二")
    assert [c.title for c in chunks] == ["标题A", "标题B"]
    assert all(c.text.startswith(c.title) for c in chunks)


def test_bm25_finds_relevant_chunk():
    r = BM25Retriever(load_kb(ROOT / "data" / "kb"))
    hits = r.search("路由器怎么恢复出厂设置", k=2)
    assert hits and hits[0].chunk.title == "恢复出厂设置"


def test_bm25_returns_nothing_for_unrelated_query():
    r = BM25Retriever(load_kb(ROOT / "data" / "kb"))
    assert r.search("zzzqqq", k=3) == []
