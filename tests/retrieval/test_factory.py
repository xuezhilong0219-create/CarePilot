from pathlib import Path

import pytest

from carepilot.config import ROOT, Settings
from carepilot.retrieval import factory
from carepilot.retrieval.rerank import LLMReranker


@pytest.mark.parametrize("kind", factory.RETRIEVER_KINDS)
def test_factory_all_kinds_are_offline_with_injected_dependencies(kind, tmp_path, monkeypatch, embedder):
    monkeypatch.setattr(factory, "FastEmbedEmbedder", lambda: embedder)
    monkeypatch.setattr(factory, "CrossEncoderReranker", lambda: LLMReranker(lambda q, p: [5] * len(p)))
    monkeypatch.setattr(factory, "make_llm_score_fn", lambda settings: lambda q, p: [5] * len(p))
    settings = Settings(llm_api_key="test", llm_base_url="http://localhost:0/v1")
    retriever = factory.build_retriever(kind, ROOT / "data/kb", index_dir=tmp_path, settings=settings)
    assert retriever.search("库存同步失败", 2)
    if kind != "bm25":
        assert (tmp_path / "embeddings.npy").exists()
        monkeypatch.setattr(embedder, "encode", lambda texts: pytest.fail("must load cached index"))
        factory.build_retriever(kind, ROOT / "data/kb", index_dir=tmp_path, settings=settings)


def test_unknown_kind_and_missing_llm_settings():
    with pytest.raises(ValueError, match="kind"):
        factory.build_retriever("missing", Path("unused"))
    with pytest.raises(ValueError, match="settings"):
        factory.build_retriever("hybrid_rerank_llm", Path("unused"))


def test_factory_rejects_stale_and_partial_index(tmp_path, monkeypatch, embedder):
    monkeypatch.setattr(factory, "FastEmbedEmbedder", lambda: embedder)
    kb = tmp_path / "kb"
    kb.mkdir()
    document = kb / "guide.md"
    document.write_text("# 标题\n第一版", encoding="utf-8")
    index_dir = tmp_path / "index"
    factory.build_retriever("vector", kb, index_dir=index_dir)
    document.write_text("# 标题\n第二版", encoding="utf-8")
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        factory.build_retriever("vector", kb, index_dir=index_dir)
    (index_dir / "embeddings.npy").unlink()
    with pytest.raises(ValueError, match="[Rr]ebuild"):
        factory.build_retriever("vector", kb, index_dir=index_dir)
