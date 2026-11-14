import json
import os
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest

from carepilot.config import ROOT, Settings
from carepilot.retrieval import embedding, rerank
from carepilot.retrieval.base import Hit


def test_fastembed_batches_normalizes_and_sets_mirror(monkeypatch):
    calls = []
    class Model:
        @staticmethod
        def list_supported_models():
            return [dict(model="BAAI/bge-small-zh-v1.5", dim=2)]

        def __init__(self, **kwargs):
            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
            calls.append(kwargs)

        def embed(self, texts, batch_size):
            calls.append((texts, batch_size))
            yield from ([3.0, 4.0] for _ in texts)

    module = ModuleType("fastembed")
    module.TextEmbedding = Model
    monkeypatch.setitem(sys.modules, "fastembed", module)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    encoder = embedding.FastEmbedEmbedder(batch_size=2, show_progress=False)
    vectors = encoder.encode(["一", "二", "三"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[.6, .8]] * 3)
    assert calls[1][1] == 2
    assert encoder.encode([]).shape == (0, 2)
    monkeypatch.setenv("HF_ENDPOINT", "https://custom.example")
    Model.__init__ = lambda self, **kwargs: None
    embedding.FastEmbedEmbedder()
    assert os.environ["HF_ENDPOINT"] == "https://custom.example"
    monkeypatch.setattr(encoder._model, "embed", lambda *a, **k: iter([[0, 0]]))
    with pytest.raises(ValueError, match="zero"):
        encoder.encode(["bad"])


def test_cross_encoder_uses_chinese_model_and_preserves_score_order(monkeypatch, chunks):
    calls = []
    class Model:
        @staticmethod
        def list_supported_models():
            return [dict(model="BAAI/bge-reranker-base")]

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def rerank(self, query, documents, batch_size):
            calls.append((query, documents, batch_size))
            return iter([-4, 2, 1, 5, 0])

    module = ModuleType("fastembed.rerank.cross_encoder")
    module.TextCrossEncoder = Model
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)
    encoder = rerank.CrossEncoderReranker()
    assert calls[0]["model_name"] == "BAAI/bge-reranker-base"
    hits = encoder.rerank("query", [Hit(c, 0) for c in chunks], 2)
    assert [h.chunk for h in hits] == [chunks[3], chunks[1]]
    assert encoder.rerank("query", [], 2) == []
    with pytest.raises(ValueError, match="Chinese"):
        rerank.CrossEncoderReranker(model_name="English-only")


def test_structured_llm_scores_all_candidates_once(monkeypatch):
    calls = []
    class Chat:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            assert kwargs == dict(method="json_schema", include_raw=True)
            return self

        def invoke(self, messages):
            calls.append(messages)
            return dict(parsed=self.schema(scores=[9, 2]), raw=None)

    def get_llm(settings, kind):
        assert kind == "fast"
        assert settings.llm_api_key == "test"
        return Chat()

    monkeypatch.setattr(rerank, "get_llm", get_llm)
    settings = Settings(llm_api_key="test", llm_base_url="http://localhost:0/v1")
    score = rerank.make_llm_score_fn(settings)
    assert score("库存", ["段落一", "段落二"]) == [9, 2]
    assert len(calls) == 1
    assert json.loads(calls[0][1].content) == dict(query="库存", passages=["段落一", "段落二"])
    assert score("query", []) == []
    with pytest.raises(ValueError, match="count"):
        score("query", ["only one"])


def test_build_index_cli_filters_and_prints_metrics(tmp_path, monkeypatch, embedder, capsys):
    kb = tmp_path / "kb"
    kb.mkdir()
    entries = []
    for directory, flags in [("keep", []), ("skipdir", []), ("skipflag", ["release_note"])]:
        path = kb / "docs" / directory / "guide.md"
        path.parent.mkdir(parents=True)
        path.write_text("# 使用手册\n库存同步", encoding="utf-8")
        entries.append(dict(doc_id=f"{directory}/guide", title="使用手册", kept=True, top_dir=directory, flags=flags))
    (kb / "manifest.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    monkeypatch.setattr(embedding, "FastEmbedEmbedder", lambda: embedder)
    script = runpy.run_path(str(ROOT / "scripts/build_index.py"))
    destination = tmp_path / "index"
    script["main"](["--kb-dir", str(kb), "--out", str(destination), "--exclude-dirs", "skipdir", "--exclude-flags", "release_note"])
    meta = json.loads((destination / "chunk_ids.json").read_text(encoding="utf-8"))
    assert meta["chunk_ids"] == ["keep/guide#0"]
    assert "chunks=1" in capsys.readouterr().out
