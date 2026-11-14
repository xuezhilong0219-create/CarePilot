from carepilot.memory.spec import MemoryRecord, NullMemoryStore
from carepilot.observability.spec import LlmCall, RunTrace, Span, clip


def _trace() -> RunTrace:
    return RunTrace(
        run_id="r1", ticket_id="T-1", category="platform_integration", started_at=1.0, ended_at=2.5, status="resolved",
        spans=[
            Span("classify", 1.0, 800.0, "客户消息…", "category=x", [LlmCall("triage", "m", 100, 20, 780.0)]),
            Span("draft_reply", 1.8, 700.0, "4 个资料", "auto_reply", [LlmCall("draft", "m", 900, 120, 690.0, retries=1)], retries=1),
        ],
        meta={"retriever": "bm25"},
    )


def test_trace_roundtrip_and_totals():
    t = _trace()
    again = RunTrace.from_dict(t.to_dict())
    assert again == t
    assert again.total_tokens == (1000, 140)
    assert [c.purpose for c in again.llm_calls] == ["triage", "draft"]


def test_clip_limits_length_and_handles_none():
    assert clip(None) == "" and clip("a" * 10, 5) == "aaaa…" and len(clip("x" * 1000)) == 300


def test_null_memory_store_never_remembers():
    s = NullMemoryStore()
    s.put("u1", "k", {"a": 1}, source_run_id="r1", ttl_seconds=60)
    assert s.get("u1", "k") is None and s.search("u1", "x") == [] and s.delete("u1", "k") is False
    assert isinstance(s.put("u1", "k", 1, source_run_id="r", ttl_seconds=None), MemoryRecord)
