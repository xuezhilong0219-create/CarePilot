import threading

import pytest

from carepilot.observability.spec import LlmCall, RunTrace
from carepilot.observability import tracer as tracer_module
from carepilot.observability.tracer import (
    NullTracer,
    RunTracer,
    TraceStore,
    llm_call_from_message,
)


class _Msg:
    def __init__(self, usage):
        if usage is not None:
            self.usage_metadata = usage


def test_concurrent_spans_are_all_recorded():
    tracer = RunTracer("run-concurrent")
    barrier = threading.Barrier(50)

    def worker(i: int) -> None:
        barrier.wait()
        with tracer.span(f"node-{i}", input_summary="x") as sp:
            sp.set_output("ok")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trace = tracer.end("resolved")
    assert len(trace.spans) == 50
    assert {s.name for s in trace.spans} == {f"node-{i}" for i in range(50)}
    assert all(s.output_summary == "ok" for s in trace.spans)


def test_span_records_error_and_reraises():
    tracer = RunTracer("run-error")
    with pytest.raises(RuntimeError, match="bad"):
        with tracer.span("draft", input_summary="in") as sp:
            sp.add_retry()
            raise RuntimeError("bad " + "x" * 500)

    trace = tracer.end("error")
    span = trace.spans[0]
    assert span.error is not None
    assert span.error.startswith("RuntimeError: bad")
    assert len(span.error) == 300
    assert span.retries == 1
    assert span.duration_ms >= 0


def test_summaries_are_clipped_and_llm_calls_kept():
    tracer = RunTracer("run-clip")
    with tracer.span("classify", input_summary="a" * 1000) as sp:
        sp.add_llm_call(LlmCall("triage", "m", 10, 5))
        sp.set_output("b" * 1000)

    span = tracer.end("resolved").spans[0]
    assert len(span.input_summary) == 300
    assert len(span.output_summary) == 300
    assert span.llm_calls[0].prompt_tokens == 10


def test_end_is_idempotent_and_category_settable():
    tracer = RunTracer("run-end", ticket_id="T-1")
    tracer.set_category("feature_howto")
    first = tracer.end("resolved")
    second = tracer.end("error")
    assert first is second
    assert first.status == "resolved"
    assert first.ticket_id == "T-1"
    assert first.category == "feature_howto"
    assert first.ended_at > 0


def test_llm_call_usage_present_and_missing():
    ok = llm_call_from_message(
        _Msg({"input_tokens": 12, "output_tokens": 7}), purpose="draft", model="m", latency_ms=1.5, retries=2
    )
    assert (ok.prompt_tokens, ok.completion_tokens) == (12, 7)
    assert ok.usage_missing is False
    assert ok.retries == 2

    missing = llm_call_from_message(_Msg(None), purpose="draft", model="m", latency_ms=1.0)
    assert (missing.prompt_tokens, missing.completion_tokens) == (0, 0)
    assert missing.usage_missing is True

    partial = llm_call_from_message(_Msg({"input_tokens": 3}), purpose="draft", model="m", latency_ms=1.0)
    assert partial.prompt_tokens == 3
    assert partial.usage_missing is True


def test_trace_store_roundtrip_and_order(tmp_path):
    store = TraceStore(tmp_path)
    store.save(RunTrace(run_id="b", started_at=2.0, status="resolved"))
    store.save(RunTrace(run_id="a", started_at=1.0, status="escalated"))

    assert store.list_runs() == ["a", "b"]
    assert [t.run_id for t in store.load_all()] == ["a", "b"]
    assert store.load("a").status == "escalated"
    assert not list(tmp_path.glob("*.tmp"))


def test_save_is_atomic_and_keeps_old_file_on_failure(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    store.save(RunTrace(run_id="r", status="running", meta={"v": 1}))

    def disk_full(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(tracer_module.os, "replace", disk_full)
    with pytest.raises(OSError):
        store.save(RunTrace(run_id="r", status="resolved", meta={"v": 2}))

    assert store.load("r").meta == {"v": 1}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("run_id", ["../x", "a/b", "a\\b", ""])
def test_trace_store_rejects_path_traversal(tmp_path, run_id):
    store = TraceStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(RunTrace(run_id=run_id))
    with pytest.raises(ValueError):
        store.load(run_id)


def test_null_tracer_is_noop():
    tracer = NullTracer("null-run")
    with tracer.span("classify", input_summary="x") as sp:
        sp.set_output("y")
        sp.add_retry()
        sp.add_llm_call(LlmCall("triage", "m"))
    trace = tracer.end("resolved")
    assert trace.run_id == "null-run"
    assert trace.spans == []


def test_snapshot_set_status_and_resume_from_a_saved_trace():
    from carepilot.observability.tracer import RunTracer

    t = RunTracer("r-resume", ticket_id="T-000001")
    with t.span("classify", input_summary="a") as sp:
        sp.set_output("b")
    t.set_status("awaiting_review")
    snap = t.snapshot()
    assert snap.status == "awaiting_review" and [s.name for s in snap.spans] == ["classify"]
    with t.span("later"):
        pass
    assert [s.name for s in snap.spans] == ["classify"]  # 快照是深拷贝，不受后续写入影响

    t2 = RunTracer.from_trace(snap)  # 模拟进程重启后审核恢复
    with t2.span("execute_actions"):
        pass
    final = t2.end("resolved")
    assert [s.name for s in final.spans] == ["classify", "execute_actions"] and final.run_id == "r-resume"
    t2.set_status("running")  # 结束后不能再改状态
    assert t2.snapshot().status == "resolved"
