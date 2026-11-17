"""节点级 trace 记录器与落盘（D1）。不调用模型、不联网。"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from .spec import LlmCall, RunTrace, Span, clip

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ERROR_MAX = 300


def validate_run_id(run_id: str) -> str:
    """run_id 会直接拼进文件名，必须拒绝路径分隔符和 ..，否则可穿越目录。"""
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"非法 run_id: {run_id!r}")
    if run_id in (".", "..") or ".." in run_id:
        raise ValueError(f"非法 run_id: {run_id!r}")
    return run_id


class SpanHandle:
    """with 块里拿到的句柄；所有写入都在 tracer 的锁内，保证并发记录不丢。"""

    def __init__(self, tracer: "RunTracer", span: Span, started_perf: float) -> None:
        self._tracer = tracer
        self._span = span
        self._started_perf = started_perf

    def set_output(self, text: str | None) -> None:
        with self._tracer._lock:
            self._span.output_summary = clip(text)

    def add_llm_call(self, call: LlmCall) -> None:
        with self._tracer._lock:
            self._span.llm_calls.append(call)

    def add_retry(self) -> None:
        with self._tracer._lock:
            self._span.retries += 1

    def __enter__(self) -> "SpanHandle":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        with self._tracer._lock:
            self._span.duration_ms = (time.perf_counter() - self._started_perf) * 1000
            if exc is not None:
                name = exc_type.__name__ if exc_type is not None else type(exc).__name__
                self._span.error = clip(f"{name}: {exc}")
            self._tracer._trace.spans.append(self._span)
        return False  # 异常继续抛给调用方


class RunTracer:
    """一次 run 一个 tracer，线程安全。"""

    def __init__(
        self,
        run_id: str,
        *,
        ticket_id: str | None = None,
        category: str | None = None,
        clock: Callable[[], float] = time.time,
        **meta: Any,
    ) -> None:
        self.run_id = run_id
        self._clock = clock
        self._lock = threading.RLock()
        self._trace = RunTrace(
            run_id=run_id,
            ticket_id=ticket_id,
            category=category,
            started_at=clock(),
            meta=dict(meta),
        )
        self._ended = False

    def span(self, name: str, *, input_summary: str | None = "") -> SpanHandle:
        with self._lock:
            span = Span(name=name, started_at=self._clock(), input_summary=clip(input_summary))
        return SpanHandle(self, span, time.perf_counter())

    def set_category(self, category: str | None) -> None:
        with self._lock:
            self._trace.category = category

    def end(self, status: str = "resolved") -> RunTrace:
        with self._lock:
            if not self._ended:
                self._trace.status = status
                self._trace.ended_at = self._clock()
                self._ended = True
            return self._trace

    def set_status(self, status: str) -> None:
        """非终态的状态更新（如 awaiting_review）。run 跨 HTTP 请求（审核恢复）时，每个阶段结束都要落一次盘。"""
        with self._lock:
            if not self._ended:
                self._trace.status = status

    def snapshot(self) -> RunTrace:
        """当前 trace 的深拷贝：落盘时用，避免序列化过程中被并发写入的 span 改动。"""
        with self._lock:
            return RunTrace.from_dict(copy.deepcopy(self._trace.to_dict()))

    @classmethod
    def from_trace(cls, trace: RunTrace, *, clock: Callable[[], float] = time.time) -> "RunTracer":
        """从落盘的 trace 续写：进程重启后审核恢复同一个 run 时，不能把之前的节点覆盖掉。"""
        tracer = cls(trace.run_id, ticket_id=trace.ticket_id, category=trace.category, clock=clock)
        tracer._trace = RunTrace.from_dict(copy.deepcopy(trace.to_dict()))
        return tracer


def llm_call_from_message(
    raw_message: Any,
    *,
    purpose: str,
    model: str,
    latency_ms: float,
    retries: int = 0,
) -> LlmCall:
    """只读 usage_metadata，没有就记 0 并标 usage_missing，不做估算。"""
    usage = getattr(raw_message, "usage_metadata", None)
    usage_missing = not isinstance(usage, Mapping)
    prompt = completion = 0
    if isinstance(usage, Mapping) and usage:
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        usage_missing = "input_tokens" not in usage or "output_tokens" not in usage
    return LlmCall(
        purpose=purpose,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=latency_ms,
        retries=retries,
        usage_missing=usage_missing,
    )


class TraceStore:
    """一次 run 一个 JSON；原子写，读路径做 run_id 校验。"""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.directory / f"{validate_run_id(run_id)}.json"

    def save(self, trace: RunTrace) -> Path:
        path = self._path(trace.run_id)
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=f".{trace.run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def load(self, run_id: str) -> RunTrace:
        path = self._path(run_id)
        with open(path, encoding="utf-8") as f:
            return RunTrace.from_dict(json.load(f))

    def _valid_files(self) -> list[tuple[float, str]]:
        found: list[tuple[float, str]] = []
        for path in self.directory.glob("*.json"):
            try:
                validate_run_id(path.stem)
                trace = self.load(path.stem)
            except (ValueError, OSError, json.JSONDecodeError, TypeError):
                continue
            found.append((trace.started_at, trace.run_id))
        return found

    def list_runs(self) -> list[str]:
        return [run_id for _, run_id in sorted(self._valid_files())]

    def load_all(self) -> list[RunTrace]:
        return [self.load(run_id) for run_id in self.list_runs()]

    def runs_for(self, ticket_id: str) -> list[RunTrace]:
        """这张工单跑过的每一轮，按开始时间倒序。

        每次调用都要把目录里的 JSON 读一遍 —— 演示规模（几十个文件）够用，
        真上线要换成按 ticket_id 建索引文件，否则 O(全部轨迹)。
        """
        runs = [t for t in self.load_all() if t.ticket_id == ticket_id]
        return sorted(runs, key=lambda t: t.started_at, reverse=True)


class _NullSpan:
    def set_output(self, text: str | None) -> None:
        return None

    def add_llm_call(self, call: LlmCall) -> None:
        return None

    def add_retry(self) -> None:
        return None

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        return False


class NullTracer:
    """不需要 trace 的调用方用它，接口与 RunTracer 一致。"""

    def __init__(self, run_id: str = "null", **meta: Any) -> None:
        self.run_id = run_id

    def span(self, name: str, *, input_summary: str | None = "") -> _NullSpan:
        return _NullSpan()

    def set_category(self, category: str | None) -> None:
        return None

    def end(self, status: str = "resolved") -> RunTrace:
        return RunTrace(run_id=self.run_id, status=status)
