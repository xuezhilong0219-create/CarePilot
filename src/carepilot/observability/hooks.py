"""节点包装层：trace 和预算守卫共用一个入口，业务节点本身不知道它们的存在。

- `instrument(nodes)`：给图的每个节点套一层。run 的 tracer / guard 通过 LangGraph 的 `config["configurable"]` 传入
  （对象不会被写进 checkpoint）。两者都没有时是直通，零开销。
- `record_llm_call(...)`：`StructuredLLM` 每次调用后记一笔到当前节点的 span（token 取自网关返回的 usage，没有就标 usage_missing，不估算）。
- `guard_step(...)`：节点内部（如工具规划循环）每一步调用一次，重复调用/超步数/超 token/超时会抛 `BudgetExceeded`。
用 ContextVar 传递"当前 span / 守卫"，因为 LLM 调用发生在节点函数内部的同一个线程里。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from ..gateway.guard import BudgetExceeded, RunGuard
from .spec import LlmCall
from .tracer import llm_call_from_message

_span: ContextVar[Any] = ContextVar("carepilot_span", default=None)
_guard: ContextVar[RunGuard | None] = ContextVar("carepilot_guard", default=None)
_tokens: ContextVar[list[int] | None] = ContextVar("carepilot_tokens", default=None)

__all__ = ["BudgetExceeded", "guard_step", "instrument", "record_llm_call", "summarize_patch", "take_tokens"]


def record_llm_call(raw: Any, *, purpose: str, model: str, latency_ms: float, retries: int = 0, error: str | None = None) -> None:
    if raw is None:
        call = LlmCall(purpose=purpose, model=model, latency_ms=latency_ms, retries=retries, usage_missing=True)
    else:
        call = llm_call_from_message(raw, purpose=purpose, model=model, latency_ms=latency_ms, retries=retries)
    call.error = error
    span, sink = _span.get(), _tokens.get()
    if span is not None:
        span.add_llm_call(call)
    if sink is not None:
        sink[0] += call.prompt_tokens + call.completion_tokens


def take_tokens() -> int:
    """取走并清零当前节点已累计的 token，避免守卫重复计费。"""
    sink = _tokens.get()
    if sink is None:
        return 0
    used, sink[0] = sink[0], 0
    return used


def guard_step(kind: str, *, tool: str | None = None, args: dict[str, Any] | None = None, tokens: int = 0) -> None:
    guard = _guard.get()
    if guard is not None:
        guard.step(kind, tool=tool, args=args, tokens=tokens)


def _clip(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize_patch(name: str, patch: Mapping[str, Any] | None) -> str:
    """节点输出的一行摘要。只挑不含敏感信息的小字段，不放回复正文和客户原话。"""
    p = patch or {}
    if name == "classify":
        t = p.get("triage") or {}
        return f"category={t.get('category')} severity={t.get('severity')} vague={t.get('is_vague')} missing={p.get('missing_fields')}"
    if name == "retrieve":
        return f"hits={len(p.get('hits') or [])} top={[h['id'] for h in (p.get('hits') or [])[:2]]}"
    if name == "plan_tools":
        tr = p.get("tool_run") or {}
        return f"outcome={tr.get('outcome')} calls={tr.get('calls')}"
    if name == "draft_reply":
        parts = [f"resolution={p.get('resolution')}", f"cited={len(p.get('citations') or [])}"]
        if p.get("escalate_reason"):
            parts.append(f"reason={p['escalate_reason']}")
        return " ".join(parts)
    if name in ("ask_for_info", "route_escalation"):
        return f"resolution={p.get('resolution')} route={p.get('route_to')}"
    if name == "execute_actions":
        return f"status={p.get('status')} actions={[a.get('tool') for a in (p.get('actions') or [])]}"
    return _clip({k: v for k, v in p.items() if k in ("resolution", "status")})


def _wrap(name: str, fn: Callable[[Any], dict]) -> Callable[..., dict]:
    def wrapped(state: Any, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {}) or {}
        tracer, guard = cfg.get("tracer"), cfg.get("guard")
        if tracer is None and guard is None:
            return fn(state)
        handle = tracer.span(name, input_summary=_clip(state.get("customer_message", ""), 120)) if tracer is not None else None
        sink = [0]
        interrupted: GraphInterrupt | None = None
        patch: dict = {}
        tokens = (_tokens.set(sink), _guard.set(guard))
        try:
            if handle is None:
                patch = fn(state)
            else:
                with handle as sp:
                    token = _span.set(sp)
                    try:
                        patch = fn(state)
                    except GraphInterrupt as exc:  # 等待人工审核不是错误，不要记成 span 的 error
                        interrupted, patch = exc, {}
                        sp.set_output("等待人工审核")
                    finally:
                        _span.reset(token)
                    if interrupted is None:
                        sp.set_output(summarize_patch(name, patch))
        finally:
            _tokens.reset(tokens[0])
            _guard.reset(tokens[1])
        if interrupted is not None:
            raise interrupted
        if guard is not None:  # 节点结束后记一步，并把这个节点里还没计过的 token 一起计入
            guard.step(f"node:{name}", tokens=sink[0])
        return patch

    wrapped.__name__ = f"instrumented_{name}"
    return wrapped


def instrument(nodes: Mapping[str, Callable[[Any], dict]]) -> dict[str, Callable[..., dict]]:
    return {name: _wrap(name, fn) for name, fn in nodes.items()}
