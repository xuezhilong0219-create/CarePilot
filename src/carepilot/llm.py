import time
from typing import Literal, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Settings
from .observability.hooks import record_llm_call

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


class StructuredLLM:
    """结构化输出的统一入口。

    实测这个网关上：qwen3.8-flash 用 function_calling 经常返回空的 `</tool_call>`
    （18 次里成功 2 次），用 json_schema 则 18/18 成功，所以统一走 json_schema；
    即便如此仍可能偶发解析失败，这里重试，仍失败就抛明确的异常，而不是把 None 传给下游。"""

    def __init__(self, llm: ChatOpenAI, schema: type[T], retries: int = 2, *, purpose: str = "llm", model: str = ""):
        self._chain = llm.with_structured_output(schema, method="json_schema", include_raw=True)
        self._retries = retries
        self._purpose, self._model = purpose, model

    def invoke(self, messages: list[BaseMessage]) -> T:
        last = None
        for attempt in range(self._retries + 1):
            started = time.perf_counter()
            try:
                out = self._chain.invoke(messages)
            except Exception as exc:  # 超时/网关错误：记到 trace 再抛，成本统计里能看到失败的调用
                record_llm_call(None, purpose=self._purpose, model=self._model,
                                latency_ms=(time.perf_counter() - started) * 1000, retries=attempt, error=type(exc).__name__)
                raise
            record_llm_call(out["raw"], purpose=self._purpose, model=self._model,
                            latency_ms=(time.perf_counter() - started) * 1000, retries=attempt)
            if out["parsed"] is not None:
                return out["parsed"]
            last = out.get("parsing_error") or repr(out["raw"].content)[:200]
        raise StructuredOutputError(f"结构化输出解析失败（已重试 {self._retries} 次）：{last}")


def get_llm(settings: Settings, kind: Literal["fast", "main"] = "main") -> ChatOpenAI:
    """网关是 OpenAI 兼容接口，所以直接用 ChatOpenAI 指向 base_url 即可。"""
    model = settings.llm_fast_model if kind == "fast" else settings.llm_main_model
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )
