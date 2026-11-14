"""测试替身：假模型 / 假检索器 / 记录调用的工单系统。"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from carepilot.graph.state import DraftResult, Triage
from carepilot.retrieval.base import Chunk, Hit
from carepilot.tools import TicketSystem


class ScriptedInvoker:
    """按脚本顺序返回预设结果，并记录每次调用。脚本用尽后继续调用会报错，避免测试悄悄多用一次模型。"""

    def __init__(self, results: BaseModel | list[BaseModel] | None = None):
        if results is None:
            results = []
        elif not isinstance(results, list):
            results = [results]
        self.results: list[BaseModel] = list(results)
        self.calls: list[list[BaseMessage]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def invoke(self, messages: list[BaseMessage]) -> BaseModel:
        self.calls.append(messages)
        if len(self.calls) > len(self.results):
            raise AssertionError(f"模型被调用了 {len(self.calls)} 次，但脚本只有 {len(self.results)} 个结果")
        return self.results[len(self.calls) - 1]


class FailingInvoker:
    """用来验证流式接口在节点抛异常时能发出 error 事件。"""

    def __init__(self, error: BaseException | None = None):
        self.error = error or RuntimeError("fake model failed")
        self.calls = 0

    def invoke(self, messages: list[BaseMessage]) -> BaseModel:
        self.calls += 1
        raise self.error


class FakeRetriever:
    def __init__(self, hits: list[Hit] | None = None):
        self.hits = list(hits or [])
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 4) -> list[Hit]:
        self.calls.append((query, k))
        return list(self.hits[:k])


@dataclass
class RecordingTicketSystem(TicketSystem):
    auth_checks: list[tuple[str, str]] = field(default_factory=list)

    def check_shop_auth(self, platform_name: str, shop_name: str) -> dict:
        self.auth_checks.append((platform_name, shop_name))
        return super().check_shop_auth(platform_name, shop_name)


def make_chunk(chunk_id: str = "doc#0", title: str = "店铺授权", text: str = "在店铺管理里重新授权。", doc_id: str = "doc") -> Chunk:
    return Chunk(id=chunk_id, title=title, text=text, doc_id=doc_id, meta={})


def make_hit(chunk_id: str = "doc#0", score: float = 1.0, **kwargs: Any) -> Hit:
    return Hit(make_chunk(chunk_id, **kwargs), score)


def make_triage(category: str = "feature_howto", severity: str = "medium", **kwargs: Any) -> Triage:
    return Triage(category=category, severity=severity, **kwargs)


def make_draft(reply: str = "请在店铺管理页重新授权。", cited_ids: list[str] | None = None, answerable: bool = True) -> DraftResult:
    return DraftResult(answerable=answerable, reply=reply if answerable else "", cited_ids=cited_ids if cited_ids is not None else ["doc#0"])
