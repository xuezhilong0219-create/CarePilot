import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from carepilot.api.app import create_app
from carepilot.config import ROOT, Settings
from carepilot.graph.build import build_graph
from carepilot.graph.nodes import Models
from carepilot.graph.state import DraftResult, Triage
from carepilot.retrieval.base import Hit

from fakes import FailingInvoker, FakeRetriever, RecordingTicketSystem, ScriptedInvoker
# 沙箱里不能写系统临时目录，但仓库根目录可写；用带 .sqlite 后缀的隐藏文件，测试结束删除。
_TMP_PREFIX = ".tmp-carepilot-test-"


def _temp_sqlite() -> Path:
    fd, name = tempfile.mkstemp(dir=ROOT, prefix=_TMP_PREFIX, suffix=".sqlite")
    os.close(fd)
    return Path(name)


@pytest.fixture
def settings():
    db = _temp_sqlite()
    yield Settings(
        llm_api_key="test",
        llm_base_url="http://localhost:0/v1",
        kb_dir=ROOT / "data" / "kb",
        checkpoint_db=db,
    )
    db.unlink(missing_ok=True)


@dataclass
class GraphEnv:
    graph: Any
    retriever: FakeRetriever
    system: RecordingTicketSystem
    triage: ScriptedInvoker
    draft: ScriptedInvoker
    settings: Settings


@pytest.fixture
def make_env(settings: Settings):
    def _make(
        triage: Triage | list[Triage],
        draft: DraftResult | list[DraftResult] | ScriptedInvoker,
        hits: list[Hit] | None = None,
        *,
        checkpointer: Any = None,
        system: RecordingTicketSystem | None = None,
    ) -> GraphEnv:
        triage_inv = triage if isinstance(triage, ScriptedInvoker) else ScriptedInvoker(triage)
        draft_inv = draft if isinstance(draft, ScriptedInvoker) else ScriptedInvoker(draft)
        retriever = FakeRetriever(list(hits or []))
        system = system or RecordingTicketSystem()
        graph = build_graph(
            settings,
            retriever,
            system,
            checkpointer or InMemorySaver(),
            models=Models(triage=triage_inv, draft=draft_inv),
        )
        return GraphEnv(graph=graph, retriever=retriever, system=system, triage=triage_inv, draft=draft_inv, settings=settings)

    return _make


@pytest.fixture
def make_app(settings: Settings):
    def _make(
        triage: Any,
        draft: Any,
        hits: list[Hit] | None = None,
        *,
        system: RecordingTicketSystem | None = None,
    ):
        triage_inv = triage if isinstance(triage, ScriptedInvoker) else ScriptedInvoker(triage)
        draft_inv = draft if isinstance(draft, (ScriptedInvoker, FailingInvoker)) else ScriptedInvoker(draft)
        return create_app(
            settings=settings,
            models=Models(triage=triage_inv, draft=draft_inv),
            retriever=FakeRetriever(list(hits or [])),
            system=system or RecordingTicketSystem(),
            checkpointer=InMemorySaver(),
        )

    return _make
