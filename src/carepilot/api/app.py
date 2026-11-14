import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..graph.build import build_graph
from ..graph.nodes import Models
from ..retrieval.base import Retriever
from ..retrieval.factory import build_retriever
from ..tools import TicketSystem


class TicketIn(BaseModel):
    message: str
    ticket_id: str | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reply: str | None = None  # action=edit 时，审核人改写后的回复
    note: str | None = None  # 升级时附带给接单团队的备注


def _view(graph, thread_id: str) -> dict:
    """把图的当前状态整理成对外的响应：已结束 / 等待人工审核。"""
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snap.created_at:
        raise HTTPException(404, "thread not found")
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    values = snap.values
    return {
        "thread_id": thread_id,
        "status": "awaiting_review" if pending else values.get("status"),
        "pending_review": pending[0] if pending else None,
        "followup_question": values.get("followup_question"),
        "triage": values.get("triage"),
        "actions": values.get("actions"),
    }


def _summary(node: str, update: dict) -> dict:
    """流式事件只带小字段，避免把整段草稿/检索正文重复推给前端。"""
    if node == "classify":
        triage = update.get("triage") or {}
        return {"category": triage.get("category"), "severity": triage.get("severity")}
    if node == "ask_for_info":
        return {"missing_fields": update.get("missing_fields")}
    if node == "retrieve":
        return {"hits": len(update.get("hits") or [])}
    if node == "draft_reply":
        return {"resolution": update.get("resolution")}
    if node == "route_escalation":
        return {"route_to": update.get("route_to")}
    if node == "human_review":
        return {"review": update.get("review")}
    if node == "execute_actions":
        return {"status": update.get("status")}
    return {}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _stream_graph(graph, graph_input: Any, config: dict, thread_id: str) -> Iterator[str]:
    try:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node, payload in update.items():
                if node == "__interrupt__":  # interrupt 不是节点，等待审核由最后的 result 表达
                    continue
                yield _sse("node", {"node": node, "summary": _summary(node, payload)})
        view = _view(graph, thread_id)
    except Exception as exc:  # 流已开始，异常只能作为事件发出，否则客户端会一直等
        yield _sse("error", {"message": str(exc)})
        return
    yield _sse("result", view)


def create_app(
    settings: Settings | None = None,
    models: Models | None = None,
    retriever: Retriever | None = None,
    system: TicketSystem | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    governed: bool = False,
    store: Any = None,
    toolbox: Any = None,
    chunks: Any = None,
) -> FastAPI:
    """governed=False：演示模式（mock 工单系统、没有身份和权限，接口按 thread_id 访问）。
    governed=True：治理模式（工单落库、三角色、网关、审计、多轮、二次确认），接口见 api/governed.py。"""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        s = settings or get_settings()
        application.state.settings = s
        application.state.system = system or TicketSystem()
        r = retriever if retriever is not None else build_retriever(
            s.retriever_kind, s.kb_dir, index_dir=s.index_dir, settings=s
        )
        conn = None
        cp = checkpointer
        if cp is None:
            s.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(s.checkpoint_db, check_same_thread=False)
            cp = SqliteSaver(conn)
            cp.setup()
        if governed:
            from ..graph.nodes import build_edition_index
            from ..retrieval.chunking import load_kb
            from ..sandbox import tools as sandbox_tools
            from ..sandbox.data import build_world
            from ..sandbox.logistics import LogisticsClient
            from ..store import TicketStore, seed_demo_users
            from ..toolbox import Toolbox

            st = store
            if st is None:
                s.tickets_db.parent.mkdir(parents=True, exist_ok=True)
                st = TicketStore(s.tickets_db)
                seed_demo_users(st)
            tb = toolbox
            if tb is None:
                world = build_world(7)
                logistics = LogisticsClient(lambda oid: sandbox_tools.query_shipment(world, oid))
                tb = Toolbox(world=world, store=st, logistics=logistics, db_path=s.gateway_db)
            kb_chunks = chunks if chunks is not None else load_kb(s.kb_dir)
            application.state.store, application.state.toolbox = st, tb
            application.state.chunks_by_id = {c.id: c for c in kb_chunks}
            application.state.graph = build_graph(s, r, application.state.system, cp, models=models, toolbox=tb,
                                                  edition_index=build_edition_index(kb_chunks))
        else:
            application.state.graph = build_graph(s, r, application.state.system, cp, models=models)
        yield
        if conn is not None:
            conn.close()

    app = FastAPI(title="CarePilot", lifespan=lifespan)

    if governed:
        from .governed import register_governed

        register_governed(app)
    else:
        @app.post("/tickets")
        def create_ticket(body: TicketIn, stream: bool = False):
            thread_id = uuid.uuid4().hex
            ticket_id = body.ticket_id or f"T-{thread_id[:6]}"
            config = {"configurable": {"thread_id": thread_id}}
            graph_input = {"ticket_id": ticket_id, "customer_message": body.message}
            if stream:
                return StreamingResponse(
                    _stream_graph(app.state.graph, graph_input, config, thread_id),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            app.state.graph.invoke(graph_input, config)
            return _view(app.state.graph, thread_id)

        @app.post("/tickets/{thread_id}/review")
        def review_ticket(thread_id: str, body: ReviewIn, stream: bool = False):
            config = {"configurable": {"thread_id": thread_id}}
            view = _view(app.state.graph, thread_id)
            if view["status"] != "awaiting_review":
                raise HTTPException(409, f"ticket is not awaiting review (status={view['status']})")
            command = Command(resume=body.model_dump(exclude_none=True))
            if stream:
                return StreamingResponse(
                    _stream_graph(app.state.graph, command, config, thread_id),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            app.state.graph.invoke(command, config)
            return _view(app.state.graph, thread_id)

        @app.get("/tickets/{thread_id}")
        def get_ticket(thread_id: str):
            return _view(app.state.graph, thread_id)

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


app = create_app(governed=True)
