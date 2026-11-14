import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from ..config import get_settings
from ..graph.build import build_graph
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.chunking import load_kb
from ..tools import TicketSystem


class TicketIn(BaseModel):
    message: str
    ticket_id: str | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reply: str | None = None  # action=edit 时，审核人改写后的回复
    note: str | None = None  # 升级时附带给接单团队的备注


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    app.state.system = TicketSystem()
    retriever = BM25Retriever(load_kb(settings.kb_dir))
    app.state.graph = build_graph(settings, retriever, app.state.system, SqliteSaver(conn))
    yield
    conn.close()


app = FastAPI(title="CarePilot", lifespan=lifespan)


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


@app.post("/tickets")
def create_ticket(body: TicketIn):
    thread_id = uuid.uuid4().hex
    ticket_id = body.ticket_id or f"T-{thread_id[:6]}"
    app.state.graph.invoke(
        {"ticket_id": ticket_id, "customer_message": body.message},
        {"configurable": {"thread_id": thread_id}},
    )
    return _view(app.state.graph, thread_id)


@app.post("/tickets/{thread_id}/review")
def review_ticket(thread_id: str, body: ReviewIn):
    view = _view(app.state.graph, thread_id)
    if view["status"] != "awaiting_review":
        raise HTTPException(409, f"ticket is not awaiting review (status={view['status']})")
    app.state.graph.invoke(
        Command(resume=body.model_dump(exclude_none=True)),
        {"configurable": {"thread_id": thread_id}},
    )
    return _view(app.state.graph, thread_id)


@app.get("/tickets/{thread_id}")
def get_ticket(thread_id: str):
    return _view(app.state.graph, thread_id)


@app.get("/health")
def health():
    return {"ok": True}
