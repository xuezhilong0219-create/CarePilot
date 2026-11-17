"""治理模式的 HTTP 接口：工单落库、三角色、多轮追问、人工审核 + 二次确认、交接摘要、审计、引用原文。

身份：请求头 `X-User-Id` 只用来**指明是哪个用户**，角色和店铺范围一律从用户表里取，不接受请求自报的角色。
这是演示用的模拟身份，**不是真正的认证**（没有密码、没有令牌），不能直接拿去上线。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from ..auth import Principal, Role
from ..gateway.guard import BudgetExceeded, RunBudget, RunGuard
from ..gateway.spec import ToolContext
from ..observability.tracer import RunTracer
from ..store import Conflict, Forbidden, IllegalTransition, NotFound, Ticket, build_handoff, can_modify
from ..toolbox import SYSTEM_PRINCIPAL, principal_to_dict
from .app import _sse, _summary
from .ui import CSP, render_ui

def _trace_summary(t: Any) -> dict[str, Any]:
    """一轮 trace 的摘要：列表用得到，详情复用同一套字段，免得两边算出不同的数。"""
    end = t.ended_at or None  # 没结束的轮次落盘是 0.0，不是 None
    if end is None and t.spans:
        end = max(s.started_at + s.duration_ms / 1000.0 for s in t.spans)
    return {
        "run_id": t.run_id,
        "status": t.status,
        "open": not t.ended_at,
        "category": t.category,
        "started_at": t.started_at,
        "ended_at": end,
        # 两个数分开给：duration_ms 是真正算掉的时间，wall_ms 含人工审核那几十分钟的等待。
        "duration_ms": round(sum(s.duration_ms for s in t.spans)),
        "wall_ms": round((end - t.started_at) * 1000) if end else None,
        "nodes": len(t.spans),
    }


PRIORITY_BY_SEVERITY = {"low": "low", "medium": "medium", "high": "high", "critical": "critical"}


class _MessageBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class TicketCreate(_MessageBody):
    shop_id: str | None = Field(default=None, pattern=r"^S\d{1,4}$")


class MessageIn(_MessageBody):
    pass


class ReviewBody(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reply: str | None = Field(default=None, max_length=4000)  # action=edit 时，审核人改写后的回复
    note: str | None = Field(default=None, max_length=500)
    confirm: bool = False  # True：批准后直接确认发送；默认两步，确认前客户收不到任何东西

    @field_validator("reply")
    @classmethod
    def reject_whitespace_only_reply(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("reply must not be blank")
        return value


def register_governed(app: FastAPI) -> None:
    @app.exception_handler(Forbidden)
    async def _forbidden(_: Request, exc: Forbidden) -> JSONResponse:
        return JSONResponse({"detail": str(exc) or "forbidden"}, status_code=403)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, exc: Conflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(IllegalTransition)
    async def _illegal(_: Request, exc: IllegalTransition) -> JSONResponse:
        return JSONResponse({"detail": f"illegal transition: {exc}"}, status_code=409)

    def current_principal(request: Request, x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> Principal:
        principal = request.app.state.store.get_principal(x_user_id) if x_user_id else None
        if principal is None:
            raise HTTPException(401, "missing or unknown X-User-Id")
        return principal

    def staff_only(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.is_staff:
            raise HTTPException(403, "staff only")
        return principal

    def admin_only(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.role is not Role.ADMIN:
            raise HTTPException(403, "admin only")
        return principal

    # ------------------------------------------------------------------ 内部工具
    def thread_config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def pending_of(snap) -> list[dict]:
        return [i.value for t in snap.tasks for i in t.interrupts]

    def history_for(ticket: Ticket) -> list[dict]:
        customer = Principal(ticket.customer_id, Role.CUSTOMER)  # 只取公开消息（客户视角）
        rows = app.state.store.list_messages(customer, ticket.ticket_id)
        return [{"role": "customer" if m.author_role == "customer" else "agent", "text": m.body}
                for m in rows if m.kind == "message"]

    def sync_after_run(ticket_id: str, thread_id: str, values: dict) -> None:
        """把本轮图的结果回写到工单：分类、优先级、累积的槽位和最近一次运行的 thread。"""
        store = app.state.store
        triage = values.get("triage") or {}
        if triage.get("category"):
            store.set_category(SYSTEM_PRINCIPAL, ticket_id, triage["category"])
        priority = PRIORITY_BY_SEVERITY.get(triage.get("severity", ""))
        if priority and store.get_ticket(SYSTEM_PRINCIPAL, ticket_id).priority != priority:
            store.set_priority(SYSTEM_PRINCIPAL, ticket_id, priority)
        store.put_context(ticket_id, slots=values.get("slots") or {}, last_thread_id=thread_id)

    def build_view(principal: Principal, ticket_id: str) -> dict[str, Any]:
        store, graph = app.state.store, app.state.graph
        ticket = store.get_ticket(principal, ticket_id)
        ctx = store.get_context(ticket_id)
        values: dict = {}
        pending: list[dict] = []
        if ctx["last_thread_id"]:
            snap = graph.get_state(thread_config(ctx["last_thread_id"]))
            values, pending = dict(snap.values), pending_of(snap)
        workflow = "awaiting_review" if pending else values.get("status")
        if ticket.status in ("resolved", "escalated", "closed") and not pending:
            workflow = ticket.status
        out: dict[str, Any] = {
            "ticket": {"ticket_id": ticket.ticket_id, "status": ticket.status, "priority": ticket.priority,
                       "category": ticket.category, "subject": ticket.subject, "created_at": ticket.created_at},
            "status": workflow,
            "followup_question": values.get("followup_question"),
            "messages": [{"id": m.id, "author_role": m.author_role, "kind": m.kind, "visibility": m.visibility,
                          "body": m.body, "created_at": m.created_at} for m in store.list_messages(principal, ticket_id)],
        }
        if principal.is_staff:  # 内部信息：处理人、SLA、待审核卡片、待确认、执行动作都只给坐席/管理员
            remaining = store.sla_remaining_seconds(ticket)
            out["ticket"].update(owner_id=ticket.owner_id,
                                 sla_remaining_minutes=None if remaining == float("inf") else round(remaining / 60, 1))
            card = dict(pending[0]) if pending else None
            if card and card.get("citation_details"):
                card["citation_details"] = [{**c, "link": f"/citations/{quote(c['id'], safe='/:')}"} for c in card["citation_details"]]
            out["pending_review"] = card
            out["pending_confirmation"] = values.get("pending_confirmation") if workflow == "awaiting_confirmation" else None
            out["reply_edited"] = bool(values.get("reply_edited"))
            out["actions"] = values.get("actions")
            out["triage"] = values.get("triage")
            out["abort_reason"] = values.get("escalate_reason") if workflow == "needs_human" else None
        return out

    def _pending_send(ticket_id: str) -> bool:
        """这张单是不是有一条已经批准、等坐席二次确认的回复。

        这种状态下工单背后有一个网关里的待确认审批（approval_id）。客户再发一句会开新一轮，确认入口跟着 last_thread_id 走，那条审批就再也没人点得开——所以这里拦住。
        """
        store, graph = app.state.store, app.state.graph
        thread_id = store.get_context(ticket_id)["last_thread_id"]
        if not thread_id:
            return False
        status = dict(graph.get_state(thread_config(thread_id)).values).get("status")
        return status == "awaiting_confirmation"

    def tracer_for(thread_id: str, ticket_id: str) -> RunTracer:
        """同一个 run 跨审核阶段共用一个 tracer；进程重启后从磁盘上的 trace 续写，不覆盖之前的节点。"""
        runs = app.state.runs
        if thread_id not in runs:
            try:
                runs[thread_id] = RunTracer.from_trace(app.state.trace_store.load(thread_id))
            except (FileNotFoundError, ValueError):
                runs[thread_id] = RunTracer(thread_id, ticket_id=ticket_id)
        return runs[thread_id]

    def new_guard() -> RunGuard:
        st = app.state.settings
        return RunGuard(RunBudget(max_steps=st.run_max_steps, max_tokens=st.run_max_tokens, max_seconds=st.run_max_seconds))

    def run_config(thread_id: str, tracer: RunTracer) -> dict:
        return {"configurable": {"thread_id": thread_id, "tracer": tracer, "guard": new_guard()}}

    def save_trace(tracer: RunTracer, view: dict[str, Any] | None, *, triage: dict | None, aborted: bool = False) -> None:
        # 分诊结果取自图的状态，不能取自 view：客户视角的 view 不含 triage（内部信息）
        tracer.set_category((triage or {}).get("category"))
        status = "budget_exceeded" if aborted else (view or {}).get("status") or "running"
        if status in ("awaiting_review", "awaiting_confirmation"):
            tracer.set_status(status)
        else:
            tracer.end(status)
        app.state.trace_store.save(tracer.snapshot())

    def abort_run(ticket_id: str, thread_id: str, cfg: dict, tracer: RunTracer, reason: str) -> None:
        """预算守卫中止：不替人做状态变更（改状态需要人工批准），只留一条内部备注，把工单标成需要人工接手。"""
        tb = app.state.toolbox
        tb.gateway.call(ToolContext(SYSTEM_PRINCIPAL, thread_id, ticket_id), "add_note",
                        {"ticket_id": ticket_id, "kind": "note", "body": f"自动处理被预算守卫中止（{reason}），请人工接手。"})
        app.state.graph.update_state(cfg, {"status": "needs_human", "escalate_reason": f"budget:{reason}"}, as_node="execute_actions")
        app.state.store.put_context(ticket_id, slots=app.state.store.get_context(ticket_id)["slots"], last_thread_id=thread_id)

    def run_graph(graph_input: Any, thread_id: str, ticket: Ticket, principal: Principal, *, stream: bool):
        graph = app.state.graph
        tracer = tracer_for(thread_id, ticket.ticket_id)
        cfg = run_config(thread_id, tracer)

        def finish(aborted_reason: str | None = None) -> dict[str, Any]:
            if aborted_reason:
                abort_run(ticket.ticket_id, thread_id, cfg, tracer, aborted_reason)
            values = dict(graph.get_state(cfg).values)
            sync_after_run(ticket.ticket_id, thread_id, values)
            view = build_view(principal, ticket.ticket_id)
            save_trace(tracer, view, triage=values.get("triage"), aborted=bool(aborted_reason))
            return view

        if not stream:
            try:
                graph.invoke(graph_input, cfg)
            except BudgetExceeded as exc:
                return finish(exc.reason)
            return finish()
        return StreamingResponse(_stream(graph_input, cfg, finish), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def run_turn(principal: Principal, ticket: Ticket, message: str, *, stream: bool):
        store = app.state.store
        ctx = store.get_context(ticket.ticket_id)
        history = history_for(ticket)  # 必须在写入本条消息之前取，否则本句会同时出现在"历史"和"本句"里
        store.add_message(ticket.ticket_id, author=principal, kind="message", body=message, visibility="public")
        thread_id = uuid.uuid4().hex
        state = {"ticket_id": ticket.ticket_id, "run_id": thread_id, "customer_message": message,
                 "principal": principal_to_dict(principal), "history": history, "slots": ctx["slots"]}
        return run_graph(state, thread_id, ticket, principal, stream=stream)

    def _stream(graph_input: Any, cfg: dict, finish) -> Iterator[str]:
        try:
            for update in app.state.graph.stream(graph_input, cfg, stream_mode="updates"):
                for node, payload in update.items():
                    if node != "__interrupt__":
                        yield _sse("node", {"node": node, "summary": _summary(node, payload)})
            view = finish()
        except BudgetExceeded as exc:
            view = finish(exc.reason)
        except Exception as exc:  # 流已开始，异常只能作为事件发出
            yield _sse("error", {"message": str(exc)})
            return
        yield _sse("result", view)

    # ------------------------------------------------------------------ 路由
    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        """工作台页面。纯静态 HTML，业务数据一律由前端用 JSON 接口取。"""
        return HTMLResponse(render_ui(), headers={"Content-Security-Policy": CSP})

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        """切换身份后用它决定界面显示哪些控件（角色和店铺范围都取自用户表）。"""
        return {"user_id": principal.user_id, "role": principal.role.value, "shops": sorted(principal.shop_ids)}

    @app.post("/tickets")
    def create_ticket(body: TicketCreate, stream: bool = False, principal: Principal = Depends(current_principal)):
        if principal.role is not Role.CUSTOMER:
            raise HTTPException(403, "only customers can open tickets")
        shop_id = body.shop_id or (sorted(principal.shop_ids)[0] if principal.shop_ids else None)
        if body.shop_id and body.shop_id not in principal.shop_ids:
            raise Forbidden("shop does not belong to caller")
        ticket = app.state.store.create_ticket(principal, subject=body.message.strip()[:40], shop_id=shop_id, thread_id="")
        return run_turn(principal, ticket, body.message.strip(), stream=stream)

    @app.post("/tickets/{ticket_id}/messages")
    def add_message(ticket_id: str, body: MessageIn, stream: bool = False, principal: Principal = Depends(current_principal)):
        """客户在同一张工单里继续说：上下文（历史 + 累积槽位）带进新一轮，槽位不丢。"""
        ticket = app.state.store.get_ticket(principal, ticket_id)
        if principal.role is not Role.CUSTOMER:
            raise HTTPException(403, "only the ticket's customer can add customer messages")
        if ticket.status == "closed":
            raise Conflict("ticket is closed")
        # 等二次确认时再发一句会开新一轮，网关里那条已批准的发送就再也没人点得开（孤儿审批）。
        # 等审核时不管：那一轮只是图停在一个 interrupt 上，开新一轮不会留下卸不掉的东西。
        if _pending_send(ticket_id):
            raise Conflict("ticket is awaiting_confirmation; wait for the agent to send or discard")
        return run_turn(principal, ticket, body.message.strip(), stream=stream)

    @app.get("/tickets")
    def list_tickets(status: str | None = None, mine: bool = False, principal: Principal = Depends(current_principal)):
        rows = app.state.store.list_tickets(principal, status=status, mine=mine)
        return [{"ticket_id": t.ticket_id, "status": t.status, "priority": t.priority, "category": t.category,
                 "subject": t.subject, "owner_id": t.owner_id if principal.is_staff else None} for t in rows]

    @app.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: str, principal: Principal = Depends(current_principal)):
        return build_view(principal, ticket_id)

    @app.post("/tickets/{ticket_id}/review")
    def review(ticket_id: str, body: ReviewBody, stream: bool = False, principal: Principal = Depends(staff_only)):
        store, graph = app.state.store, app.state.graph
        ticket = store.get_ticket(principal, ticket_id)
        thread_id = store.get_context(ticket_id)["last_thread_id"]
        if not thread_id or not pending_of(graph.get_state(thread_config(thread_id))):
            raise Conflict("ticket is not awaiting review")
        if principal.role is Role.AGENT:  # 坐席审核前先认领；别人负责的不能碰
            if ticket.owner_id is None:
                store.claim(principal, ticket_id)
            elif not can_modify(principal, ticket):
                raise Forbidden("ticket is owned by another agent")
        resume = Command(resume={**body.model_dump(exclude_none=True), "reviewer": principal_to_dict(principal)})
        return run_graph(resume, thread_id, ticket, principal, stream=stream)

    @app.post("/tickets/{ticket_id}/confirm")
    def confirm(ticket_id: str, principal: Principal = Depends(staff_only)):
        """对外发送的二次确认。幂等：重复确认返回同一结果，不会产生第二条消息。"""
        store, graph, tb = app.state.store, app.state.graph, app.state.toolbox
        ticket = store.get_ticket(principal, ticket_id)
        thread_id = store.get_context(ticket_id)["last_thread_id"]
        values = dict(graph.get_state(thread_config(thread_id)).values) if thread_id else {}
        pc = values.get("pending_confirmation")
        if not pc:
            raise Conflict("nothing to confirm")
        if not can_modify(principal, ticket):
            raise Forbidden("only the ticket owner (or an admin) can confirm")
        from ..gateway.spec import Executed, Rejected

        out = tb.gateway.confirm(pc["approval_id"], principal)
        if isinstance(out, Rejected):
            raise HTTPException(403 if out.gate.value == "permission" else 409, f"{out.gate.value}:{out.code}")
        assert isinstance(out, Executed)
        if store.get_ticket(SYSTEM_PRINCIPAL, ticket_id).status not in ("resolved", "closed"):
            tb.resolve_ticket(principal, thread_id, ticket_id)
        graph.update_state(thread_config(thread_id), {"status": "resolved"}, as_node="execute_actions")
        return build_view(principal, ticket_id)

    @app.get("/tickets/{ticket_id}/handoff")
    def handoff(ticket_id: str, principal: Principal = Depends(staff_only)):
        store, tb = app.state.store, app.state.toolbox
        ticket = store.get_ticket(principal, ticket_id)
        actions = [{"tool": r["tool"], "summary": r["args_summary"], "at": r["ts"]}
                   for r in tb.gateway.audit(ticket_id=ticket_id, limit=500) if r["decision"] == "executed"]
        h = build_handoff(ticket, store.list_messages(principal, ticket_id), actions)
        return {k: getattr(h, k) for k in ("ticket_id", "escalation_reason", "actions_done", "open_questions",
                                           "last_customer_message", "owner_id", "status")}

    @app.get("/tickets/{ticket_id}/audit")
    def audit(ticket_id: str, principal: Principal = Depends(staff_only)):
        app.state.store.get_ticket(principal, ticket_id)
        return app.state.toolbox.gateway.audit(ticket_id=ticket_id, limit=500)

    @app.get("/tickets/{ticket_id}/traces")
    def traces(ticket_id: str, principal: Principal = Depends(staff_only)) -> dict[str, Any]:
        """这张工单跑过的每一轮（新的在前），只给摘要；点开某一轮再取详情。"""
        app.state.store.get_ticket(principal, ticket_id)
        return {"runs": [_trace_summary(t) for t in app.state.trace_store.runs_for(ticket_id)]}

    @app.get("/tickets/{ticket_id}/trace")
    def trace(ticket_id: str, run_id: str | None = None,
              principal: Principal = Depends(staff_only)) -> dict[str, Any]:
        """某一轮的节点级 trace：每个节点花了多久、模型调用多少 token、有没有报错。

        不给 `run_id` 就取最近一轮。只读磁盘上已经落好的那份 JSON，不重跑图、
        不碰网关账本，所以随便点不产生副作用。
        """
        store = app.state.store
        store.get_ticket(principal, ticket_id)
        if run_id is None:
            thread_id = store.get_context(ticket_id)["last_thread_id"]
            if not thread_id:
                raise NotFound(f"{ticket_id} 还没有跑过")
            run_id = thread_id
        try:
            t = app.state.trace_store.load(run_id)
        except (FileNotFoundError, ValueError):
            raise NotFound(f"{ticket_id} 这一轮没有 trace") from None
        if t.ticket_id != ticket_id:
            raise NotFound(f"{run_id} 不属于 {ticket_id}")  # 别的工单的 run_id 不能串门
        prompt, completion = t.total_tokens
        return {
            **_trace_summary(t),
            "tokens": {
                "prompt": prompt,
                "completion": completion,
                "total": prompt + completion,
                "usage_missing": any(c.usage_missing for c in t.llm_calls),
            },
            "nodes": [
                {
                    "node": s.name,
                    "started_at": s.started_at,
                    "duration_ms": round(s.duration_ms, 1),
                    "input": s.input_summary,
                    "output": s.output_summary,
                    "error": s.error,
                    "llm": [
                        {"purpose": c.purpose, "model": c.model, "prompt_tokens": c.prompt_tokens,
                         "completion_tokens": c.completion_tokens, "latency_ms": round(c.latency_ms, 1),
                         "usage_missing": c.usage_missing}
                        for c in s.llm_calls
                    ],
                }
                for s in t.spans
            ],
        }

    @app.get("/admin/inflight")
    def admin_inflight(_principal: Principal = Depends(admin_only)) -> dict[str, Any]:
        """started 还没结束的调用，附「有没有活租约」。只看，不动。"""
        return {"inflight": app.state.toolbox.gateway.inflight()}

    @app.post("/admin/reconcile")
    def admin_reconcile(min_age_s: float = 0.0, dry_run: bool = False,
                        _principal: Principal = Depends(admin_only)) -> dict[str, Any]:
        """手动对账。`dry_run=true` 只列候选：认领副作用不可回退，先看一眼再动手。"""
        if min_age_s < 0:
            raise HTTPException(400, "min_age_s must be >= 0")
        outcomes = app.state.toolbox.gateway.reconcile(min_age_s=min_age_s, dry_run=dry_run)
        return {"dry_run": dry_run, "outcomes": outcomes}

    @app.get("/citations/{chunk_id:path}")
    def citation(chunk_id: str, principal: Principal = Depends(staff_only)):
        """引用点开后看到的原文片段。只给坐席：里面有内部 wiki 的来源链接。"""
        chunk = app.state.chunks_by_id.get(chunk_id)
        if chunk is None:
            raise HTTPException(404, "citation not found")
        head, _, body = chunk.text.partition("\n")
        parts = [p.strip() for p in head.split(" > ")]
        return {"id": chunk.id, "doc_id": chunk.doc_id, "doc_title": parts[0], "section": " > ".join(parts[1:]),
                "edition": chunk.meta.get("top_dir", ""), "source_url": chunk.meta.get("source_url", ""), "text": chunk.text}
