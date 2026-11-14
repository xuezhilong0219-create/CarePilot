"""工具箱：把沙箱查询和工单存储操作注册成网关里的 ToolSpec，并提供「规划 → 来源校验 → 网关执行」的入口。

结构性防护（不依赖模型听话）：
1. 规划模型（planner）只看客户消息和对话历史，**永远看不到知识库原文和订单留言**；
   留言只能通过 query_order_remark 读取，而这个工具没有注册给 planner，所以留言里的注入文本无法影响工具选择。
2. planner 给出的订单号/工单号/店铺号必须**逐字出现在客户消息里**才会被执行（来源校验），防止凭空捏造标识符。
3. 模型只能提议只读工具。写操作（改状态、发送、升级）由确定性代码在人工批准后经网关执行，模型没有这个入口。
4. 所有调用都过网关的四道闸并写审计；订单/物流/店铺在沙箱里按调用者的店铺范围过滤，无权和不存在返回完全相同的结果。

SYSTEM_PRINCIPAL 只给服务端自己的代码用（建单、回写分诊、系统备注），不对外暴露，也不能通过请求头声明。
所有数据都是合成的沙箱数据，不代表对接了真实系统。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .auth import Principal, Role
from .gateway.gateway import ToolGateway
from .gateway.spec import CallOutcome, Executed, Gate, NeedsApproval, Rejected, Risk, RuleViolation, ToolContext, ToolSpec
from .graph.state import ToolPlan
from .sandbox import tools as sbx
from .sandbox.data import World
from .sandbox.logistics import LogisticsClient
from .store import STATUSES, NotFound, TicketStore, can_modify, can_transition, can_view

SYSTEM_PRINCIPAL = Principal("system", Role.ADMIN)
STAFF = frozenset({Role.AGENT, Role.ADMIN})
KNOWN_TEAMS = frozenset(
    {
        "product_support_team",
        "platform_integration_team",
        "warehouse_logistics_team",
        "data_support_team",
        "tier2_support",
        "oncall",
    }
)
LOGISTICS_DEGRADED_REPLY = "您好，物流信息暂时查询不到，请您稍后再试；如果长时间仍然查询不到，我们会安排同事继续跟进。"

_TICKET_ID = r"^T-\d{6}$"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrderArgs(_Args):
    order_id: str = Field(pattern=r"^[A-Za-z0-9-]{6,32}$")


class ShopArgs(_Args):
    shop_id: str = Field(pattern=r"^S\d{1,4}$")


class TicketArgs(_Args):
    ticket_id: str = Field(pattern=_TICKET_ID)


class NoteArgs(_Args):
    ticket_id: str = Field(pattern=_TICKET_ID)
    body: str = Field(min_length=1, max_length=4000)
    kind: Literal["note", "draft"]


class StatusArgs(_Args):
    ticket_id: str = Field(pattern=_TICKET_ID)
    to_status: Literal["new", "open", "pending_customer", "on_hold", "escalated", "resolved", "closed"]
    reason: str = Field(default="", max_length=500)


class SendArgs(_Args):
    ticket_id: str = Field(pattern=_TICKET_ID)
    text: str = Field(min_length=1, max_length=4000)


class EscalateArgs(_Args):
    ticket_id: str = Field(pattern=_TICKET_ID)
    team: str = Field(pattern=r"^[a-z][a-z0-9_]{2,40}$")
    reason: str = Field(min_length=1, max_length=500)
    actions_summary: str = Field(default="", max_length=1000)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class ToolRun:
    """一次规划后的工具执行结果。facts_text 是给回复节点看的事实（确定性渲染，不含留言和明文敏感信息）。"""

    outcome: Literal["ok", "not_found", "unavailable", "denied", "invalid", "none", "error"]
    tool: str | None = None
    facts_text: str = ""
    source_id: str = ""  # 当作"资料"引用的伪 chunk id，如 tool:query_order:123
    detail: str = ""  # invalid 时是缺失的参数名；error 时是原因码
    data: dict[str, Any] = field(default_factory=dict)


def render_facts(tool: str, data: dict[str, Any]) -> str:
    if tool == "query_order":
        return (
            f"订单 {data['order_id']}：平台 {data['platform']}，状态【{data['status']}】，"
            f"金额 {data['amount_yuan']} 元，下单时间 {data['created_at']}。"
        )
    if tool == "query_shipment":
        return (
            f"订单 {data['order_id']} 的物流：承运商 {data['carrier']}，最新节点「{data['latest_node']}」"
            f"（{data['latest_time']}），预计送达 {data['eta']}。"
        )
    if tool == "query_shop_auth":
        state = "已过期，需要重新授权" if data["expired"] else "有效"
        return f"店铺「{data['name']}」（{data['platform']}）的授权状态：{state}，授权到期日 {data['auth_expires_at']}。"
    if tool == "query_ticket":
        parts = [f"工单 {data['ticket_id']}：状态【{data['status']}】，优先级 {data['priority']}"]
        if "sla_remaining_minutes" in data and data["sla_remaining_minutes"] is not None:
            parts.append(f"SLA 剩余 {data['sla_remaining_minutes']} 分钟")
        parts.append("已分配处理人" if data.get("has_owner") or data.get("owner_id") else "尚未分配处理人")
        return "，".join(parts) + "。"
    return ""


class Toolbox:
    def __init__(
        self,
        *,
        world: World,
        store: TicketStore,
        logistics: LogisticsClient,
        db_path: str | Path,
        clock: Callable[[], float] = time.time,
        approval_ttl_s: float | None = 86400.0,
    ):
        self.world, self.store, self.logistics = world, store, logistics
        self.registry: dict[str, ToolSpec] = self._build_registry()
        self.gateway = ToolGateway(db_path, self.registry, clock=clock, approval_ttl_s=approval_ttl_s)

    # ------------------------------------------------------------------ 注册表
    def _scope(self, ctx: ToolContext) -> frozenset[str]:
        if ctx.principal.role is Role.CUSTOMER:
            return frozenset(ctx.principal.shop_ids)
        return frozenset(self.world.shops)

    def _ticket_or_none(self, ticket_id: str):
        try:
            return self.store.get_ticket(SYSTEM_PRINCIPAL, ticket_id)
        except NotFound:
            return None

    def _can_modify(self, ctx: ToolContext, args: Any) -> bool:
        ticket = self._ticket_or_none(args.ticket_id)
        return ticket is not None and can_modify(ctx.principal, ticket)

    def _can_view(self, ctx: ToolContext, args: Any) -> bool:
        ticket = self._ticket_or_none(args.ticket_id)
        return True if ticket is None else can_view(ctx.principal, ticket)  # 不存在交给工具返回 found=False

    def _build_registry(self) -> dict[str, ToolSpec]:
        world, store, logistics = self.world, self.store, self.logistics

        def query_order(ctx: ToolContext, a: OrderArgs) -> dict[str, Any]:
            return sbx.query_order(world, self._scope(ctx), a.order_id)

        def query_shipment(ctx: ToolContext, a: OrderArgs) -> dict[str, Any]:
            if not sbx.query_order(world, self._scope(ctx), a.order_id)["found"]:
                return {"found": False}  # 无权和不存在不可区分
            r = logistics.query(a.order_id)
            if r.get("status") != "ok":
                return {"found": None, "unavailable": True, "message": r.get("message", "")}
            r = {k: v for k, v in r.items() if k not in ("status", "attempts")}
            return r if r.get("found") else {"found": False}

        def query_shop_auth(ctx: ToolContext, a: ShopArgs) -> dict[str, Any]:
            if a.shop_id not in self._scope(ctx):
                return {"found": False}
            return sbx.query_shop_auth(world, a.shop_id)

        def query_ticket(ctx: ToolContext, a: TicketArgs) -> dict[str, Any]:
            ticket = self._ticket_or_none(a.ticket_id)
            if ticket is None:
                return {"found": False}
            out: dict[str, Any] = {"found": True, "ticket_id": ticket.ticket_id, "status": ticket.status,
                                   "priority": ticket.priority, "subject": ticket.subject}
            if ctx.principal.is_staff:  # 处理人和 SLA 是内部信息，客户只知道有没有人接手
                out["owner_id"] = ticket.owner_id
                remaining = store.sla_remaining_seconds(ticket)
                out["sla_remaining_minutes"] = None if remaining == float("inf") else round(remaining / 60, 1)
            else:
                out["has_owner"] = ticket.owner_id is not None
            return out

        def add_note(ctx: ToolContext, a: NoteArgs) -> dict[str, Any]:
            m = store.add_message(a.ticket_id, author=ctx.principal, kind=a.kind, body=a.body,
                                  visibility="internal", idem_key=ctx.idem_key)
            return {"ticket_id": a.ticket_id, "message_id": m.id, "kind": a.kind}

        def status_rule(ctx: ToolContext, a: StatusArgs) -> None:
            ticket = self._ticket_or_none(a.ticket_id)
            if ticket is None:
                raise RuleViolation("ticket_not_found")
            if not can_transition(ticket.status, a.to_status):
                raise RuleViolation("illegal_transition", f"{ticket.status} -> {a.to_status}")

        def set_status(ctx: ToolContext, a: StatusArgs) -> dict[str, Any]:
            t = store.transition(ctx.principal, a.ticket_id, a.to_status, reason=a.reason)
            return {"ticket_id": t.ticket_id, "status": t.status}

        def status_verify(ctx: ToolContext, a: StatusArgs) -> dict[str, Any] | None:
            t = self._ticket_or_none(a.ticket_id)
            return {"ticket_id": a.ticket_id, "status": t.status} if t and t.status == a.to_status else None

        def send_reply(ctx: ToolContext, a: SendArgs) -> dict[str, Any]:
            m = store.add_message(a.ticket_id, author=ctx.principal, kind="message", body=a.text,
                                  visibility="public", idem_key=ctx.idem_key)
            return {"ticket_id": a.ticket_id, "message_id": m.id}

        def send_verify(ctx: ToolContext, a: SendArgs) -> dict[str, Any] | None:
            m = store.find_message_by_idem_key(ctx.idem_key or "")
            return {"ticket_id": a.ticket_id, "message_id": m.id} if m else None

        def escalate_rule(ctx: ToolContext, a: EscalateArgs) -> None:
            ticket = self._ticket_or_none(a.ticket_id)
            if ticket is None:
                raise RuleViolation("ticket_not_found")
            if ticket.status not in ("new", "open"):
                raise RuleViolation("illegal_transition", f"{ticket.status} -> escalated")
            if a.team not in KNOWN_TEAMS:
                raise RuleViolation("unknown_team", a.team)

        def escalate(ctx: ToolContext, a: EscalateArgs) -> dict[str, Any]:
            p = ctx.principal
            if store.get_ticket(p, a.ticket_id).status == "new":
                store.transition(p, a.ticket_id, "open", reason="分诊后进入处理")
            store.transition(p, a.ticket_id, "escalated", reason=a.reason)
            store.add_message(a.ticket_id, author=p, kind="system", body=f"升级原因：{a.reason}（转交 {a.team}）",
                              visibility="internal", idem_key=f"{ctx.idem_key}:reason")
            store.add_message(a.ticket_id, author=p, kind="note", body=f"已完成动作：{a.actions_summary or '无'}",
                              visibility="internal", idem_key=f"{ctx.idem_key}:actions")
            return {"ticket_id": a.ticket_id, "status": "escalated", "team": a.team}

        def escalate_verify(ctx: ToolContext, a: EscalateArgs) -> dict[str, Any] | None:
            t = self._ticket_or_none(a.ticket_id)
            return {"ticket_id": a.ticket_id, "status": "escalated", "team": a.team} if t and t.status == "escalated" else None

        everyone = frozenset(Role)
        return {
            "query_order": ToolSpec("query_order", OrderArgs, query_order, Risk.READ, everyone,
                                    summarize_args=lambda a: f"order_id={a.order_id}"),
            "query_shipment": ToolSpec("query_shipment", OrderArgs, query_shipment, Risk.READ, everyone,
                                       summarize_args=lambda a: f"order_id={a.order_id}"),
            "query_shop_auth": ToolSpec("query_shop_auth", ShopArgs, query_shop_auth, Risk.READ, everyone,
                                        summarize_args=lambda a: f"shop_id={a.shop_id}"),
            "query_ticket": ToolSpec("query_ticket", TicketArgs, query_ticket, Risk.READ, everyone,
                                     owner_check=self._can_view, summarize_args=lambda a: f"ticket_id={a.ticket_id}"),
            "add_note": ToolSpec("add_note", NoteArgs, add_note, Risk.WRITE_INTERNAL, STAFF,
                                 owner_check=lambda ctx, a: self._ticket_or_none(a.ticket_id) is not None
                                 and can_view(ctx.principal, self._ticket_or_none(a.ticket_id)),
                                 summarize_args=lambda a: f"ticket={a.ticket_id}, kind={a.kind}, len={len(a.body)}, sha1={_digest(a.body)}"),
            "set_ticket_status": ToolSpec("set_ticket_status", StatusArgs, set_status, Risk.STATE_CHANGE, STAFF,
                                          rules=(status_rule,), owner_check=self._can_modify, verify=status_verify,
                                          summarize_args=lambda a: f"ticket={a.ticket_id}, to={a.to_status}"),
            "send_reply": ToolSpec("send_reply", SendArgs, send_reply, Risk.EXTERNAL_SEND, STAFF,
                                   owner_check=self._can_modify, verify=send_verify,
                                   summarize_args=lambda a: f"ticket={a.ticket_id}, len={len(a.text)}, sha1={_digest(a.text)}"),
            "escalate_ticket": ToolSpec("escalate_ticket", EscalateArgs, escalate, Risk.STATE_CHANGE, STAFF,
                                        rules=(escalate_rule,), owner_check=self._can_modify, verify=escalate_verify,
                                        summarize_args=lambda a: f"ticket={a.ticket_id}, team={a.team}, reason_len={len(a.reason)}"),
        }

    # ------------------------------------------------------------------ 只读工具的规划执行
    _PLAN_ARG = {"query_order": "order_id", "query_shipment": "order_id", "query_shop_auth": "shop_id", "query_ticket": "ticket_id"}

    def run_plan(self, principal: Principal, run_id: str, ticket_id: str | None, plan: ToolPlan, *, source_text: str) -> ToolRun:
        """执行 planner 提议的一次只读调用。参数必须逐字出现在 source_text（客户消息+历史）里，否则当成缺参。"""
        if plan.tool == "none":
            return ToolRun("none")
        arg_name = self._PLAN_ARG[plan.tool]
        value = getattr(plan, arg_name)
        if not value or str(value) not in source_text:
            return ToolRun("invalid", plan.tool, detail=arg_name)
        ctx = ToolContext(principal, run_id, ticket_id)
        out = self.gateway.call(ctx, plan.tool, {arg_name: str(value)})
        return self._interpret(plan.tool, out)

    @staticmethod
    def _interpret(tool: str, out: CallOutcome) -> ToolRun:
        if isinstance(out, Rejected):
            if out.gate is Gate.SCHEMA:
                return ToolRun("invalid", tool, detail="format")
            if out.gate is Gate.PERMISSION:
                return ToolRun("denied", tool, detail=out.code)
            return ToolRun("error", tool, detail=out.code)
        if not isinstance(out, Executed):
            return ToolRun("error", tool, detail="unexpected_outcome")
        data = out.result
        if data.get("unavailable"):
            return ToolRun("unavailable", tool, facts_text=LOGISTICS_DEGRADED_REPLY, source_id=f"tool:{tool}:unavailable", data=data)
        if not data.get("found"):
            return ToolRun("not_found", tool, data=data)
        ref = data.get("order_id") or data.get("shop_id") or data.get("ticket_id") or ""
        return ToolRun("ok", tool, facts_text=render_facts(tool, data), source_id=f"tool:{tool}:{ref}", data=data)

    # ------------------------------------------------------------------ 写操作（人工批准后由确定性代码执行）
    def propose_and_approve(self, requester: Principal, approver: Principal, run_id: str, ticket_id: str,
                            tool: str, args: dict[str, Any]) -> CallOutcome:
        """提议并立即用人工审核的结果批准。STATE_CHANGE 到此为止已执行；EXTERNAL_SEND 返回 NeedsApproval(stage=confirm)。"""
        out = self.gateway.call(ToolContext(requester, run_id, ticket_id), tool, args)
        if isinstance(out, NeedsApproval) and out.stage == "approve":
            out = self.gateway.approve(out.approval_id, approver)
        return out
