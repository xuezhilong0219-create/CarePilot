"""MCP 通道：把工具箱里的工具按 Model Context Protocol 暴露出去，执行仍然落在网关。

为什么 MCP 之上还要网关：
1. MCP 规范里没有权限模型——没有角色、没有审批流、没有审计表。服务器只能决定
   「暴露哪些工具」，决定不了「这一次调用能不能做」。谁连上谁就能调。
2. 本项目的九个工具已经全部注册在 ``Toolbox.registry`` 里，每次调用都要过网关四道闸
   （schema / 规则 / 权限 / 风险）+ 审计 + 幂等账本。所以这里刻意只做传输层：
   MCP 负责把 schema 和结果搬到对面，``gateway.call()`` 仍然决定能不能执行。
3. 风险等级 ≥ STATE_CHANGE 的工具在这条通道上永远拿到 ``needs_approval``：
   审批需要人，而 MCP 通道上没有人。这不是限制，是让「必须有人批」无法被绕过。

身份问题：MCP 客户端不携带用户身份，所以身份来自服务器启动参数（``--user``），
按用户表还原角色和店铺范围——和 HTTP 接口一样，不信任调用方自报的角色。
MCP 服务器没有认证这件事本身，就是「谁连上谁就能调」的来源，这一点写在这里而不是藏起来。

一次服务器进程 = 一个 run_id。同一进程内重复的等价调用会命中幂等账本并回放，
不重复产生副作用。
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable

from mcp.server.mcpserver import MCPServer

from .auth import Principal
from .config import Settings
from .gateway.spec import Executed, NeedsApproval, Rejected, ToolContext
from .sandbox import tools as sandbox_tools
from .sandbox.data import build_world
from .sandbox.logistics import LogisticsClient
from .store import TicketStore, seed_demo_users
from .toolbox import Toolbox

log = logging.getLogger(__name__)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "query_order": "按订单号查询订单：平台、状态、金额、下单时间。只能查调用者店铺范围内的订单。",
    "query_shipment": "按订单号查询物流：承运商、最新节点、预计送达。无权和不存在返回相同结果。",
    "query_shop_auth": "按店铺号查询授权状态：是否过期、到期日。只能查调用者自己的店铺。",
    "query_ticket": "按工单号查询工单：状态、优先级、是否已分配处理人。",
    "add_note": "给工单加一条内部备注或草稿，客户看不到。",
    "post_followup": "用系统模板向客户追问信息。只能使用固定模板开头，不能任意发言。",
    "set_ticket_status": "变更工单状态，受状态机约束。这是写操作，需要人工批准。",
    "send_reply": "把回复发送给客户。对外发送，需要人工批准 + 二次确认。",
    "escalate_ticket": "把工单升级并转交给指定团队。需要人工批准。",
}

INSTRUCTIONS = (
    "CarePilot 售后工单助手的数据查询与工单操作接口。"
    "只读工具会立即执行并写审计；写操作（改状态、发送、升级、追问）会返回 needs_approval，"
    "因为审批必须由人完成，MCP 通道上没有人可以批准。"
)


def _as_parameter(name: str, info: Any) -> inspect.Parameter:
    """把 pydantic 的 FieldInfo 变回一个函数参数：约束（pattern/max_length）原样带进 MCP 的 inputSchema。"""
    annotation = Annotated[info.annotation, info]
    default = inspect.Parameter.empty if info.is_required() else info.default
    return inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=default)


def tool_signature(args_model: type) -> inspect.Signature:
    """MCP 的 inputSchema 由函数签名生成，所以签名必须从 ToolSpec.args_model 反推——一份定义，两处使用。"""
    return inspect.Signature([_as_parameter(n, f) for n, f in args_model.model_fields.items()])


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def execute(toolbox: Toolbox, principal: Principal, run_id: str, tool: str, arguments: dict[str, Any]) -> str:
    """一次 MCP 调用 → 一次 gateway.call()。返回给对面的永远是一段 JSON，拒绝和待审批都不是异常。"""
    ticket_id = arguments.get("ticket_id")
    ctx = ToolContext(principal, run_id, ticket_id if isinstance(ticket_id, str) else None)
    outcome = toolbox.gateway.call(ctx, tool, arguments)

    if isinstance(outcome, Executed):
        return _dumps({"outcome": "executed", "tool": tool, "replayed": outcome.replayed, "result": outcome.result})
    if isinstance(outcome, NeedsApproval):
        return _dumps({
            "outcome": "needs_approval",
            "tool": tool,
            "approval_id": outcome.approval_id,
            "stage": outcome.stage,
            "preview": outcome.preview,
            "note": "写操作必须由人工批准；这条通道上没有人可以批准，请在工单界面处理。",
        })
    assert isinstance(outcome, Rejected)
    return _dumps({"outcome": "rejected", "tool": tool, "gate": outcome.gate.value, "code": outcome.code,
                   "message": outcome.message})


def _make_handler(tool: str, args_model: type, dispatch: Callable[[str, dict[str, Any]], str]) -> Callable[..., Any]:
    """每个工具一个具名函数，签名由 args_model 生成。MCP 靠签名推 schema，所以不能用 **kwargs 糊过去。"""

    async def handler(**kwargs: Any) -> str:
        return dispatch(tool, kwargs)

    handler.__name__ = tool
    handler.__qualname__ = tool
    handler.__doc__ = TOOL_DESCRIPTIONS.get(tool)
    handler.__signature__ = tool_signature(args_model)  # type: ignore[attr-defined]
    return handler


def build_server(toolbox: Toolbox, principal: Principal, *, run_id: str, name: str = "carepilot") -> MCPServer:
    server = MCPServer(name=name, instructions=INSTRUCTIONS)
    dispatch = partial(execute, toolbox, principal, run_id)
    for tool, spec in toolbox.registry.items():
        server.add_tool(
            _make_handler(tool, spec.args_model, dispatch),
            name=tool,
            description=TOOL_DESCRIPTIONS.get(tool),
            structured_output=False,
        )
    return server


def build_toolbox(settings: Settings | None = None, data_dir: Path | None = None) -> tuple[Toolbox, TicketStore]:
    """和 api/app.py 的 governed 分支同一套装配，只是不需要图、检索和 checkpoint。

    `Settings` 把 LLM 密钥设成必填，但这条通道从不调用模型——一个只想用工具、
    手上没有模型的客户端不该被迫填一个假密钥。`model_construct` 跳过校验，
    这里只用得上那些有默认值的路径字段。
    """
    s = settings or Settings.model_construct()
    tickets_db = (data_dir / "tickets.sqlite") if data_dir else s.tickets_db
    gateway_db = (data_dir / "gateway.sqlite") if data_dir else s.gateway_db
    tickets_db.parent.mkdir(parents=True, exist_ok=True)
    gateway_db.parent.mkdir(parents=True, exist_ok=True)
    store = TicketStore(tickets_db)
    seed_demo_users(store)
    world = build_world(7)
    logistics = LogisticsClient(lambda order_id: sandbox_tools.query_shipment(world, order_id))
    return Toolbox(world=world, store=store, logistics=logistics, db_path=gateway_db), store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carepilot-mcp", description="把 CarePilot 的工具暴露成 MCP server（stdio）")
    parser.add_argument("--user", required=True, help="以哪个演示用户的身份执行：c1/c2/c3/a1/a2/admin1")
    parser.add_argument("--data-dir", type=Path, default=None, help="SQLite 落盘目录，默认取配置")
    parser.add_argument("--run-id", default=None, help="幂等键的 run 部分，默认每次启动新生成")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    try:
        toolbox, store = build_toolbox(data_dir=args.data_dir)
    except OSError as exc:
        # 目录不可写（挂载只读、被别的用户占了）时给一句话，不要甩 traceback：
        # MCP 客户端只会看到进程退出，栈对它没有意义。
        print(f"cannot prepare data dir: {exc}", file=sys.stderr)
        return 2
    principal = store.get_principal(args.user)
    if principal is None:
        print(f"unknown user: {args.user}（可用：c1/c2/c3/a1/a2/admin1）", file=sys.stderr)
        return 2
    run_id = args.run_id or f"mcp-{uuid.uuid4().hex[:12]}"
    log.info("MCP server 启动：user=%s role=%s run_id=%s", principal.user_id, principal.role.value, run_id)
    build_server(toolbox, principal, run_id=run_id).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
