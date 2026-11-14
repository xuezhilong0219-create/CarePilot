"""工具网关的冻结接口。实现在 gateway/gateway.py（由网关 worker 完成），调用方只依赖这里。

核心约定：**模型只能"提议"一次工具调用（工具名 + 参数），是否放行由网关里的确定性代码决定，网关内没有任何模型调用。**
四道闸按顺序执行，第一道不过就返回，且每次调用（包括被拒绝的）都写审计：
    1. schema      参数能否通过 pydantic 校验（未知工具、多余字段都算不通过）
    2. rule        业务规则（如工单状态迁移是否合法），由 ToolSpec.rules 提供
    3. permission  角色是否允许调用该工具 + 资源归属（如客户只能查自己店铺的订单）
    4. risk        按风险等级决定直接执行，还是要人工批准、二次确认
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Protocol

from pydantic import BaseModel

from ..auth import Principal, Role


class Risk(IntEnum):
    READ = 0  # 只读：自动执行
    WRITE_INTERNAL = 1  # 内部写（备注、草稿）：自动执行，必须审计
    STATE_CHANGE = 2  # 改业务状态：必须有人工批准
    EXTERNAL_SEND = 3  # 对外发送：必须批准 + 二次确认，且幂等


class Gate(str, Enum):
    SCHEMA = "schema"
    RULE = "rule"
    PERMISSION = "permission"
    RISK = "risk"


class RuleViolation(Exception):
    """业务规则不通过。code 是稳定的机器可读标识（如 illegal_transition），message 给人看。"""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass(frozen=True)
class ToolContext:
    principal: Principal
    run_id: str  # 一次工单处理的运行标识；恢复（resume）同一次运行时保持不变，幂等键依赖它
    ticket_id: str | None = None
    idem_key: str | None = None  # 由网关在执行时填入，工具可以把它透传给下游做去重


ToolFn = Callable[[ToolContext, Any], dict[str, Any]]  # 返回值必须可 JSON 序列化
RuleFn = Callable[[ToolContext, Any], None]  # 违反时 raise RuleViolation
OwnerFn = Callable[[ToolContext, Any], bool]  # 资源归属检查，True 表示允许
VerifyFn = Callable[[ToolContext, Any], "dict[str, Any] | None"]  # 对账：副作用已发生则返回其结果，否则 None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[BaseModel]  # 应设 model_config = ConfigDict(extra="forbid")
    fn: ToolFn
    risk: Risk
    roles: frozenset[Role]  # 允许调用的角色
    rules: tuple[RuleFn, ...] = ()
    owner_check: OwnerFn | None = None
    verify: VerifyFn | None = None
    # 审计里的参数摘要。涉及手机号等敏感字段的工具必须提供，保证明文不进审计日志
    summarize_args: Callable[[Any], str] | None = None


@dataclass(frozen=True)
class Executed:
    result: dict[str, Any]
    replayed: bool = False  # True 表示这次是从幂等账本里回放的结果，没有再次产生副作用


@dataclass(frozen=True)
class NeedsApproval:
    approval_id: str
    tool: str
    preview: dict[str, Any]  # 给审批人看的预览（参数摘要，不含敏感明文）
    stage: str  # "approve"（等批准）或 "confirm"（已批准，等二次确认）


@dataclass(frozen=True)
class Rejected:
    gate: Gate
    code: str
    message: str


CallOutcome = Executed | NeedsApproval | Rejected


class Gateway(Protocol):
    def call(self, ctx: ToolContext, tool: str, raw_args: dict[str, Any]) -> CallOutcome:
        """走四道闸。READ / WRITE_INTERNAL 直接执行；STATE_CHANGE / EXTERNAL_SEND 返回 NeedsApproval。"""

    def approve(self, approval_id: str, approver: Principal) -> CallOutcome:
        """批准。STATE_CHANGE：执行并返回 Executed；EXTERNAL_SEND：返回 NeedsApproval(stage="confirm")。
        只有 agent / admin 可以批准，否则返回 Rejected(Gate.PERMISSION, ...)。"""

    def confirm(self, approval_id: str, approver: Principal) -> CallOutcome:
        """二次确认（仅 EXTERNAL_SEND）。首次确认执行；重复确认返回 Executed(replayed=True)，不产生第二次副作用。"""

    def reject(self, approval_id: str, approver: Principal, reason: str) -> None: ...

    def reconcile(self) -> list[dict[str, Any]]:
        """进程重启后调用：对"已预留、未完成"的账本条目调用 ToolSpec.verify 对账，决定是认领已发生的副作用还是允许重试。"""

    def audit(
        self,
        *,
        run_id: str | None = None,
        ticket_id: str | None = None,
        actor_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """按时间顺序返回审计记录。"""
