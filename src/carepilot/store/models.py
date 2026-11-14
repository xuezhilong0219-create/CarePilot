"""工单存储的数据结构：全部是不可变值对象。"""

from dataclasses import dataclass, field
from typing import Callable

PRIORITIES = ("critical", "high", "medium", "low")


@dataclass(frozen=True)
class SlaPolicy:
    """按优先级的响应时限（秒）。可覆盖默认值。

    简化：只按自然时间算，不考虑工作日、节假日和上下班时间。
    """

    critical: float = 1 * 3600
    high: float = 4 * 3600
    medium: float = 8 * 3600
    low: float = 24 * 3600

    def seconds_for(self, priority: str) -> float:
        if priority not in PRIORITIES:
            raise ValueError(f"未知优先级: {priority!r}")
        return float(getattr(self, priority))


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    customer_id: str
    shop_id: str | None
    subject: str
    status: str
    priority: str
    owner_id: str | None
    thread_id: str
    category: str | None
    created_at: float
    updated_at: float
    sla_due_at: float
    version: int


@dataclass(frozen=True)
class Message:
    id: int
    ticket_id: str
    author_id: str
    author_role: str
    kind: str
    visibility: str
    body: str
    idem_key: str | None
    created_at: float


@dataclass(frozen=True)
class TicketEvent:
    id: int
    ticket_id: str
    ts: float
    actor_id: str
    event: str
    from_status: str | None
    to_status: str | None
    detail: str


@dataclass(frozen=True)
class Handoff:
    ticket_id: str
    escalation_reason: str
    actions_done: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    last_customer_message: str = ""
    owner_id: str | None = None
    status: str = "new"


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


class Conflict(Exception):
    pass


class IllegalTransition(Exception):
    pass


Clock = Callable[[], float]
