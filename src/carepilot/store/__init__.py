"""工单存储：SQLite 持久化 + 状态机 + SLA + 三角色权限 + 交接摘要。不调用任何模型。"""

from .handoff import build_handoff
from .models import (
    PRIORITIES,
    Conflict,
    Forbidden,
    Handoff,
    IllegalTransition,
    Message,
    NotFound,
    SlaPolicy,
    Ticket,
    TicketEvent,
)
from .rules import STATUSES, TRANSITIONS, can_modify, can_transition, can_view
from .service import TicketStore, seed_demo_users

__all__ = [
    "PRIORITIES",
    "STATUSES",
    "TRANSITIONS",
    "Conflict",
    "Forbidden",
    "Handoff",
    "IllegalTransition",
    "Message",
    "NotFound",
    "SlaPolicy",
    "Ticket",
    "TicketEvent",
    "TicketStore",
    "build_handoff",
    "can_modify",
    "can_transition",
    "can_view",
    "seed_demo_users",
]
