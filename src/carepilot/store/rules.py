"""纯函数：状态机迁移表、权限判定。store 和网关共用同一份，避免两套口径。"""

from ..auth import Principal, Role
from .models import Ticket

STATUSES = ("new", "open", "pending_customer", "on_hold", "escalated", "resolved", "closed")

# 任何非终态都可直接结案。
_RESOLVE_FROM = tuple(s for s in STATUSES if s != "closed")

TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("new", "open"),
        ("open", "pending_customer"),
        ("pending_customer", "open"),
        ("open", "on_hold"),
        ("on_hold", "open"),
        ("open", "escalated"),
        ("escalated", "open"),
        ("resolved", "closed"),
        ("resolved", "open"),
    }
    | {(s, "resolved") for s in _RESOLVE_FROM}
)


def can_transition(from_status: str, to_status: str) -> bool:
    return (from_status, to_status) in TRANSITIONS


def is_staff(who: Principal) -> bool:
    return who.role in (Role.AGENT, Role.ADMIN)


def can_view(who: Principal, ticket: Ticket) -> bool:
    if who.role == Role.ADMIN:
        return True
    if who.role == Role.AGENT:
        return True
    return ticket.customer_id == who.user_id


def can_modify(who: Principal, ticket: Ticket) -> bool:
    """管理员全可改；坐席只能改自己负责的；未分配的工单必须先 claim，不能直接改。"""
    if who.role == Role.ADMIN:
        return True
    if who.role == Role.AGENT:
        return ticket.owner_id == who.user_id
    return False
