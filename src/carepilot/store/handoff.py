"""B7 交接摘要：完全确定性的纯函数，不调用模型。缺信息时返回空值，不抛异常。"""

from collections.abc import Sequence
from typing import Any

from .models import Handoff, Message, Ticket

ESCALATION_PREFIX = "升级原因："
QUESTION_PREFIX = "待确认："
CUSTOMER_ROLE = "customer"


def _actions_done(actions: Sequence[dict[str, Any]]) -> list[str]:
    done: list[str] = []
    for action in actions:
        tool = str(action.get("tool") or "").strip()
        summary = str(action.get("summary") or "").strip()
        if tool and summary:
            done.append(f"{tool}：{summary}")
        elif summary:
            done.append(summary)
        elif tool:
            done.append(tool)
    return done


def _escalation_reason(messages: Sequence[Message]) -> str:
    candidates = [m for m in messages if m.kind == "system" and m.body.startswith(ESCALATION_PREFIX)]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda m: (m.created_at, m.id))
    return latest.body[len(ESCALATION_PREFIX) :].strip()


def _open_questions(messages: Sequence[Message]) -> list[str]:
    # 越新的备注越靠前；同一问题只留一条。
    notes = sorted((m for m in messages if m.kind == "note"), key=lambda m: (m.created_at, m.id), reverse=True)
    questions: list[str] = []
    seen: set[str] = set()
    for note in notes:
        for raw in note.body.splitlines():
            line = raw.strip()
            if not line.startswith(QUESTION_PREFIX):
                continue
            question = line[len(QUESTION_PREFIX) :].strip()
            if question and question not in seen:
                seen.add(question)
                questions.append(question)
    return questions


def _last_customer_message(messages: Sequence[Message]) -> str:
    candidates = [m for m in messages if m.kind == "message" and m.author_role == CUSTOMER_ROLE]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda m: (m.created_at, m.id))
    return latest.body


def build_handoff(ticket: Ticket, messages: Sequence[Message], actions: Sequence[dict[str, Any]]) -> Handoff:
    return Handoff(
        ticket_id=ticket.ticket_id,
        escalation_reason=_escalation_reason(messages),
        actions_done=_actions_done(actions),
        open_questions=_open_questions(messages),
        last_customer_message=_last_customer_message(messages),
        owner_id=ticket.owner_id,
        status=ticket.status,
    )
