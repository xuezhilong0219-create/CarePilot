import pytest

from carepilot.auth import Principal
from carepilot.store import Forbidden, TicketStore, build_handoff

from conftest import make_open_ticket, make_ticket


def test_build_handoff_collects_deterministic_context(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.add_message(ticket.ticket_id, author=c1, kind="message", body="店铺授权一直失败")
    store.add_message(
        ticket.ticket_id, author=a1, kind="system", body="升级原因：需要财务确认退款", visibility="internal"
    )
    store.add_message(
        ticket.ticket_id,
        author=a1,
        kind="note",
        body="待确认：订单号\n待确认：退款金额\n其他备注",
        visibility="internal",
    )
    messages = store.list_messages(a1, ticket.ticket_id)
    actions = [{"tool": "query_order", "summary": "查到订单 A1", "at": 123.0}]
    handoff = build_handoff(ticket, messages, actions)
    assert handoff.ticket_id == ticket.ticket_id
    assert handoff.escalation_reason == "需要财务确认退款"
    assert handoff.actions_done == ["query_order：查到订单 A1"]
    assert handoff.open_questions == ["订单号", "退款金额"]
    assert handoff.last_customer_message == "店铺授权一直失败"
    assert handoff.owner_id is None
    assert handoff.status == "new"


def test_build_handoff_without_context_is_safe(store: TicketStore, c1: Principal):
    ticket = make_ticket(store, c1)
    handoff = build_handoff(ticket, [], [])
    assert handoff.escalation_reason == ""
    assert handoff.actions_done == []
    assert handoff.open_questions == []
    assert handoff.last_customer_message == ""


def test_handoff_lets_next_agent_resume(store: TicketStore, c1: Principal, a1: Principal, a2: Principal):
    ticket = make_open_ticket(store, c1, a1)
    store.add_message(ticket.ticket_id, author=a1, kind="system", body="升级原因：需要主管特批", visibility="internal")
    store.add_message(ticket.ticket_id, author=c1, kind="message", body="麻烦尽快")
    store.transition(a1, ticket.ticket_id, "escalated")
    released = store.release(a1, ticket.ticket_id)
    assert released.owner_id is None
    assert released.status == "open"

    claimed = store.claim(a2, ticket.ticket_id)
    assert claimed.owner_id == "a2"
    handoff = build_handoff(claimed, store.list_messages(a2, ticket.ticket_id), [{"tool": "t", "summary": "s", "at": 1}])
    assert handoff.escalation_reason == "需要主管特批"
    assert handoff.last_customer_message == "麻烦尽快"
    assert handoff.owner_id == "a2"


def test_release_requires_ownership(store: TicketStore, c1: Principal, a1: Principal, a2: Principal):
    ticket = make_open_ticket(store, c1, a1)
    with pytest.raises(Forbidden):
        store.release(a2, ticket.ticket_id)
