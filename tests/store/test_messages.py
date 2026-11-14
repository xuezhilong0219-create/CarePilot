import pytest

from carepilot.auth import Principal
from carepilot.store import Forbidden, TicketStore

from conftest import make_ticket


def test_draft_must_be_internal(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(ValueError):
        store.add_message(ticket.ticket_id, author=a1, kind="draft", body="草稿", visibility="public")
    draft = store.add_message(ticket.ticket_id, author=a1, kind="draft", body="草稿", visibility="internal")
    assert draft.visibility == "internal"


def test_idem_key_deduplicates_message(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    first = store.add_message(
        ticket.ticket_id, author=a1, kind="message", body="已处理", idem_key="run-1:send"
    )
    second = store.add_message(
        ticket.ticket_id, author=a1, kind="message", body="已处理（重试）", idem_key="run-1:send"
    )
    assert second.id == first.id
    assert second.body == "已处理"
    assert len(store.list_messages(a1, ticket.ticket_id)) == 1


def test_different_idem_keys_create_two_messages(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.add_message(ticket.ticket_id, author=a1, kind="message", body="a", idem_key="k1")
    store.add_message(ticket.ticket_id, author=a1, kind="message", body="b", idem_key="k2")
    assert len(store.list_messages(a1, ticket.ticket_id)) == 2


def test_customer_cannot_message_other_ticket(store: TicketStore, c1: Principal, c2: Principal):
    ticket = make_ticket(store, c2, shop_id="S3")
    with pytest.raises(Forbidden):
        store.add_message(ticket.ticket_id, author=c1, kind="message", body="越权")


def test_events_cover_create_claim_transition_priority(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    store.transition(a1, ticket.ticket_id, "open")
    store.set_priority(a1, ticket.ticket_id, "high")
    events = [e.event for e in store.events(a1, ticket.ticket_id)]
    assert events == ["created", "claimed", "transition", "priority_changed"]


def test_events_hidden_from_other_customer(store: TicketStore, c1: Principal, c2: Principal):
    ticket = make_ticket(store, c2, shop_id="S3")
    with pytest.raises(Forbidden):
        store.events(c1, ticket.ticket_id)
