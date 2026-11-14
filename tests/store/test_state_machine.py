import itertools

import pytest

from carepilot.store import (
    STATUSES,
    TRANSITIONS,
    IllegalTransition,
    can_transition,
    TicketStore,
)
from carepilot.auth import Principal

from conftest import make_open_ticket, make_ticket


@pytest.mark.parametrize("from_status,to_status", list(itertools.product(STATUSES, repeat=2)))
def test_transition_table_matches_constant(from_status: str, to_status: str):
    assert can_transition(from_status, to_status) is ((from_status, to_status) in TRANSITIONS)


def test_expected_transitions_are_present():
    for pair in [
        ("new", "open"),
        ("open", "pending_customer"),
        ("pending_customer", "open"),
        ("open", "on_hold"),
        ("on_hold", "open"),
        ("open", "escalated"),
        ("escalated", "open"),
        ("resolved", "closed"),
        ("resolved", "open"),
    ]:
        assert pair in TRANSITIONS


def test_every_non_closed_state_can_resolve():
    for status in STATUSES:
        assert can_transition(status, "resolved") is (status != "closed")


def test_closed_is_terminal():
    assert all(not can_transition("closed", to) for to in STATUSES)


def test_store_rejects_illegal_transition(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_open_ticket(store, c1, a1)
    with pytest.raises(IllegalTransition):
        store.transition(a1, ticket.ticket_id, "closed")


def test_store_allows_new_to_open_and_resolved_to_closed(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    opened = store.transition(a1, ticket.ticket_id, "open")
    assert opened.status == "open"
    resolved = store.transition(a1, ticket.ticket_id, "resolved")
    assert resolved.status == "resolved"
    closed = store.transition(a1, ticket.ticket_id, "closed")
    assert closed.status == "closed"


def test_transition_records_event(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_open_ticket(store, c1, a1)
    store.transition(a1, ticket.ticket_id, "escalated", reason="需要财务")
    events = store.events(a1, ticket.ticket_id)
    last = events[-1]
    assert (last.event, last.from_status, last.to_status, last.detail) == (
        "transition",
        "open",
        "escalated",
        "需要财务",
    )
    assert last.actor_id == "a1"
