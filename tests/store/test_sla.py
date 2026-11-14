import math

import pytest

from carepilot.auth import Principal
from carepilot.store import SlaPolicy, TicketStore

from conftest import FakeClock, make_open_ticket, make_ticket


def test_default_deadline_by_priority(store: TicketStore, clock, c1: Principal, a1: Principal):
    created = make_ticket(store, c1, priority="medium")
    assert created.sla_due_at == created.created_at + 8 * 3600
    assert store.sla_remaining_seconds(created) == 8 * 3600
    clock.advance(3 * 3600)
    assert store.sla_remaining_seconds(created) == 5 * 3600


def test_remaining_can_be_negative_when_overdue(store: TicketStore, clock, c1: Principal):
    created = make_ticket(store, c1, priority="critical")
    clock.advance(2 * 3600)
    assert store.sla_remaining_seconds(created) == -1 * 3600


@pytest.mark.parametrize(
    "priority,seconds",
    [("critical", 3600), ("high", 4 * 3600), ("medium", 8 * 3600), ("low", 24 * 3600)],
)
def test_all_default_priorities(store: TicketStore, c1: Principal, priority: str, seconds: float):
    created = make_ticket(store, c1, priority=priority)
    assert created.sla_due_at - created.created_at == seconds


def test_priority_change_recomputes_from_created_at(store: TicketStore, clock, c1: Principal, a1: Principal):
    ticket = make_open_ticket(store, c1, a1, priority="medium")
    clock.advance(3 * 3600)
    updated = store.set_priority(a1, ticket.ticket_id, "high")
    assert updated.sla_due_at == updated.created_at + 4 * 3600
    assert store.sla_remaining_seconds(updated) == 1 * 3600


def test_resolved_and_closed_have_no_sla(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_open_ticket(store, c1, a1)
    resolved = store.transition(a1, ticket.ticket_id, "resolved")
    assert math.isinf(store.sla_remaining_seconds(resolved))
    closed = store.transition(a1, ticket.ticket_id, "closed")
    assert math.isinf(store.sla_remaining_seconds(closed))


def test_custom_policy_overrides_defaults(tmp_path, clock: FakeClock, c1: Principal):
    store = TicketStore(tmp_path / "custom.sqlite", clock=clock, sla_policy=SlaPolicy(medium=60))
    ticket = make_ticket(store, c1, priority="medium")
    assert ticket.sla_due_at - ticket.created_at == 60


def test_unknown_priority_rejected(store: TicketStore, c1: Principal):
    with pytest.raises(ValueError):
        make_ticket(store, c1, priority="urgent")
