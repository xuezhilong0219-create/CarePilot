import pytest

from carepilot.auth import Principal, Role
from carepilot.store import Conflict, Forbidden, NotFound, TicketStore, can_modify, can_view

from conftest import make_open_ticket, make_ticket


def test_customer_cannot_read_other_customers_ticket(store: TicketStore, c1: Principal, c2: Principal):
    ticket = make_ticket(store, c2, shop_id="S3")
    with pytest.raises(Forbidden):
        store.get_ticket(c1, ticket.ticket_id)


def test_customer_without_shop_scope_cannot_create_ticket_for_a_shop(store: TicketStore):
    customer = Principal("unassigned-customer", Role.CUSTOMER)

    with pytest.raises(Forbidden):
        store.create_ticket(customer, subject="synthetic", shop_id="S999", thread_id="thread-unscoped")


def test_missing_ticket_is_not_found_not_forbidden(store: TicketStore, c1: Principal):
    with pytest.raises(NotFound):
        store.get_ticket(c1, "T-999999")


def test_customer_list_only_own_tickets(store: TicketStore, c1: Principal, c2: Principal):
    mine = make_ticket(store, c1)
    make_ticket(store, c2, shop_id="S3")
    listed = store.list_tickets(c1)
    assert [t.ticket_id for t in listed] == [mine.ticket_id]


def test_agent_can_view_all_tickets(store: TicketStore, c1: Principal, c2: Principal, a2: Principal):
    first = make_ticket(store, c1)
    second = make_ticket(store, c2, shop_id="S3")
    ids = {t.ticket_id for t in store.list_tickets(a2)}
    assert {first.ticket_id, second.ticket_id} <= ids
    assert store.get_ticket(a2, first.ticket_id).ticket_id == first.ticket_id


def test_customer_cannot_transition_claim_or_set_priority(store: TicketStore, c1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(Forbidden):
        store.transition(c1, ticket.ticket_id, "open")
    with pytest.raises(Forbidden):
        store.claim(c1, ticket.ticket_id)
    with pytest.raises(Forbidden):
        store.set_priority(c1, ticket.ticket_id, "high")


def test_agent_cannot_modify_ticket_owned_by_someone_else(
    store: TicketStore, c1: Principal, a1: Principal, a2: Principal
):
    ticket = make_open_ticket(store, c1, a1)
    with pytest.raises(Forbidden):
        store.transition(a2, ticket.ticket_id, "on_hold")
    with pytest.raises(Forbidden):
        store.set_priority(a2, ticket.ticket_id, "high")


def test_agent_cannot_claim_someone_elses_ticket(store: TicketStore, c1: Principal, a1: Principal, a2: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    with pytest.raises(Conflict):
        store.claim(a2, ticket.ticket_id)


def test_unassigned_ticket_must_be_claimed_before_modified(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(Forbidden):
        store.set_priority(a1, ticket.ticket_id, "high")
    claimed = store.claim(a1, ticket.ticket_id)
    assert claimed.owner_id == "a1"
    assert store.set_priority(a1, ticket.ticket_id, "high").priority == "high"


def test_admin_can_modify_any_ticket(store: TicketStore, c1: Principal, a1: Principal, admin: Principal):
    ticket = make_open_ticket(store, c1, a1)
    assert store.transition(admin, ticket.ticket_id, "on_hold").status == "on_hold"
    assert store.set_priority(admin, ticket.ticket_id, "critical").priority == "critical"
    assert store.release(admin, ticket.ticket_id).owner_id is None


def test_internal_messages_hidden_from_customer(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    store.add_message(ticket.ticket_id, author=c1, kind="message", body="客户问题")
    store.add_message(ticket.ticket_id, author=a1, kind="note", body="内部备注", visibility="internal")
    store.add_message(ticket.ticket_id, author=a1, kind="message", body="对外回复")
    customer_bodies = [m.body for m in store.list_messages(c1, ticket.ticket_id)]
    assert customer_bodies == ["客户问题", "对外回复"]
    agent_bodies = [m.body for m in store.list_messages(a1, ticket.ticket_id)]
    assert "内部备注" in agent_bodies


def test_customer_cannot_create_internal_message(store: TicketStore, c1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(Forbidden):
        store.add_message(ticket.ticket_id, author=c1, kind="message", body="x", visibility="internal")


def test_can_view_and_can_modify_are_consistent(store: TicketStore, c1: Principal, c2: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    assert can_view(c1, ticket) is True
    assert can_view(c2, ticket) is False
    assert can_modify(c1, ticket) is False
    assert can_modify(a1, ticket) is False
    store.claim(a1, ticket.ticket_id)
    owned = store.get_ticket(a1, ticket.ticket_id)
    assert can_modify(a1, owned) is True
