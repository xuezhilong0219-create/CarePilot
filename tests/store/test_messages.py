import pytest

from carepilot.auth import Principal
from carepilot.store import Forbidden, NotFound, TicketStore

from conftest import make_ticket


def test_draft_must_be_internal(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(ValueError):
        store.add_message(ticket.ticket_id, author=a1, kind="draft", body="草稿", visibility="public")
    draft = store.add_message(ticket.ticket_id, author=a1, kind="draft", body="草稿", visibility="internal")
    assert draft.visibility == "internal"


def test_idem_key_deduplicates_message(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)  # 坐席要先认领，才能向客户发公开消息
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
    store.claim(a1, ticket.ticket_id)
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


def test_non_owner_agent_cannot_post_public_message_but_can_write_internal_note(
    store: TicketStore, c1: Principal, a1: Principal, a2: Principal, admin: Principal
):
    # 回归：worker S 原先允许任何坐席在别人负责的工单里发客户可见的消息，等于改别人的工单（C1）
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    with pytest.raises(Forbidden):
        store.add_message(ticket.ticket_id, author=a2, kind="message", body="我来回复")
    store.add_message(ticket.ticket_id, author=a2, kind="note", body="同事备注", visibility="internal")
    store.add_message(ticket.ticket_id, author=admin, kind="message", body="管理员代回复")
    store.add_message(ticket.ticket_id, author=a1, kind="message", body="负责人回复")
    public = [m.body for m in store.list_messages(c1, ticket.ticket_id)]
    assert public == ["管理员代回复", "负责人回复"]


def test_unclaimed_ticket_agent_must_claim_before_public_message(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    with pytest.raises(Forbidden):
        store.add_message(ticket.ticket_id, author=a1, kind="message", body="未认领就回复")
    store.claim(a1, ticket.ticket_id)
    store.add_message(ticket.ticket_id, author=a1, kind="message", body="认领后回复")


def test_find_message_by_idem_key(store: TicketStore, c1: Principal, a1: Principal):
    ticket = make_ticket(store, c1)
    store.claim(a1, ticket.ticket_id)
    msg = store.add_message(ticket.ticket_id, author=a1, kind="message", body="已处理", idem_key="run-9:send")
    assert store.find_message_by_idem_key("run-9:send").id == msg.id
    assert store.find_message_by_idem_key("nope") is None


def test_ticket_context_roundtrip_and_category(store: TicketStore, c1: Principal, a1: Principal, admin: Principal):
    ticket = make_ticket(store, c1)
    assert store.get_context(ticket.ticket_id) == {"slots": {}, "last_thread_id": None}
    store.put_context(ticket.ticket_id, slots={"platform_name": "京东", "feature_name": None}, last_thread_id="th-1")
    store.put_context(ticket.ticket_id, slots={"platform_name": "京东", "feature_name": "库存同步"}, last_thread_id="th-2")
    assert store.get_context(ticket.ticket_id) == {
        "slots": {"platform_name": "京东", "feature_name": "库存同步"}, "last_thread_id": "th-2"}
    assert store.set_category(admin, ticket.ticket_id, "platform_integration").category == "platform_integration"
    with pytest.raises(Forbidden):
        store.set_category(c1, ticket.ticket_id, "other")
    with pytest.raises(NotFound):
        store.get_context("T-999999")
