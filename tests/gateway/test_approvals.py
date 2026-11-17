from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest

from carepilot.gateway.spec import Executed, Gate, NeedsApproval, Rejected, ToolContext
from conftest import AGENT, ADMIN, CUSTOMER


def test_approve_requires_pending_including_already_approved(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "send", {"shop_id": "mine"})
    assert gateway.approve(pending.approval_id, AGENT).stage == "confirm"
    repeated = gateway.approve(pending.approval_id, AGENT)
    assert repeated.gate == Gate.RISK and repeated.code == "approval_not_pending"
    assert calls == []


def test_confirm_replays_saved_result_and_parameters(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    raw = {"shop_id": "mine", "value": 7}
    pending = gateway.call(ToolContext(AGENT, "run", "ticket"), "send", raw)
    raw["value"] = 99
    gateway.approve(pending.approval_id, AGENT)
    results = [gateway.confirm(pending.approval_id, ADMIN) for _ in range(3)]
    assert all(isinstance(result, Executed) and result.result["value"] == 7 for result in results)
    assert [result.replayed for result in results] == [False, True, True]
    assert len(calls) == 1 and calls[0][1].value == 7


def test_state_change_executes_only_on_approval(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "state", {"shop_id": "mine"})
    assert calls == []
    assert isinstance(gateway.approve(pending.approval_id, AGENT), Executed)
    assert gateway.approve(pending.approval_id, ADMIN).code == "approval_not_pending"
    assert gateway.confirm(pending.approval_id, AGENT).code == "confirmation_not_required"
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["rejected", "expired"])
def test_closed_approvals_cannot_be_approved(setup_gateway, status):
    gateway, calls, _, path = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "state", {"shop_id": "mine"})
    if status == "rejected":
        gateway.reject(pending.approval_id, AGENT, "not allowed")
    else:
        with sqlite3.connect(path) as db:
            db.execute("UPDATE approvals SET status='expired' WHERE approval_id=?", (pending.approval_id,))
    assert gateway.approve(pending.approval_id, AGENT).code == "approval_not_pending"
    assert calls == []


def test_replaying_a_rejected_approval_returns_rejected(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    ctx = ToolContext(AGENT, "run")
    args = {"shop_id": "mine"}
    pending = gateway.call(ctx, "state", args)
    gateway.reject(pending.approval_id, AGENT, "not allowed")

    replay = gateway.call(ctx, "state", args)

    assert isinstance(replay, Rejected)
    assert replay.code == "approval_rejected"
    assert calls == []


def test_staff_still_needs_tool_permission_on_approval(setup_gateway):
    _, calls, registry, path = setup_gateway
    from carepilot.gateway.gateway import ToolGateway

    registry["state"] = replace(registry["state"], owner_check=lambda ctx, args: ctx.principal.user_id == AGENT.user_id)
    gateway = ToolGateway(path, registry)
    pending = gateway.call(ToolContext(AGENT, "run"), "state", {"shop_id": "mine"})
    assert gateway.approve(pending.approval_id, ADMIN).code == "not_owner"
    assert calls == []


def test_reject_customer_cannot_cancel_and_missing_id_is_audited(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "send", {"shop_id": "mine"})
    gateway.reject(pending.approval_id, CUSTOMER, "cancel")
    assert gateway.approve(pending.approval_id, AGENT).stage == "confirm"
    assert gateway.approve("unknown", AGENT).code == "unknown_approval"
    assert gateway.confirm("unknown", AGENT).code == "unknown_approval"
    assert calls == []


def test_approval_is_atomic_with_execution_reservation(setup_gateway, monkeypatch):
    gateway, calls, _, path = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "state", {"shop_id": "mine"})
    append = gateway._audit.append
    def fail_intent(db, event):
        if event["code"] == "execution_intent":
            raise sqlite3.OperationalError("disk full")
        append(db, event)
    with monkeypatch.context() as patch:
        patch.setattr(gateway._audit, "append", fail_intent)
        with pytest.raises(sqlite3.OperationalError):
            gateway.approve(pending.approval_id, AGENT)
    assert calls == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM approvals").fetchone()[0] == "pending"
    assert isinstance(gateway.approve(pending.approval_id, AGENT), Executed)


def test_completed_state_action_is_replayed_on_same_call(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    ctx = ToolContext(AGENT, "run")
    pending = gateway.call(ctx, "state", {"shop_id": "mine"})
    first = gateway.approve(pending.approval_id, AGENT)
    assert gateway.call(ctx, "state", {"shop_id": "mine"}) == Executed(first.result, replayed=True)
    assert len(calls) == 1


def test_reject_requires_tool_ownership(setup_gateway):
    _, calls, registry, path = setup_gateway
    from carepilot.gateway.gateway import ToolGateway
    registry["state"] = replace(registry["state"], owner_check=lambda ctx, args: ctx.principal.user_id == AGENT.user_id)
    gateway = ToolGateway(path, registry)
    pending = gateway.call(ToolContext(AGENT, "run"), "state", {"shop_id": "mine"})
    gateway.reject(pending.approval_id, ADMIN, "cancel")
    assert isinstance(gateway.approve(pending.approval_id, AGENT), Executed)
    assert len(calls) == 1


def test_concurrent_approvals_and_confirmations_execute_once(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    pending = gateway.call(ToolContext(AGENT, "run"), "send", {"shop_id": "mine"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        barrier = threading.Barrier(4)
        def approve(index):
            barrier.wait(timeout=5)
            return gateway.approve(pending.approval_id, AGENT)
        outcomes = list(pool.map(approve, range(4)))
        assert sum(isinstance(outcome, NeedsApproval) for outcome in outcomes) == 1
        assert calls == []
        barrier = threading.Barrier(4)
        def confirm(index):
            barrier.wait(timeout=5)
            return gateway.confirm(pending.approval_id, AGENT)
        outcomes = list(pool.map(confirm, range(4)))
    assert all(isinstance(outcome, Executed) for outcome in outcomes)
    assert sum(not outcome.replayed for outcome in outcomes) == 1
    assert len(calls) == 1
