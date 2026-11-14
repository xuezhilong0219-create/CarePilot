from dataclasses import replace
import json
import sqlite3
import time

import pytest

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import Executed, ToolContext
from conftest import AGENT, ADMIN


def test_sensitive_args_and_results_never_enter_any_audit_column(setup_gateway):
    gateway, _, _, path = setup_gateway
    ctx = ToolContext(AGENT, "run", "ticket")
    phone = "13812345678"
    assert isinstance(gateway.call(ctx, "read", {"shop_id": "mine", "phone": phone}), Executed)
    send = gateway.call(ctx, "send", {"shop_id": "mine", "phone": phone})
    gateway.approve(send.approval_id, AGENT)
    gateway.confirm(send.approval_id, ADMIN)
    gateway.call(ctx, "read", {"shop_id": "mine", "extra": phone})
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT * FROM audit_log").fetchall()
    assert rows and all(phone not in str(value) for row in rows for value in row)
    assert phone not in json.dumps(send.preview)
    assert any("phone=***" in row["args_summary"] for row in gateway.audit())


def test_six_calls_reconstruct_run_chain_and_filters(setup_gateway):
    gateway, _, _, _ = setup_gateway
    started = time.monotonic()
    ctx = ToolContext(AGENT, "run", "ticket")
    gateway.call(ctx, "missing", {})
    pending = gateway.call(ctx, "send", {"shop_id": "mine", "value": 7})
    gateway.approve(pending.approval_id, AGENT)
    gateway.confirm(pending.approval_id, ADMIN)
    gateway.confirm(pending.approval_id, ADMIN)
    gateway.call(ctx, "read", {"shop_id": "mine"})
    rows = gateway.audit(run_id="run")
    assert rows == sorted(rows, key=lambda row: (row["ts"], row["id"]))
    assert {row["decision"] for row in rows} >= {"rejected", "needs_approval", "approved", "confirmed", "executed", "replayed"}
    assert rows[0]["actor_id"] == AGENT.user_id and rows[0]["gate"] == "schema"
    approved = next(row for row in rows if row["code"] == "approval_granted")
    assert approved["approver_id"] == AGENT.user_id
    executed = next(row for row in rows if row["decision"] == "executed" and row["tool"] == "send")
    assert executed["actor_id"] == ADMIN.user_id and executed["approver_id"] == AGENT.user_id
    assert '7' in executed["result_summary"]
    assert all(row["ticket_id"] == "ticket" for row in gateway.audit(ticket_id="ticket"))
    assert all(row["actor_id"] == ADMIN.user_id for row in gateway.audit(actor_id=ADMIN.user_id))
    assert len(gateway.audit(run_id="run", limit=2)) == 2
    assert gateway.audit(run_id="other") == []
    assert time.monotonic() - started < 30


def test_audit_failure_before_effect_aborts_and_rolls_back_reservation(setup_gateway, monkeypatch):
    gateway, calls, _, path = setup_gateway
    def broken(db, event):
        raise sqlite3.OperationalError("audit disk failed")
    with monkeypatch.context() as patch:
        patch.setattr(gateway._audit, "append", broken)
        with pytest.raises(sqlite3.OperationalError, match="audit disk failed"):
            gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"})
    assert calls == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM action_ledger").fetchone()[0] == 0


def test_post_effect_audit_failure_keeps_started_and_reconcile_adopts(setup_gateway, monkeypatch):
    _, calls, registry, path = setup_gateway
    registry["internal"] = replace(registry["internal"], verify=lambda ctx, args: {"value": args.value, "phone": args.phone} if calls else None)
    gateway = ToolGateway(path, registry)
    append = gateway._audit.append
    def broken_after_effect(db, event):
        if event["decision"] == "executed":
            raise sqlite3.OperationalError("audit full")
        append(db, event)
    ctx = ToolContext(AGENT, "run")
    with monkeypatch.context() as patch:
        patch.setattr(gateway._audit, "append", broken_after_effect)
        with pytest.raises(sqlite3.OperationalError, match="audit full"):
            gateway.call(ctx, "internal", {"shop_id": "mine"})
    assert len(calls) == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"
        assert db.execute("SELECT code FROM audit_log").fetchone()[0] == "execution_intent"
    assert gateway.reconcile()[0]["code"] == "reconciled_adopted"
    assert gateway.call(ctx, "internal", {"shop_id": "mine"}).replayed
    assert len(calls) == 1


def test_intent_is_committed_before_tool_runs(setup_gateway):
    _, calls, registry, path = setup_gateway
    def effect(ctx, args):
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT code FROM audit_log WHERE idem_key=?", (ctx.idem_key,)).fetchone()[0] == "execution_intent"
            assert db.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"
        calls.append(1)
        return {"ok": True}
    registry["internal"] = replace(registry["internal"], fn=effect)
    assert isinstance(ToolGateway(path, registry).call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"}), Executed)
    assert calls == [1]


def test_reconcile_does_not_swallow_non_sqlite_audit_failure(setup_gateway, monkeypatch):
    _, calls, registry, path = setup_gateway
    verifies = []
    def verify(ctx, args):
        verifies.append(1)
        return None
    registry["internal"] = replace(registry["internal"], verify=verify)
    gateway = ToolGateway(path, registry)
    def interrupt(name):
        if name == "before_effect":
            raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr("carepilot.gateway.gateway._crash_point", interrupt)
        with pytest.raises(KeyboardInterrupt):
            gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"})
    append = gateway._audit.append
    def fail_intent(db, event):
        if event["code"] == "reconciliation_intent":
            raise RuntimeError("audit failed")
        append(db, event)
    monkeypatch.setattr(gateway._audit, "append", fail_intent)
    with pytest.raises(RuntimeError, match="audit failed"):
        gateway.reconcile()
    assert verifies == [] and calls == []
