from dataclasses import replace
import sqlite3

import pytest

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import ToolContext
from conftest import AGENT


def leave_started(gateway, monkeypatch):
    def stop(name):
        if name == "before_effect":
            raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr("carepilot.gateway.gateway._crash_point", stop)
        with pytest.raises(KeyboardInterrupt):
            gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"})


def test_started_call_requires_reconciliation_before_retry(setup_gateway, monkeypatch):
    gateway, calls, _, _ = setup_gateway
    leave_started(gateway, monkeypatch)
    assert gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"}).code == "reconciliation_required"
    assert calls == []


@pytest.mark.parametrize("verification", ["error", "invalid", "missing_tool"])
def test_uncertain_verification_stays_manual(setup_gateway, monkeypatch, verification):
    gateway, calls, registry, path = setup_gateway
    leave_started(gateway, monkeypatch)
    def verify(ctx, args):
        if verification == "error":
            raise RuntimeError("backend unavailable")
        return []
    registry["internal"] = replace(registry["internal"], verify=verify)
    if verification == "missing_tool":
        registry.pop("internal")
    recovery = ToolGateway(path, registry).reconcile()
    assert recovery[0]["status"] == "needs_manual"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"
    assert calls == []


def test_empty_dict_verify_is_an_existing_result(setup_gateway, monkeypatch):
    gateway, calls, registry, path = setup_gateway
    leave_started(gateway, monkeypatch)
    registry["internal"] = replace(registry["internal"], verify=lambda ctx, args: {})
    gateway = ToolGateway(path, registry)
    assert gateway.reconcile()[0]["status"] == "done"
    replay = gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"})
    assert replay.replayed and replay.result == {} and calls == []
