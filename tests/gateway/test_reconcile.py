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


def audit_count(gateway) -> int:
    with sqlite3.connect(gateway.ledger.path) as db:
        return db.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]


def ledger_status(gateway) -> str:
    with sqlite3.connect(gateway.ledger.path) as db:
        return db.execute("SELECT status FROM action_ledger").fetchone()[0]


def test_inflight_call_is_not_taken_over_by_reconcile(setup_gateway, monkeypatch):
    """工具挪到锁外执行之后，对账不能再靠"拿锁"判断谁在跑——租约才是那个事实。"""
    import threading

    _, calls, registry, path = setup_gateway
    running, release = threading.Event(), threading.Event()

    def slow(ctx, args):
        running.set()
        release.wait(timeout=10)
        return {"value": args.value}

    registry["internal"] = replace(registry["internal"], fn=slow)
    gateway = ToolGateway(path, registry)  # ToolGateway 复制一份 registry，改完要重建
    worker = threading.Thread(target=lambda: gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"}))
    worker.start()
    try:
        assert running.wait(timeout=10)
        outcome = gateway.reconcile()
        assert [o["status"] for o in outcome] == ["skipped_inflight"]
        assert outcome[0]["code"] == "inflight"
        assert ledger_status(gateway) == "started"  # 没有被记账抢走
        assert gateway.inflight()[0]["live"] is True
    finally:
        release.set()
        worker.join(timeout=10)
    assert ledger_status(gateway) == "done"
    assert gateway.reconcile() == []  # 跑完了就没有待对账的东西


def test_reconcile_can_ignore_rows_that_are_too_fresh(setup_gateway, monkeypatch):
    gateway, calls, _, _ = setup_gateway
    leave_started(gateway, monkeypatch)
    assert gateway.reconcile(min_age_s=60)[0]["status"] == "skipped_recent"
    assert ledger_status(gateway) == "started"


def test_dry_run_lists_candidates_and_writes_nothing(setup_gateway, monkeypatch):
    gateway, calls, registry, _ = setup_gateway
    leave_started(gateway, monkeypatch)
    verified: list[str] = []
    registry["internal"] = replace(registry["internal"], verify=lambda ctx, args: verified.append("called") or {"value": 1})
    before = audit_count(gateway)
    outcome = gateway.reconcile(dry_run=True)
    assert outcome[0]["status"] == "would_reconcile"
    assert verified == []  # 连 verify 都不调：认领副作用不可回退，先看一眼
    assert audit_count(gateway) == before
    assert ledger_status(gateway) == "started"
