"""周期对账：只捡「够老」且「没人握着租约」的 started 行。"""

from dataclasses import replace
import time

import pytest

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import ToolContext
from carepilot.gateway.sweeper import ReconcileSweeper
from conftest import AGENT


def leave_started(gateway, monkeypatch):
    def stop(name):
        if name == "before_effect":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr("carepilot.gateway.gateway._crash_point", stop)
        with pytest.raises(KeyboardInterrupt):
            gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"})


def wait_until(predicate, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_interval_must_be_positive(setup_gateway):
    gateway, *_ = setup_gateway
    with pytest.raises(ValueError):
        ReconcileSweeper(gateway, interval_s=0)


def test_tick_leaves_too_fresh_rows_alone(setup_gateway, monkeypatch):
    gateway, *_ = setup_gateway
    leave_started(gateway, monkeypatch)
    outcomes = ReconcileSweeper(gateway, interval_s=60, min_age_s=3600).tick()
    assert [o["status"] for o in outcomes] == ["skipped_recent"]


def test_tick_handles_an_orphaned_row(setup_gateway, monkeypatch):
    gateway, calls, registry, path = setup_gateway
    leave_started(gateway, monkeypatch)
    gateway = ToolGateway(path, registry)
    outcomes = ReconcileSweeper(gateway, interval_s=60, min_age_s=0).tick()
    assert [o["status"] for o in outcomes] == ["needs_manual"]
    assert calls == []


def test_tick_skips_a_call_that_is_still_running(setup_gateway, monkeypatch):
    _, calls, registry, path = setup_gateway
    import threading

    running, release = threading.Event(), threading.Event()

    def slow(ctx, args):
        running.set()
        release.wait(timeout=10)
        return {"value": args.value}

    registry["internal"] = replace(registry["internal"], fn=slow)
    gateway = ToolGateway(path, registry)
    worker = threading.Thread(target=lambda: gateway.call(ToolContext(AGENT, "run"), "internal", {"shop_id": "mine"}))
    worker.start()
    try:
        assert running.wait(timeout=10)
        assert [o["status"] for o in ReconcileSweeper(gateway, interval_s=60, min_age_s=0).tick()] == ["skipped_inflight"]
    finally:
        release.set()
        worker.join(timeout=10)


def test_thread_ticks_on_its_own_and_stops(setup_gateway):
    gateway, *_ = setup_gateway
    sweeper = ReconcileSweeper(gateway, interval_s=0.02, min_age_s=0)
    ticks: list[int] = []
    sweeper.tick = lambda: ticks.append(1) or []  # type: ignore[method-assign]
    sweeper.start()
    try:
        assert wait_until(lambda: len(ticks) >= 2)
    finally:
        sweeper.stop()
    assert sweeper._thread is None
