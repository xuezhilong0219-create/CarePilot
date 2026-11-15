from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha1
import sqlite3
import threading
import time

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.ledger import canonical_json, idempotency_key
from carepilot.gateway.spec import Executed, ToolContext
from conftest import AGENT


def test_key_matches_frozen_formula_and_unicode():
    args = {"z": [1, {"b": "中文"}], "a": 2}
    canonical = '{"a":2,"z":[1,{"b":"中文"}]}'
    assert canonical_json(args) == canonical
    expected = sha1(("run\0tool\0" + canonical).encode("utf-8")).hexdigest()
    assert idempotency_key("run", "tool", args) == expected
    assert idempotency_key("run", "tool", {"a": 2, "z": args["z"]}) == expected
    assert idempotency_key("run\0tool", "other", args) != expected


def test_semantic_defaults_reordered_args_and_three_replays(setup_gateway):
    gateway, calls, _, path = setup_gateway
    ctx = ToolContext(AGENT, "run", "ticket", "untrusted-key")
    first = gateway.call(ctx, "internal", {"shop_id": "mine"})
    for _ in range(3):
        replay = gateway.call(ctx, "internal", {"phone": "", "value": 1, "shop_id": "mine"})
        assert replay == Executed(first.result, replayed=True)
    assert len(calls) == 1
    assert calls[0][0].idem_key == idempotency_key("run", "internal", calls[0][1].model_dump(mode="json"))
    assert ctx.idem_key == "untrusted-key"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status, COUNT(*) FROM action_ledger").fetchone() == ("done", 1)
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_two_thread_barrier_fifty_rounds_execute_once(setup_gateway):
    _, calls, registry, path = setup_gateway
    original = registry["internal"].fn
    def slow(ctx, args):
        time.sleep(0.002)
        return original(ctx, args)
    registry["internal"] = replace(registry["internal"], fn=slow)
    gateways = [ToolGateway(path, registry), ToolGateway(path, registry)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(50):
            barrier = threading.Barrier(2)
            def call(worker, run=index, barrier=barrier):
                barrier.wait(timeout=5)
                return gateways[worker].call(ToolContext(AGENT, f"round-{run}"), "internal", {"shop_id": "mine"})
            results = list(pool.map(call, (0, 1)))
            assert all(isinstance(result, Executed) for result in results)
            assert sorted(result.replayed for result in results) == [False, True]
            assert len(calls) == index + 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM action_ledger WHERE status='done'").fetchone()[0] == 50


def test_tool_error_is_abandoned_audited_and_retryable(setup_gateway):
    _, calls, registry, path = setup_gateway
    effect = registry["internal"].fn
    attempts = []
    def flaky(ctx, args):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("13812345678")
        return effect(ctx, args)
    registry["internal"] = replace(registry["internal"], fn=flaky)
    gateway = ToolGateway(path, registry)
    ctx = ToolContext(AGENT, "run")
    assert gateway.call(ctx, "internal", {"shop_id": "mine"}).code == "tool_error"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM action_ledger").fetchone()[0] == "abandoned"
    assert isinstance(gateway.call(ctx, "internal", {"shop_id": "mine"}), Executed)
    assert len(calls) == 1
    assert any(row["code"] == "tool_error" for row in gateway.audit())
