"""并发下的网关：锁只夹住占位和记账，工具在锁外跑，副作用仍然恰好一次。

这些测试钉的是「锁的宽度」这件事本身——不测并发不出门的问题，而是测：
慢工具会不会把别的调用堵住、同一条调用进来两次会不会执行两遍、等不到落账时报什么。
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sqlite3
import threading
import time

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.ledger import canonical_json, idempotency_key
from carepilot.gateway.spec import Executed, NeedsApproval, Rejected, ToolContext
from conftest import AGENT

SLEEP = 0.25


def slow_everything(registry, calls, seconds=SLEEP, *, only_run=None):
    """把四个工具换成「睡一会儿再干活」，睡的时候就是锁有没有被占住的观察窗口。

    only_run 只让某一条 run 变慢：这样「新来的调用有多快」才不会被它自己的工具耗时污染。
    """
    original = registry["read"].fn

    def slow(ctx, args):
        if only_run is None or ctx.run_id == only_run:
            time.sleep(seconds)
        return original(ctx, args)

    for name in list(registry):
        registry[name] = replace(registry[name], fn=slow)
    return registry


def test_distinct_keys_do_not_queue_behind_each_other(setup_gateway):
    _, calls, registry, path = setup_gateway
    slow_everything(registry, calls)
    gateway = ToolGateway(path, registry)
    workers = 4

    def one(index: int) -> None:
        gateway.call(ToolContext(AGENT, f"distinct-{index}"), "read", {"shop_id": "mine"})

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(workers)))
    elapsed = time.perf_counter() - started
    # 串行要 4 × 0.25 = 1.0s；真并发只该是「一次工具 + 几次记账」
    assert len(calls) == workers
    assert elapsed < workers * SLEEP * 0.75


def test_slow_inflight_call_does_not_block_a_new_key(setup_gateway):
    _, calls, registry, path = setup_gateway
    slow_everything(registry, calls, seconds=0.4, only_run="holder")
    gateway = ToolGateway(path, registry)
    worker = threading.Thread(target=gateway.call, args=(ToolContext(AGENT, "holder"), "read", {"shop_id": "mine"}))
    worker.start()
    time.sleep(0.1)  # 让它先占到 started 并进入工具
    started = time.perf_counter()
    outcome = gateway.call(ToolContext(AGENT, "newcomer"), "read", {"shop_id": "mine"})
    waited = time.perf_counter() - started
    worker.join(timeout=5)
    assert isinstance(outcome, Executed)
    # 新 key 不该等那个 0.4s 的工具跑完
    assert waited < 0.25


def test_same_key_concurrent_callers_all_get_the_same_result(setup_gateway):
    _, calls, registry, path = setup_gateway
    slow_everything(registry, calls, seconds=0.1)
    gateway = ToolGateway(path, registry)
    workers = 8
    barrier = threading.Barrier(workers)

    def one(_index: int):
        barrier.wait()
        return gateway.call(ToolContext(AGENT, "same-run"), "read", {"shop_id": "mine"})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(workers)))
    assert len(calls) == 1
    assert all(isinstance(result, Executed) for result in results)
    # 恰好一次执行，其余全部回放——不是报错、也不是各跑一遍
    assert sorted(result.replayed for result in results) == [False] + [True] * (workers - 1)
    assert len({canonical_json(result.result) for result in results}) == 1


def test_inflight_wait_times_out_into_reconciliation_required(setup_gateway):
    _, calls, registry, path = setup_gateway
    gateway = ToolGateway(path, registry, inflight_wait_s=0.05)
    args = {"shop_id": "mine"}
    key = idempotency_key("stuck-run", "read", {"phone": "", "shop_id": "mine", "value": 1})
    with gateway.ledger.transaction() as db:
        db.execute("""INSERT INTO action_ledger (idem_key,run_id,tool,status,created_at,updated_at)
            VALUES (?,?,?,'started',?,?)""", (key, "stuck-run", "read", time.time(), time.time()))
        db.execute("""INSERT INTO action_inputs
            (idem_key,principal_json,ticket_id,args_json,approver_id,approval_id,recovery_status)
            VALUES (?,?,NULL,?,NULL,NULL,NULL)""", (key, '{"role":"agent","shop_ids":[],"user_id":"agent-1"}', canonical_json(args)))
    outcome = gateway.call(ToolContext(AGENT, "stuck-run"), "read", args)
    assert isinstance(outcome, Rejected)
    assert outcome.code == "reconciliation_required"
    # 未决不等于没发生：不能替那个执行者重跑一遍
    assert calls == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM action_ledger").fetchone()[0] == "started"


def test_approved_state_change_runs_once_even_though_it_holds_the_lock(setup_gateway):
    _, calls, registry, path = setup_gateway
    slow_everything(registry, calls, seconds=0.05)
    gateway = ToolGateway(path, registry)
    ctx = ToolContext(AGENT, "approve-run")
    pending = gateway.call(ctx, "state", {"shop_id": "mine"})
    assert isinstance(pending, NeedsApproval)
    first = gateway.approve(pending.approval_id, AGENT)
    second = gateway.approve(pending.approval_id, AGENT)
    assert isinstance(first, Executed) and isinstance(second, Rejected)
    assert second.code == "approval_not_pending"
    assert len(calls) == 1
