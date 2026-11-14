"""网关黑盒对抗测试：只看 spec.py / auth.py / requirements.md，不读实现。

所有用例都通过公开接口 ``ToolGateway`` 交互；失败时打印可复现的 seed 与操作序列。
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from carepilot.auth import Principal, Role
from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import (
    Executed,
    NeedsApproval,
    Rejected,
    Risk,
    ToolContext,
    ToolSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = REPO_ROOT / ".tmp_test"
PHONE = "13812345678"


@pytest.fixture
def work_dir():
    path = TMP_ROOT / f"gw-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- 假工具


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q: str
    amount: float | None = None


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class StateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


class SendArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: str
    message: str
    priority: int = 1


def normalize(args: dict) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=repr)


class FakeTools:
    """四个风险档的假工具，记录真实调用次数（副作用发生的唯一证据）。"""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.arg_log: dict[tuple, list] = {}
        self.lock = threading.Lock()

    def _hit(self, tool: str, ctx: ToolContext, args: BaseModel, result: dict) -> dict:
        key = (ctx.run_id, tool, normalize(args.model_dump()))
        with self.lock:
            self.counts[key] += 1
            self.arg_log.setdefault(key, []).append(args.model_dump())
        return result

    def read(self, ctx: ToolContext, args: ReadArgs) -> dict:
        return self._hit("read", ctx, args, {"value": len(args.q)})

    def note(self, ctx: ToolContext, args: NoteArgs) -> dict:
        return self._hit("note", ctx, args, {"noted": True})

    def state(self, ctx: ToolContext, args: StateArgs) -> dict:
        return self._hit("state", ctx, args, {"status": args.status})

    def send(self, ctx: ToolContext, args: SendArgs) -> dict:
        # 结果里绝不回显手机号，避免测试自己制造敏感信息泄漏
        return self._hit("send", ctx, args, {"delivered": True})

    @staticmethod
    def summarize_send(args: SendArgs) -> str:
        return json.dumps({"phone": "1**********", "message": args.message}, ensure_ascii=False)

    def specs(self) -> dict[str, ToolSpec]:
        everyone = frozenset(Role)
        return {
            "read": ToolSpec("read", ReadArgs, self.read, Risk.READ, everyone),
            "note": ToolSpec("note", NoteArgs, self.note, Risk.WRITE_INTERNAL, everyone),
            "state": ToolSpec("state", StateArgs, self.state, Risk.STATE_CHANGE, everyone),
            "send": ToolSpec(
                "send",
                SendArgs,
                self.send,
                Risk.EXTERNAL_SEND,
                frozenset({Role.AGENT, Role.ADMIN}),
                summarize_args=self.summarize_send,
            ),
        }

    def snapshot(self) -> Counter:
        with self.lock:
            return Counter(self.counts)

    def total(self) -> int:
        with self.lock:
            return sum(self.counts.values())


def new_gateway(path: Path, tools: FakeTools, **kwargs) -> ToolGateway:
    return ToolGateway(path / "gateway.db", tools.specs(), **kwargs)


def principal(role: Role, user_id: str | None = None) -> Principal:
    return Principal(user_id or f"u-{role.value}", role)


def ctx_for(principal_: Principal, run_id: str = "run-1") -> ToolContext:
    return ToolContext(principal=principal_, run_id=run_id)


def make_tools() -> FakeTools:
    return FakeTools()


# ---------------------------------------------------------------- 随机序列不变量


TOOL_ARGS = {
    "read": [
        {"q": "库存同步"},
        {"q": "x" * 20},
        {"q": 123},
        {},
        {"q": "订单", "extra": 1},
        {"q": "授权", "amount": 1.5},
    ],
    "note": [{"text": "内部备注"}, {}, {"text": 5}, {"text": "备注", "extra": True}],
    "state": [{"status": "done"}, {"status": "pending"}, {}, {"status": 0}],
    "send": [
        {"phone": PHONE, "message": "您好"},
        {"phone": PHONE},
        {"phone": PHONE, "message": "您好", "extra": 1},
        {"phone": 123, "message": "您好"},
    ],
    "ghost": [{"q": "x"}, {}],
}
TOOL_NAMES = list(TOOL_ARGS)


def _random_step(rng: random.Random, known_approvals: list[str]) -> tuple:
    kind = rng.choices(
        ["call", "approve", "confirm", "reject"], weights=[62, 13, 13, 12]
    )[0]
    if kind == "call":
        tool = rng.choice(TOOL_NAMES)
        args = dict(rng.choice(TOOL_ARGS[tool]))
        return kind, principal(rng.choice(list(Role))), rng.choice(["run-1", "run-2"]), tool, args
    approval_id = rng.choice(known_approvals + [uuid.uuid4().hex, "", "missing"])
    return kind, principal(rng.choice(list(Role))), None, approval_id, None


def _run_random_sequence(seed: int, work_dir: Path) -> tuple[FakeTools, list[dict], list[str]]:
    rng = random.Random(seed)
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    known: list[str] = []
    trace: list[dict] = []
    log: list[str] = []
    steps = rng.randint(5, 10)
    audit_len = 0

    for step in range(steps):
        kind, who, run_id, a, b = _random_step(rng, known)
        before = tools.snapshot()
        audit_before = audit_len
        if kind == "call":
            tool, raw_args = a, b
            log.append(
                f"[{step}] call tool={tool!r} role={who.role.value} run={run_id!r} "
                f"args={json.dumps(raw_args, ensure_ascii=False, default=repr)}"
            )
            try:
                outcome = gw.call(ctx_for(who, run_id), tool, raw_args)
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"seed={seed} 第 {step} 步 call 抛异常 {type(exc).__name__}: {exc}\n"
                    + "\n".join(log)
                ) from exc
            if isinstance(outcome, NeedsApproval):
                known.append(outcome.approval_id)
        elif kind == "approve":
            log.append(f"[{step}] approve id={a!r} role={who.role.value}")
            try:
                outcome = gw.approve(a, who)
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"seed={seed} 第 {step} 步 approve 抛异常 {type(exc).__name__}: {exc}\n"
                    + "\n".join(log)
                ) from exc
            if isinstance(outcome, NeedsApproval):
                known.append(outcome.approval_id)
        elif kind == "confirm":
            log.append(f"[{step}] confirm id={a!r} role={who.role.value}")
            try:
                outcome = gw.confirm(a, who)
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"seed={seed} 第 {step} 步 confirm 抛异常 {type(exc).__name__}: {exc}\n"
                    + "\n".join(log)
                ) from exc
        else:
            log.append(f"[{step}] reject id={a!r} role={who.role.value}")
            try:
                gw.reject(a, who, "adversarial")
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"seed={seed} 第 {step} 步 reject 抛异常 {type(exc).__name__}: {exc}\n"
                    + "\n".join(log)
                ) from exc
            outcome = None

        after = tools.snapshot()
        audit_after = len(gw.audit(limit=100000))
        audit_len = audit_after
        trace.append(
            {
                "step": step,
                "kind": kind,
                "outcome": outcome,
                "delta": after - before,
                "audit_delta": audit_after - audit_before,
            }
        )
        # I4：每一次 call/approve/confirm/reject 至少一条审计
        assert audit_after > audit_before, (
            f"seed={seed} 第 {step} 步 {kind} 没有留下审计记录\n" + "\n".join(log)
        )
        # I6：被拒绝的调用不能执行工具
        if isinstance(outcome, Rejected):
            assert sum((after - before).values()) == 0, (
                f"seed={seed} 第 {step} 步被拒绝却执行了工具: {outcome}\n" + "\n".join(log)
            )
        # I3：EXTERNAL_SEND 只能在确认阶段产生副作用
        for (run, tool, _), count in (after - before).items():
            if tool == "send":
                assert kind == "confirm" and isinstance(outcome, Executed), (
                    f"seed={seed} 第 {step} 步 send 在非 confirm 阶段执行: kind={kind} "
                    f"outcome={outcome}\n" + "\n".join(log)
                )

    # I1：同一 (run_id, 工具, 规范化参数) 至多执行一次
    for key, count in tools.counts.items():
        assert count <= 1, f"seed={seed} 同一动作执行了 {count} 次: {key}\n" + "\n".join(log)

    audit = gw.audit(limit=100000)
    # I5：敏感明文不出现在审计任何一列
    blob = json.dumps(audit, ensure_ascii=False, default=repr)
    assert PHONE not in blob, f"seed={seed} 审计里出现手机号明文\n" + "\n".join(log)

    # I2 / I3：副作用前必须有批准（发送还要 confirm）记录
    for tool in ("state", "send"):
        for rec in audit:
            if rec.get("tool") != tool or rec.get("decision") != "executed":
                continue
            pane = [
                r
                for r in audit
                if r.get("tool") == tool
                and r.get("run_id") == rec.get("run_id")
                and r.get("idem_key") == rec.get("idem_key")
                and r.get("id") < rec.get("id")
            ]
            assert any(
                r.get("decision") == "approved" and r.get("code") == "approval_granted"
                for r in pane
            ), f"seed={seed} {tool} 执行前没有批准记录: {rec}\n" + "\n".join(log)
            if tool == "send":
                assert any(
                    r.get("decision") == "confirmed" for r in pane
                ), f"seed={seed} send 执行前没有确认记录: {rec}\n" + "\n".join(log)
    return tools, trace, log


def test_random_sequences_keep_invariants(work_dir):
    total_steps = 0
    for seed in range(300):
        tools, trace, _ = _run_random_sequence(seed, work_dir)
        total_steps += len(trace)
    assert total_steps >= 1500


# ---------------------------------------------------------------- 并发


def test_concurrent_same_key_executes_once(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    threads = 16
    rounds = 30
    started = time.monotonic()

    for r in range(rounds):
        round_started = time.monotonic()
        state_key = ("run-1", "state", normalize({"status": f"s{r}"}))
        barrier = threading.Barrier(threads)
        ids: list[str] = []
        ids_lock = threading.Lock()

        def state_worker() -> None:
            ctx = ctx_for(principal(Role.AGENT), "run-1")
            barrier.wait()
            outcome = gw.call(ctx, "state", {"status": f"s{r}"})
            if isinstance(outcome, NeedsApproval):
                with ids_lock:
                    ids.append(outcome.approval_id)

        workers = [threading.Thread(target=state_worker) for _ in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=30)
        assert ids, f"第 {r} 轮没有任何批准单"
        barrier2 = threading.Barrier(threads)

        def approve_worker(index: int) -> None:
            approval_id = ids[index % len(ids)]
            barrier2.wait()
            gw.approve(approval_id, principal(Role.ADMIN))

        workers = [threading.Thread(target=approve_worker, args=(i,)) for i in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=30)
        assert tools.counts[state_key] == 1, f"第 {r} 轮 STATE_CHANGE 副作用 {tools.counts[state_key]} 次"

        send_key = ("run-1", "send", normalize({"phone": PHONE, "message": f"m{r}", "priority": 1}))
        barrier3 = threading.Barrier(threads)
        send_ids: list[str] = []

        def send_call_worker() -> None:
            ctx = ctx_for(principal(Role.AGENT), "run-1")
            barrier3.wait()
            outcome = gw.call(ctx, "send", {"phone": PHONE, "message": f"m{r}"})
            if isinstance(outcome, NeedsApproval):
                with ids_lock:
                    send_ids.append(outcome.approval_id)

        workers = [threading.Thread(target=send_call_worker) for _ in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=30)
        assert send_ids, f"第 {r} 轮发送没有批准单"
        barrier4 = threading.Barrier(threads)

        def send_approve_worker(index: int) -> None:
            barrier4.wait()
            gw.approve(send_ids[index % len(send_ids)], principal(Role.ADMIN))

        workers = [threading.Thread(target=send_approve_worker, args=(i,)) for i in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=30)
        barrier5 = threading.Barrier(threads)

        def confirm_worker(index: int) -> None:
            barrier5.wait()
            gw.confirm(send_ids[index % len(send_ids)], principal(Role.ADMIN))

        workers = [threading.Thread(target=confirm_worker, args=(i,)) for i in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=30)
        assert tools.counts[send_key] == 1, f"第 {r} 轮 EXTERNAL_SEND 副作用 {tools.counts[send_key]} 次"
        assert time.monotonic() - round_started < 30, f"第 {r} 轮耗时超过 30 秒，疑似死锁"

    elapsed = time.monotonic() - started
    assert elapsed < 90, f"并发测试总耗时 {elapsed:.1f}s，疑似死锁"


def test_concurrent_distinct_keys_do_not_deadlock(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []
    started = time.monotonic()

    def worker(i: int) -> None:
        try:
            barrier.wait()
            outcome = gw.call(ctx_for(principal(Role.AGENT), "run-1"), "read", {"q": f"item-{i}"})
            assert isinstance(outcome, Executed)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert tools.total() == 16
    assert time.monotonic() - started < 30


# ---------------------------------------------------------------- 定向攻击


def test_customer_cannot_approve_own_state_change(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    customer = principal(Role.CUSTOMER)
    outcome = gw.call(ctx_for(customer), "state", {"status": "closed"})
    assert isinstance(outcome, NeedsApproval)

    approval = gw.approve(outcome.approval_id, customer)
    assert isinstance(approval, Rejected)
    assert approval.gate.value == "permission"
    assert tools.total() == 0


def test_confirm_uses_approved_parameters_not_new_ones(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    agent = principal(Role.AGENT)
    first = gw.call(ctx_for(agent), "send", {"phone": PHONE, "message": "原始内容"})
    assert isinstance(first, NeedsApproval)
    assert isinstance(gw.approve(first.approval_id, agent), NeedsApproval)

    # 批准后换一份参数再次提议：这是新的幂等键，不能复用旧批准
    second = gw.call(ctx_for(agent), "send", {"phone": PHONE, "message": "被替换的内容"})
    assert isinstance(second, NeedsApproval)
    assert second.approval_id != first.approval_id

    confirmed = gw.confirm(first.approval_id, agent)
    assert isinstance(confirmed, Executed)
    executed = [args for (run, tool, _), values in tools.arg_log.items() if tool == "send" for args in values]
    assert executed == [{"phone": PHONE, "message": "原始内容", "priority": 1}]
    assert isinstance(gw.confirm(second.approval_id, agent), Rejected)


def test_reject_then_approve_same_id_does_not_execute(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    agent = principal(Role.AGENT)
    outcome = gw.call(ctx_for(agent), "state", {"status": "done"})
    assert isinstance(outcome, NeedsApproval)

    gw.reject(outcome.approval_id, agent, "不要执行")
    assert isinstance(gw.approve(outcome.approval_id, agent), Rejected)
    assert tools.total() == 0


def test_confirm_on_read_tool_is_cleanly_rejected(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    agent = principal(Role.AGENT)
    # READ 自动执行，不会产生 approval_id；用任意 id confirm 必须是干净的 Rejected
    assert isinstance(gw.call(ctx_for(agent), "read", {"q": "订单"}), Executed)
    outcome = gw.confirm("not-an-approval-id", agent)
    assert isinstance(outcome, Rejected)
    # 后续调用仍然正常
    assert isinstance(gw.call(ctx_for(agent), "read", {"q": "库存"}), Executed)


def test_approval_cannot_be_replayed_across_run_ids(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    agent = principal(Role.AGENT)
    first = gw.call(ctx_for(agent, "run-1"), "state", {"status": "done"})
    second = gw.call(ctx_for(agent, "run-2"), "state", {"status": "done"})
    assert isinstance(first, NeedsApproval) and isinstance(second, NeedsApproval)
    assert first.approval_id != second.approval_id  # 幂等键含 run_id

    assert isinstance(gw.approve(first.approval_id, agent), Executed)
    assert isinstance(gw.approve(first.approval_id, agent), Rejected)
    assert tools.counts[("run-1", "state", normalize({"status": "done"}))] == 1
    # run-2 的单子仍然待批，没有被 run-1 的批准顺带执行
    assert tools.counts[("run-2", "state", normalize({"status": "done"}))] == 0
    assert isinstance(gw.approve(second.approval_id, agent), Executed)


def test_hostile_argument_shapes_are_clean(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    agent = principal(Role.AGENT)
    deep: object = "x"
    for _ in range(200):
        deep = [deep]

    cases = [
        ("read", {"q": "a" * 100_000}),
        ("read", {"q": deep}),
        ("read", {"q": "订单", "amount": float("nan")}),
        ("read", {"q": "订单", "amount": float("inf")}),
        ("read", {"q": "a\x00b"}),
        ("read", {"q": "😀" * 50}),
        ("state", {"status": "a\x00b"}),
        ("send", {"phone": PHONE, "message": "😀" * 50}),
    ]
    for tool, args in cases:
        try:
            outcome = gw.call(ctx_for(agent), tool, args)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{tool} 对异常参数抛未捕获异常 {type(exc).__name__}: {exc}") from exc
        assert isinstance(outcome, (Executed, NeedsApproval, Rejected))

    # 恶心参数之后，正常调用仍然可用
    assert isinstance(gw.call(ctx_for(agent), "read", {"q": "正常"}), Executed)
    blob = json.dumps(gw.audit(limit=100000), ensure_ascii=False, default=repr)
    assert PHONE not in blob


def test_audit_limit_zero_and_huge_are_clean(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    gw.call(ctx_for(principal(Role.AGENT)), "read", {"q": "x"})
    for limit in (0, 10**9):
        rows = gw.audit(limit=limit)
        assert isinstance(rows, list)


# 回归：黑盒测试发现 audit(limit=-1) 抛 ValueError（最初是严格 xfail），修复为钳到 0 返回空列表
def test_audit_limit_negative_does_not_crash(work_dir):
    tools = make_tools()
    gw = new_gateway(work_dir, tools)
    gw.call(ctx_for(principal(Role.AGENT)), "read", {"q": "x"})
    rows = gw.audit(limit=-1)
    assert rows == []  # 不能因为 SQLite 的 LIMIT -1 = 不限条数，就把整张审计表倒出来
