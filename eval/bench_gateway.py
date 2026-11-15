"""网关并发账：锁只夹住「占位」和「记账」两段，工具本身在锁外跑。

同一个工作负载跑两遍，唯一变量是「锁有没有把工具执行也圈进去」：

  串行 = 调用方自己持锁包住整次调用（等于改动前的行为，`ledger.exclusive()` 跨过副作用）
  并发 = 交给网关管锁（改动后的行为，`_reserve` 只锁占位、`_execute` 只锁记账）

还会验证并发下副作用仍然恰好一次：同一 run + 同一参数打 16 次，效果文件只能有一行。

用法：
    uv run python eval/bench_gateway.py
    uv run python eval/bench_gateway.py --workers 16 --sleep-ms 100 --procs 8 --keep

这个脚本不调用模型，不需要 LLM_API_KEY。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from carepilot.auth import Principal, Role  # noqa: E402
from carepilot.gateway.gateway import ToolGateway  # noqa: E402
from carepilot.gateway.ledger import idempotency_key  # noqa: E402
from carepilot.gateway.spec import Risk, ToolContext, ToolSpec  # noqa: E402


class SlowArgs(BaseModel):
    order_id: str


def build_gateway(db: Path, effects: Path, sleep_s: float) -> ToolGateway:
    def slow_read(ctx: ToolContext, args: SlowArgs) -> dict:
        time.sleep(sleep_s)
        with effects.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"order_id": args.order_id}) + "\n")
        return {"order_id": args.order_id}

    spec = ToolSpec(
        name="slow_read",
        args_model=SlowArgs,
        fn=slow_read,
        risk=Risk.READ,
        roles=frozenset({Role.CUSTOMER, Role.AGENT, Role.ADMIN}),
    )
    return ToolGateway(db, {spec.name: spec})


ADMIN = Principal("admin1", Role.ADMIN, frozenset({"S1"}))


def one_call(gateway: ToolGateway, ctx: ToolContext) -> float:
    """跑一次调用，返回它自己的墙钟秒数。"""
    started = time.perf_counter()
    gateway.call(ctx, "slow_read", {"order_id": ctx.run_id})
    return time.perf_counter() - started


def one_call_like_before(gateway: ToolGateway, ctx: ToolContext) -> float:
    """改动前 call() 的路径：调用方持着全局锁，一直走到工具跑完才放。

    这里直接调 _execute(locked=True) 而不是 call()，否则会在同一线程上二次申请同一把
    非重入锁——那正是改动前不存在的开销，不能拿它冒充基线。
    """
    spec = gateway.registry["slow_read"]
    args = spec.args_model.model_validate({"order_id": ctx.run_id})
    key = idempotency_key(ctx.run_id, "slow_read", args.model_dump(mode="json"))
    started = time.perf_counter()
    with gateway.ledger.exclusive():
        gateway._execute(replace(ctx, idem_key=key), "slow_read", spec, args, locked=True)
    return time.perf_counter() - started


def load_of(gateway: ToolGateway, workers: int, run_prefix: str, sleep_s: float, *, serialize: bool) -> dict:
    """workers 个不同 run 同时打进来。serialize=True 走改动前的路径。"""
    barrier = threading.Barrier(workers)

    def task(index: int) -> float:
        ctx = ToolContext(ADMIN, f"{run_prefix}-{index}")
        barrier.wait()
        return one_call_like_before(gateway, ctx) if serialize else one_call(gateway, ctx)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_call = list(pool.map(task, range(workers)))
    wall = time.perf_counter() - started
    total_work = workers * sleep_s
    return {
        "wall": wall,
        "p50": statistics.median(per_call),
        "max": max(per_call),
        "overlap": total_work / wall if wall else 0.0,
    }


def same_key(gateway: ToolGateway, workers: int, effects: Path) -> dict:
    """同一 run + 同一参数打 workers 次：只允许执行一次，其余的要么回放要么报未决。"""
    ctx = ToolContext(ADMIN, "same-key-run")
    barrier = threading.Barrier(workers)
    outcomes: list[str] = []
    lock = threading.Lock()

    def task(_index: int) -> None:
        barrier.wait()
        outcome = gateway.call(ctx, "slow_read", {"order_id": "same-key-run"})
        with lock:
            outcomes.append(getattr(outcome, "code", None) or ("replayed" if outcome.replayed else "executed"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(task, range(workers)))
    lines = effects.read_text(encoding="utf-8").splitlines() if effects.exists() else []
    return {"outcomes": outcomes, "effects": len(lines)}


def _child(arguments: tuple[str, str, float, int, str, str, object]) -> None:
    db, effects, sleep_s, index, ready, release, queue = arguments
    gateway = build_gateway(Path(db), Path(effects), sleep_s)
    Path(ready).write_text("ready", encoding="utf-8")
    while not Path(release).exists():
        time.sleep(0.005)
    started = time.perf_counter()
    gateway.call(ToolContext(ADMIN, f"proc-{index}"), "slow_read", {"order_id": f"proc-{index}"})
    queue.put(time.perf_counter() - started)


def across_processes(db: Path, effects: Path, sleep_s: float, procs: int, work: Path) -> dict:
    """真多进程：文件锁是跨进程的，所以这条才证明得了「锁没有把执行串起来」。

    进程启动开销不算在工作负载里——都先建好库、报 ready，父进程发令之后才计时。
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    release = work / "release"
    children = []
    for index in range(procs):
        ready = work / f"ready-{index}"
        process = context.Process(target=_child, args=((str(db), str(effects), sleep_s, index, str(ready), str(release), queue),))
        process.start()
        children.append((process, ready))
    for _process, ready in children:
        while not ready.exists():
            time.sleep(0.005)
    started = time.perf_counter()
    release.write_text("go", encoding="utf-8")
    for process, _ready in children:
        process.join(timeout=120)
    wall = time.perf_counter() - started
    per_call = [queue.get(timeout=5) for _ in children]
    return {"wall": wall, "p50": statistics.median(per_call), "max": max(per_call),
            "overlap": (procs * sleep_s) / wall if wall else 0.0}


def line(name: str, workers: int, sleep_ms: int, result: dict) -> str:
    serial = workers * sleep_ms / 1000
    return (f"| {name} | {workers} | {sleep_ms} | {result['wall']:.2f}s | {result['p50'] * 1000:.0f}ms | "
            f"{result['overlap']:.2f}× | 串行下界 {serial:.2f}s |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="网关并发账：锁是否把工具执行也串起来了")
    parser.add_argument("--workers", type=int, default=16, help="线程数")
    parser.add_argument("--sleep-ms", type=int, default=100, help="工具自身的耗时")
    parser.add_argument("--procs", type=int, default=8, help="跨进程那一档的进程数，0 表示跳过")
    parser.add_argument("--keep", action="store_true", help="保留临时目录，便于自己看 sqlite")
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix=".tmp-bench-", dir=ROOT))
    sleep_s = args.sleep_ms / 1000
    try:
        # 两档各自独立的库和效果文件，互不干扰
        serial_db, serial_fx = work / "serial.sqlite", work / "serial.jsonl"
        conc_db, conc_fx = work / "concurrent.sqlite", work / "concurrent.jsonl"
        serial_gateway = build_gateway(serial_db, serial_fx, sleep_s)
        conc_gateway = build_gateway(conc_db, conc_fx, sleep_s)

        serial = load_of(serial_gateway, args.workers, "serial", sleep_s, serialize=True)
        concurrent = load_of(conc_gateway, args.workers, "concurrent", sleep_s, serialize=False)

        print(f"\n## 同一份负载（{args.workers} 个不同 run，工具每次睡 {args.sleep_ms}ms）\n")
        print("| 场景 | 并发数 | 工具耗时 | 墙钟 | 单次 p50 | 并行度 | 对照 |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        print(line("改动前（锁跨过执行）", args.workers, args.sleep_ms, serial))
        print(line("改动后（锁只夹记账）", args.workers, args.sleep_ms, concurrent))

        dup_db, dup_fx = work / "dup.sqlite", work / "dup.jsonl"
        dup_gateway = build_gateway(dup_db, dup_fx, sleep_s)
        dup = same_key(dup_gateway, args.workers, dup_fx)
        effects_total = len(dup_fx.read_text(encoding="utf-8").splitlines())
        counts = {name: dup["outcomes"].count(name) for name in sorted(set(dup["outcomes"]))}
        print(f"\n### 同一 run + 同一参数打 {args.workers} 次（带 {args.sleep_ms}ms 工具）\n")
        print(f"- 结果分布：{counts}")
        print(f"- 效果文件共 {effects_total} 行（{args.workers} 次调用之后）")

        if args.procs:
            proc_db, proc_fx = work / "procs.sqlite", work / "procs.jsonl"
            proc_gateway = build_gateway(proc_db, proc_fx, sleep_s)
            del proc_gateway  # 子进程各自开库；父进程先建好 schema
            proc = across_processes(proc_db, proc_fx, sleep_s, args.procs, work)
            print(f"\n### 跨进程（{args.procs} 个进程，各查一单，工具睡 {args.sleep_ms}ms）\n")
            print(f"- 墙钟 {proc['wall']:.2f}s，单次 p50 {proc['p50'] * 1000:.0f}ms，并行度 {proc['overlap']:.2f}×")
            print(f"- 串行下界 {args.procs * sleep_s:.2f}s，效果文件 {len(proc_fx.read_text(encoding='utf-8').splitlines())} 行")
        return 0
    finally:
        if not args.keep:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"\n临时目录：{work}")


if __name__ == "__main__":
    raise SystemExit(main())
