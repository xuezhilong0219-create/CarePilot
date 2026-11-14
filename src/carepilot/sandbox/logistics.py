"""不可靠的外部物流接口客户端：限时、重试、降级。

只有外部故障（超时 / 连接失败）才降级；其他异常视为 bug，原样抛给上层。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

UNAVAILABLE = {"status": "unavailable", "message": "暂时查不到物流信息，请稍后再试"}


def _call_with_timeout(
    transport: Callable[[str], dict], order_id: str, timeout_s: float
) -> dict:
    """线程 + 队列实现限时，超时后不等待阻塞的 transport。"""
    result: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result.put(("ok", transport(order_id)))
        except BaseException as exc:  # 需要把异常带回调用线程再判断
            result.put(("error", exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        kind, value = result.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise TimeoutError(f"物流接口 {timeout_s}s 未返回") from exc
    if kind == "error":
        raise value
    return value


class LogisticsClient:
    """`transport(order_id)` 可以是任意会超时、阻塞或抛外部故障的函数。"""

    def __init__(
        self,
        transport: Callable[[str], dict],
        *,
        timeout_s: float = 5.0,
        retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if retries < 0:
            raise ValueError("retries 不能为负数")
        self.transport = transport
        self.timeout_s = timeout_s
        self.retries = retries
        self.sleep = sleep

    def query(self, order_id: str) -> dict:
        """最多调用 transport `retries + 1` 次；全部外部故障时返回降级结果，永不抛外部异常。"""
        attempts = 0
        for attempt in range(self.retries + 1):
            attempts += 1
            try:
                payload = _call_with_timeout(self.transport, order_id, self.timeout_s)
                return {"status": "ok", **payload, "attempts": attempts}
            except (TimeoutError, ConnectionError):
                if attempt < self.retries:
                    self.sleep(0.05 * (attempt + 1))
        return {**UNAVAILABLE, "attempts": attempts}
