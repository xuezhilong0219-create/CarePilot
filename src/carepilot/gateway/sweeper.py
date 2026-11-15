"""周期性对账：让「卡在 started 的调用」自己被人捡起来，而不是等运维手写脚本。

默认关闭（`RECONCILE_INTERVAL_S=0`）。对账可能会认领一条副作用并把它记成 done，这是
写库的动作；多久扫一次、多久才算「卡住」，属于部署环境的决定，不该由代码替它决定。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .gateway import ToolGateway

log = logging.getLogger(__name__)


class ReconcileSweeper:
    def __init__(self, gateway: ToolGateway, *, interval_s: float = 60.0, min_age_s: float = 300.0):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.gateway = gateway
        self.interval_s = interval_s
        self.min_age_s = min_age_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> list[dict[str, Any]]:
        """跑一轮。返回 reconcile 的原始结果，日志里只报有处理动作的那些。"""
        outcomes = self.gateway.reconcile(min_age_s=self.min_age_s)
        handled = [o for o in outcomes if o["status"] in ("done", "abandoned", "needs_manual")]
        if handled:
            log.warning("reconcile: 处理 %d 条卡住的调用，跳过 %d 条（在飞或太新）", len(handled), len(outcomes) - len(handled))
        return outcomes

    def start(self) -> ReconcileSweeper:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="carepilot-reconcile", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.tick()
            except Exception:  # 一轮失败不该带走整个进程，下一轮继续
                log.exception("reconcile 扫描失败，下一轮重试")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
