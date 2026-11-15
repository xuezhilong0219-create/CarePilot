"""在飞租约：执行期间对租约文件加一把操作系统锁，对账据此跳过正在跑的调用。

为什么不用账本时间戳：工具执行期间账本一行都不写，`updated_at` 停在占位那一刻，
分不清「刚跑 3 秒」和「崩了 3 分钟」。

为什么不用 pid 判活：Windows 上 `os.kill(pid, 0)` 不是探测而是 TerminateProcess；
`OpenProcess` 也不可靠——实测进程刚被 `Popen.kill()` 杀掉后它仍返回一个句柄；pid 还会复用。
而文件锁由操作系统在进程消失时释放，正好等于我们要的语义：**锁还在 = 真的在跑，
锁没了 = 上次那个执行者已经不在了**。

这层是给 `reconcile()` 用的：它原来靠「对账和工具执行持同一把锁」避开在飞调用，
锁收窄到只夹记账之后（见 DESIGN.md），就得有别的办法知道自己不该碰谁。
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import time


def _lock_nonblock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LeaseSet:
    """`<db>.leases/<idem_key>`：持锁 = 这条调用正在执行。"""

    def __init__(self, db_path: str | Path):
        path = Path(db_path).resolve()
        self.dir = path.with_name(path.name + ".leases")
        self._held: dict[str, tuple[object, Path]] = {}

    def _path(self, key: str) -> Path:
        return self.dir / key

    def acquire(self, key: str, *, timeout_s: float = 5.0) -> None:
        """占住这个 key 的租约，返回时保证锁在手上。

        走到这里意味着账本里那行已经归我们了（刚插入，或刚把 abandoned 抢回来），
        所以正常情况下一试就成功；等不到只能说明有人跳过了账本直接占租约，那是 bug，要炸出来。
        """
        if key in self._held:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        handle = path.open("a+b")
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                _lock_nonblock(handle)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    handle.close()
                    raise
                if time.monotonic() > deadline:
                    handle.close()
                    raise RuntimeError(f"lease busy: {key}") from exc
                time.sleep(0.005)
        self._held[key] = (handle, path)

    def release(self, key: str) -> None:
        item = self._held.pop(key, None)
        if item is None:
            return
        handle, path = item
        try:
            _unlock(handle)
        finally:
            handle.close()
        try:
            path.unlink()
        except OSError:
            pass  # 还有别人开着这个文件（正在探活），留着也无害：没锁就是没锁

    def live(self, key: str) -> bool:
        """现在有没有人握着这个 key 的租约。只探，不改。"""
        path = self._path(key)
        try:
            handle = path.open("r+b")
        except (FileNotFoundError, OSError):
            return False
        try:
            if path.stat().st_size == 0:
                return False
            try:
                _lock_nonblock(handle)
            except OSError:
                return True
            _unlock(handle)
            return False
        finally:
            handle.close()

    def live_keys(self) -> set[str]:
        if not self.dir.is_dir():
            return set()
        return {p.name for p in self.dir.iterdir() if self.live(p.name)}

    def clear(self) -> None:
        for key in list(self._held):
            self.release(key)
        if self.dir.is_dir():
            for path in self.dir.iterdir():
                try:
                    path.unlink()
                except OSError:
                    pass
