from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha1
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def idempotency_key(run_id: str, tool: str, args: dict[str, Any]) -> str:
    return sha1(f"{run_id}\0{tool}\0{canonical_json(args)}".encode("utf-8")).hexdigest()


class Ledger:
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        with self._locks_guard:
            self._lock = self._locks.setdefault(os.path.normcase(str(self.path)), threading.Lock())

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        # 跨进程互斥，只用来夹住账本的短事务（占位、记账、审批状态）。工具本身在锁外执行：
        # 「恰好执行一次」由 action_ledger 里那行 started 保证，不是靠这把锁一直握着。
        # 进程被强杀时由操作系统释放。
        with self._lock, self.lock_path.open("a+b") as handle:
            if self.lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import errno
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def create_schema(db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS action_ledger (
            idem_key TEXT PRIMARY KEY, run_id TEXT NOT NULL, tool TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('started','done','abandoned')),
            result_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS action_inputs (
            idem_key TEXT PRIMARY KEY, principal_json TEXT NOT NULL, ticket_id TEXT,
            args_json TEXT NOT NULL, approver_id TEXT, approval_id TEXT, recovery_status TEXT)""")
        db.execute("CREATE INDEX IF NOT EXISTS ledger_status ON action_ledger(status)")
