import re
import sqlite3
from typing import Any

from .ledger import Ledger, canonical_json


def safe_summary(value: Any) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    # 兜底覆盖结果和异常里的大陆手机号；其他敏感字段由工具摘要函数负责。
    return re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)[:200]


class AuditLog:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @staticmethod
    def create_schema(db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
            actor_id TEXT NOT NULL, actor_role TEXT NOT NULL, run_id TEXT NOT NULL, ticket_id TEXT,
            tool TEXT NOT NULL, args_summary TEXT NOT NULL, decision TEXT NOT NULL,
            gate TEXT, code TEXT, result_summary TEXT NOT NULL, approver_id TEXT, idem_key TEXT)""")
        for field in ("run_id", "ticket_id", "actor_id"):
            db.execute(f"CREATE INDEX IF NOT EXISTS audit_{field} ON audit_log({field}, ts, id)")

    @staticmethod
    def append(db: sqlite3.Connection, event: dict[str, Any]) -> None:
        columns = ("ts", "actor_id", "actor_role", "run_id", "ticket_id", "tool",
                   "args_summary", "decision", "gate", "code", "result_summary", "approver_id", "idem_key")
        db.execute(
            f"INSERT INTO audit_log ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(event.get(field) for field in columns),
        )

    def query(self, *, run_id: str | None = None, ticket_id: str | None = None,
              actor_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        filters = {key: value for key, value in {"run_id": run_id, "ticket_id": ticket_id, "actor_id": actor_id}.items() if value is not None}
        where = " AND ".join(f"{field}=?" for field in filters) or "1"
        with self.ledger.connect() as db:
            return [dict(row) for row in db.execute(
                f"SELECT * FROM audit_log WHERE {where} ORDER BY ts, id LIMIT ?", (*filters.values(), limit)
            )]
