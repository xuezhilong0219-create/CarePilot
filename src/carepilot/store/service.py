"""SQLite 工单存储。每个操作独立连接，配合 WAL + busy_timeout 支持多线程。
写状态用 `UPDATE ... WHERE version=?` 做乐观锁，认领用 `WHERE owner_id IS NULL` 保证原子。"""

import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from ..auth import Principal, Role
from .models import (
    Conflict,
    Forbidden,
    IllegalTransition,
    Message,
    NotFound,
    SlaPolicy,
    Ticket,
    TicketEvent,
)
from .rules import STATUSES, can_modify, can_transition, can_view, is_staff

KINDS = ("message", "draft", "note", "system")
VISIBILITIES = ("public", "internal")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    display_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_shops (
    user_id TEXT NOT NULL,
    shop_id TEXT NOT NULL,
    PRIMARY KEY (user_id, shop_id)
);
CREATE TABLE IF NOT EXISTS ticket_alloc (
    seq INTEGER PRIMARY KEY AUTOINCREMENT
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    shop_id TEXT,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    owner_id TEXT,
    thread_id TEXT NOT NULL,
    category TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sla_due_at REAL NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ticket_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_role TEXT NOT NULL,
    kind TEXT NOT NULL,
    visibility TEXT NOT NULL,
    body TEXT NOT NULL,
    idem_key TEXT UNIQUE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    ts REAL NOT NULL,
    actor_id TEXT NOT NULL,
    event TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS ticket_context (
    ticket_id TEXT PRIMARY KEY,
    slots_json TEXT NOT NULL DEFAULT '{}',
    last_thread_id TEXT,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_messages_ticket ON ticket_messages(ticket_id);
CREATE INDEX IF NOT EXISTS idx_events_ticket ON ticket_events(ticket_id);
"""


def _ticket(row: sqlite3.Row) -> Ticket:
    return Ticket(
        ticket_id=row["ticket_id"],
        customer_id=row["customer_id"],
        shop_id=row["shop_id"],
        subject=row["subject"],
        status=row["status"],
        priority=row["priority"],
        owner_id=row["owner_id"],
        thread_id=row["thread_id"],
        category=row["category"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sla_due_at=row["sla_due_at"],
        version=row["version"],
    )


def _message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        ticket_id=row["ticket_id"],
        author_id=row["author_id"],
        author_role=row["author_role"],
        kind=row["kind"],
        visibility=row["visibility"],
        body=row["body"],
        idem_key=row["idem_key"],
        created_at=row["created_at"],
    )


def _event(row: sqlite3.Row) -> TicketEvent:
    return TicketEvent(
        id=row["id"],
        ticket_id=row["ticket_id"],
        ts=row["ts"],
        actor_id=row["actor_id"],
        event=row["event"],
        from_status=row["from_status"],
        to_status=row["to_status"],
        detail=row["detail"],
    )


class TicketStore:
    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        sla_policy: SlaPolicy | None = None,
    ):
        self._db_path = Path(db_path)
        self._clock = clock
        self.policy = sla_policy or SlaPolicy()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> float:
        return float(self._clock())

    def _fetch_ticket(self, conn: sqlite3.Connection, ticket_id: str) -> Ticket:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if row is None:
            raise NotFound(ticket_id)
        return _ticket(row)

    def upsert_user(
        self,
        user_id: str,
        role: Role,
        display_name: str,
        shop_ids: Sequence[str] = (),
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users(user_id, role, display_name) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET role=excluded.role, display_name=excluded.display_name",
                (user_id, role.value, display_name),
            )
            for shop_id in shop_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO user_shops(user_id, shop_id) VALUES(?,?)",
                    (user_id, shop_id),
                )

    def create_ticket(
        self,
        customer: Principal,
        *,
        subject: str,
        shop_id: str | None,
        thread_id: str,
        priority: str = "medium",
        category: str | None = None,
    ) -> Ticket:
        if customer.role != Role.CUSTOMER:
            raise Forbidden("只有客户能创建工单")
        if customer.shop_ids and shop_id is not None and shop_id not in customer.shop_ids:
            raise Forbidden(f"客户无权操作店铺 {shop_id}")
        due_seconds = self.policy.seconds_for(priority)
        now = self._now()
        with self._connect() as conn:
            seq = conn.execute("INSERT INTO ticket_alloc DEFAULT VALUES").lastrowid
            ticket_id = f"T-{seq:06d}"
            conn.execute(
                "INSERT INTO tickets(ticket_id, customer_id, shop_id, subject, status, priority, owner_id, "
                "thread_id, category, created_at, updated_at, sla_due_at, version) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (
                    ticket_id,
                    customer.user_id,
                    shop_id,
                    subject,
                    "new",
                    priority,
                    None,
                    thread_id,
                    category,
                    now,
                    now,
                    now + due_seconds,
                ),
            )
            conn.execute(
                "INSERT INTO ticket_events(ticket_id, ts, actor_id, event, from_status, to_status, detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (ticket_id, now, customer.user_id, "created", None, "new", ""),
            )
            return self._fetch_ticket(conn, ticket_id)

    def get_ticket(self, who: Principal, ticket_id: str) -> Ticket:
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
        if not can_view(who, ticket):
            raise Forbidden(f"无权查看工单 {ticket_id}")
        return ticket

    def list_tickets(
        self,
        who: Principal,
        *,
        status: str | None = None,
        mine: bool = False,
    ) -> list[Ticket]:
        clauses: list[str] = []
        params: list[object] = []
        if who.role == Role.CUSTOMER:
            clauses.append("customer_id=?")
            params.append(who.user_id)
        if mine:
            clauses.append("owner_id=?")
            params.append(who.user_id)
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"未知状态: {status!r}")
            clauses.append("status=?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tickets{where} ORDER BY created_at, ticket_id", params
            ).fetchall()
        return [_ticket(row) for row in rows]

    def add_message(
        self,
        ticket_id: str,
        *,
        author: Principal,
        kind: str,
        body: str,
        visibility: str = "public",
        idem_key: str | None = None,
    ) -> Message:
        if kind not in KINDS:
            raise ValueError(f"未知消息类型: {kind!r}")
        if visibility not in VISIBILITIES:
            raise ValueError(f"未知可见性: {visibility!r}")
        if kind == "draft" and visibility != "internal":
            raise ValueError("草稿（draft）必须只对内部可见")
        now = self._now()
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if not can_view(author, ticket):
                raise Forbidden(f"无权在工单 {ticket_id} 上留言")
            if not is_staff(author) and (kind != "message" or visibility != "public"):
                raise Forbidden("客户只能发公开消息")
            # 客户可见的消息等于"替这张工单对外说话"，属于修改：坐席必须是负责人（或管理员）。
            # 内部备注/草稿是同事间协作，任何能看到工单的坐席都可以写。
            if is_staff(author) and visibility == "public" and not can_modify(author, ticket):
                raise Forbidden(f"无权以坐席身份向客户发布 {ticket_id} 的公开消息（需先认领，或由负责人发布）")
            if idem_key is not None:
                existing = conn.execute(
                    "SELECT * FROM ticket_messages WHERE idem_key=?", (idem_key,)
                ).fetchone()
                if existing is not None:
                    return _message(existing)
            try:
                cur = conn.execute(
                    "INSERT INTO ticket_messages(ticket_id, author_id, author_role, kind, visibility, body, "
                    "idem_key, created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (ticket_id, author.user_id, author.role.value, kind, visibility, body, idem_key, now),
                )
            except sqlite3.IntegrityError:
                # 并发重试：幂等键唯一约束兜底，返回先写入的那条。
                existing = conn.execute(
                    "SELECT * FROM ticket_messages WHERE idem_key=?", (idem_key,)
                ).fetchone()
                if existing is None:
                    raise
                return _message(existing)
            conn.execute("UPDATE tickets SET updated_at=? WHERE ticket_id=?", (now, ticket_id))
            row = conn.execute("SELECT * FROM ticket_messages WHERE id=?", (cur.lastrowid,)).fetchone()
            return _message(row)

    def list_messages(self, who: Principal, ticket_id: str) -> list[Message]:
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if not can_view(who, ticket):
                raise Forbidden(f"无权查看工单 {ticket_id}")
            sql = "SELECT * FROM ticket_messages WHERE ticket_id=?"
            params: list[object] = [ticket_id]
            if who.role == Role.CUSTOMER:
                sql += " AND visibility='public'"
            sql += " ORDER BY id"
            rows = conn.execute(sql, params).fetchall()
        return [_message(row) for row in rows]

    def transition(self, who: Principal, ticket_id: str, to_status: str, *, reason: str = "") -> Ticket:
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if not can_modify(who, ticket):
                raise Forbidden(f"无权修改工单 {ticket_id}")
            if not can_transition(ticket.status, to_status):
                raise IllegalTransition(f"{ticket.status} -> {to_status}")
            now = self._now()
            cur = conn.execute(
                "UPDATE tickets SET status=?, updated_at=?, version=version+1 "
                "WHERE ticket_id=? AND version=?",
                (to_status, now, ticket_id, ticket.version),
            )
            if cur.rowcount == 0:
                raise Conflict(f"工单 {ticket_id} 已被并发修改")
            conn.execute(
                "INSERT INTO ticket_events(ticket_id, ts, actor_id, event, from_status, to_status, detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (ticket_id, now, who.user_id, "transition", ticket.status, to_status, reason),
            )
            return self._fetch_ticket(conn, ticket_id)

    def claim(self, who: Principal, ticket_id: str) -> Ticket:
        if not is_staff(who):
            raise Forbidden("只有坐席或管理员能认领工单")
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if ticket.owner_id == who.user_id:
                return ticket
            if ticket.owner_id is not None:
                raise Conflict(f"工单 {ticket_id} 已被 {ticket.owner_id} 认领")
            now = self._now()
            cur = conn.execute(
                "UPDATE tickets SET owner_id=?, updated_at=?, version=version+1 "
                "WHERE ticket_id=? AND owner_id IS NULL AND version=?",
                (who.user_id, now, ticket_id, ticket.version),
            )
            if cur.rowcount == 0:
                raise Conflict(f"工单 {ticket_id} 已被其他坐席认领")
            conn.execute(
                "INSERT INTO ticket_events(ticket_id, ts, actor_id, event, from_status, to_status, detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (ticket_id, now, who.user_id, "claimed", ticket.status, ticket.status, ""),
            )
            return self._fetch_ticket(conn, ticket_id)

    def release(self, who: Principal, ticket_id: str) -> Ticket:
        """释放工单：清空处理人；能和 open 迁移时顺带回 open，方便下一个坐席接续。"""
        if not is_staff(who):
            raise Forbidden("只有坐席或管理员能释放工单")
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if who.role != Role.ADMIN and ticket.owner_id != who.user_id:
                raise Forbidden(f"无权释放工单 {ticket_id}")
            to_status = "open" if can_transition(ticket.status, "open") else ticket.status
            now = self._now()
            cur = conn.execute(
                "UPDATE tickets SET owner_id=NULL, status=?, updated_at=?, version=version+1 "
                "WHERE ticket_id=? AND version=?",
                (to_status, now, ticket_id, ticket.version),
            )
            if cur.rowcount == 0:
                raise Conflict(f"工单 {ticket_id} 已被并发修改")
            conn.execute(
                "INSERT INTO ticket_events(ticket_id, ts, actor_id, event, from_status, to_status, detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (ticket_id, now, who.user_id, "released", ticket.status, to_status, ""),
            )
            return self._fetch_ticket(conn, ticket_id)

    def set_priority(self, who: Principal, ticket_id: str, priority: str) -> Ticket:
        seconds = self.policy.seconds_for(priority)
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if not can_modify(who, ticket):
                raise Forbidden(f"无权修改工单 {ticket_id}")
            now = self._now()
            cur = conn.execute(
                "UPDATE tickets SET priority=?, sla_due_at=?, updated_at=?, version=version+1 "
                "WHERE ticket_id=? AND version=?",
                (priority, ticket.created_at + seconds, now, ticket_id, ticket.version),
            )
            if cur.rowcount == 0:
                raise Conflict(f"工单 {ticket_id} 已被并发修改")
            conn.execute(
                "INSERT INTO ticket_events(ticket_id, ts, actor_id, event, from_status, to_status, detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (ticket_id, now, who.user_id, "priority_changed", ticket.status, ticket.status,
                 f"{ticket.priority}->{priority}"),
            )
            return self._fetch_ticket(conn, ticket_id)

    def get_principal(self, user_id: str) -> Principal | None:
        """按用户 id 还原调用者身份：角色和店铺范围来自用户表，**不信任请求里自报的角色**。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
            if row is None:
                return None
            shops = [r["shop_id"] for r in conn.execute("SELECT shop_id FROM user_shops WHERE user_id=?", (user_id,))]
        return Principal(user_id, Role(row["role"]), frozenset(shops))

    def find_message_by_idem_key(self, idem_key: str) -> Message | None:
        """按幂等键找消息，供网关崩溃恢复时核实"这条消息到底有没有发出去"。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ticket_messages WHERE idem_key=?", (idem_key,)).fetchone()
        return _message(row) if row else None

    def get_context(self, ticket_id: str) -> dict:
        """工单级会话上下文：跨轮累积的槽位（平台/店铺/功能/仓库…）和最近一次运行的 thread_id。"""
        with self._connect() as conn:
            self._fetch_ticket(conn, ticket_id)
            row = conn.execute("SELECT * FROM ticket_context WHERE ticket_id=?", (ticket_id,)).fetchone()
        if row is None:
            return {"slots": {}, "last_thread_id": None}
        return {"slots": json.loads(row["slots_json"]), "last_thread_id": row["last_thread_id"]}

    def put_context(self, ticket_id: str, *, slots: dict, last_thread_id: str | None) -> None:
        with self._connect() as conn:
            self._fetch_ticket(conn, ticket_id)
            conn.execute(
                "INSERT INTO ticket_context(ticket_id, slots_json, last_thread_id, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET slots_json=excluded.slots_json, "
                "last_thread_id=excluded.last_thread_id, updated_at=excluded.updated_at",
                (ticket_id, json.dumps(slots, ensure_ascii=False, sort_keys=True), last_thread_id, self._now()),
            )

    def set_category(self, who: Principal, ticket_id: str, category: str) -> Ticket:
        """分诊结果回写。只有坐席/管理员（含服务端的 system 主体）可以写。"""
        if not is_staff(who):
            raise Forbidden("只有坐席或管理员能修改工单分类")
        with self._connect() as conn:
            self._fetch_ticket(conn, ticket_id)
            conn.execute("UPDATE tickets SET category=?, updated_at=? WHERE ticket_id=?", (category, self._now(), ticket_id))
            return self._fetch_ticket(conn, ticket_id)

    def sla_remaining_seconds(self, ticket: Ticket, now: float | None = None) -> float:
        if ticket.status in ("resolved", "closed"):
            return float("inf")
        current = self._now() if now is None else now
        return ticket.sla_due_at - current

    def events(self, who: Principal, ticket_id: str) -> list[TicketEvent]:
        with self._connect() as conn:
            ticket = self._fetch_ticket(conn, ticket_id)
            if not can_view(who, ticket):
                raise Forbidden(f"无权查看工单 {ticket_id}")
            rows = conn.execute(
                "SELECT * FROM ticket_events WHERE ticket_id=? ORDER BY id", (ticket_id,)
            ).fetchall()
        return [_event(row) for row in rows]


def seed_demo_users(store: TicketStore) -> None:
    store.upsert_user("c1", Role.CUSTOMER, "客户一", ("S1", "S2"))
    store.upsert_user("c2", Role.CUSTOMER, "客户二", ("S3",))
    store.upsert_user("c3", Role.CUSTOMER, "客户三", ("S4",))
    store.upsert_user("a1", Role.AGENT, "坐席一")
    store.upsert_user("a2", Role.AGENT, "坐席二")
    store.upsert_user("admin1", Role.ADMIN, "管理员")
