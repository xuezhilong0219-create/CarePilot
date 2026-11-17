from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from ..auth import Principal, Role
from .audit import AuditLog, safe_summary
from .lease import LeaseSet
from .ledger import Ledger, canonical_json, idempotency_key
from .spec import CallOutcome, Executed, Gate, NeedsApproval, Rejected, Risk, RuleViolation, ToolContext, ToolSpec


def _crash_point(name: str) -> None:
    if os.environ.get("CAREPILOT_CRASH_AT") == name:
        marker = Path(os.environ["CAREPILOT_CRASH_MARKER"])
        marker.write_text(name, encoding="utf-8")
        time.sleep(60)


def _principal_json(principal: Principal) -> str:
    return canonical_json({"user_id": principal.user_id, "role": principal.role.value, "shop_ids": sorted(principal.shop_ids)})


def _principal(raw: str) -> Principal:
    value = json.loads(raw)
    return Principal(value["user_id"], Role(value["role"]), frozenset(value["shop_ids"]))


class ToolGateway:
    def __init__(
        self,
        db_path: str | Path,
        registry: Mapping[str, ToolSpec],
        *,
        clock: Callable[[], float] = time.time,
        approval_ttl_s: float | None = 86400.0,
        inflight_wait_s: float = 10.0,
    ):
        self.registry = dict(registry)
        self.clock = clock
        # 待批准、以及"已批准待二次确认"的审批超过这个时长就作废（从上一个阶段的时间起算）；None 表示永不过期
        self.approval_ttl_s = approval_ttl_s
        # 同一条调用已经有别的执行者在跑时，重复调用最多等这么久再报未决；等到了就回放，绝不会再执行一次
        self.inflight_wait_s = inflight_wait_s
        self.ledger = Ledger(db_path)
        # 在飞租约：让 reconcile 知道哪条调用正在执行（工具在锁外跑，账本时间戳分不清"在跑"和"崩了"）
        self.leases = LeaseSet(db_path)
        self._audit = AuditLog(self.ledger)
        with self.ledger.exclusive(), self.ledger.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            Ledger.create_schema(db)
            AuditLog.create_schema(db)
            db.execute("""CREATE TABLE IF NOT EXISTS approvals (
                approval_id TEXT PRIMARY KEY, idem_key TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL,
                ticket_id TEXT, tool TEXT NOT NULL, args_json TEXT NOT NULL, principal_json TEXT NOT NULL,
                risk INTEGER NOT NULL, status TEXT NOT NULL, approver_id TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL)""")

    @staticmethod
    def _summary(spec: ToolSpec | None, args: BaseModel | None) -> str:
        if args is None:
            return "[invalid arguments]"
        try:
            value = spec.summarize_args(args) if spec and spec.summarize_args else args.model_dump(mode="json")
            return safe_summary(value)
        except Exception:
            return "[summary unavailable]"

    def _event(self, db: sqlite3.Connection, ctx: ToolContext, tool: str, summary: str, decision: str,
               *, gate: Gate = Gate.RISK, code: str = "", result: Any = "", approver_id: str | None = None) -> None:
        self._audit.append(db, {
            "ts": self.clock(), "actor_id": ctx.principal.user_id, "actor_role": ctx.principal.role.value,
            "run_id": ctx.run_id, "ticket_id": ctx.ticket_id, "tool": tool, "args_summary": summary,
            "decision": decision, "gate": gate.value, "code": code, "result_summary": safe_summary(result),
            "approver_id": approver_id, "idem_key": ctx.idem_key,
        })

    def _reject(self, ctx: ToolContext, tool: str, summary: str, gate: Gate, code: str, message: str = "",
                *, approver_id: str | None = None) -> Rejected:
        message = safe_summary(message or code)
        with self.ledger.transaction() as db:
            self._event(db, ctx, tool, summary, "rejected", gate=gate, code=code, result=message, approver_id=approver_id)
        return Rejected(gate, code, message)

    @staticmethod
    def _permission(ctx: ToolContext, spec: ToolSpec, args: BaseModel) -> str | None:
        if ctx.principal.role not in spec.roles:
            return "role_denied"
        if spec.owner_check is not None:
            try:
                if not spec.owner_check(ctx, args):
                    return "not_owner"
            except Exception:
                return "not_owner"
        return None

    def call(self, ctx: ToolContext, tool: str, raw_args: dict[str, Any]) -> CallOutcome:
        ctx = replace(ctx, idem_key=None)
        spec = self.registry.get(tool)
        if spec is None:
            return self._reject(ctx, tool, self._summary(None, None), Gate.SCHEMA, "unknown_tool")
        try:
            args = spec.args_model.model_validate(raw_args, strict=True, extra="forbid")
            params = args.model_dump(mode="json")
            key = idempotency_key(ctx.run_id, tool, params)
        except (ValidationError, ValueError, TypeError):
            return self._reject(ctx, tool, self._summary(spec, None), Gate.SCHEMA, "invalid_args")
        ctx = replace(ctx, idem_key=key)
        summary = self._summary(spec, args)
        for rule in spec.rules:
            try:
                rule(ctx, args)
            except RuleViolation as exc:
                return self._reject(ctx, tool, summary, Gate.RULE, exc.code, exc.message)
            except Exception:
                return self._reject(ctx, tool, summary, Gate.RULE, "rule_error")
        if code := self._permission(ctx, spec, args):
            return self._reject(ctx, tool, summary, Gate.PERMISSION, code)
        # 只读 / 内部写的工具不在全局锁里执行：锁只夹住「占位」和「记账」两段，
        # 工具本身（可能很慢）在锁外跑。exactly-once 由 action_ledger 的 started 行保证，
        # 不是由锁的宽度保证。审批相关的分支仍然整段持锁（它们不做副作用，只有几行读写）。
        if spec.risk in (Risk.READ, Risk.WRITE_INTERNAL):
            return self._execute(ctx, tool, spec, args)
        if spec.risk not in (Risk.STATE_CHANGE, Risk.EXTERNAL_SEND):
            return self._reject(ctx, tool, summary, Gate.RISK, "unknown_risk")
        with self.ledger.exclusive():
            with self.ledger.connect() as db:
                completed = db.execute("SELECT result_json FROM action_ledger WHERE idem_key=? AND status='done'", (key,)).fetchone()
            if completed is not None:
                result = json.loads(completed["result_json"])
                with self.ledger.transaction() as db:
                    self._event(db, ctx, tool, summary, "replayed", result=result)
                return Executed(result, replayed=True)
            with self.ledger.transaction() as db:
                now = self.clock()
                db.execute("""INSERT OR IGNORE INTO approvals
                    (approval_id,idem_key,run_id,ticket_id,tool,args_json,principal_json,risk,status,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,'pending',?,?)""",
                    (uuid4().hex, key, ctx.run_id, ctx.ticket_id, tool, canonical_json(params), _principal_json(ctx.principal), int(spec.risk), now, now))
                row = db.execute("SELECT * FROM approvals WHERE idem_key=?", (key,)).fetchone()
                if row["status"] == "expired":
                    # 过期的提议在同一个 run 里重新提出时重新打开，否则同样的动作会被永久卡死
                    db.execute("UPDATE approvals SET status='pending', approver_id=NULL, updated_at=? WHERE idem_key=?", (now, key))
                    row = db.execute("SELECT * FROM approvals WHERE idem_key=?", (key,)).fetchone()
                if row["status"] == "rejected":
                    self._event(db, ctx, tool, summary, "rejected", code="approval_rejected")
                    return Rejected(Gate.RISK, "approval_rejected", "approval_rejected")
                self._event(db, ctx, tool, summary, "needs_approval", code="approval_required")
            stage = "confirm" if row["risk"] == Risk.EXTERNAL_SEND and row["status"] in ("approved", "confirmed") else "approve"
            return NeedsApproval(row["approval_id"], tool, {"args_summary": summary}, stage)

    @contextmanager
    def _critical(self, locked: bool) -> Iterator[sqlite3.Connection]:
        """一次短事务。locked=True 表示调用方（审批路径）已经持有全局锁，这里不再重入。"""
        if locked:
            with self.ledger.transaction() as db:
                yield db
        else:
            with self.ledger.exclusive(), self.ledger.transaction() as db:
                yield db

    def _reserve(self, ctx: ToolContext, tool: str, spec: ToolSpec, args: BaseModel, summary: str, *,
                 approver_id: str | None, approval_id: str | None, grant_approval: bool,
                 locked: bool) -> CallOutcome | None:
        """占位：把 (run, 工具, 参数) 标成 started 并记下主体和参数。返回 None 表示「轮到我来执行」。

        同一个 idem_key 已经有执行者在跑时，这里不在锁里空等——那会把那个执行者的记账也堵住——
        而是让出锁去轮询账本：它落成 done 我就回放，等到超时还没落账我就报未决。
        """
        deadline = time.monotonic() + self.inflight_wait_s
        while True:
            waiting = False
            with self._critical(locked) as db:
                now = self.clock()
                inserted = db.execute("""INSERT OR IGNORE INTO action_ledger
                    (idem_key,run_id,tool,status,created_at,updated_at) VALUES (?,?,?,'started',?,?)""",
                    (ctx.idem_key, ctx.run_id, tool, now, now)).rowcount == 1
                row = db.execute("SELECT * FROM action_ledger WHERE idem_key=?", (ctx.idem_key,)).fetchone()
                if row["status"] == "done":
                    if approver_id is None and spec.risk in (Risk.READ, Risk.WRITE_INTERNAL):
                        # 直接执行路径上调用者就是执行者：回放时必须还是同一个主体，否则等于把别人的缓存结果发给了另一个人
                        first = db.execute("SELECT principal_json FROM action_inputs WHERE idem_key=?", (ctx.idem_key,)).fetchone()
                        if first is not None and first["principal_json"] != _principal_json(ctx.principal):
                            self._event(db, ctx, tool, summary, "rejected", code="principal_mismatch")
                            return Rejected(Gate.RISK, "principal_mismatch", "principal_mismatch")
                    result = json.loads(row["result_json"])
                    self._event(db, ctx, tool, summary, "replayed", result=result, approver_id=approver_id)
                    return Executed(result, replayed=True)
                if not inserted and row["status"] == "started":
                    meta = db.execute("SELECT recovery_status FROM action_inputs WHERE idem_key=?", (ctx.idem_key,)).fetchone()
                    if meta and meta["recovery_status"] == "needs_manual":
                        self._event(db, ctx, tool, summary, "rejected", code="needs_manual", approver_id=approver_id)
                        return Rejected(Gate.RISK, "needs_manual", "needs_manual")
                    waiting = True
                if not waiting:
                    if not inserted:
                        # abandoned：上一个执行者把这次调用让了出来，重新捡起来；被别人抢先就回去再看一眼
                        waiting = db.execute(
                            "UPDATE action_ledger SET status='started', updated_at=? WHERE idem_key=? AND status='abandoned'",
                            (now, ctx.idem_key)).rowcount != 1
                    if not waiting:
                        db.execute("""INSERT OR REPLACE INTO action_inputs
                            (idem_key,principal_json,ticket_id,args_json,approver_id,approval_id,recovery_status)
                            VALUES (?,?,?,?,?,?,NULL)""",
                            (ctx.idem_key, _principal_json(ctx.principal), ctx.ticket_id,
                             canonical_json(args.model_dump(mode="json")), approver_id, approval_id))
                        if grant_approval:
                            db.execute("UPDATE approvals SET status='approved', approver_id=?, updated_at=? WHERE approval_id=?",
                                       (approver_id, now, approval_id))
                            self._event(db, ctx, tool, summary, "approved", code="approval_granted", approver_id=approver_id)
                        self._event(db, ctx, tool, summary, "approved", code="execution_intent", approver_id=approver_id)
                        # 占位与租约在同一个临界区里完成：对账要么看到"有行无锁"（崩了，可以接管），
                        # 要么看到"有行有锁"（在跑，别碰），不会看到中间态。
                        self.leases.acquire(ctx.idem_key)
                        return None
            if locked or time.monotonic() > deadline:
                # 拿着锁的人不能等（会挡住执行者落账），到点也只能报未决
                return self._reject(ctx, tool, summary, Gate.RISK, "reconciliation_required", approver_id=approver_id)
            while time.monotonic() < deadline:
                with self.ledger.connect() as db:
                    current = db.execute("SELECT status FROM action_ledger WHERE idem_key=?", (ctx.idem_key,)).fetchone()
                if current is None or current["status"] != "started":
                    break
                time.sleep(0.005)
            else:
                return self._reject(ctx, tool, summary, Gate.RISK, "reconciliation_required", approver_id=approver_id)

    def _execute(self, ctx: ToolContext, tool: str, spec: ToolSpec, args: BaseModel,
                 *, approver_id: str | None = None, approval_id: str | None = None,
                 grant_approval: bool = False, locked: bool = False) -> CallOutcome:
        summary = self._summary(spec, args)
        early = self._reserve(ctx, tool, spec, args, summary, approver_id=approver_id,
                              approval_id=approval_id, grant_approval=grant_approval, locked=locked)
        if early is not None:
            return early
        try:
            return self._apply(ctx, tool, spec, args, summary, approver_id=approver_id,
                               approval_id=approval_id, grant_approval=grant_approval, locked=locked)
        finally:
            self.leases.release(ctx.idem_key)

    def _apply(self, ctx: ToolContext, tool: str, spec: ToolSpec, args: BaseModel, summary: str,
               *, approver_id: str | None, approval_id: str | None,
               grant_approval: bool, locked: bool) -> CallOutcome:
        _crash_point("before_effect")
        exec_code = ""
        try:
            result = spec.fn(ctx, args)
            if not isinstance(result, dict):
                raise TypeError("tool result must be a dict")
            result_json = canonical_json(result)
        except Exception as exc:
            # 工具抛异常 ≠ 副作用没发生（请求发出后连接被重置就是典型）。直接标"可重试"会重复执行，所以先核实。
            adopted, unresolved_manual = self._check_effect_after_error(ctx, spec, args)
            if adopted is not None:
                result, result_json, exec_code = adopted, canonical_json(adopted), "adopted_after_error"
            elif adopted is None and unresolved_manual is not None:
                # 无法确认副作用有没有发生：保持 started，不自动重试。无 verify → 需人工；verify 自己出错 → 留给 reconcile 再核实
                with self._critical(locked) as db:
                    if unresolved_manual:
                        db.execute("UPDATE action_inputs SET recovery_status='needs_manual' WHERE idem_key=?", (ctx.idem_key,))
                    self._event(db, ctx, tool, summary, "rejected", code="tool_error_unverified", result=type(exc).__name__, approver_id=approver_id)
                return Rejected(Gate.RISK, "tool_error_unverified", type(exc).__name__)
            else:
                with self._critical(locked) as db:
                    db.execute("UPDATE action_ledger SET status='abandoned', updated_at=? WHERE idem_key=?", (self.clock(), ctx.idem_key))
                    if grant_approval:
                        db.execute("UPDATE approvals SET status='pending', updated_at=? WHERE approval_id=?", (self.clock(), approval_id))
                    self._event(db, ctx, tool, summary, "rejected", code="tool_error", result=type(exc).__name__, approver_id=approver_id)
                return Rejected(Gate.RISK, "tool_error", type(exc).__name__)
        _crash_point("after_effect")
        with self._critical(locked) as db:
            db.execute("UPDATE action_ledger SET status='done', result_json=?, updated_at=? WHERE idem_key=?", (result_json, self.clock(), ctx.idem_key))
            if approval_id:
                status = "confirmed" if spec.risk == Risk.EXTERNAL_SEND else "executed"
                db.execute("UPDATE approvals SET status=?, updated_at=? WHERE approval_id=?", (status, self.clock(), approval_id))
            self._event(db, ctx, tool, summary, "executed", code=exec_code, result=result, approver_id=approver_id)
        return Executed(result)

    @staticmethod
    def _check_effect_after_error(ctx: ToolContext, spec: ToolSpec, args: BaseModel) -> tuple[dict | None, bool | None]:
        """工具抛异常后核实副作用。返回 (已发生时的结果, 是否需要保持未决)：
        (dict, None)   verify 确认副作用已发生 → 认领结果，不重试；
        (None, None)   确认没发生，或这是没有 verify 的只读/内部写工具 → 允许重试；
        (None, True)   有副作用但没有 verify → 无法确认，需人工；
        (None, False)  verify 自己出错 → 无法确认，留给 reconcile。"""
        if spec.verify is None:
            return (None, True) if spec.risk >= Risk.STATE_CHANGE else (None, None)
        try:
            found = spec.verify(ctx, args)
            if found is not None:
                if not isinstance(found, dict):
                    raise TypeError("verify result must be a dict")
                canonical_json(found)
            return found, None
        except Exception:
            return None, False

    def _expired(self, row: sqlite3.Row) -> bool:
        return self.approval_ttl_s is not None and self.clock() - row["updated_at"] > self.approval_ttl_s

    def _expire(self, ctx: ToolContext, row: sqlite3.Row, approver: Principal) -> Rejected:
        with self.ledger.transaction() as db:
            db.execute("UPDATE approvals SET status='expired', updated_at=? WHERE approval_id=?", (self.clock(), row["approval_id"]))
            self._event(db, ctx, row["tool"], "[approval]", "rejected", gate=Gate.RISK, code="approval_expired",
                        result="approval expired", approver_id=approver.user_id)
        return Rejected(Gate.RISK, "approval_expired", "approval expired")

    @staticmethod
    def _approval_context(row: sqlite3.Row, principal: Principal | None = None) -> ToolContext:
        return ToolContext(principal or _principal(row["principal_json"]), row["run_id"], row["ticket_id"], row["idem_key"])

    def _load_approval(self, approval_id: str, approver: Principal) -> sqlite3.Row | Rejected:
        with self.ledger.connect() as db:
            row = db.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        ctx = self._approval_context(row, approver) if row else ToolContext(approver, "")
        tool = row["tool"] if row else ""
        if not approver.is_staff:
            return self._reject(ctx, tool, "[approval]", Gate.PERMISSION, "approver_role_denied", approver_id=approver.user_id)
        if row is None:
            return self._reject(ctx, tool, "[approval]", Gate.RISK, "unknown_approval", approver_id=approver.user_id)
        return row

    def _approved_inputs(self, row: sqlite3.Row, approver: Principal) -> tuple[ToolSpec, BaseModel] | Rejected:
        ctx = self._approval_context(row, approver)
        spec = self.registry.get(row["tool"])
        if spec is None or spec.risk != row["risk"]:
            return self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "tool_changed", approver_id=approver.user_id)
        try:
            args = spec.args_model.model_validate(json.loads(row["args_json"]), extra="forbid")
        except (ValidationError, ValueError, TypeError):
            return self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "tool_changed", approver_id=approver.user_id)
        if code := self._permission(ctx, spec, args):
            return self._reject(ctx, row["tool"], self._summary(spec, args), Gate.PERMISSION, code, approver_id=approver.user_id)
        return spec, args

    def approve(self, approval_id: str, approver: Principal) -> CallOutcome:
        with self.ledger.exclusive():
            row = self._load_approval(approval_id, approver)
            if isinstance(row, Rejected):
                return row
            ctx = self._approval_context(row, approver)
            if row["status"] != "pending":
                return self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "approval_not_pending", approver_id=approver.user_id)
            if self._expired(row):
                return self._expire(ctx, row, approver)
            inputs = self._approved_inputs(row, approver)
            if isinstance(inputs, Rejected):
                return inputs
            spec, args = inputs
            if row["risk"] == Risk.EXTERNAL_SEND:
                with self.ledger.transaction() as db:
                    db.execute("UPDATE approvals SET status='approved', approver_id=?, updated_at=? WHERE approval_id=?", (approver.user_id, self.clock(), approval_id))
                    self._event(db, ctx, row["tool"], self._summary(spec, args), "approved", code="approval_granted", approver_id=approver.user_id)
                return NeedsApproval(approval_id, row["tool"], {"args_summary": self._summary(spec, args)}, "confirm")
            return self._execute(ctx, row["tool"], spec, args, approver_id=approver.user_id, approval_id=approval_id,
                                 grant_approval=True, locked=True)

    def confirm(self, approval_id: str, approver: Principal) -> CallOutcome:
        with self.ledger.exclusive():
            row = self._load_approval(approval_id, approver)
            if isinstance(row, Rejected):
                return row
            ctx = self._approval_context(row, approver)
            if row["risk"] != Risk.EXTERNAL_SEND:
                return self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "confirmation_not_required", approver_id=approver.user_id)
            if row["status"] not in ("approved", "confirmed"):
                return self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "approval_not_approved", approver_id=approver.user_id)
            if row["status"] == "approved" and self._expired(row):  # 已确认的是回放，不受过期影响
                return self._expire(ctx, row, approver)
            inputs = self._approved_inputs(row, approver)
            if isinstance(inputs, Rejected):
                return inputs
            spec, args = inputs
            with self.ledger.transaction() as db:
                self._event(db, ctx, row["tool"], self._summary(spec, args), "confirmed", code="confirmation_granted", approver_id=row["approver_id"])
            return self._execute(ctx, row["tool"], spec, args, approver_id=row["approver_id"], approval_id=approval_id,
                                 locked=True)

    def reject(self, approval_id: str, approver: Principal, reason: str) -> None:
        with self.ledger.exclusive():
            row = self._load_approval(approval_id, approver)
            if isinstance(row, Rejected):
                return
            ctx = self._approval_context(row, approver)
            if row["status"] != "pending":
                self._reject(ctx, row["tool"], "[approval]", Gate.RISK, "approval_not_pending", approver_id=approver.user_id)
                return
            if isinstance(self._approved_inputs(row, approver), Rejected):
                return
            with self.ledger.transaction() as db:
                db.execute("UPDATE approvals SET status='rejected', updated_at=? WHERE approval_id=?", (self.clock(), approval_id))
                self._event(db, ctx, row["tool"], "[approval]", "rejected", code="approval_rejected", result=reason, approver_id=approver.user_id)

    def inflight(self) -> list[dict[str, Any]]:
        """还没结束的调用（started），附带「有没有活租约」。只读，不写库、不写审计。"""
        now = self.clock()
        with self.ledger.connect() as db:
            rows = db.execute("""SELECT l.idem_key, l.tool, l.run_id, l.created_at, i.ticket_id
                FROM action_ledger l LEFT JOIN action_inputs i USING(idem_key)
                WHERE l.status='started' ORDER BY l.created_at, l.idem_key""").fetchall()
        return [{"idem_key": row["idem_key"], "tool": row["tool"], "run_id": row["run_id"],
                 "ticket_id": row["ticket_id"], "age_s": round(now - float(row["created_at"]), 3),
                 "live": self.leases.live(row["idem_key"])} for row in rows]

    def reconcile(self, *, min_age_s: float = 0.0, dry_run: bool = False) -> list[dict[str, Any]]:
        """扫描 started 并逐条对账。

        - `min_age_s`：只处理占位早于这么多秒的行（默认 0，立刻处理）。
        - `dry_run`：只列候选，不调 verify、不写库、不写审计——认领副作用不可回退，值得先看一眼。
        - 有活租约的行一律跳过：它在跑，不是在崩。工具已挪到锁外执行，这里不能再靠锁判断。
        """
        now = self.clock()
        outcomes: list[dict[str, Any]] = []
        with self.ledger.exclusive():
            with self.ledger.connect() as db:
                rows = db.execute("""SELECT l.*, i.principal_json, i.ticket_id, i.args_json, i.approver_id, i.approval_id
                    FROM action_ledger l JOIN action_inputs i USING(idem_key) WHERE l.status='started' ORDER BY l.created_at, l.idem_key""").fetchall()
            for row in rows:
                age = now - float(row["created_at"])
                skip = None
                if self.leases.live(row["idem_key"]):
                    skip = ("skipped_inflight", "inflight")
                elif age < min_age_s:
                    skip = ("skipped_recent", "too_recent")
                elif dry_run:
                    skip = ("would_reconcile", "dry_run")
                if skip is not None:
                    outcomes.append({"idem_key": row["idem_key"], "tool": row["tool"], "status": skip[0],
                                     "code": skip[1], "age_s": round(age, 3)})
                    continue
                ctx = self._approval_context(row)
                spec = self.registry.get(row["tool"])
                result = None
                status, code = "needs_manual", "reconciled_needs_manual"
                summary = "[recovery]"
                if spec is not None and spec.verify is not None:
                    try:
                        args = spec.args_model.model_validate(json.loads(row["args_json"]), extra="forbid")
                        summary = self._summary(spec, args)
                    except Exception:
                        code = "reconciliation_error"
                    else:
                        # 审计失败不能被当成 verify 失败吞掉。
                        with self.ledger.transaction() as db:
                            self._event(db, ctx, row["tool"], summary, "approved", code="reconciliation_intent", approver_id=row["approver_id"])
                        try:
                            result = spec.verify(ctx, args)
                            if result is not None:
                                if not isinstance(result, dict):
                                    raise TypeError("verify result must be a dict")
                                canonical_json(result)
                                status, code = "done", "reconciled_adopted"
                            else:
                                status, code = "abandoned", "reconciled_retry_allowed"
                        except Exception:
                            result = None
                            code = "reconciliation_error"
                with self.ledger.transaction() as db:
                    if status == "needs_manual":
                        db.execute("UPDATE action_inputs SET recovery_status='needs_manual' WHERE idem_key=?", (ctx.idem_key,))
                    else:
                        db.execute("UPDATE action_ledger SET status=?, result_json=?, updated_at=? WHERE idem_key=?",
                                   (status, canonical_json(result) if result is not None else None, self.clock(), ctx.idem_key))
                        db.execute("UPDATE action_inputs SET recovery_status=NULL WHERE idem_key=?", (ctx.idem_key,))
                        if row["approval_id"]:
                            approval_status = "confirmed" if spec.risk == Risk.EXTERNAL_SEND else "executed"
                            if status == "abandoned":
                                approval_status = "approved" if spec.risk == Risk.EXTERNAL_SEND else "pending"
                            db.execute("UPDATE approvals SET status=?, updated_at=? WHERE approval_id=?", (approval_status, self.clock(), row["approval_id"]))
                    self._event(db, ctx, row["tool"], summary, "replayed" if status == "done" else "rejected", code=code, result=result, approver_id=row["approver_id"])
                outcomes.append({"idem_key": ctx.idem_key, "tool": row["tool"], "status": status, "code": code})
        return outcomes

    def audit(self, *, run_id: str | None = None, ticket_id: str | None = None,
              actor_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self._audit.query(run_id=run_id, ticket_id=ticket_id, actor_id=actor_id, limit=limit)
