# 任务 S：工单存储（SQLite）、状态机、SLA、可见性与权限判定、交接摘要

负责人：opencode。先读 `.team/TEAM.md` 和 `docs/requirements.md`（重点看 B1/B4/B5/B7/C1）。

## 背景

现在的"工单"只存在于 LangGraph 的 checkpoint 里，没有真正的工单存储：查不了状态和 SLA，没有处理人，客户和坐席看到的内容也没有区分。你要做一个**自包含的工单服务**。**不调用任何 LLM。**
身份用 `src/carepilot/auth.py` 里的 `Principal` / `Role`（冻结，别改）。

## 数据与表（`src/carepilot/store/`）

所有表在构造时幂等创建；SQLite 开 WAL 和 `busy_timeout`；多线程安全（每次操作自己开连接或用锁，别共享一个连接跨线程）。

- `users(user_id PK, role, display_name)`、`user_shops(user_id, shop_id)`：`seed_demo_users(store)` 写入固定演示数据：客户 `c1`→店铺 `S1,S2`、`c2`→`S3`、`c3`→`S4`；坐席 `a1`、`a2`；管理员 `admin1`。
- `tickets(ticket_id PK, customer_id, shop_id, subject, status, priority, owner_id NULL, thread_id, category NULL, created_at, updated_at, sla_due_at, version)`。`ticket_id` 形如 `T-000123`（自增，不要用随机数）。
- `ticket_messages(id PK, ticket_id, author_id, author_role, kind, visibility, body, idem_key NULL UNIQUE, created_at)`：`kind ∈ message/draft/note/system`，`visibility ∈ public/internal`。**`idem_key` 有唯一约束**：网关重试同一个动作时，`add_message(..., idem_key=k)` 第二次返回已存在的那条，不插入第二条（这是副作用的最后一道防线）。
- `ticket_events(id, ticket_id, ts, actor_id, event, from_status, to_status, detail)`：每次状态变化和认领都记一条。

## 要提供的接口（名字和签名是约定，集成时我会直接调用）

```python
class TicketStore:
    def __init__(self, db_path: Path, *, clock: Callable[[], float] = time.time, sla_policy: SlaPolicy | None = None): ...
    def create_ticket(self, customer: Principal, *, subject: str, shop_id: str | None, thread_id: str,
                      priority: str = "medium", category: str | None = None) -> Ticket
    def get_ticket(self, who: Principal, ticket_id: str) -> Ticket            # 无权 → 抛 Forbidden；不存在 → 抛 NotFound
    def list_tickets(self, who: Principal, *, status: str | None = None, mine: bool = False) -> list[Ticket]
    def add_message(self, ticket_id: str, *, author: Principal, kind: str, body: str,
                    visibility: str = "public", idem_key: str | None = None) -> Message
    def list_messages(self, who: Principal, ticket_id: str) -> list[Message]    # 客户看不到 internal
    def transition(self, who: Principal, ticket_id: str, to_status: str, *, reason: str = "") -> Ticket
    def claim(self, who: Principal, ticket_id: str) -> Ticket                   # 坐席认领；已被别人认领 → 抛 Conflict
    def set_priority(self, who: Principal, ticket_id: str, priority: str) -> Ticket
    def sla_remaining_seconds(self, ticket: Ticket, now: float | None = None) -> float   # 可为负（已超时）
    def events(self, who: Principal, ticket_id: str) -> list[TicketEvent]
# 纯函数，供网关的 rule / owner_check 直接用
def can_transition(from_status: str, to_status: str) -> bool
def can_view(who: Principal, ticket: Ticket) -> bool
def can_modify(who: Principal, ticket: Ticket) -> bool
def build_handoff(ticket: Ticket, messages: Sequence[Message], actions: Sequence[dict]) -> Handoff   # B7
class NotFound(Exception); class Forbidden(Exception); class Conflict(Exception); class IllegalTransition(Exception)
```
`Ticket`、`Message`、`TicketEvent`、`Handoff`、`SlaPolicy` 用 `@dataclass(frozen=True)`，`Ticket` 带 `sla_remaining_seconds` 的便捷属性不要有（用上面的方法）。

## 规则（每条都要有测试）

**状态机**：`new → open`；`open ⇄ pending_customer`；`open ⇄ on_hold`；`open → escalated`；`escalated → open`；任何非 `closed` 状态 → `resolved`；`resolved → closed`；`resolved → open`（重开）；`closed` 是终态。其余一律 `IllegalTransition`（`can_transition` 返回 False）。把整张迁移表写成一个常量并**对所有 6×6 状态对**做参数化测试。

**权限（C1）**：
- 客户：只能 `get_ticket` / `list_tickets` / `list_messages` 自己的工单（`customer_id == who.user_id`）；看别人的 → `Forbidden`（不是 NotFound，**越权统一 403 口径**；但对不存在的工单返回 NotFound）。客户不能 `transition` / `claim` / `set_priority`；`list_messages` 看不到 `visibility='internal'`。
- 坐席：可以查看所有工单；**只能修改自己负责的（`owner_id == who.user_id`）**，或者未分配的（只能 `claim`）。改别人负责的 → `Forbidden`。
- 管理员：全部可以。
- `can_view` / `can_modify` 是纯函数，`TicketStore` 内部调用的是同一份，不要写两套。

**SLA（B1）**：按优先级定义时限，默认 `critical=1h, high=4h, medium=8h, low=24h`（`SlaPolicy` 可覆盖）。`sla_due_at = created_at + 时限`；优先级变化时重算（从创建时间起算，不是从修改时间）；工单进入 `resolved` / `closed` 后 `sla_remaining_seconds` 返回 `float("inf")`。**不考虑工作日和节假日**——在 docstring 里写明这是简化。时间用注入的 `clock`，测试里不要真的等。

**B4 草稿不可见**：`kind='draft'` 的消息**强制** `visibility='internal'`（调用方传 public 也要被纠正或抛错，选抛 `ValueError` 并写测试）。

**B7 交接摘要**：`build_handoff(ticket, messages, actions)` 是**确定性的纯函数，不用模型**。返回 `Handoff(ticket_id, escalation_reason, actions_done: list[str], open_questions: list[str], last_customer_message: str, owner_id, status)`。`actions` 是形如 `{"tool": "...", "summary": "...", "at": ts}` 的字典列表；`escalation_reason` 取最近一条 `kind='system'` 且 body 以 `升级原因：` 开头的消息；`open_questions` 取最近的 `kind='note'` 且 body 以 `待确认：` 开头的行。缺信息时字段给空值，不能抛异常。再测一个"下一个坐席接续"：坐席 a1 认领、升级、a1 释放（`transition` 回 `open` 且 `owner_id` 置空——为此给 `claim` 配一个 `release(who, ticket_id)`），坐席 a2 能认领并通过 `build_handoff` 看到同样的上下文。

**并发**：两个坐席同时 `claim` 同一张工单，恰好一个成功、另一个 `Conflict`（用线程 + Barrier，跑 50 轮）。状态更新用 `version` 做乐观锁或 `UPDATE ... WHERE status=?` 保证原子。

## 文件归属

`src/carepilot/store/`、`tests/store/`。不要改 `auth.py`、`gateway/`、`graph/`、`api/`、`pyproject.toml`。

## 验收

- `uv run pytest -q` 全绿。
- 手动改坏下面 4 处，各自确认有测试变红再改回，并写进报告：①客户能看到 internal 消息；②坐席能改别人负责的工单；③`resolved → closed` 之外多放行了一条迁移；④`claim` 去掉原子性检查。
- 越权用例：客户 c1 访问 c2 的工单 → `Forbidden`；坐席 a2 `transition` a1 负责的工单 → `Forbidden`；两个都要有测试。

## 不要做

不要调用 LLM；不要引入新依赖；不要在这里做 HTTP 接口。
