# 任务 G：工具网关内核（四道闸、审计、幂等、批准流程、崩溃恢复、预算守卫）

负责人：Codex。先读 `.team/TEAM.md`（规则）和 `docs/requirements.md`（背景，重点看 C2/C3/C4/C5/C6/B5/B6）。

## 背景

我们要把"检索问答"升级成"可治理的工单 Agent"。这个任务是整个治理层的内核。核心原则：**模型只能提议一次工具调用，是否放行由确定性代码决定；网关内没有任何模型调用，也不 import 任何 LLM 相关模块。**
冻结接口在 `src/carepilot/auth.py` 和 `src/carepilot/gateway/spec.py`，**不能改**（需要改就写进报告）。你实现 `Gateway` 协议。

## 要做的事

### 1. `gateway/gateway.py`：`ToolGateway(db_path, registry, *, clock=time.time)`
- `registry: Mapping[str, ToolSpec]`。所有表在构造时幂等创建（`CREATE TABLE IF NOT EXISTS`），SQLite 开 WAL、设 `busy_timeout`，多线程和多进程同时访问同一个 db 文件要安全。
- **四道闸按 schema → rule → permission → risk 的顺序，第一道不过立即返回 `Rejected`**：
  - schema：未知工具 → `Rejected(SCHEMA, "unknown_tool")`；参数用 `spec.args_model.model_validate` 校验，多余字段、缺字段、类型错都是 `Rejected(SCHEMA, "invalid_args")`。后面所有闸用的都是**校验后的模型对象**。
  - rule：依次执行 `spec.rules`，`RuleViolation` → `Rejected(RULE, e.code, e.message)`；规则函数抛了别的异常 → `Rejected(RULE, "rule_error")`，不能让异常穿透网关。
  - permission：`ctx.principal.role not in spec.roles` → `Rejected(PERMISSION, "role_denied")`；`spec.owner_check` 返回 False → `Rejected(PERMISSION, "not_owner")`；owner_check 抛异常按拒绝处理。
  - risk：`READ`、`WRITE_INTERNAL` 直接执行；`STATE_CHANGE`、`EXTERNAL_SEND` 返回 `NeedsApproval(stage="approve")`，并持久化待批准记录（含校验后参数、幂等键、发起人）。
- `approve(approval_id, approver)`：只有 agent/admin 可以批准，否则 `Rejected(PERMISSION, "approver_role_denied")`；批准人**重新**过一遍 permission 之外的检查没有意义，但必须校验审批记录存在、状态是 pending（已批准/已拒绝/已过期的返回 `Rejected(RISK, "approval_not_pending")`）。`STATE_CHANGE`：批准即执行，返回 `Executed`。`EXTERNAL_SEND`：返回 `NeedsApproval(stage="confirm")`。
- `confirm(approval_id, approver)`：仅 `EXTERNAL_SEND`，且必须先经过 approve。首次确认执行；**重复确认返回 `Executed(同一结果, replayed=True)`，副作用只发生一次**。确认的参数是批准时存下的那份，调用方无法在确认阶段改参数。
- `reject(...)`：记录拒绝并写审计。

### 2. 幂等（`gateway/ledger.py`）
- 幂等键 = `sha1(run_id + "\0" + tool + "\0" + canonical_json(校验后的参数))`；canonical_json = 键排序、无多余空白、`ensure_ascii=False`。参数顺序不同但语义相同，键必须相同。
- 账本表 `action_ledger(idem_key PRIMARY KEY, run_id, tool, status, result_json, created_at, updated_at)`，`status ∈ started/done/abandoned`。
- 执行协议：事务里 `INSERT OR IGNORE` 预留 → 插入成功才执行 `spec.fn`（把 `idem_key` 放进新建的 `ToolContext` 传下去）→ 成功后 `UPDATE status='done', result_json=...`。已 `done` → 直接返回 `Executed(result, replayed=True)`；已 `started` 且当前没有别的执行者 → 见恢复。
- **并发**：两个线程同时用同一个键调用，只能有一个真正执行（用 `threading.Barrier` 写测试，跑 50 轮）。
- `spec.fn` 抛异常：账本记为 `abandoned`（允许重试），异常不穿透，返回 `Rejected(RISK, "tool_error", 摘要)` 并审计。

### 3. 审计（`gateway/audit.py`）
- 表 `audit_log`：`id, ts, actor_id, actor_role, run_id, ticket_id, tool, args_summary, decision(executed/needs_approval/rejected/approved/confirmed/replayed), gate, code, result_summary, approver_id, idem_key`。
- **每一次调用都写，包括被拒绝的**；写审计失败不能吞掉（要让调用失败，宁可不执行也不能没有记录）——请在设计说明里写清你怎么保证"先有审计意图，再有副作用"。
- `args_summary` 用 `spec.summarize_args`，没有则用 JSON 截断到 200 字符；`result_summary` 截断到 200 字符。**敏感信息不能以明文进审计**：写一个测试，工具参数里带手机号 `13812345678`，`summarize_args` 返回脱敏串，断言整张 audit_log 表的任何列里都找不到明文号码。
- `audit(run_id=..., ticket_id=..., actor_id=..., limit=...)` 按时间顺序返回。验收要求"任一动作 30 秒内从日志回溯完整链路"：用一个 run 里的 6 次调用（含被拒绝的、批准的、确认的）写测试，断言 `audit(run_id=...)` 能还原出"谁发起 → 哪道闸 → 谁批准 → 最终结果"。

### 4. 崩溃恢复（`reconcile()` + 真实的进程强杀测试）
- `reconcile()`：扫描 `status='started'` 的账本条目，对每条调用对应 `spec.verify`：返回结果 → 认领（`done`，写入该结果，审计 `reconciled_adopted`）；返回 `None` → 标为 `abandoned`（允许同一个键重新执行，审计 `reconciled_retry_allowed`）；没有 `verify` 的工具 → 保守处理，标 `needs_manual`，**不自动重试**（宁可不做也不重复做），返回的列表里要体现。
- **必须用真实子进程强杀测试**（`tests/gateway/test_crash_recovery.py` + `tests/gateway/crash_child.py`）：子进程执行一次调用，在网关内部预留的**测试钩子点**（`_crash_point(name)`，生产环境是空操作，由环境变量 `CAREPILOT_CRASH_AT` 触发，写出标记文件后 `time.sleep(60)`）停住；父进程等到标记文件出现后 `Popen.kill()`；再启动新进程 `reconcile()` + 重新执行同一调用。两种崩溃点都要测：`before_effect`（预留后、副作用前）和 `after_effect`（副作用后、账本记 done 前）。**两种情况最终副作用都必须恰好 1 次。**副作用用一个会向文件追加一行的工具来计数。
- 说明：本机是 Windows，`Popen.kill()` 是 TerminateProcess，等价于不可捕获的强杀；在报告里写这一点。

### 5. 预算与死循环守卫（`gateway/guard.py`）
- `RunBudget(max_steps=20, max_tokens=50_000, max_seconds=120.0)`，`RunGuard(budget, *, clock=time.monotonic)`。
- `guard.step(kind: str, *, tool: str | None = None, args: dict | None = None, tokens: int = 0)`：每次图节点执行或工具调用前调用一次。超限抛 `BudgetExceeded(reason)`，`reason ∈ {"steps","tokens","time","loop"}`。
- 死循环检测：**同一个 (tool, 规范化参数) 连续出现 3 次**，或同一个键累计出现 4 次 → `reason="loop"`。
- 验收：写一个"假模型"测试，它永远提议同一个工具调用，驱动一个 while 循环，断言在 20 步之内被 `BudgetExceeded` 打断，且 `reason` 是 `loop`（不是 `steps`）。另写一个每次提议不同参数的假模型，断言被 `steps` 在第 21 次打断。时间预算用注入的假时钟测，不要真的 sleep。

### 6. 对抗性验收测试（你自己写，另一个 worker 之后会独立再写一份）
`tests/gateway/test_gates.py`：四类非法输入**各至少 2 例**，全部断言在网关层被拒且**工具函数没有被调用**（用计数的假工具）：
- schema：未知工具名；多余字段 / 类型错误。
- rule：非法状态迁移；违反业务规则的参数。
- permission：客户调用只有坐席能用的工具；客户查别的店铺的资源（`owner_check` 为 False）。
- risk：`STATE_CHANGE` 工具在没有批准时不执行；`EXTERNAL_SEND` 批准后不确认也不执行；客户尝试批准。
再加一个"恶意模型"测试：一个脚本化的假模型按顺序提议 10 个各种非法调用，断言没有任何一个产生副作用，审计里有 10 条记录。

## 文件归属

`src/carepilot/gateway/`（**除 `spec.py` 外**）、`tests/gateway/`。不要改 `auth.py`、`spec.py`、`graph/`、`api/`、`retrieval/`、`pyproject.toml`。

## 验收

- `uv run pytest -q` 全绿，包含真实子进程强杀测试（不允许 skip）。
- 你**手动改坏下面 5 处中的每一处**，确认对应测试变红再改回，并在报告里列出：①去掉并发下的 `INSERT OR IGNORE`；②审批通过后不校验 pending 状态；③`confirm` 重复调用时重新执行；④审计里写明文参数；⑤`reconcile` 对没有 verify 的工具直接重试。
- 报告里写：幂等和审计的先后顺序设计、并发压力测试的轮数与结果、你认为这个设计仍然存在的风险（至少 3 条，诚实写）。

## 不要做

不要引入新依赖；不要在 gateway 里 import `langchain` / `langgraph` / `carepilot.llm` / `carepilot.graph`；不要连网络；不要改冻结接口。
