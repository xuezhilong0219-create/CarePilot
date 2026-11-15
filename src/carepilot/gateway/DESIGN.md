# 网关执行与恢复约定

`ToolGateway` 实现冻结的 `Gateway` 协议；仅使用标准库、Pydantic 和 `auth/spec`，没有模型或网络调用。

## 执行顺序

1. schema → rules → permission → risk；规则、权限和工具函数都收到验证后的模型。schema 强制严格类型和禁止额外字段，拒绝不记原始无效参数。
2. 参数转 JSON 后计算 `sha1(run_id + NUL + tool + NUL + canonical_json(args))`，默认值也参加规范化。
3. 同一个数据库的线程锁 + 操作系统文件锁只覆盖账本事务：预留、记账、对账、审批状态。**工具本身在锁外执行**，「恰好一次」由 `action_ledger` 里那行 `started` 保证，不是靠锁一直握着。SQLite 使用 WAL、FULL synchronous、30 秒 busy_timeout；每个连接独立创建和关闭。
4. `BEGIN IMMEDIATE` 中 `INSERT OR IGNORE` 预留 `started`，保存恢复参数，并写 `execution_intent`；这整个事务成功提交后才调用工具。审计写失败向上抛出，预留一起回滚，不执行工具。
5. 工具成功后，在一个事务里写账本 `done`、结果、审批完成状态和 `executed` 审计。此时审计失败会回滚完成记账，保留已提交的 `started` 和意图，重启后对账。
6. STATE_CHANGE 的 `approved` 状态和预留在同一事务；EXTERNAL_SEND 先持久化批准，再确认执行。重复确认走账本回放，不执行第二次。批准、确认、拒绝都重新检查操作者的工具角色和 owner_check；不重新运行业务规则。

同一幂等键的并发调用既不报错也不重复执行：第一个调用者占位后到锁外跑工具，其余调用者在锁外轮询账本（默认最多 `inflight_wait_s` = 10 秒），看到 `done` 就回放；等不到返回 `reconciliation_required`，留给对账。审批路径（`approve` / `confirm`）仍持锁执行 STATE_CHANGE / EXTERNAL_SEND——低频、人工触发的写操作，状态机语义因此逐字未变。

一条成功调用可能有意图和完成两条审计，不能用日志行数代替调用数；被拦截调用只有一条拒绝审计。`gate/code` 表明拒绝位置，`actor_id` 表示当前操作者，`approver_id` 保留批准人；用 `run_id/ticket_id/idem_key` 串联。

## 恢复

`reconcile()` 扫描 started 并逐条核实。工具挪到锁外执行之后，"谁还在跑"不能靠锁判断了，改由**在飞租约**回答：执行期间对 `<db>.leases/<idem_key>` 加一把操作系统文件锁，对账探到锁在就跳过这条（`skipped_inflight`），锁没了说明上一个执行者已经不在了，可以接管。不用 pid 判活：Windows 上 `os.kill(pid, 0)` 不是探测而是 TerminateProcess，`OpenProcess` 在进程刚被强杀后仍可能拿到句柄（实测过），pid 还会复用；文件锁由操作系统在进程消失时释放，正好等于要的语义。

`verify` 是只读检查：结果字典（包括空字典）表示副作用已存在，None 才表示确定没有副作用。前者记 done 并认领，后者记 abandoned 允许原键重试；异常、工具缺失或没有 verify 一律人工处理。

对账有两个开关：`min_age_s` 只处理占位早于这么多秒的行（防止刚起的调用被顺手接管），`dry_run` 只列候选——不调 verify、不写库、不写审计。认领副作用不可回退，所以入口给的是「先看再动」。

账本 status 只用冻结任务规定的 started/done/abandoned；`needs_manual` 放 `action_inputs.recovery_status`，返回列表也包含该值。直接重放 started 返回 `reconciliation_required`，须显式 reconcile；没有 verify 的动作始终保持 started 并拒绝自动重试。

测试用 `CAREPILOT_CRASH_AT=before_effect|after_effect` 和 `CAREPILOT_CRASH_MARKER` 控制钩子；生产默认空操作，部署时应清理这两个环境变量。Windows 测试使用真正的 `Popen.kill()`（TerminateProcess），不是模拟异常或 Linux SIGKILL。

## 集成边界与现存风险

- **外部 exactly-once 有前提**：verify 必须能以 idem_key 可靠判断最终副作用；若下游仍在异步提交或 verify 读到陈旧状态，None 后重试仍可能重复。应把 idem_key 传给下游唯一约束；工具「先完成副作用再抛异常」按任务要求记 abandoned，也可能在重试时重复。
- **吞吐与卡住风险**：只读与内部写已经不再串行——`eval/bench_gateway.py` 用 16 个不同 run、工具耗时 100ms：改动前墙钟 1.88s（并行度 0.85×，就是串行），改动后 0.25s（6.50×）；跨进程 8 个 0.23s（3.49×）。仍然串行的部分：审批后执行的 STATE_CHANGE / EXTERNAL_SEND 在锁内，SQLite 的所有写入本身也按库串行。工具或 verify 卡住只占住它自己那条调用（只读/内部写），但 `reconcile()` 仍要持锁扫表。RunGuard 只能在节点前检查，不能抢占同步工具；工具必须自带超时，且不能递归调用同一网关。
- **存储含敏感数据**：审批和恢复需要完整参数，action_inputs/approvals/action_ledger 可能存明文；审计只写 summarize_args、截断结果并兜底屏蔽大陆手机号，其他敏感字段必须由工具自行脱敏，当前没有加密、访问控制或保留期清理。
- **权限和业务一致性**：身份来自模拟 header；审批不重跑业务规则，工具自身必须在业务事务里防止过期状态迁移。run_id 必须可信且绑定工单/用户，幂等键按冻结算法不含 actor_id 或 ticket_id。
- **部署与人工恢复**：文件锁（含租约）依赖本机文件系统，同一个库不能通过不同硬链接或不可靠网络共享访问；无 verify 的 started 需要人判断，入口是 `POST /admin/reconcile`（仅管理员，支持 `dry_run`）。`ReconcileSweeper` 可以按 `RECONCILE_INTERVAL_S` 定时扫，默认关闭——认领副作用是写库动作，开不开由部署环境决定。
- **生命周期限制**：冻结接口未定义 TTL；支持拒绝已标 expired 的记录，但不自动过期；已 rejected/expired 的原键不会生成新审批，需要新 run_id。没有 schema 迁移、审计防篡改、日志轮转。

## 验收证据

`tests/gateway/` 覆盖四道闸、恶意模型 10 次提议、六次调用链回溯、手机号全表检查、审计前后失败、50 轮双线程 Barrier、四个独立进程、两处真实强杀和审批动作恢复。`uv run python -W ignore tests/gateway/check_mutations.py` 会逐一实际改坏五处源代码，要求对应测试失败后逐字节恢复；不能和其他测试并行运行。

RunGuard 独立于网关，由图集成层在节点/工具前调用；BudgetExceeded.reason 供集成层转人工，模块自身不写工单或调用模型。
