# CarePilot 使用手册

这份文档回答三件事：**怎么跑起来**、**每个功能怎么用**、**出了问题看哪里**。
想要设计理由和评估数字，看 [README](../README.md) 和 [status.md](status.md)；想要每个模块为什么这么写，看 [architecture.md](architecture.md)。

---

## 1. 五分钟跑起来

### 前置

- Python 3.11+（我用 3.12 验证）
- [uv](https://docs.astral.sh/uv/)（仓库里有 `uv.lock`，装依赖是确定性的）
- 一个 **OpenAI 兼容**的模型网关地址和 key（`LLM_BASE_URL` 要能接受 `POST /v1/chat/completions`）

### 安装与自检

```bash
uv sync                      # 装依赖（含 dev 组）
cp .env.example .env         # 然后编辑 .env，填 LLM_API_KEY / LLM_BASE_URL
uv run pytest -q             # 713 passed：不联网、不调模型，用的是假模型
uv run python scripts/check.py   # 门禁 = ruff + 全量测试，CI 跑的就是这一条
```

> 跑测试不需要 key：所有图测试都注入脚本化的假模型。只有真的起服务、或在 `eval/` 下跑评测时才需要。

### 起服务

```bash
uv run uvicorn carepilot.api.app:app --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

默认用仓库自带的示例知识库 `data/kb/`（合成的小样本，够演示）。要用真实知识库：

```bash
uv run python scripts/build_kb.py --raw <语雀导出目录> --out data/kb_real   # 纯规则清洗，几秒
KB_DIR=data/kb_real uv run uvicorn carepilot.api.app:app --port 8000
```

### 演示身份（没有任何登录，就是请求头）

| 身份 | 角色 | 能看到的店铺 | 用来看什么 |
|---|---|---|---|
| `c1` | 客户 | S1、S2 | 开单、追问、看回复 |
| `c2` | 客户 | S3 | 验证「客户之间互相看不到」 |
| `c3` | 客户 | S4 | 同上 |
| `a1` / `a2` | 坐席 | 全部 | 审核、二次确认、交接、审计、轨迹 |
| `admin1` | 管理员 | 全部 | 对账入口、在飞调用 |

> 角色和店铺范围从 `users` 表读，请求里自报 `X-Role: admin` 无效（有测试）。

---

## 2. 界面怎么用

右上角下拉就是切换身份（等价于换 `X-User-Id`，会记在 localStorage 里）。页面三栏：**左边工单列表 / 中间会话 / 右边处理台**。

### 客户视角（c1）

1. 点「**＋ 新建**」，写一句话，例如：`企业中台导出订单时报错「库存同步失败」，导不出来，麻烦看下怎么处理`。
2. 中间栏会逐个亮起节点：分诊 → 检索 → 起草 → 待审核。**亮的是真的 SSE 事件**（每个节点跑完推一条），不是假进度条。
3. 等整轮跑完，右侧会显示「系统正在处理…」；等坐席处理完，回复才会出现在会话里。
4. 想接着追问，直接在底部输入框发——历史和已识别的平台/店铺/功能会带进下一轮。

### 坐席视角（a1）

1. 左边列表出现客户的工单（状态徽章：`待审核` / `待二次确认` / `已解决` / `已升级`）。
2. 点开工单，右侧从上到下：
   - **工单状态**：编号、分类、优先级、负责人、SLA。
   - **分诊结果**：类别、严重度、是否笼统、识别到的槽位。
   - **待审核卡片**：草稿全文（`pre-wrap`）、引用 chips、审核备注、三个按钮。
   - **二次确认卡片**：批准后出现，说明「批准」和「确认」为什么要分开。
   - **本轮执行**：工具调用的结构化结果（只显示成功的那几条）。
   - **运行与审计**：三个 tab。
3. 三个按钮的含义：
   - **通过并发送**（草稿是 `auto_reply`）→ 工单变「待二次确认」，客户**还看不到**任何东西。
   - **改后发送** → 同上，但用你改过的文案；引用会被清空（因为你改过的内容不再是知识库原文），审计里标 `edited_by_human`。
   - **驳回并升级** → 直接执行 `escalate_ticket`，工单变「已升级」，没有二次确认这一步。
   - 如果这张卡本来就是**升级卡**（`resolution=escalate`），按钮是「确认升级 / 驳回」。
4. 点「**确认发送（真实动作）**」→ 消息进会话、工单置 `resolved`。重复点不会多发（网关按幂等键回放）。
5. **运行与审计**三个 tab：
   - **轨迹**：这一轮每个节点一行——横条长度 = 该节点耗时 / 最慢节点，右边是毫秒 + 输出摘要。顶部两个数必须分开看：`算力`（各节点之和）和 `全程`（含等人审核的时间），`T-000001` 是 61.41 s 算力 / 762.69 s 全程。token 显示 `1862（972 进 / 890 出）`；网关没给 usage 时写「只统计到有返回的那几次」，不估。
   - **轮次下拉**：一单会被追问很多轮，**每轮一个 trace 文件**，默认最近一轮，可以往回翻。
   - **交接**：升级原因、已执行动作、待确认的问题、客户最后一句话——下一个坐席靠这个接手。
   - **审计**：表格，时间 / 决策徽章（`needs_approval` 黄、`approved`/`confirmed`/`executed` 绿、`rejected` 红）/ 工具 / 参数摘要。看的就是 `needs_approval → approved → confirmed → executed → set_ticket_status` 这一串。

### 引用卡片

草稿里的 `[1] [2]` 对应 chips，点开是知识库原文（`GET /citations/{chunk_id}`，只有坐席能取）。客户点不到。

### 管理员视角（admin1）

界面里没有对账按钮，用命令行：

```bash
# 谁卡着、有没有活租约
curl localhost:8000/admin/inflight -H 'X-User-Id: admin1'
# 只列候选，不调 verify、不写库、不写审计
curl -X POST 'localhost:8000/admin/reconcile?dry_run=true' -H 'X-User-Id: admin1'
# 真对账（早于 300 秒的才算卡住）
curl -X POST 'localhost:8000/admin/reconcile?min_age_s=300' -H 'X-User-Id: admin1'
```

---

## 3. 六类问法，分别会发生什么

想快速体验，直接拿这几句开单：

| 你发的话 | 分诊 | 接下来发生什么 | 你在界面看到 |
|---|---|---|---|
| `店铺授权过期了怎么办` | 平台对接 | 检索 → 起草 → 待审核 | 草稿 + 引用，等坐席 |
| `我的店铺有问题`（笼统） | 笼统 | 追问「具体遇到什么问题」 | 一条模板追问，工单 `need_info` |
| `库存同步一直失败，日志里显示「请求过于频繁」，要怎么排查` | 功能 | 检索到知识库 → 草稿 | 草稿 + 引用 chip 可点开原文 |
| `订单不能导出` | 功能 | 知识库答不了 → 升级建议 | 升级卡，按钮「确认升级」 |
| `帮我看下订单 139931582794 发货了没` | 需查数据 | 规划只读工具 → 查订单/物流 → 用事实起草 | 审核卡里引用写「系统查询结果」 |
| `查一下账单为什么少了三笔` | 需查数据 | 账单类不查库，直接升级 | 升级卡 + 交接原因 |

几个必须知道的语义：

- **越权和不存在不可区分**：查别人店铺的订单、查不存在的订单，返回的是**逐字相同**的「没有找到您说的记录…」。不是偷懒，是防探测。
- **一条工具降级**（比如物流超时）→ 整轮走确定性降级文案，**不交给模型发挥**。
- **拒答优先**：资料里找不到原文依据就升级，绝不硬编。评估里可答样本的误拒率 ~5–7%，这是有意的取舍。

---

## 4. 命令行 / API 一览

所有 `/tickets*`、`/citations`、`/admin` 接口都要 `X-User-Id`。

| 方法与路径 | 谁能调 | 说明 |
|---|---|---|
| `GET /` | 谁都能 | 单文件工单台（不内嵌业务数据，CSP 收紧） |
| `GET /health` | 谁都能 | `{"ok":true}` |
| `GET /whoami` | 已登录 | `{user_id, role, shops}` |
| `POST /tickets` | 客户 | 开单；body `{message, shop_id?}`；`?stream=true` 走 SSE |
| `GET /tickets?status=&mine=` | 已登录 | 列表（客户只看自己的，坐席看全部） |
| `GET /tickets/{id}` | 有权限的 | 完整视图（客户看不到草稿、处理人、SLA、待审核） |
| `POST /tickets/{id}/messages` | 该工单客户 | 追加一句，历史和槽位带进下一轮 |
| `POST /tickets/{id}/review` | 坐席/管理员 | `{action: approve\|edit\|reject, reply?, note?, confirm?}` |
| `POST /tickets/{id}/confirm` | 工单负责人/管理员 | 对外发送的二次确认，**幂等** |
| `GET /tickets/{id}/handoff` | 坐席/管理员 | 交接摘要 |
| `GET /tickets/{id}/audit` | 坐席/管理员 | 网关审计（固定取 500 行） |
| `GET /tickets/{id}/trace?run_id=` | 坐席/管理员 | 某一轮的节点级 trace，只读磁盘 |
| `GET /tickets/{id}/traces` | 坐席/管理员 | 这一单的所有轮次（摘要） |
| `GET /citations/{chunk_id}` | 坐席/管理员 | 引用原文 |
| `GET /admin/inflight` | 管理员 | 卡在 `started` 的调用 + 租约是否还活着 |
| `POST /admin/reconcile` | 管理员 | 对账（`dry_run` / `min_age_s`） |

### SSE 事件格式

`?stream=true` 时返回 `text/event-stream`，每个事件两行：

```
event: node
data: {"node":"classify","summary":{...}}

event: result
data: {"ticket":{...},"messages":[...],"pending_review":{...}}
```

最后一条一定是 `result`（内容和同步响应一模一样），出错是 `error`。用 `curl -N` 或 `fetch` + `ReadableStream` 读；`EventSource` 不能用，因为它不支持 POST + 自定义头。

### Windows 终端坑

在 Git Bash / cmd 里 `curl -d '{"message":"中文"}'` 可能被按 GBK 编码，服务端返回 `400 There was an error parsing the body`（服务端只吃 UTF-8）。办法：把 JSON 写进 UTF-8 文件用 `--data-binary @msg.json`，或者直接用 Python 的 `httpx`。

---

## 5. 配置

`.env`（`pydantic-settings` 读，未列出的见 `src/carepilot/config.py`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` | 必填 | OpenAI 兼容网关 |
| `LLM_FAST_MODEL` / `LLM_MAIN_MODEL` | `deepseek-v4.1-flash-1017` | 分诊/抽取用快模型，起草用主模型 |
| `KB_DIR` | `data/kb` | 知识库目录（`build_kb.py` 的产出） |
| `RETRIEVER_KIND` | `bm25` | `bm25` / `vector` / `hybrid` / `hybrid_rerank_ce` / `hybrid_rerank_llm` |
| `RETRIEVAL_TOP_K` | 4 | 回复节点实际看到的 chunk 数 |
| `TICKETS_DB` / `GATEWAY_DB` / `CHECKPOINT_DB` | `data/*.sqlite` | 工单 / 账本审计 / LangGraph checkpoint |
| `TRACE_DIR` | `data/traces` | 每轮一个 JSON |
| `CONTEXT_BUDGET_TOKENS` | 3000 | 对话历史预算：最近 3 轮逐字，更早每轮一行摘要 |
| `RUN_MAX_STEPS` / `RUN_MAX_TOKENS` / `RUN_MAX_SECONDS` | 20 / 50000 / 120 | 每个 HTTP 阶段的预算守卫 |
| `RECONCILE_INTERVAL_S` | `0`（关） | 自动对账间隔，`docker-compose.yml` 里默认 60 |
| `RECONCILE_MIN_AGE_S` | 300 | 早于这个秒数的 `started` 不算卡住 |

**重置演示数据**：停服务后删 `data/tickets.sqlite*`、`data/gateway.sqlite*`、`data/checkpoints.sqlite*`、`data/traces/*`，下次启动会重新 seed 演示用户。别在服务运行时删——checkpoint 库被占着会 `Device or resource busy`。

---

## 6. 评估与度量

```bash
uv run python eval/eval_all.py --quick      # 约 5 分钟：测试 + 检索 + 流水线 + 工具选择
uv run python eval/eval_all.py --full       # 全 279 条，一次几十分钟
uv run python eval/eval_all.py --quick --skip tests   # 跳过测试，只跑数据部分
```

其它按需：

| 命令 | 量什么 |
|---|---|
| `uv run python eval/latency_profile.py` | 逐节点耗时（串行、40 条可答），拆成模型时间 / 本地时间 |
| `uv run python eval/ttft_probe.py` | 流式首字延迟 vs 整段完成（结论：结构化输出不能增量） |
| `uv run python eval/run_retrieval.py --dataset <jsonl> --kb-dir <dir>` | 检索指标（hit@k / MRR / chunk_hit@4） |
| `uv run python eval/run_injection.py` | 60 条注入用例，`--no-defense` 是对照臂 |
| `uv run python eval/bench_gateway.py` | 网关并发（假工具），验证锁只夹记账 |
| `uv run python eval/run_edition_filter.py` | 产品线过滤陷阱（企业版 / 极速版同名文档） |
| `uv run python eval/judge_replies.py <items.jsonl>` | 回复质量逐条评审（同模型自评，有偏好） |
| `uv run python scripts/check_viewer.py` | 无头浏览器验 trace 查看器（缺标记退出 1，没浏览器退出 2） |
| `uv run python scripts/trace_viewer.py --traces data/traces --out viewer.html` | 生成单文件 trace 查看器 |
| `uv run python scripts/cost_report.py` | token 报表（金额需要你填 `pricing.json` 的单价） |

---

## 7. Docker

```bash
docker compose up --build          # 起在 8000，挂 ./data 到 /app/data
docker compose down
```

镜像细节：双阶段（builder 装依赖 → runtime 只拷 `.venv`/`src`/`data/kb`），以非 root 用户 `app` 运行，`/app/data` 已 chown 给 `app`（删掉那行会 `PermissionError`，实测过）。compose 里默认开着自动对账（60 秒）。

只构建镜像：

```bash
docker build -t carepilot-api:verify .
```

> 国内网络：`# syntax=docker/dockerfile:1` 的解析发生在 **buildx 客户端**，只给 daemon 配代理不够，要给 `docker build` 命令本身设 `HTTP(S)_PROXY`。

---

## 8. 常见问题

**端口被占 / 改了代码没生效**
Windows 下 `netstat | findstr :8000` 经常什么都查不到，但端口确实被占。用：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '8000' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

`uvicorn` 不加 `--reload` 时改 `ui.py` 必须重启；`taskkill //F` 在 Git Bash 里会报参数错误，用 `Stop-Process -Force`。

**返回 409 是什么意思**
全是业务状态冲突，不是崩了：`ticket is not awaiting review`（这张单现在没有待审核的东西）、`nothing to confirm`（没有待确认的发送）、`ticket is closed`、`approval_not_pending`（这个审批已经被处理过）、`reconciliation_required`（同一幂等键有一次没跑完，需要先对账）。

**重复点「确认发送」会不会发两遍**
不会。`POST /confirm` 是幂等的：网关按幂等键回放上一次结果，审计里能看到 `replayed`。端到端测过重复确认 3 次只产生 1 条消息。

**升级卡为什么没有二次确认**
因为升级没有「对外发送」这一步，`approve` 直接执行 `escalate_ticket`。这时再调 `/confirm` 返回 409 是正常的。

**界面里的身份是真的登录吗**
不是。就是换一个请求头，后端从用户表读角色（自报无效），但整个认证本来就是模拟的。不要拿它当生产认证。

**SSE 断了怎么办**
不自动重连，刷新页面即可。进度条在结果回来后清空，要看历史节点用右侧「轨迹」tab。

**为什么有时要等十几秒**
两次模型调用（分诊 + 起草）占了整单的 99%：p50 6 s、p95 9 s；网关本身每次调用约 16 ms。等人工审核的时间不算在「算力」里，所以轨迹顶部给了两个数。

---

## 9. 边界（别对外承诺这些）

- **身份是模拟的**：没有登录、没有令牌、没有会话。只有「角色取自用户表」这一点是真的。
- **数据是合成的**：订单、物流、店铺授权都在 `src/carepilot/sandbox/` 里生成；只有知识库是真实语雀导出的。
- **模型可能编**：引用能校验、拒答有兜底，但不等于回答正确。回复质量是**同模型自评**，有自我偏好。
- **审计可查不可信**：没有哈希链，能改库的人能改审计。
- **没有端到端压测**：只量了网关层并发；真实链路的秒级延迟在模型。
- **单文件前端是演示台**：没有框架、没有构建链、SSE 不重连、审计只显示最后 40 条。
- **`eval_all --full` 只跑过一次**：279 条一次几十分钟，样本波动比 quick 大。

---

## 10. 想改代码，从哪进

| 想改什么 | 去哪 |
|---|---|
| 加一个工具（还有权限、风险等级、幂等） | `src/carepilot/toolbox.py` 注册 → `src/carepilot/gateway/spec.py` 看 `ToolSpec` |
| 改工作流分支 | `src/carepilot/graph/build.py`（连边）+ `graph/routing.py`（判断条件） |
| 改提示词 | `src/carepilot/graph/nodes.py` 顶部常量 |
| 改检索 | `src/carepilot/retrieval/`（`bm25.py` / `hybrid.py` / `factory.py`） |
| 改界面 | `src/carepilot/api/ui.py`（一个 Python 字符串，改完重启服务） |
| 改 API | `src/carepilot/api/governed.py`（注意是 CRLF，插入代码时别混行尾） |
| 加评估 | `eval/`（`eval_all.py` 是总入口，各 `run_*.py` 单项） |

改完一律：

```bash
uv run python scripts/check.py
```
