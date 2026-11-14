# 任务 O：可观测性（trace 记录器、成本统计）、一键评测、badcase 回流、记忆约定

负责人：opencode。先读 `.team/TEAM.md`（含「第二阶段补充」）和 `docs/requirements.md`（重点看 D1–D5）。

## 背景

D1–D5 的数据结构已冻结在 `src/carepilot/observability/spec.py`（`LlmCall` / `Span` / `RunTrace`，含 `to_dict` / `from_dict`）和 `src/carepilot/memory/spec.py`。**这两个文件不能改。**
你实现记录器、落盘、成本统计和评测编排。**不调用任何 LLM**，测试不联网。

## 要做的事

### 1. `observability/tracer.py`（D1）
- `RunTracer(run_id, *, ticket_id=None, category=None, clock=time.time, **meta)`：一次 run 一个对象，**线程安全**（评估时多个 run 并发，同一个 run 内也可能并发记录）。
  - `with tracer.span("classify", input_summary="...") as sp:`：进入时记开始时间，退出时记 `duration_ms`；`sp.set_output("...")`、`sp.add_llm_call(LlmCall(...))`、`sp.add_retry()`；**with 块内抛异常时记录 `error`（类型名 + 截断的消息）并继续抛出**。所有摘要进入 `Span` 前都用 `clip()` 截断。
  - `tracer.end(status)` → 返回 `RunTrace` 并设置 `ended_at`；重复调用 `end` 只生效第一次。`tracer.set_category(...)` 在分诊之后调用。
- `llm_call_from_message(raw_message, *, purpose, model, latency_ms, retries=0) -> LlmCall`：从 LangChain 的 `AIMessage.usage_metadata`（`input_tokens` / `output_tokens`）读 token；**没有 usage 就记 0 并设 `usage_missing=True`，不估算**。用一个带 `usage_metadata` 属性的假对象写测试，不要 import langchain。
- `TraceStore(directory)`：`save(trace)` 写 `<dir>/<run_id>.json`（原子写：先写临时文件再 `os.replace`），`load(run_id)`、`list_runs() -> list[str]`（按开始时间排序）、`load_all()`。`run_id` 要校验（只允许 `[A-Za-z0-9_.-]`，禁止路径分隔符和 `..`），非法抛 `ValueError`；这是路径穿越防护，要测。
- 一个 `NullTracer`（所有方法空操作，接口相同），供不需要 trace 的调用方使用。

### 2. `observability/cost.py`（D2）
- 单价**只能来自用户配置**，不能写死、不能猜。`load_pricing(path) -> Pricing`：JSON 形如
  `{"models": {"deepseek-v4.1-flash-1017": {"input_per_mtoken": null, "output_per_mtoken": null}}, "currency": "CNY", "source": "", "as_of": ""}`。
  单价为 `null` 或文件不存在 → 该模型的成本为 `None`（不是 0）。**要给出价格时，`source` 和 `as_of` 必须非空，否则抛 `PricingError`**（验收标准：报告中的成本 = 实测 token × 当前单价，注明来源与日期）。
- `summarize_cost(traces, pricing) -> CostReport`：包含
  - 总体：run 数、各 purpose 的调用次数、prompt/completion token 合计；
  - **按工单**（每个 run）：token 和成本（成本可能为 `None`）；
  - **按意图**（`RunTrace.category`，缺失记为 `unknown`）：平均每单 token、平均成本、run 数；
  - **按模型**：调用次数、token；
  - `usage_missing` 的调用数量单独列出，并在文本报告里警示"token 统计不完整"；
  - 成本为 `None` 时报告里写"未配置单价：仅统计 token"，**不显示 0**。
- `CostReport.to_markdown()` 输出表格；`.to_dict()` 可 JSON 序列化。
- 同时提供示例文件 `pricing.example.json`（单价为 null，`source`/`as_of` 为空，里面写注释说明怎么填；JSON 不支持注释，就放一个 `_comment` 字段），放仓库根目录。

### 3. 一键评测 `eval/eval_all.py`（D3）+ `Makefile`
- 本机**没有 `make`**，所以主入口是跨平台的 Python 脚本，Makefile 只是转发。`python eval/eval_all.py --quick | --full [--report-dir DIR] [--skip tests,retrieval,pipeline,tools]`。
- 依次运行（每一步是一个**可注入的函数**，测试里用假实现，不跑真模型）：
  1. `tests`：`pytest -q`，记录通过/失败数；
  2. `retrieval`：对评估集的可答样本跑 BM25 检索，报 Recall@5（= hit@5 严格口径）、hit@10、chunk_hit@4、MRR；
  3. `pipeline`：对评估集跑全流程，用已有的 `evaluation/runners.py::run_pipeline`，报「答对率」（放行且引用命中的占可答样本比例）、「拒答准确率」（负样本不被自动回复的比例）、「误拒率」（可答样本被拒的比例）、笼统消息追问率；
  4. `tools`：工具选择准确率。**这一步由我（Claude）稍后提供 runner**，你先定义接口 `run_tool_selection(dataset_path, settings) -> dict`（返回 `{"n":..., "tool_accuracy":..., "args_accuracy":..., "denied_correct":...}`），在 eval_all 里用 `importlib` 尝试导入 `carepilot.evaluation.tool_eval.run_tool_selection`，导入不到就标记为"未提供"，不报错。
- `--quick`：流水线和工具选择只跑**分层子集**（固定 seed，按 `可答/需查数据/无关/含糊` 四组各至少 3 条，共约 60 条）；`--full` 跑全部。分层抽样写成纯函数 `stratified_subset(items, n, seed)`，要测。
- 最后输出一张汇总 markdown 表 + 写 `report-dir/eval_all_<时间戳>.json`；每个指标旁标注样本数；某一步失败不影响后面步骤，汇总里标 `FAILED` 并给出原因，**退出码非 0**。
- **真实评测集在 `eval/real/`（gitignore）**，文件不存在时对应步骤标记"数据集缺失"而不是崩溃。

### 4. badcase 回流 `eval/export_badcases.py`（D4，骨架即可）
读 `run_pipeline` 产出的 `*.items.jsonl` + 评估集，把「实际去向与期望不符」的样本导出为候选集 `eval/real/badcase_candidates.jsonl`：每行 `{qid, question, expected, actual, reason, source_report}`；期望去向的口径写成纯函数 `expected_outcome(item) -> str`（可答→auto_reply；需查数据→escalate；无关→escalate/need_info 之一；含糊→need_info）。不要把候选直接并入评估集（需要人工确认），脚本里写明。

### 5. 记忆约定（D5）
`memory/spec.py` 已经冻结，你只需要写 `docs/memory.md`：用 10–20 行说明读写约定（直接引用 spec 里的 5 条）、为什么本项目不实现、接入时的第一步。**不要写任何实现代码。**

## 文件归属

`src/carepilot/observability/`（**除 `spec.py` 外**）、`eval/eval_all.py`、`eval/export_badcases.py`、`Makefile`、`pricing.example.json`、`docs/memory.md`、`tests/observability/`（**除已有的 `test_trace_spec.py` 外**）、`tests/test_eval_all.py`。
不要改 `observability/spec.py`、`memory/spec.py`、`graph/`、`api/`、`gateway/`、`store/`、`pyproject.toml`。

## 验收

- 全部测试通过。
- 测试必须覆盖：多线程并发记录同一个 run（50 个线程各记一个 span，断言一个都不丢）；`with` 块内异常被记录且继续抛出；`run_id` 路径穿越（`../x`、`a/b`、`a\b`、空串）全部被拒；`usage_missing` 路径；单价缺失时成本是 `None` 而不是 0、给了价格但缺来源/日期时抛 `PricingError`；按意图/按模型汇总的手算期望值；`stratified_subset` 的分组下限和 seed 可复现；eval_all 某一步抛异常时其余步骤照常运行且退出码非 0。
- 手动改坏 3 处各确认变红再改回，写进报告：①trace 落盘不是原子写；②成本在单价为 null 时返回 0；③`stratified_subset` 丢掉某个小组。

## 不要做

不要调用 LLM；不要联网；不要写真实价格；不要改冻结接口。
