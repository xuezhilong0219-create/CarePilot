# 任务 C：工作流测试 + API 重构与流式输出 + 容器化

负责人：DeepSeek Harness（dsh）。先读 `.team/TEAM.md`。

## 背景

`src/carepilot/graph/` 是 LangGraph 工作流（节点见 `nodes.py`，图结构见 `build.py`），`api/app.py` 是 FastAPI 薄层。
目前只有手动冒烟脚本 `eval/smoke.py`（会调真实模型），没有自动化测试。工作流里有几个**容易写错、必须被测试锁住**的行为，这是这个任务的核心。

## 要做的事

### 1. 测试夹具（`tests/conftest.py`、`tests/fakes.py`）
- 假的 `Models`：`triage` 和 `draft` 按预设的脚本返回 `Triage` / `DraftResult` 对象，并记录被调用的次数。
- 假的 `Retriever`：返回预设的 `Hit` 列表（可以为空）。
- 用 `langgraph.checkpoint.memory.InMemorySaver` 或临时目录里的 `SqliteSaver` 构图（`build_graph(..., models=假模型)`）。

### 2. 工作流测试（`tests/test_graph.py`）——每条一个用例
1. 分类后缺必填字段（如 `platform_integration` 但 `platform_name=None`）→ 走 `ask_for_info`，`status="need_info"`，检索器**没有被调用**。
2. 正常回复 + 人工 `approve` → 检索被调用、`draft` 被调用、`send_reply` 发出的内容等于草稿、`status="resolved"`。
3. 人工 `edit` → 发送的是审核人改写的文本，不是草稿。
4. 人工 `reject` → 创建升级单、`status="escalated"`，不发送回复。
5. 检索结果为空 → 不调用 draft 模型，`resolution="escalate"`，理由是"知识库无相关资料"。
6. 模型返回 `answerable=False` → 升级。
7. 模型返回的 `cited_ids` 全是检索结果里不存在的编号 → **升级，不放行**；部分存在则只保留存在的。
8. `severity="critical"` → 升级路由里同时包含对应团队和 `oncall`。
9. `category="platform_integration"` 且 `platform_name`、`shop_name` 都有 → `shop_status` 被写入状态；缺 `shop_name` → 不调用 `check_shop_auth`。
10. **持久化恢复**：用同一个 sqlite 文件、**新建一个**图实例，对第一个图实例停在 interrupt 的 thread 做 `Command(resume=...)`，能继续并正确完成。（这是 checkpoint 的价值所在，必须测。）
11. 同一个 thread 在 `resolved` 之后再 resume 的行为：写一个用例把当前行为固定下来（先观察实际表现，不要猜）。

### 3. API 重构（`api/app.py`，必要时新增 `api/` 下的文件）
- 改成工厂 `create_app(settings=None, models=None, retriever=None, system=None, checkpointer=None) -> FastAPI`，依赖都可注入；保留模块级 `app = create_app()`（`eval/smoke.py` 和 `uvicorn carepilot.api.app:app` 在用）。真正的重资源（读知识库、建图）仍在 lifespan 里做。
- `settings.kb_dir` 的知识库加载改用 `carepilot.retrieval.chunking.load_kb` 和 `BM25Retriever` 这一对**现有**的接口，不要引用 `factory`（别的 worker 在写，还没合并）。
- 现有三个端点的请求/响应格式**不变**。
- **SSE 流式**：`POST /tickets?stream=true` 和 `POST /tickets/{thread_id}/review?stream=true` 返回 `text/event-stream`。图每完成一个节点发一个 `event: node`，数据是 `{"node": "<节点名>", "summary": {...}}`（summary 只放小字段：如 `classify` 给 category 和 severity，`retrieve` 给命中数，`draft_reply` 给 resolution）；最后发一个 `event: result`，数据和非流式响应完全相同。用 `graph.stream(..., stream_mode="updates")`。流中途出现异常要发 `event: error` 再结束，不要让连接挂着。
- 新增 `GET /health` 已有，保留。

### 4. API 测试（`tests/test_api.py`，用 `fastapi.testclient.TestClient` + 假模型）
覆盖：创建 → 查询 → 审核的完整流程；不存在的 thread 返回 404；对非 `awaiting_review` 的 thread 审核返回 409；非法的 `action` 返回 422；SSE 的事件顺序（`node` 若干条在前，`result` 在最后，且 `result` 的内容等于非流式响应）；SSE 中途异常发出 `error` 事件。

### 5. 容器化
`Dockerfile`（多阶段，基于 `python:3.12-slim` 和 `uv`，非 root 运行，`HEALTHCHECK` 调 `/health`）、`docker-compose.yml`（一个 `api` 服务；`./data` 挂载为卷；`env_file: .env`；端口 8000）、`.dockerignore`（排除 `.env`、`.venv`、`data/raw`、`data/kb_real`、`data/index`、`data/models`、`eval/real`、`.git`）。
**本机可能没有 docker**：如果有，跑 `docker compose config` 校验；没有就在报告里明确写"未验证"。不要声称跑通了没跑过的东西。

## 文件归属

`tests/conftest.py`、`tests/fakes.py`、`tests/test_graph.py`、`tests/test_api.py`、`src/carepilot/api/`、`Dockerfile`、`docker-compose.yml`、`.dockerignore`。
**不要改** `graph/` 下的任何文件。如果测试里发现图的逻辑有 bug，**不要自己修**：把该用例标成 `pytest.mark.xfail(reason=...)`，在报告里写清楚复现步骤，由 Claude 修。

## 验收

- `uv run pytest -q` 全绿（xfail 允许，但每个都要在报告里列出）。
- 11 个工作流用例一个不少；你手动把节点里的某个关键判断改坏一次（比如去掉引用校验），确认对应用例会变红，再改回来——在报告里写你验证了哪几个。
- `eval/smoke.py` 不需要你改。

## 不要做

不要调用真实 LLM；不要改 `graph/`、`retrieval/`、`evaluation/`、`pyproject.toml`；不要写 README。
