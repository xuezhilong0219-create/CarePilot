# CarePilot 团队协作规则

CarePilot 是一个面试作品：旺店通 ERP 的**售后工单助手**。LangGraph 编排（分类 → 缺信息追问 → 检索 → 带引用回复 → 人工审核 interrupt → 执行动作）+ 自建 RAG + FastAPI。
卖点是**可评估、可控、能讲清楚取舍**，不是功能堆砌。所以：每个决定要有数据或理由，宁可少做也不要做出没验证过的东西。

你是团队里的一个 worker，只负责分配给你的那份任务（`.team/tasks/<你的编号>.md`）。统筹由 Claude 负责。

## 铁律

1. **只改你任务文件里"文件归属"列出的路径。** 别人在并行改别的文件，越界会造成合并冲突。
2. **不改 `pyproject.toml` / `uv.lock`。** 缺依赖就在报告里写明，由 Claude 统一加。只用已安装的依赖和标准库。
3. **不提交任何真实数据或衍生物**：`.env`、`data/raw/`、`data/kb_real/`、`data/index/`、`data/models/`、`eval/real/`（都已在 `.gitignore`）。真实知识库是内部资料。
4. **不 push，不动别的 worktree，不动主仓库目录。** 你在自己的 git worktree 和分支里工作。
5. **单元测试不能依赖网络、API key 或真实知识库**，用 `data/kb/` 的示例数据和你自己造的小夹具。需要真实模型/数据的验证，单独写成脚本或用环境变量开关跳过的测试。
   你的 worktree 里**没有 `.env`**：测试里构造配置要显式传假值，`Settings(llm_api_key="test", llm_base_url="http://localhost:0/v1")`，不要调用 `get_settings()`。
6. **完成标准**：`uv run pytest -q` 全绿 → 在你自己的分支上 `git add` 你的文件并 `git commit`（英文一行描述）→ 输出 REPORT。测试没过就不要提交，如实写在报告里。
7. 遇到任务说明没覆盖、或说明有矛盾的地方：选最简单的合理做法继续，并写进报告的"需要决定的事"。不要停下来等。
8. 不要写样板注释和长篇 docstring。注释只解释"为什么这么做"。类型标注要有。中文注释即可。

## 运行方式

- 运行：`uv run python -W ignore <脚本>`；测试：`uv run pytest -q`。首次进 worktree 先 `uv sync`。
- Windows 上终端输出中文要设 `PYTHONIOENCODING=utf-8`；读写文件一律显式 `encoding="utf-8"`。
- 原始知识库（只读，绝对路径）：`D:/ORCApj/CarePilot/data/raw/慧策wiki-2`
- 网关是 OpenAI 兼容接口，配置在 `.env`（`LLM_API_KEY` / `LLM_BASE_URL`），通过 `carepilot.config.get_settings()` 读取。**不要把 key 写进任何文件或输出。**
- HuggingFace 直连不通，用镜像：`HF_ENDPOINT=https://hf-mirror.com`。

## 冻结的接口（不能改，要改请在报告里提）

**检索**（`src/carepilot/retrieval/base.py`）：`Chunk(id, title, text, doc_id, meta)`、`Hit(chunk, score)`、`Retriever.search(query, k) -> list[Hit]`。
`Chunk.meta` 的键：`top_dir`（顶层目录名）、`title_key`、`flags`（逗号分隔字符串）、`source_url`。

**检索器工厂**（`src/carepilot/retrieval/factory.py`，由任务 R 实现，任务 E1 按这个签名调用）：

```python
RETRIEVER_KINDS = ("bm25", "vector", "hybrid", "hybrid_rerank_ce", "hybrid_rerank_llm")

def build_retriever(
    kind: str,
    kb_dir: Path,
    *,
    index_dir: Path | None = None,        # 向量索引目录，None 则用 data/index/<kb_dir 名>
    exclude_dirs: Sequence[str] = (),     # 按 top_dir 排除
    exclude_flags: Sequence[str] = (),    # 按 manifest flags 排除
    settings: Settings | None = None,
) -> Retriever: ...
```

**知识库目录格式**（由任务 P 产出，任务 R 读取）：

```
<kb_dir>/manifest.jsonl     每个原始文档一行，所有文档都在（包括被丢弃的）
<kb_dir>/docs/<doc_id>.md   只有 kept=true 的文档，内容是清洗后的 markdown
```

`doc_id` = 原始文件相对 `慧策wiki-2/` 的路径，POSIX 斜杠，去掉 `.md`，如 `平台对接_店铺授权/淘宝`。
manifest 每行：

```json
{"doc_id": "...", "title": "淘宝", "title_key": "淘宝", "top_dir": "平台对接_店铺授权",
 "source_url": "https://www.yuque.com/...", "chars": 1234,
 "kept": true, "drop_reason": null, "dup_of": null, "flags": ["release_note", "short", "dup_title"]}
```

`drop_reason` 取值：`tiny` | `exact_dup`（此时 `dup_of` 是保留下来的那份的 doc_id）。
`title_key`：NFKC 归一化 → 去掉结尾的 `(2)`/`（2）` 这类副本后缀 → 去掉所有空白 → 小写。
**目录没有 manifest.jsonl 时**（如 `data/kb/` 的示例数据），`load_kb` 退化为直接读所有 `*.md`。

**图的节点**（`src/carepilot/graph/nodes.py`）：`make_nodes(settings, retriever, system, models=None)` 返回 `dict[str, Callable[[TicketState], dict]]`，
键为 `classify / ask_for_info / retrieve / draft_reply / route_escalation / human_review / execute_actions`；`Models(triage, draft)` 可注入假模型；`build_query(triage, message)` 是检索词的唯一来源。
`TicketState` 字段见 `graph/state.py`。

**评估集格式**：`src/carepilot/evaluation/schema.py` 的 `EvalItem`（含严格/宽松两档答案对齐的说明）。

## REPORT 格式（完成后作为你最后的输出，控制在 40 行内）

```
## REPORT <任务编号>
状态: 完成 | 部分完成 | 失败
提交: <分支名> <commit hash>
做了什么: 3-6 行
验证: 跑了什么命令、结果（贴 pytest 汇总行；有实测数据就贴数据）
改动文件: 列表
遗留问题 / 与说明不一致之处:
需要 Claude 决定的事:
```


## 第二阶段补充（治理层）

- 新增冻结接口：`src/carepilot/auth.py`、`src/carepilot/gateway/spec.py`。**不能改**，需要改就写进报告。
- 这一阶段每个 worker 只做**自包含模块**（网关 / 工单存储 / 业务沙箱），**不改 `graph/`、`api/`**，集成由 Claude 做。各自的文件归属见任务说明，越界会造成冲突。
- 这些模块里**不能有任何 LLM 调用**；测试不联网、不依赖 `.env`。
- SQLite 测试文件放 `tmp_path`；如果你的沙箱不允许用 `tmp_path`，放在仓库根的 `.tmp_test/`（已 gitignore），测试结束要清理。
- 需求背景和验收口径在 `docs/requirements.md`；那里写的"谁做"只对你的任务编号负责，别替别人做。
