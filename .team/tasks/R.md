# 任务 R：检索层（BM25 优化 + 向量 + 混合 + 重排）

负责人：Codex。先读 `.team/TEAM.md`。

## 背景

真实知识库约 4000 篇、预计切出 2 到 4 万个 chunk。客户的查询方式是**直接拿报错原文、平台名、功能名搜**，所以精确匹配（BM25）很重要，但同义表达需要向量补充。
现有的 `BM25Retriever` 每次查询遍历全部 chunk，规模上去会很慢，也没有标题加权。

## 要做的事

1. **`load_kb(kb_dir, exclude_dirs=(), exclude_flags=()) -> list[Chunk]`**（放 `chunking.py`）：有 `manifest.jsonl` 就读 `kept=true` 的 `docs/<doc_id>.md`，填好 `doc_id` 和 `meta`（键见 TEAM.md）；没有 manifest 就读所有 `*.md`（示例数据）。`exclude_*` 按 manifest 的 `top_dir` / `flags` 过滤。
2. **分块**（`chunking.py`）：按标题层级切；超长的段按空行切，目标 `max_chars` 默认 600，表格和代码块不要从中间切断；过短的块并入相邻块。每个 chunk 的 `text` 以面包屑开头 `文档标题 > 小节标题`，让 chunk 脱离上下文也能被检索到和被读懂。chunk id 为 `{doc_id}#{序号}`。
3. **BM25**（`bm25.py`，保持 `BM25Retriever(chunks, ...)` 构造方式兼容，现有测试要继续通过）：
   - 改成**倒排索引**；3 万个 chunk 建索引 < 60 秒，单次查询 < 50 ms。
   - 标题/小节标题加权（参数可调，默认标题词权重翻倍或等价做法）。
   - jieba 加载领域词典 `src/carepilot/retrieval/userdict.txt`（你从 `D:/ORCApj/CarePilot/data/raw/慧策wiki-2` 的目录名和文件名里挑出平台名、物流商名、功能名，几十到几百条即可，**不要整库灌进去**）；报错码、型号、英文数字串保持整体不拆。
   - 提供 `tokenizer="jieba" | "bigram"` 两种分词，bigram 是中文字二元组，用于对比报错原文的精确匹配效果。
4. **向量检索**：
   - `embedding.py`：`Embedder` 协议（`encode(texts) -> np.ndarray`，已归一化）和 `FastEmbedEmbedder`，默认模型 `BAAI/bge-small-zh-v1.5`。模型缓存到 `data/models/`；下载前若未设置 `HF_ENDPOINT` 则默认设为 `https://hf-mirror.com`。分批编码，显示进度。
   - `vector.py`：`VectorIndex`（numpy float32 矩阵 + chunk id 列表，落盘到 `index_dir`：`embeddings.npy` + `chunk_ids.json`）和 `VectorRetriever`。余弦相似度，用 `argpartition` 取 top-k。索引文件里记录 embedder 名称和 chunk 数，加载时和当前知识库不一致要报明确的错。
   - 3 万 chunk 规模用 numpy 暴力检索就够，这是有意的选择（零运维），保持 `VectorIndex` 接口干净以便将来换向量库。
5. **混合检索**（`hybrid.py`）：`HybridRetriever(retrievers, weights=None, rrf_k=60, max_chunks_per_doc=2)`，用 RRF 融合；`max_chunks_per_doc` 防止一篇长文占满结果。
6. **重排**（`rerank.py`）：`Reranker` 协议 `rerank(query, hits, k) -> list[Hit]`；
   - `CrossEncoderReranker`：用 fastembed 的 `TextCrossEncoder`。先用 `TextCrossEncoder.list_supported_models()` 看有哪些，选一个支持中文的（优先 `BAAI/bge-reranker-base`）；如果没有可用的中文 cross-encoder，实现留空并在报告里说明，不要硬塞英文模型。
   - `LLMReranker(score_fn)`：`score_fn(query, passages) -> list[float]` 由外部注入（单测用假函数）；另提供 `make_llm_score_fn(settings)`，用 `carepilot.llm.get_llm(settings, "fast")` + `StructuredLLM` 做列表式打分（一次调用给全部候选打 0 到 10 分）。
   - `RerankedRetriever(base, reranker, fetch_k=20)`。
7. **`factory.py`**：按 TEAM.md 里冻结的签名实现 `build_retriever` 和 `RETRIEVER_KINDS`。向量索引不存在时自动构建并落盘。
8. **`scripts/build_index.py`**：`--kb-dir --out --exclude-dirs --exclude-flags`，构建并保存向量索引，打印 chunk 数、耗时、吞吐。

## 文件归属

`src/carepilot/retrieval/`（**除 `base.py` 外**）、`scripts/build_index.py`、`tests/retrieval/`（含 `__init__.py` 或 conftest 如需要）。
`tests/test_retrieval.py` 是现有测试，你可以把里面的用例迁到 `tests/retrieval/` 并删除原文件。

## 验收

- `uv run pytest -q` 全绿。单测用**假 embedder**（对文本哈希出确定的向量），不联网。
- 一个用环境变量 `RUN_MODEL_TESTS=1` 才运行的集成测试：真实下载 bge-small，验证"库存没同步上去"能检索到示例知识库里的"库存同步失败"，且这个查询 BM25 单独找不到时向量能找到（如果你找不到这样的例子，挑一个同义改写的查询即可）。**这个测试你要自己跑一次**（`RUN_MODEL_TESTS=1`），确认镜像下载可行。
- 倒排 BM25 与旧实现在示例数据上**排序结果一致**（相同分词配置下），写成测试。
- 性能：用脚本造 3 万个随机 chunk，测 BM25 建索引时间和查询 p50/p95、向量检索 p50/p95，写进报告。
- 报告里写：embedding 在本机的吞吐（chunk/秒）、模型体积、首次加载耗时；cross-encoder 选了哪个及理由。

## 不要做

不要引入新的依赖；不要做 ima 适配器；不要改 `base.py`、`graph/`、`api/`；不要做评估脚本（别的 worker 在做）。
