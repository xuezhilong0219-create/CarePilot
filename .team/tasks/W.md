# 任务 W：产品线陷阱评测集生成器 + 对网关的独立黑盒对抗测试

负责人：DeepSeek Harness（dsh）。先读 `.team/TEAM.md`（运行环境限制同之前：不要 git commit、不要用 uv，测试用 `.venv/Scripts/python.exe -W ignore -m pytest -q -p no:cacheprovider`）和 `docs/requirements.md`（重点看 A2、C2–C4）。

## 部分一：产品线陷阱评测集（A2）

背景：知识库里企业版和极速版有 67 组**同名但内容不同**的文档。客户明确说"我们用的是极速版"时，回复必须引用极速版的文档，不能引用企业版的同名文档（反之亦然）。要构造能"咬住"这个错误的评测集。

### 要做的事（`eval/gen_edition_traps.py`、`src/carepilot/evaluation/edition_traps.py`）
- 复用 `src/carepilot/evaluation/gen_core.py` 里已有的 `load_manifest`、`ManifestEntry`、`CachedGenerator`、`GeneratedTicket`、`check_answerable`、`read_doc`、`StructuredLLM` 用法（先读 `eval/gen_dataset.py` 看它是怎么调用的，保持同样的缓存和并发方式）。**不要改 `gen_core.py`**，在你的文件里 import 使用。
- 纯函数（`edition_traps.py`，要有测试）：
  - `find_edition_pairs(entries) -> list[tuple[ManifestEntry, ManifestEntry]]`：按 `title_key` 找 `top_dir == "企业版"` 与 `top_dir == "极速版2.0"` 各一篇的配对；两篇都 `kept`、`chars >= 300`、flags 不含 `release_note` 和 `short`；同一个 `title_key` 在某一侧有多篇时跳过该组（有歧义）。
  - `unique_lines(doc_a: str, doc_b: str, *, min_len=10, cap=15) -> list[str]`：在 `doc_a` 里出现、但**不在 `doc_b` 里**的行（去首尾空白后比较，忽略空行和纯标点行、markdown 标题行），保持原顺序，最多 `cap` 条。
  - `is_discriminative(a, b, *, min_unique=3) -> bool`：两个方向的 `unique_lines` 都 ≥ `min_unique` 条才算"能区分"的配对（太相似的文档生成不出有区分度的问题）。
- 生成：对每个能区分的配对、**每一侧**各生成 1 条问题，共 `2 × n_pairs` 条。提示词要点：
  1. 你是使用旺店通 {edition} 的商家，给售后写一条消息；
  2. **必须在消息里自然地说出自己用的版本**（如"我们用的极速版…"），不能只放在括号里；
  3. 问题必须只能靠【只出现在这个版本文档里的内容】回答——把 `unique_lines` 的结果作为"本版本特有内容"提供给模型，要求问题的答案取自这些行；
  4. 15–80 字，口语化，不能照抄任何一行原文（沿用 `check_answerable` 的 12 字防照抄校验，参照的"正文"用该版本文档）；
  5. 不要出现"文档""资料"等词。
  结构化输出用 `GeneratedTicket`（沿用它的字段）。拒收后最多重试 2 次，仍失败就丢掉该配对并记录原因。
- 命令行：`--kb-dir data/kb_real --n-pairs 30 --seed 42 --model deepseek-v4.1-flash-1017 --concurrency 8 --out eval/real/dataset_edition_traps.jsonl [--dry-run]`。`--dry-run` 不调用 LLM，打印配对统计（总配对数、可区分配对数、抽中的配对标题）和一条完整的提示词样例。
- 输出每行：`{qid, question, edition, gold_doc_id, forbidden_doc_ids: [另一版本同名文档的 doc_id], title_key, unique_lines_used: [...] (最多 5 条)}`，`qid` 形如 `e0001`。
- 同时写 `eval/real/dataset_edition_traps_review.md`：随机 12 条，列出问题、版本、金标文档路径、被禁止的文档路径、用到的特有行，供人工抽查。
- **缓存**：沿用 `CachedGenerator`，缓存目录 `eval/real/cache/gen`，重跑不重复花钱。

### 验收
- 纯函数测试：配对规则（一侧多篇时跳过、缺一侧时跳过、release_note 排除）；`unique_lines` 的各种边界（空、完全相同、标题行不计、cap 生效、顺序保持）；`is_discriminative`。
- 生成流程用**假 LLM** 测试（参照 `tests/test_gen_dataset.py` 的做法）：断言每个配对产出两条、方向正确（`gold_doc_id` 与 `edition` 对应、`forbidden_doc_ids` 是另一侧）、问题里**包含版本名**（不含则拒收）、防照抄拒收生效、同 seed 一致、缓存命中不再调用。
- 在真实知识库上 `--dry-run` 跑一次（只读 `D:/ORCApj/CarePilot/data/kb_real`），把"总配对数 / 可区分配对数"写进报告。**不要调用真实 LLM**（你也没有 key）。

## 部分二：对网关的独立黑盒对抗测试（C2–C4）

背景：`src/carepilot/gateway/` 是另一个 worker 写的工具网关，它自己也写了测试，但作者的测试可能带着同样的盲区。你要**只根据接口说明**独立写一套攻击性测试。

### 硬性规则
- **只允许阅读** `src/carepilot/gateway/spec.py`、`src/carepilot/auth.py`、`docs/requirements.md` 里 C2–C4 的描述。**不要读** `gateway/gateway.py`、`ledger.py`、`audit.py`、`guard.py`，也不要读 `tests/gateway/` 里的任何文件。读了会让你的测试带上作者的假设，失去独立性。如果你不确定某个行为（比如某种情形下应该返回什么），按 `spec.py` 里的文档注释推断，推断不出来就在报告里写"接口未规定"，不要去翻实现。
- 用 `from carepilot.gateway.gateway import ToolGateway`（构造签名：`ToolGateway(db_path, registry, *, clock=time.time, approval_ttl_s=86400.0)`）。

### 要做的事（`tests/gateway_adversarial/test_blackbox.py`，必要时加 `conftest.py`）
自己定义一组 `ToolSpec`（用你自己写的带调用计数的假工具函数，覆盖 `Risk.READ / WRITE_INTERNAL / STATE_CHANGE / EXTERNAL_SEND` 四档，参数模型 `extra="forbid"`），然后：

1. **不变量的随机序列测试**（用 `random.Random(seed)`，不要装 hypothesis）：生成 ≥300 条随机操作序列（每条 5–25 步），每步随机选择：以随机角色（customer / agent / admin）调用随机工具（含不存在的工具、缺字段、多余字段、类型错误的参数、重复的参数）、批准、确认、拒绝（含对不存在的 approval_id 操作、用 customer 身份批准）。每条序列用独立的临时 db。**序列结束后检查这些不变量，任何一条不成立就让测试失败并打印出能复现的 seed 和操作序列**：
   - (I1) 对同一个 `(run_id, 工具名, 规范化参数)`，工具函数**至多被真正调用一次**（READ 重试情形除外——只读工具允许在工具报错后重试，你的假工具不要抛错就能避开）；
   - (I2) `STATE_CHANGE` / `EXTERNAL_SEND` 工具的副作用发生之前，审计里必须已经有对应的批准记录；
   - (I3) `EXTERNAL_SEND` 的副作用只在**批准且确认之后**发生；
   - (I4) 每一次 `call` / `approve` / `confirm` / `reject`（包括被拒绝的）在审计里**至少留下一条记录**；
   - (I5) 标记为敏感的参数（比如一个叫 `phone` 的字段，值是 `13812345678`，工具的 `summarize_args` 返回脱敏串）的**明文**不出现在审计里任何一列；
   - (I6) 被拒绝（`Rejected`）的调用，其工具函数**没有**被执行。
2. **并发测试**：同一个 db、同一个键，用 `threading.Barrier` 让 16 个线程同时 `call`→`approve`（STATE_CHANGE）和 `confirm`（EXTERNAL_SEND），重复 30 轮，断言每轮副作用恰好 1 次；多个**不同**的键并发时互不阻塞到永久死锁（整轮必须在 30 秒内完成）。
3. **定向攻击用例**（每个都写成独立的测试函数，名字说明攻击意图）：客户尝试批准自己的 `STATE_CHANGE`；坐席批准后换一份参数去 `confirm`（参数应以批准时存下的为准）；对同一个 `approval_id` 先 `reject` 再 `approve`；对 `READ` 工具调用 `confirm`；用另一个 run_id 重放同一个 approval_id；参数里放超长字符串（100KB）、嵌套很深的结构、`NaN`/`Infinity`、`\x00`、非 BMP 字符（emoji）——必须是干净的 `Rejected` 或正常处理，**不能抛出未捕获的异常**，也不能让后续调用失败；审计查询的 `limit` 为 0、负数、超大值。
4. **发现了网关的 bug 时**：不要改网关。把对应用例标成 `pytest.mark.xfail(strict=True, reason=...)`，并在报告里写清楚复现步骤和你认为的正确行为。**这是这部分最有价值的产出**，别为了让测试变绿而放宽断言。

### 验收
- 全部测试通过（`xfail(strict=True)` 允许，但每一个都要在报告里列出并解释）。
- 测试运行时间：整套 < 120 秒（`pytest` 的 `-p no:cacheprovider` 下）。
- 报告里写：随机序列测试的序列数和总步数、发现的 bug 数量（0 也要如实写，并说明你尝试过哪些攻击面）、"接口未规定"的行为清单。

## 文件归属

`eval/gen_edition_traps.py`、`src/carepilot/evaluation/edition_traps.py`、`tests/test_edition_traps.py`、`tests/gateway_adversarial/`。
不要改 `gen_core.py`、`gateway/`、`store/`、`sandbox/`、`graph/`、`api/`、`observability/`、`pyproject.toml`、`tests/gateway/`。

## 不要做

不要调用真实 LLM；不要联网；不要读被禁止阅读的网关实现文件。
