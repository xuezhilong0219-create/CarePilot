# 任务 E2b：评估集增加「报错原文」样本

负责人：DeepSeek Harness（dsh），继续你在 E2 里写的 `gen_dataset.py` / `gen_core.py`。先读 `.team/TEAM.md`，运行环境限制同 E2（不要 git commit、不要用 uv，测试用 `.venv/Scripts/python.exe -W ignore -m pytest -q -p no:cacheprovider`）。

## 背景

用户说真实客户的典型查询方式是**拿报错原文、平台名、功能名直接搜**。但当前生成的 150 条可答样本里只有 9 条带 `error_text`，评估集严重欠缺最关键的查询类型。
知识库里有现成的报错原文，例如 `📒常见问题库/库存同步常见报错及解决方案` 里的：`库存同步失败提示“商品服务调用失败 #isv.item-quantity-item-update-service-error”`、`oauth-error-code[30111]description[access token invalid]`。

## 要做的事

1. **`gen_core.py` 新增 `extract_error_candidates(doc_text: str) -> list[str]`**（纯函数）：从文档里抽出"像报错提示"的原文串，要求**逐字来自文档**。两类：
   - 引号内的片段：`“…”` 或 `"…"`，长度 6 到 80 个字符，且在它前面 30 个字符以内出现 `报错|提示|失败|异常|错误|返回` 之一；
   - 英文/代码形态的报错标识：包含 `error|fail|invalid|exception|code`（不分大小写）、长度 ≥ 12、由字母数字和 `-_.#:[]` 组成的串（如 `oauth-error-code[30111]`、`isv.item-quantity-item-update-service-error`）。
   - 去重，保持出现顺序；过滤掉纯数字、纯标点、含换行的。
2. **`plan_error_samples(entries, kb_dir, n, seed, exclude_doc_ids)`**：候选 = 与可答样本同样的过滤条件（`candidate_docs`），再读取文档、要求 `extract_error_candidates` 非空；**排除已被可答样本选中的文档**；分层方式沿用 `allocate_quotas` 的 sqrt 规则；每篇文档用独立种子从候选报错串里随机选一条。
3. **生成**：新增提示词 `build_error_prompt(entry, doc_text, error_text)`：告诉模型"商家遇到了这条报错，请把下面这句报错原文**逐字**贴进消息里，再用一两句口语描述当时在做什么、有什么现象；消息 20 到 100 字；不要照抄文档里其他句子"。输出沿用 `GeneratedTicket`。校验沿用 `check_answerable`（它已经把 `error_text` 排除在防照抄比对之外），**另加强制检查**：`ticket.error_text` 必须**等于**我们给它的那条报错串（不是模型自己改写的版本），否则拒收。
4. **命令行**：`gen_dataset.py` 增加 `--n-error`（默认 0，这样不影响现有行为）。这些样本 `answerable=True`、`gold_doc_ids=[doc_id]`、`gold_title_keys=[title_key]`、`error_text=<那条报错串>`，`category` 仍用 `category_for`。`qid` 接在可答和不可答之后继续编号。`--dry-run` 要同时打印报错样本的抽样结果和每个样本选中的报错串。
5. 同时修改 `dataset_review.md` 的输出：报错样本单独列一节，最多抽 10 条。

## 文件归属

`eval/gen_dataset.py`、`src/carepilot/evaluation/gen_core.py`、`tests/test_gen_dataset.py`（只追加，别删现有用例）。

## 验收

- 全部测试通过（现有 27 条不能变红）。
- 新增测试：`extract_error_candidates`——上面两个真实例子能被抽出、纯数字和无关键词的引号串不被抽出、结果是文档的逐字子串；`plan_error_samples`——排除已选文档、seed 可复现、没有报错串的文档不会入选；强制检查——模型改写了报错串时被拒收；`--n-error 0` 时输出与改动前完全一致。
- 用 `data/kb_real` 之外的夹具做 dry-run 即可，**不要调用真实 LLM**（你也没有 key）。如果你的 worktree 里能读到 `D:/ORCApj/CarePilot/data/kb_real`（只读），额外在真实知识库上跑一次 `--dry-run --n-error 40`，把抽到的前 10 条报错串贴进报告，我要看它们像不像真的报错。

## 不要做

不要调用真实 LLM；不要改其他文件；不要改 `schema.py`。
