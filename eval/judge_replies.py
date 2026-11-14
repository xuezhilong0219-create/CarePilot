"""回复质量检查：对 run_pipeline 里被自动放行的回复，逐条让模型只依据「被引用的资料原文」判断两件事：
  faithful    回复里的每条事实性陈述，都能在被引用的资料里找到依据（没有编造）
  answers     回复回答了客户问的问题

局限（面试时要主动讲）：
  - 评审模型和被评的回复是同一个模型（deepseek-v4.1-flash-1017），存在自我偏好，数字偏乐观；
  - 它只检查「回复 ↔ 被引用资料」，不检查「被引用资料本身对不对」「有没有更合适的文档」；
  - 所以这里的数字用来发现明显的编造，不是准确率；用 --show-fail 把没通过的拉出来人工读才是重点。

用法：
    uv run python eval/judge_replies.py --dataset eval/real/dataset_full.jsonl --kb-dir data/kb_real eval/real/reports/*_ds_after.items.jsonl
"""

import argparse
import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from carepilot.config import get_settings
from carepilot.evaluation.schema import load_dataset
from carepilot.llm import StructuredLLM, get_llm
from carepilot.retrieval.chunking import load_kb

SYSTEM = """你是严格的审核员。给你【客户问题】【回复】和回复所引用的【资料原文】。只依据【资料原文】判断，不要用你自己的知识。
unsupported_level 的含义（取最严重的一处）：
- "none"：回复里的事实性陈述都能在资料里找到依据。
- "minor"：只有无害的泛泛内容没有依据，例如「建议及时操作」「如有疑问请联系客服」「界面可能略有差异」这类提醒或客套，
  或把资料里的步骤换了说法、补了显而易见的连接语；客户照着做不会出错。
- "major"：出现资料里没有、或与资料矛盾的【具体事实】：菜单路径、字段名、编码、数字、时间、条件、状态名、是否支持某项能力。
  客户照着做可能出错的，一律是 major。
answers：回复是否直接回答了客户问的问题；只复述了背景、没回答所问的填 false。
leaks_ref：回复正文里是否出现「资料1」「[2]」这类内部资料编号。"""


class Verdict(BaseModel):
    unsupported_level: Literal["none", "minor", "major"] = Field(description="见系统提示")
    unsupported: str = Field("", description="level 不是 none 时，必须逐字摘出回复里没有依据的那一句话（不是写 minor/major 这类词）；none 时留空")
    answers: bool = Field(description="回复直接回答了客户的问题")
    leaks_ref: bool = Field(False, description="回复正文里出现内部资料编号")


_REF = re.compile(r"资料\s*[\[【(]?\s*\d|[\[【]\s*\d{1,2}\s*[\]】]")


def r_leak(text: str) -> bool:
    return bool(_REF.search(text or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--kb-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--show-fail", type=int, default=12, help="打印多少条没通过的，供人工阅读")
    ap.add_argument("report", type=Path, help="run_pipeline 产出的 .items.jsonl")
    args = ap.parse_args()

    settings = get_settings().model_copy(update={"llm_timeout": 120})
    judge = StructuredLLM(get_llm(settings, "fast"), Verdict)
    items = {i.qid: i for i in load_dataset(args.dataset)}
    chunks = {c.id: c for c in load_kb(args.kb_dir)}
    rows = [json.loads(line) for line in open(args.report, encoding="utf-8") if line.strip()]
    todo = [r for r in rows if r.get("resolution") == "auto_reply" and r.get("draft") and r["qid"] in items]

    def run(r: dict) -> dict:
        ctx = "\n\n".join(f"[资料{i}] {chunks[c].text}" for i, c in enumerate(r["citations"], 1) if c in chunks)
        try:
            v = judge.invoke([SystemMessage(SYSTEM), HumanMessage(f"【客户问题】\n{r['question']}\n\n【回复】\n{r['draft']}\n\n【资料原文】\n{ctx}")])
            return {"qid": r["qid"], "level": v.unsupported_level, "answers": v.answers, "unsupported": v.unsupported, "error": None}
        except Exception as exc:
            return {"qid": r["qid"], "error": type(exc).__name__}

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(run, todo))
    ok = [x for x in res if not x["error"]]
    by_qid = {r["qid"]: r for r in rows}
    split = lambda q: "dev" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 2 == 0 else "test"
    gold = lambda x: bool(set(by_qid[x["qid"]]["cited_doc_ids"]) & set(items[x["qid"]].gold_doc_ids))

    def line(sub):
        n = max(1, len(sub))
        c = lambda f: sum(1 for x in sub if f(x))
        return (f"n={len(sub):3d} | 无编造(none) {c(lambda x: x['level']=='none')/n:5.1%} | 仅无害多余(minor) {c(lambda x: x['level']=='minor')/n:5.1%}"
                f" | 具体事实无依据(major) {c(lambda x: x['level']=='major')/n:5.1%} | 答了所问 {c(lambda x: x['answers'])/n:5.1%}")

    print(f"被评的自动放行回复 {len(todo)} 条，评审出错 {len(res) - len(ok)} 条")
    for name, sub in (("全部", ok), ("dev 半边", [x for x in ok if split(x["qid"]) == "dev"]),
                      ("test 半边", [x for x in ok if split(x["qid"]) == "test"]),
                      ("引用命中标准答案", [x for x in ok if gold(x)]), ("引用未命中标准答案", [x for x in ok if not gold(x)])):
        print(f"  {name:<10} {line(sub)}")
    leak = [x for x in ok if r_leak(by_qid[x["qid"]]["draft"])]
    print(f"  正则统计：回复正文含内部资料编号 {len(leak)}/{len(ok)} = {len(leak) / max(1, len(ok)):.1%}")

    fails = [x for x in ok if x["level"] == "major"]
    print(f"\n--- major 共 {len(fails)} 条，前 {min(args.show_fail, len(fails))} 条（请人工读，别只看数字）：")
    for x in fails[: args.show_fail]:
        r = by_qid[x["qid"]]
        print(f"\n[{x['qid']}] ({split(x['qid'])}) 客户: {r['question'][:70]}")
        print(f"    评审指出无依据: {x['unsupported'][:140]}")
        print(f"    回复: {(r['draft'] or '')[:150].replace(chr(10), ' ')}")
    out = args.report.with_name(args.report.name.replace(".items.jsonl", ".judge.json"))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
