"""交叉评审：同一批回复，让两个不同家族的模型各评一遍，报各自的比例和两者的
一致率（raw agreement + Cohen's kappa）。

为什么要有这个脚本：
    judge_replies.py 的评审模型和被评的回复是同一个模型（deepseek-v4.1-flash-1017），
    存在自我偏好，README 里那句「数字偏乐观」一直没有度量。这里不换评分标准
    （SYSTEM 提示和 Verdict 结构完全照抄），只换评审人，所以差异只可能来自模型。

它证明什么、不证明什么：
    - 证明：结论不是某一个模型的口味。如果两个家族都判「major 很低」，那个数字
      至少不是单模型幻觉。
    - 不证明：真值。两个 LLM 一致也可能一起错；rubric 本身偏松的问题没被修掉。
    - kappa 是给「一致率」去水分用的：两个模型都 95% 判 none 时，一致率天然就高，
      kappa 会把随机一致的部分扣掉。

用法：
    uv run python eval/judge_cross.py --dataset eval/real/dataset_full.jsonl \
        --kb-dir data/kb_real --judge-b qwen3.8-max eval/real/reports/<run>.items.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from carepilot.config import get_settings
from carepilot.evaluation.schema import load_dataset
from carepilot.llm import StructuredLLM, get_llm
from carepilot.retrieval.chunking import load_kb

# 和 judge_replies.py 逐字一致：换的只是评审人，不是评分标准。
# 脚本目录加进 sys.path 是因为测试会用 spec_from_file_location 加载本文件，
# 那时 sys.path[0] 是仓库根，不是 eval/。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_replies import SYSTEM, Verdict, quote_missing  # noqa: E402

LEVELS = ("none", "minor", "major")


def cohen_kappa(a: list, b: list, labels: tuple) -> float | None:
    """两个标注者的一致性，扣掉随机一致。

    kappa = (po - pe) / (1 - pe)。两人都只用同一个标签时 pe = 1，除零，返回 None
    （此时一致率是 100%，但 kappa 无定义 —— 不要把它当成「完全一致」的证据）。
    """
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def agreement(a: list, b: list) -> float | None:
    if not a or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def rate(values: list, pred) -> float:
    return sum(1 for v in values if pred(v)) / max(1, len(values))


def _sample(rows: list, limit: int) -> list:
    """按 qid 的 md5 排序取前 limit 条 —— 同一个 --limit 每次抽到同一批。"""
    if limit <= 0 or limit >= len(rows):
        return rows
    return sorted(rows, key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--kb-dir", type=Path, required=True)
    ap.add_argument("--judge-b", default="qwen3.8-max", help="第二个评审模型；--judge-a 默认是被评回复的生成模型")
    ap.add_argument("--judge-a", default="", help="留空则用 .env 里的 LLM_FAST_MODEL")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（按 qid 的 md5 稳定抽样），0 = 全部")
    ap.add_argument("--show-disagree", type=int, default=8)
    ap.add_argument("report", type=Path, help="run_pipeline 产出的 .items.jsonl")
    args = ap.parse_args()

    base = get_settings().model_copy(update={"llm_timeout": 120})
    judge_a_model = args.judge_a or base.llm_fast_model
    if judge_a_model == args.judge_b:
        print("两个评审模型相同，那就不是交叉评审了。")
        return 2
    judges = {
        "a": (judge_a_model, StructuredLLM(get_llm(base.model_copy(update={"llm_fast_model": judge_a_model}), "fast"), Verdict)),
        "b": (args.judge_b, StructuredLLM(get_llm(base.model_copy(update={"llm_fast_model": args.judge_b}), "fast"), Verdict)),
    }

    items = {i.qid: i for i in load_dataset(args.dataset)}
    chunks = {c.id: c for c in load_kb(args.kb_dir)}
    rows = [json.loads(line) for line in open(args.report, encoding="utf-8") if line.strip()]
    todo = [
        r
        for r in rows
        if r.get("resolution") == "auto_reply" and r.get("draft") and r.get("citations") and r["qid"] in items
    ]
    total = len(todo)
    todo = _sample(todo, args.limit)
    print(f"待评 {len(todo)}/{total} 条；评审 A = {judge_a_model}，评审 B = {args.judge_b}")

    def ask(judge: StructuredLLM, prompt: str) -> dict:
        try:
            v = judge.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)])
            return {"level": v.unsupported_level, "answers": v.answers, "unsupported": v.unsupported}
        except Exception as exc:  # 单个评审失败不算整体失败，但要在报告里数出来
            return {"error": type(exc).__name__}

    def run(r: dict) -> dict:
        ctx = "\n\n".join(f"[资料{i}] {chunks[c].text}" for i, c in enumerate(r["citations"], 1) if c in chunks)
        prompt = f"【客户问题】\n{r['question']}\n\n【回复】\n{r['draft']}\n\n【资料原文】\n{ctx}"
        return {"qid": r["qid"], "a": ask(judges["a"][1], prompt), "b": ask(judges["b"][1], prompt)}

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(run, todo))

    both = [x for x in res if "error" not in x["a"] and "error" not in x["b"]]
    la = [x["a"]["level"] for x in both]
    lb = [x["b"]["level"] for x in both]
    aa = [x["a"]["answers"] for x in both]
    ba = [x["b"]["answers"] for x in both]

    print(f"\n两边都评成功 {len(both)} 条；失败 A={sum(1 for x in res if 'error' in x['a'])} B={sum(1 for x in res if 'error' in x['b'])}")
    print(f"{'指标':<26}{'评审 A（同族）':>14}{'评审 B（异族）':>14}")
    rows_out = (
        ("无编造 none", rate(la, lambda v: v == "none"), rate(lb, lambda v: v == "none")),
        ("仅无害多余 minor", rate(la, lambda v: v == "minor"), rate(lb, lambda v: v == "minor")),
        ("具体事实 major", rate(la, lambda v: v == "major"), rate(lb, lambda v: v == "major")),
        ("答了所问 answers", rate(aa, bool), rate(ba, bool)),
    )
    for name, va, vb in rows_out:
        print(f"{name:<26}{va:>14.1%}{vb:>14.1%}")

    k_level = cohen_kappa(la, lb, LEVELS)
    k_answers = cohen_kappa(aa, ba, (True, False))
    fmt = lambda v: "无定义" if v is None else f"{v:.3f}"
    print(f"\n一致率（level 三分类）：{fmt(agreement(la, lb))}  kappa：{fmt(k_level)}")
    print(f"一致率（answers 布尔） ：{fmt(agreement(aa, ba))}  kappa：{fmt(k_answers)}")

    # 判了非 none 却摘不出那句话 —— 这段是给「能不能人工复核」用的，不是给好看用的。
    qa = [x for x in both if x["a"]["level"] != "none"]
    qb = [x for x in both if x["b"]["level"] != "none"]
    miss_a = sum(1 for x in qa if quote_missing(x["a"]["level"], x["a"]["unsupported"]))
    miss_b = sum(1 for x in qb if quote_missing(x["b"]["level"], x["b"]["unsupported"]))
    print(f"摘句缺失：A {miss_a}/{len(qa)}，B {miss_b}/{len(qb)}（缺了等于 --show-fail 打出来是废话）")

    disagree = [x for x in both if x["a"]["level"] != x["b"]["level"]]
    print(f"\n--- level 不一致 {len(disagree)} 条，前 {min(args.show_disagree, len(disagree))} 条（请人工读）：")
    for x in disagree[: args.show_disagree]:
        r = next(r for r in rows if r["qid"] == x["qid"])
        print(f"\n[{x['qid']}] A={x['a']['level']} B={x['b']['level']}")
        print(f"    客户: {r['question'][:60]}")
        print(f"    A 指出: {(x['a']['unsupported'] or '')[:100]}")
        print(f"    B 指出: {(x['b']['unsupported'] or '')[:100]}")

    summary = {
        "judge_a": judge_a_model,
        "judge_b": args.judge_b,
        "n": len(both),
        "n_total": total,
        "rates": {
            "a": {"none": rate(la, lambda v: v == "none"), "minor": rate(la, lambda v: v == "minor"),
                  "major": rate(la, lambda v: v == "major"), "answers": rate(aa, bool)},
            "b": {"none": rate(lb, lambda v: v == "none"), "minor": rate(lb, lambda v: v == "minor"),
                  "major": rate(lb, lambda v: v == "major"), "answers": rate(ba, bool)},
        },
        "agreement": {"level": agreement(la, lb), "answers": agreement(aa, ba)},
        "kappa": {"level": k_level, "answers": k_answers},
        "quote_missing": {"a": f"{miss_a}/{len(qa)}", "b": f"{miss_b}/{len(qb)}"},
        "errors": {"a": sum(1 for x in res if "error" in x["a"]), "b": sum(1 for x in res if "error" in x["b"])},
    }
    out = args.report.with_name(args.report.name.replace(".items.jsonl", ".judge_cross.json"))
    out.write_text(json.dumps({"summary": summary, "rows": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
