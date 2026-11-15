"""多轮评估：同一工单连续两轮，第二轮的指代只能靠第一轮理解。

为什么单独一个脚本：`eval_all` 的数据集是单轮的，所以「上下文预算」这项改动
在端到端指标上是盲的（见 README「这次改动的局限」）。这里补的就是那个盲区。

做法：从评估集里取可答样本，把它当第一轮客户原话；第二轮接一句只有指代的
追问（「这个还有别的办法吗」——单看这句话，人也不知道在说什么）。两臂跑同一
批会话，唯一差别是第二轮带不带对话历史：

    历史开：第二轮 classify 能看到第一轮，standalone_question 会把「这个」补成
            原来的问题，检索词因此还能落在同一篇文档上。
    历史关：第二轮只看这一句话，检索词里没有主题。

判分用评估集自带的 gold 标签（第二轮检索到不在同一篇文档上就算没命中），再加
两条确定性检查：有没有可校验引用、有没有重新索要第一轮已经给过的信息。不调
模型评判——同模型自评已经被诟病过一次（见 README），不引入第二个不可信判定器。
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from carepilot.config import ROOT, get_settings
from carepilot.evaluation.report import markdown_table, save_report
from carepilot.evaluation.schema import EvalItem, load_dataset
from carepilot.graph.nodes import make_nodes
from carepilot.graph.routing import after_classify, after_draft
from carepilot.retrieval.base import Chunk, Hit, Retriever
from carepilot.tools import TicketSystem

DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"
DEFAULT_DATASET = ROOT / "eval" / "real" / "dataset_full.jsonl"

# 只有指代、没有主题的第二轮。轮着用，免得十二段会话全是同一句话。
FOLLOWUPS = (
    "具体要在哪儿改",
    "这个设置叫什么名字",
    "步骤能再说一遍吗",
)

# 客户第一轮已经交代过的信息，第二轮再要一次就是没读历史。
_REASK = re.compile(r"(请|麻烦)?(提供|发一下|给一下)[^。！？]{0,12}(单号|订单号|原文|报错|截图)")


class _Recorder:
    """记录本轮检索到的 chunk，用来把引用 id 还原成标题 / 文档 id。"""

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.chunks: dict[str, Chunk] = {}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        hits = self._inner.search(query, k=k)
        for hit in hits:
            self.chunks[hit.chunk.id] = hit.chunk
        return hits


def run_turn(nodes: dict, message: str, history: list[dict]) -> dict[str, Any]:
    """跑一轮 classify →（追问 / 升级 / 检索+起草），路由和图同一份函数。"""
    state: dict[str, Any] = {"ticket_id": "mt", "customer_message": message, "history": history}
    try:
        state.update(nodes["classify"](state))
        route = after_classify(state)
        if route == "ask_for_info":
            state.update(nodes["ask_for_info"](state))
        elif route == "route_escalation":
            state.update(nodes["route_escalation"](state))
        else:
            state.update(nodes["retrieve"](state))
            state.update(nodes["draft_reply"](state))
            if after_draft(state) == "ask_for_info":
                state.update(nodes["ask_for_info"](state))
    except Exception as exc:  # 单条失败不拖垮整批
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "error": None,
        "query": state.get("query") or "",
        "resolution": state.get("resolution"),
        "hits": [h["id"] for h in (state.get("hits") or [])],
        "citations": list(state.get("citations") or []),
        "reply": state.get("draft") or state.get("followup_question") or "",
    }


def judge(item: EvalItem, turn: dict[str, Any], recorder: _Recorder) -> dict[str, Any]:
    """第二轮命中的文档是否还是第一轮那篇（用评估集自带的 gold 标签）。"""
    if turn.get("error"):
        return {"strict_hit": False, "lenient_hit": False, "has_citation": False, "reasks_known": False}
    chunks = [recorder.chunks[h] for h in turn["hits"] if h in recorder.chunks]
    strict = any(c.doc_id in item.gold_doc_ids for c in chunks)
    lenient = any((c.meta.get("title_key") or c.title) in item.gold_title_keys for c in chunks)
    return {
        "strict_hit": strict,
        "lenient_hit": lenient,
        "has_citation": bool(turn["citations"]),
        "reasks_known": bool(_REASK.search(turn["reply"])),
    }


def _rate(rows: list[dict], key: str) -> float | None:
    values = [r["judged"][key] for r in rows if r.get("judged")]
    return round(sum(1 for v in values if v) / len(values), 3) if values else None


def first_turns(pairs: list[tuple[EvalItem, str]], nodes: dict) -> list[dict]:
    """第一轮在两臂里完全一样（没有历史），只跑一次，省一半调用。"""
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda p: run_turn(nodes, p[0].question, []), pairs))


def run_arm(name: str, pairs: list[tuple[EvalItem, str]], firsts: list[dict], nodes: dict,
            recorder: _Recorder, *, with_history: bool) -> dict:
    def one(args: tuple[tuple[EvalItem, str], dict]) -> dict:
        (item, followup), first = args
        history = [] if not with_history else [
            {"role": "customer", "text": item.question},
            {"role": "agent", "text": first.get("reply") or ""},
        ]
        second = run_turn(nodes, followup, history)
        return {
            "qid": item.qid,
            "category": item.category,
            "followup": followup,
            "turn1": {k: first.get(k) for k in ("resolution", "query")},
            "turn2": second,
            "judged": judge(item, second, recorder),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, list(zip(pairs, firsts, strict=True))))
    summary = {
        "arm": name,
        "n": len(rows),
        "errors": sum(1 for r in rows if (r["turn2"] or {}).get("error")),
        "strict_hit": _rate(rows, "strict_hit"),
        "lenient_hit": _rate(rows, "lenient_hit"),
        "has_citation": _rate(rows, "has_citation"),
        "reasks_known": _rate(rows, "reasks_known"),
        "answered": round(sum(1 for r in rows if r["turn2"].get("resolution") == "auto_reply") / len(rows), 3),
    }
    return {"summary": summary, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多轮评估：第二轮带不带历史，各跑一遍。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kb-dir", type=Path, default=None, help="知识库目录，默认取 settings.kb_dir")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--limit", type=int, default=12, help="跑多少段会话（等距抽样可答样本）")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--name", default="multiturn")
    return parser


def _pairs(path: Path, limit: int) -> list[tuple[EvalItem, str]]:
    items = [it for it in load_dataset(path) if it.answerable and it.gold_title_keys]
    if not items:
        raise SystemExit(f"{path} 里没有可答样本")
    step = max(1, len(items) // limit)
    picked = items[::step][:limit]
    return [(item, FOLLOWUPS[i % len(FOLLOWUPS)]) for i, item in enumerate(picked)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    from carepilot.retrieval.factory import build_retriever

    kb_dir = args.kb_dir or Path(settings.kb_dir)
    retriever = build_retriever(args.retriever, kb_dir, settings=settings)
    recorder = _Recorder(retriever)
    nodes = make_nodes(settings, recorder, TicketSystem(), None)
    pairs = _pairs(args.dataset, args.limit)

    firsts = first_turns(pairs, nodes)
    arms = [
        run_arm("history_on", pairs, firsts, nodes, recorder, with_history=True),
        run_arm("history_off", pairs, firsts, nodes, recorder, with_history=False),
    ]
    rows = []
    for a in arms:
        s = a["summary"]
        rows.append([s["arm"], s["n"], s["lenient_hit"], s["strict_hit"], s["has_citation"],
                     s["answered"], s["reasks_known"], s["errors"]])
    print(markdown_table(
        ["臂", "n", "命中同文档(宽)", "命中同文档(严)", "有可校验引用", "自动回复", "重新索要已知信息", "错误"],
        rows,
    ))
    for a in arms:
        missed = [r["qid"] for r in a["rows"] if not (r["judged"] or {}).get("lenient_hit")]
        print(f"{a['summary']['arm']} 第二轮没落在原文档上：{len(missed)}/{len(a['rows'])}")

    json_path, items_path = save_report(
        args.report_dir,
        args.name,
        {a["summary"]["arm"]: a["summary"] for a in arms},
        [r for a in arms for r in a["rows"]],
        {"kb_dir": str(kb_dir), "retriever": args.retriever, "dataset": str(args.dataset),
         "conversations": len(pairs)},
    )
    print(f"明细：{json_path} / {items_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
