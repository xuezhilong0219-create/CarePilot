"""回复节点的 A/B：固定检索结果，只换提示词/引用方式，看放行率、引用命中率、错误放行率。

只调回复节点，不跑分类，所以比全流程便宜得多，适合迭代提示词。
    uv run python eval/ab_draft.py --dataset eval/real/dataset.jsonl --kb-dir data/kb_real --limit 60
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from carepilot.config import get_settings
from carepilot.evaluation.schema import load_dataset
from carepilot.graph.nodes import DRAFT_PROMPT, build_query, resolve_citations
from carepilot.graph.state import DraftResult
from carepilot.llm import StructuredLLM, get_llm
from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.chunking import load_kb

OLD_PROMPT = """你是旺店通 ERP 的售后客服。只能依据【资料】回答客户，禁止使用资料以外的知识。
- 资料没有覆盖客户的问题时，answerable 填 false，reply 留空。
- 资料可能来自不同产品版本（企业版/极速版等）或较早的版本；若资料适用的版本与客户不一致，要在回复里说明。
- 回复要具体、可操作、分步骤，语气礼貌；用到哪条资料就把它的编号写进 cited_ids。"""


def variant_old(hits, message):
    ctx = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
    return OLD_PROMPT, f"【资料】\n{ctx}\n\n【客户消息】\n{message}", lambda cited: [c for c in cited if c in {h.chunk.id for h in hits}]


def variant_new(hits, message):
    ctx = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
    ids = [h.chunk.id for h in hits]
    return DRAFT_PROMPT, f"【资料】\n{ctx}\n\n【客户消息】\n{message}", lambda cited: resolve_citations(cited, ids)


VARIANTS = {"old(长id)": variant_old, "new(短编号)": variant_new}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--kb-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=60, help="可答样本取前 N 条；不可答样本固定全取")
    ap.add_argument("--model", default=None, help="覆盖回复模型，默认用 LLM_MAIN_MODEL")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("eval/real/reports/ab_draft.json"))
    args = ap.parse_args()

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"llm_main_model": args.model})
    settings = settings.model_copy(update={"llm_timeout": 120})
    llm = StructuredLLM(get_llm(settings, "main"), DraftResult)

    items = load_dataset(args.dataset)
    answerable = [i for i in items if i.answerable][: args.limit]
    unanswerable = [i for i in items if not i.answerable]
    retriever = BM25Retriever(load_kb(args.kb_dir))

    def run(job):
        item, name = job
        # 用期望实体拼检索词，隔离分诊误差，让这里只反映回复节点的差别
        query = build_query(
            {"platform_name": item.platform_name, "feature_name": item.feature_name,
             "partner_name": item.partner_name, "error_text": item.error_text}, item.question)
        hits = retriever.search(query, k=settings.retrieval_top_k)
        system, user, resolve = VARIANTS[name](hits, item.question)
        try:
            d = llm.invoke([SystemMessage(system), HumanMessage(user)])
        except Exception as exc:  # 超时等基础设施错误单独计，不当成模型行为
            return {"qid": item.qid, "name": name, "error": type(exc).__name__}
        cited = resolve(d.cited_ids)
        chunk_by_id = {h.chunk.id: h.chunk for h in hits}
        cited_docs = {chunk_by_id[c].doc_id for c in cited}
        passed = d.answerable and bool(d.reply.strip()) and bool(cited)
        return {
            "qid": item.qid, "name": name, "error": None, "answerable_flag": d.answerable,
            "raw_cited": d.cited_ids, "cited_valid": bool(cited), "auto_reply": passed,
            "citation_hit": bool(cited_docs & set(item.gold_doc_ids)) if item.answerable else None,
            "retrieved_gold": any(h.chunk.doc_id in item.gold_doc_ids for h in hits) if item.answerable else None,
        }

    jobs = [(i, n) for i in answerable + unanswerable for n in VARIANTS]
    with ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(run, jobs))

    summary = {}
    print(f"\n可答 {len(answerable)} 条 / 不可答 {len(unanswerable)} 条，回复模型 {settings.llm_main_model}")
    print(f"{'变体':<14}{'放行率':>8}{'引用有效率':>11}{'放行里引用命中':>14}{'错误放行率':>11}{'超时':>6}")
    for name in VARIANTS:
        a = [r for r in results if r["name"] == name and r["qid"] in {i.qid for i in answerable}]
        u = [r for r in results if r["name"] == name and r["qid"] in {i.qid for i in unanswerable}]
        a_ok, u_ok = [r for r in a if not r["error"]], [r for r in u if not r["error"]]
        passed = [r for r in a_ok if r["auto_reply"]]
        s = {
            "n_answerable": len(a_ok), "n_unanswerable": len(u_ok), "errors": len(a) + len(u) - len(a_ok) - len(u_ok),
            "pass_rate": sum(r["auto_reply"] for r in a_ok) / max(1, len(a_ok)),
            "claimed_answerable_rate": sum(r["answerable_flag"] for r in a_ok) / max(1, len(a_ok)),
            "cited_valid_rate_when_claimed": sum(r["cited_valid"] for r in a_ok if r["answerable_flag"]) / max(1, sum(r["answerable_flag"] for r in a_ok)),
            "citation_hit_among_passed": sum(bool(r["citation_hit"]) for r in passed) / max(1, len(passed)),
            "wrong_auto_rate": sum(r["auto_reply"] for r in u_ok) / max(1, len(u_ok)),
        }
        summary[name] = s
        print(f"{name:<14}{s['pass_rate']:>8.3f}{s['cited_valid_rate_when_claimed']:>11.3f}{s['citation_hit_among_passed']:>14.3f}{s['wrong_auto_rate']:>11.3f}{s['errors']:>6}")
    args.out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
