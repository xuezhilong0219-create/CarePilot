"""在评估集上跑检索和完整流水线。不发真实网络请求，模型由调用方注入。"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import Settings
from ..graph.nodes import Models, build_query, make_nodes
from ..retrieval.base import Chunk, Hit, Retriever
from ..tools import TicketSystem
from .metrics import (
    aggregate_retrieval,
    percentile,
    reply_metrics,
    retrieval_metrics,
    triage_metrics,
)
from .schema import EvalItem


def _hit_dict(hit: Hit) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "id": chunk.id,
        "title": chunk.title,
        "doc_id": chunk.doc_id,
        "title_key": chunk.meta.get("title_key") or chunk.title,
        "score": float(hit.score),
    }


def _query_for(item: EvalItem, query_mode: str) -> str:
    if query_mode == "raw":
        return item.question
    if query_mode == "triaged":
        # 用数据集的期望实体拼检索词，隔离分诊误差，专看检索本身
        expected = {
            "platform_name": item.platform_name,
            "feature_name": item.feature_name,
            "partner_name": item.partner_name,
            "error_text": item.error_text,
        }
        return build_query(expected, item.question)
    raise ValueError(f"unknown query_mode: {query_mode}")


def run_retrieval(
    items: Sequence[EvalItem],
    retriever: Retriever,
    k: int = 10,
    query_mode: str = "raw",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        query = _query_for(item, query_mode)
        started = time.perf_counter()
        hits = retriever.search(query, k=k)
        latency_ms = (time.perf_counter() - started) * 1000
        metrics = retrieval_metrics(hits, item.gold_doc_ids, item.gold_title_keys, k, latency_ms)
        records.append(
            {
                "qid": item.qid,
                "question": item.question,
                "query": query,
                "query_mode": query_mode,
                "k": k,
                "latency_ms": latency_ms,
                "hits": [_hit_dict(hit) for hit in hits],
                "metrics": metrics,
            }
        )
    return records


def summarize_retrieval(records: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    return {
        "n": len(records),
        "k": k,
        **aggregate_retrieval([record["metrics"] for record in records]),
    }


class _RecordingRetriever:
    """包一层记录 chunk，用于把回复里的引用 id 还原成 doc_id / title_key。
    retrieve 节点只把 id/title/text 放进 state，评估需要文档级的 gold 对齐信息。"""

    def __init__(self, inner: Retriever):
        self._inner = inner
        self.chunks: dict[str, Chunk] = {}
        self._lock = threading.Lock()

    def search(self, query: str, k: int = 4) -> list[Hit]:
        hits = self._inner.search(query, k=k)
        with self._lock:
            for hit in hits:
                self.chunks[hit.chunk.id] = hit.chunk
        return hits


def _cited_docs(cited_ids: Sequence[str], chunks: Mapping[str, Chunk]) -> tuple[list[str], list[str]]:
    doc_ids: list[str] = []
    title_keys: list[str] = []
    for cited_id in cited_ids:
        chunk = chunks.get(cited_id)
        if chunk is None:
            continue
        if chunk.doc_id and chunk.doc_id not in doc_ids:
            doc_ids.append(chunk.doc_id)
        title_key = chunk.meta.get("title_key") or chunk.title
        if title_key and title_key not in title_keys:
            title_keys.append(title_key)
    return doc_ids, title_keys


def run_pipeline(
    items: Sequence[EvalItem],
    settings: Settings,
    retriever: Retriever,
    models: Models | None = None,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """依次跑 classify →（缺信息则 ask_for_info 停住）→ retrieve → draft_reply，不走 interrupt。"""
    recorder = _RecordingRetriever(retriever)
    nodes = make_nodes(settings, recorder, TicketSystem(), models)

    def run_one(item: EvalItem) -> dict[str, Any]:
        state: dict[str, Any] = {"ticket_id": item.qid, "customer_message": item.question}
        timings: dict[str, float] = {}
        started = time.perf_counter()
        try:
            t0 = time.perf_counter()
            state.update(nodes["classify"](state))
            timings["classify_ms"] = (time.perf_counter() - t0) * 1000

            if state.get("missing_fields"):
                t0 = time.perf_counter()
                state.update(nodes["ask_for_info"](state))
                timings["ask_for_info_ms"] = (time.perf_counter() - t0) * 1000
            else:
                t0 = time.perf_counter()
                state.update(nodes["retrieve"](state))
                timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                state.update(nodes["draft_reply"](state))
                timings["draft_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # 单条失败不能拖垮整批
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return {
                "qid": item.qid,
                "question": item.question,
                "error": f"{type(exc).__name__}: {exc}",
                "triage": state.get("triage"),
                "resolution": state.get("resolution"),
                "timings_ms": timings,
            }

        timings["total_ms"] = (time.perf_counter() - started) * 1000
        citations = list(state.get("citations") or [])
        cited_doc_ids, cited_title_keys = _cited_docs(citations, recorder.chunks)
        return {
            "qid": item.qid,
            "question": item.question,
            "error": None,
            "query": state.get("query"),
            "triage": state.get("triage"),
            "missing_fields": state.get("missing_fields") or [],
            "n_hits": len(state.get("hits") or []),
            "resolution": state.get("resolution"),
            "escalate_reason": state.get("escalate_reason"),
            "draft": state.get("draft"),
            "citations": citations,
            "cited_doc_ids": cited_doc_ids,
            "cited_title_keys": cited_title_keys,
            "timings_ms": timings,
        }

    workers = max(1, int(concurrency))
    if workers == 1:
        return [run_one(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, items))


def _latency_stats(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = [float(r["timings_ms"][key]) for r in records if key in (r.get("timings_ms") or {})]
    return {"mean": sum(values) / len(values) if values else 0.0, "p90": percentile(values, 90)}


def summarize_pipeline(
    items: Sequence[EvalItem], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    pairs = list(zip(items, records))
    ok = [(item, rec) for item, rec in pairs if not rec.get("error")]
    errors = len(pairs) - len(ok)

    samples = []
    reply_rows = []
    for item, rec in ok:
        samples.append(
            {
                "gold": {
                    "category": item.category,
                    "severity": item.severity,
                    "platform_name": item.platform_name,
                    "feature_name": item.feature_name,
                    "partner_name": item.partner_name,
                    "error_text": item.error_text,
                },
                "predicted": rec.get("triage") or {},
                "message": item.question,
            }
        )
        reply_rows.append(
            {
                "answerable": item.answerable,
                "resolution": rec.get("resolution"),
                "cited_doc_ids": rec.get("cited_doc_ids") or [],
                "cited_title_keys": rec.get("cited_title_keys") or [],
                "gold_doc_ids": item.gold_doc_ids,
                "gold_title_keys": item.gold_title_keys,
            }
        )

    return {
        "n": len(records),
        "n_evaluated": len(ok),
        "errors": errors,
        "error_rate": errors / len(records) if records else 0.0,
        "resolutions": dict(Counter(rec.get("resolution") for _, rec in ok)),
        "triage": triage_metrics(samples),
        "reply": reply_metrics(reply_rows, total=len(records)),
        "latency_ms": {
            "classify_mean": _latency_stats(records, "classify_ms")["mean"],
            "classify_p90": _latency_stats(records, "classify_ms")["p90"],
            "retrieve_mean": _latency_stats(records, "retrieve_ms")["mean"],
            "retrieve_p90": _latency_stats(records, "retrieve_ms")["p90"],
            "draft_mean": _latency_stats(records, "draft_ms")["mean"],
            "draft_p90": _latency_stats(records, "draft_ms")["p90"],
            "total_mean": _latency_stats(records, "total_ms")["mean"],
            "total_p90": _latency_stats(records, "total_ms")["p90"],
        },
    }
