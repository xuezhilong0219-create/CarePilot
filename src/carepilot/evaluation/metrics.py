"""评估指标：全部是纯函数，不做 IO。期望值都在 tests/test_evaluation.py 里手算过。

检索指标先把 chunk 级结果折叠成文档级排序（同一文档只保留最靠前的一次），
再用 doc_id（严格）或 title_key（宽松）对齐 gold。两档都报，因为知识库里同名文档很多。
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..retrieval.base import Hit

STRICT = "strict"
LENIENT = "lenient"
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
ENTITY_FIELDS = ("platform_name", "feature_name", "partner_name")


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: Sequence[float]) -> float:
    return _ratio(sum(values), len(values))


def fold_ranking(hits: Sequence[Hit], mode: str = STRICT) -> list[str]:
    """把 chunk 级结果折叠成文档级排序，同一文档只算最靠前的那次。"""
    if mode not in (STRICT, LENIENT):
        raise ValueError(f"unknown mode: {mode}")
    ranking: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        chunk = hit.chunk
        key = chunk.doc_id if mode == STRICT else (chunk.meta.get("title_key") or chunk.title)
        if not key or key in seen:
            continue
        seen.add(key)
        ranking.append(key)
    return ranking


def hit_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    return 1.0 if any(key in gold_set for key in ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for rank, key in enumerate(ranking, start=1):
        if key in gold_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    found = gold_set & set(ranking[:k])
    return len(found) / len(gold_set)


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数，和 numpy.percentile 默认一致。空集返回 0，避免汇总时除零。"""
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    p = min(100.0, max(0.0, float(p)))
    pos = (len(vals) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def retrieval_metrics(
    hits: Sequence[Hit],
    gold_doc_ids: Iterable[str],
    gold_title_keys: Iterable[str],
    k: int,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    doc_gold = list(gold_doc_ids)
    title_gold = list(gold_title_keys)
    ranking_strict = fold_ranking(hits, STRICT)
    ranking_lenient = fold_ranking(hits, LENIENT)
    out: dict[str, Any] = {
        "k": k,
        "n_hits": len(hits),
        "hit@k_strict": hit_at_k(ranking_strict, doc_gold, k),
        "hit@k_lenient": hit_at_k(ranking_lenient, title_gold, k),
        "mrr_strict": reciprocal_rank(ranking_strict, doc_gold),
        "mrr_lenient": reciprocal_rank(ranking_lenient, title_gold),
        "recall@k_strict": recall_at_k(ranking_strict, doc_gold, k),
        "recall@k_lenient": recall_at_k(ranking_lenient, title_gold, k),
    }
    if latency_ms is not None:
        out["latency_ms"] = latency_ms
    return out


def aggregate_retrieval(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("hit@k_strict", "hit@k_lenient", "mrr_strict", "mrr_lenient", "recall@k_strict", "recall@k_lenient")
    out: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        out[key] = _mean([float(r[key]) for r in rows if key in r])
    latencies = [float(r["latency_ms"]) for r in rows if "latency_ms" in r]
    out["latency_ms_mean"] = _mean(latencies)
    out["latency_ms_p50"] = percentile(latencies, 50)
    out["latency_ms_p90"] = percentile(latencies, 90)
    return out


def normalize_entity(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(unicodedata.normalize("NFKC", str(value)).split()).lower()


def entity_match(predicted: str | None, expected: str | None) -> bool:
    pred, gold = normalize_entity(predicted), normalize_entity(expected)
    if not pred or not gold:
        return False
    return pred == gold or pred in gold or gold in pred


def overlap_ratio(predicted: str, expected: str) -> float:
    """预测值覆盖了多少比例的期望文本（按字符多重集，忽略顺序）。"""
    pred, gold = normalize_entity(predicted), normalize_entity(expected)
    if not pred or not gold:
        return 0.0
    common = sum((Counter(pred) & Counter(gold)).values())
    return common / len(gold)


def error_text_match(predicted: str | None, expected: str | None, message: str) -> bool:
    pred, gold, msg = normalize_entity(predicted), normalize_entity(expected), normalize_entity(message)
    if not pred or not gold:
        return False
    return pred in msg and overlap_ratio(pred, gold) >= 0.8


def triage_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """samples 每条：{"gold": {...}, "predicted": {...}, "message": 客户原话}。"""
    confusion: dict[str, dict[str, int]] = {}
    cat_ok = 0
    sev_ok = sev_near = sev_n = 0
    entity_ok = {f: 0 for f in ENTITY_FIELDS}
    entity_n = {f: 0 for f in ENTITY_FIELDS}
    err_ok = err_n = 0

    for sample in samples:
        gold = sample.get("gold") or {}
        pred = sample.get("predicted") or {}
        gold_cat = gold.get("category")
        pred_cat = pred.get("category") or "missing"
        if gold_cat is not None:
            confusion.setdefault(gold_cat, {})
            confusion[gold_cat][pred_cat] = confusion[gold_cat].get(pred_cat, 0) + 1
            cat_ok += int(pred_cat == gold_cat)

        gold_sev = gold.get("severity")
        if gold_sev is not None:
            sev_n += 1
            pred_sev = pred.get("severity")
            if pred_sev == gold_sev:
                sev_ok += 1
            if pred_sev in SEVERITY_ORDER and gold_sev in SEVERITY_ORDER:
                if abs(SEVERITY_ORDER[pred_sev] - SEVERITY_ORDER[gold_sev]) <= 1:
                    sev_near += 1

        for field in ENTITY_FIELDS:
            expected = gold.get(field)
            if expected is None:
                continue
            entity_n[field] += 1
            entity_ok[field] += int(entity_match(pred.get(field), expected))

        expected_err = gold.get("error_text")
        if expected_err is not None:
            err_n += 1
            err_ok += int(error_text_match(pred.get("error_text"), expected_err, sample.get("message", "")))

    return {
        "n": len(samples),
        "category_accuracy": _ratio(cat_ok, len(samples)),
        "category_confusion": confusion,
        "severity_accuracy": _ratio(sev_ok, sev_n),
        "severity_within_one_accuracy": _ratio(sev_near, sev_n),
        "severity_n": sev_n,
        "entity_accuracy": {f: _ratio(entity_ok[f], entity_n[f]) for f in ENTITY_FIELDS},
        "entity_n": dict(entity_n),
        "error_text_accuracy": _ratio(err_ok, err_n),
        "error_text_n": err_n,
    }


def citation_hit(
    cited_doc_ids: Iterable[str],
    cited_title_keys: Iterable[str],
    gold_doc_ids: Iterable[str],
    gold_title_keys: Iterable[str],
) -> bool:
    return bool(set(cited_doc_ids) & set(gold_doc_ids) or set(cited_title_keys) & set(gold_title_keys))


def reply_metrics(rows: Sequence[Mapping[str, Any]], total: int | None = None) -> dict[str, Any]:
    """rows 每条：answerable / resolution / cited_* / gold_*。
    total 是错误放行率的分母（默认用 rows 条数），调用方可以把抛异常的样本也算进总数。"""
    rows = list(rows)
    answerable = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]
    citations_ok = sum(
        int(
            citation_hit(
                r.get("cited_doc_ids") or [],
                r.get("cited_title_keys") or [],
                r.get("gold_doc_ids") or [],
                r.get("gold_title_keys") or [],
            )
        )
        for r in answerable
    )
    wrong_auto = [r for r in unanswerable if r.get("resolution") == "auto_reply"]
    return {
        "n": len(rows),
        "answerable_n": len(answerable),
        "answer_rate": _ratio(sum(r.get("resolution") == "auto_reply" for r in answerable), len(answerable)),
        "citation_hit_rate": _ratio(citations_ok, len(answerable)),
        "correct_refusal_rate": _ratio(
            sum(r.get("resolution") != "auto_reply" for r in unanswerable), len(unanswerable)
        ),
        "wrong_auto_rate": _ratio(len(wrong_auto), len(rows) if total is None else total),
    }
