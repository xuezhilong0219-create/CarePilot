import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

from carepilot.config import ROOT, Settings
from carepilot.evaluation import metrics as m
from carepilot.evaluation.report import load_report, markdown_table, save_report
from carepilot.evaluation.runners import (
    run_pipeline,
    run_retrieval,
    summarize_pipeline,
    summarize_retrieval,
)
from carepilot.evaluation.schema import EvalItem, dump_dataset
from carepilot.graph.nodes import Models
from carepilot.graph.state import DraftResult, Triage
from carepilot.retrieval.base import Chunk, Hit
from carepilot.retrieval.bm25 import BM25Retriever
from carepilot.retrieval.chunking import load_kb


def make_hit(doc_id: str, title_key: str | None = None, score: float = 1.0, idx: int = 0) -> Hit:
    meta = {"title_key": title_key} if title_key else {}
    chunk = Chunk(
        id=f"{doc_id}#{idx}",
        title=title_key or doc_id,
        text="正文",
        doc_id=doc_id,
        meta=meta,
    )
    return Hit(chunk, score)


def test_fold_ranking_keeps_first_chunk_per_doc():
    hits = [
        make_hit("A", "ta", 9.0, idx=0),
        make_hit("A", "ta", 8.0, idx=1),
        make_hit("B", "tb", 7.0, idx=0),
    ]
    assert m.fold_ranking(hits, "strict") == ["A", "B"]
    assert m.fold_ranking(hits, "lenient") == ["ta", "tb"]


def test_hit_at_k_strict_and_lenient_disagree_on_duplicate_titles():
    hits = [make_hit("retrieval/other", "店铺授权", 2.0)]
    metrics = m.retrieval_metrics(hits, ["platform/auth"], ["店铺授权"], k=5)
    assert metrics["hit@k_strict"] == 0.0
    assert metrics["hit@k_lenient"] == 1.0


def test_hit_mrr_recall_edge_cases():
    ranking = ["A", "B"]
    # k 大于结果数
    assert m.hit_at_k(ranking, ["B"], 10) == 1.0
    assert m.hit_at_k(ranking, ["B"], 1) == 0.0
    assert m.recall_at_k(ranking, ["A", "C"], 1) == 0.5
    assert m.recall_at_k(ranking, ["A", "C"], 5) == 0.5
    assert m.reciprocal_rank(ranking, ["B"]) == 0.5
    assert m.reciprocal_rank(ranking, ["C"]) == 0.0
    # 空结果 / 空 gold
    assert m.hit_at_k([], ["A"], 5) == 0.0
    assert m.hit_at_k(ranking, [], 5) == 0.0
    assert m.recall_at_k(ranking, [], 5) == 0.0


def test_retrieval_metrics_on_empty_hits():
    metrics = m.retrieval_metrics([], ["A"], ["ta"], k=10, latency_ms=12.5)
    assert metrics["n_hits"] == 0
    assert metrics["hit@k_strict"] == metrics["hit@k_lenient"] == 0.0
    assert metrics["mrr_strict"] == metrics["mrr_lenient"] == 0.0
    assert metrics["recall@k_strict"] == 0.0
    assert metrics["latency_ms"] == 12.5


def test_aggregate_retrieval_means_and_percentiles():
    rows = [
        m.retrieval_metrics([make_hit("A", "ta")], ["A"], ["ta"], 2, latency_ms=10.0),
        m.retrieval_metrics([make_hit("B", "tb")], ["A"], ["ta"], 2, latency_ms=20.0),
    ]
    agg = m.aggregate_retrieval(rows)
    assert agg["n"] == 2
    assert agg["hit@k_strict"] == 0.5
    assert agg["mrr_strict"] == 0.5
    assert agg["latency_ms_mean"] == 15.0
    assert agg["latency_ms_p50"] == 15.0
    assert agg["latency_ms_p90"] == pytest.approx(19.0)


def test_percentile():
    assert m.percentile([1, 2, 3, 4], 0) == 1.0
    assert m.percentile([1, 2, 3, 4], 50) == 2.5
    assert m.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert m.percentile([1, 2, 3, 4], 100) == 4.0
    assert m.percentile([], 90) == 0.0


def test_entity_normalization_whitespace_case_and_containment():
    assert m.entity_match(" 淘宝 ", "淘宝")
    assert m.entity_match("Order", "order")
    assert m.entity_match("菜鸟物流", "菜鸟")
    assert m.entity_match("菜鸟", "菜鸟物流")
    assert not m.entity_match(None, "淘宝")
    assert not m.entity_match("京东", "淘宝")


def test_error_text_requires_in_message_and_overlap():
    message = "系统提示 session expired，下载不了订单"
    assert m.error_text_match("session expired", "session expired", message)
    assert not m.error_text_match("网络超时", "session expired", message)
    assert not m.error_text_match("session", "session expired now please retry", message)
    assert m.overlap_ratio("session expired", "session expired") == 1.0


def test_triage_metrics_hand_computed():
    samples = [
        {
            "gold": {"category": "feature_howto", "severity": "medium", "platform_name": "淘宝", "error_text": "session expired"},
            "predicted": {"category": "feature_howto", "severity": "high", "platform_name": " 淘宝 ", "error_text": "session expired"},
            "message": "系统提示 session expired，下载不了订单",
        },
        {
            "gold": {"category": "platform_integration", "severity": "critical", "feature_name": "Order", "error_text": "请求过于频繁"},
            "predicted": {"category": "feature_howto", "severity": "critical", "feature_name": "order", "error_text": "网络超时"},
            "message": "库存同步一直提示请求过于频繁",
        },
        {
            "gold": {"category": "other", "severity": "low", "partner_name": "菜鸟"},
            "predicted": {"category": "other", "severity": None, "partner_name": "菜鸟物流"},
            "message": "电子面单问题",
        },
    ]
    result = m.triage_metrics(samples)
    assert result["category_accuracy"] == pytest.approx(2 / 3)
    assert result["category_confusion"] == {
        "feature_howto": {"feature_howto": 1},
        "platform_integration": {"feature_howto": 1},
        "other": {"other": 1},
    }
    assert result["severity_accuracy"] == pytest.approx(1 / 3)
    assert result["severity_within_one_accuracy"] == pytest.approx(2 / 3)
    assert result["entity_accuracy"] == {"platform_name": 1.0, "feature_name": 1.0, "partner_name": 1.0}
    assert result["error_text_accuracy"] == 0.5


def test_triage_metrics_no_expected_fields_does_not_divide_by_zero():
    result = m.triage_metrics(
        [{"gold": {"category": "other"}, "predicted": {"category": "other"}, "message": "hi"}]
    )
    assert result["category_accuracy"] == 1.0
    assert result["severity_n"] == 0
    assert result["severity_accuracy"] == 0.0
    assert result["entity_accuracy"] == {"platform_name": 0.0, "feature_name": 0.0, "partner_name": 0.0}
    assert result["error_text_accuracy"] == 0.0


def test_reply_metrics_hand_computed():
    rows = [
        {"answerable": True, "resolution": "auto_reply", "cited_doc_ids": ["A"], "cited_title_keys": [], "gold_doc_ids": ["A"], "gold_title_keys": []},
        {"answerable": True, "resolution": "auto_reply", "cited_doc_ids": ["B"], "cited_title_keys": [], "gold_doc_ids": ["A"], "gold_title_keys": ["ta"]},
        {"answerable": True, "resolution": "escalate", "cited_doc_ids": [], "cited_title_keys": [], "gold_doc_ids": ["A"], "gold_title_keys": []},
        {"answerable": False, "resolution": "escalate", "cited_doc_ids": [], "cited_title_keys": [], "gold_doc_ids": [], "gold_title_keys": []},
        {"answerable": False, "resolution": "auto_reply", "cited_doc_ids": ["X"], "cited_title_keys": [], "gold_doc_ids": [], "gold_title_keys": []},
    ]
    result = m.reply_metrics(rows)
    assert result["answer_rate"] == pytest.approx(2 / 3)
    assert result["citation_hit_rate"] == pytest.approx(1 / 3)
    assert result["correct_refusal_rate"] == 0.5
    assert result["wrong_auto_rate"] == 0.2
    assert m.reply_metrics(rows, total=10)["wrong_auto_rate"] == 0.1


def test_reply_metrics_no_answerable_samples_does_not_divide_by_zero():
    result = m.reply_metrics(
        [{"answerable": False, "resolution": "escalate", "gold_doc_ids": [], "gold_title_keys": []}]
    )
    assert result["answer_rate"] == 0.0
    assert result["citation_hit_rate"] == 0.0
    assert result["correct_refusal_rate"] == 1.0
    empty = m.reply_metrics([])
    assert empty["answer_rate"] == empty["citation_hit_rate"] == empty["wrong_auto_rate"] == 0.0


def test_save_and_load_report(tmp_path: Path):
    summary = {"hit@k_strict": 0.5, "n": 2}
    items = [{"qid": "q1", "hits": []}]
    json_path, items_path = save_report(
        tmp_path, "exp", summary, items, {"retriever": "bm25"}, timestamp=datetime(2026, 1, 2, 3, 4, 5)
    )
    assert json_path.name == "20260102_030405_exp.json"
    assert items_path.exists()
    loaded = load_report(json_path)
    assert loaded["name"] == "exp"
    assert loaded["n_items"] == 1
    assert loaded["summary"] == summary
    assert "| metric |" in markdown_table(["metric"], [[1]])


class FakeTriage:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses

    def invoke(self, messages):
        text = messages[-1].content
        for marker, value in self.responses.items():
            if marker in text:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"no triage response for: {text}")


class FakeDraft:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses

    def invoke(self, messages):
        text = messages[-1].content
        for marker, value in self.responses.items():
            if marker in text:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"no draft response for: {text}")


class FakeRetriever:
    def __init__(self, mapping: dict[str, list[Hit]]):
        self.mapping = mapping

    def search(self, query: str, k: int = 4) -> list[Hit]:
        for marker, hits in self.mapping.items():
            if marker in query:
                return hits[:k]
        return []


def _pipeline_items() -> list[EvalItem]:
    return [
        EvalItem(
            qid="q1",
            question="京东店铺授权失效，提示 session expired，订单下载不了",
            category="platform_integration",
            answerable=True,
            gold_doc_ids=["platform/auth"],
            gold_title_keys=["店铺授权"],
            severity="medium",
            platform_name="京东",
            error_text="session expired",
        ),
        EvalItem(
            qid="q2",
            question="我想设置一下订单打印功能",
            category="feature_howto",
            answerable=True,
            gold_doc_ids=["feature/print"],
            gold_title_keys=["打印设置"],
            severity="low",
        ),
        EvalItem(qid="q3", question="zzzqqq 完全无关", category="other", answerable=False, severity="low"),
        EvalItem(
            qid="q4",
            question="帮我查一下昨天订单数据为什么少了",
            category="data_exception",
            answerable=False,
            severity="high",
        ),
        EvalItem(qid="q5", question="这条会抛异常", category="other", answerable=True, severity="low"),
        EvalItem(
            qid="q6",
            question="淘宝平台规则变了会有什么影响",
            category="rule_change",
            answerable=False,
            severity="medium",
            platform_name="淘宝",
        ),
    ]


def _pipeline_models() -> Models:
    triage = FakeTriage(
        {
            "京东店铺授权": Triage(category="platform_integration", severity="medium", platform_name="京东", error_text="session expired"),
            "订单打印功能": Triage(category="feature_howto", severity="low"),
            "zzzqqq": Triage(category="other", severity="low"),
            "昨天订单数据": Triage(category="data_exception", severity="high", fault_description="订单数据缺失"),
            "这条会抛异常": RuntimeError("boom"),
            "平台规则变了": Triage(category="rule_change", severity="medium", platform_name="淘宝"),
        }
    )
    draft = FakeDraft(
        {
            "京东店铺授权": DraftResult(answerable=True, reply="请重新授权", cited_ids=["platform/auth#0"]),
            "昨天订单数据": DraftResult(answerable=False, reply=""),
            "平台规则变了": DraftResult(answerable=True, reply="按新规则处理", cited_ids=["rules/new#0"]),
        }
    )
    return Models(triage=triage, draft=draft)


def _pipeline_retriever() -> FakeRetriever:
    return FakeRetriever(
        {
            "京东店铺授权": [make_hit("platform/auth", "店铺授权", 0.9)],
            "昨天订单数据": [make_hit("data/orders", "订单查询", 0.8)],
            "平台规则变了": [make_hit("rules/new", "规则变更", 0.7)],
        }
    )


def test_run_pipeline_six_samples_hand_computed():
    items = _pipeline_items()
    settings = Settings(
        llm_api_key="test",
        llm_base_url="http://localhost:0/v1",
        retrieval_top_k=5,
    )
    records = run_pipeline(items, settings, _pipeline_retriever(), models=_pipeline_models(), concurrency=2)
    summary = summarize_pipeline(items, records)

    by_qid = {r["qid"]: r for r in records}
    assert by_qid["q1"]["resolution"] == "auto_reply"
    assert by_qid["q2"]["resolution"] == "need_info"
    assert by_qid["q3"]["resolution"] == "escalate"
    assert by_qid["q4"]["resolution"] == "escalate"
    assert "boom" in by_qid["q5"]["error"]
    assert by_qid["q6"]["resolution"] == "auto_reply"

    assert summary["n"] == 6
    assert summary["n_evaluated"] == 5
    assert summary["errors"] == 1
    assert summary["error_rate"] == pytest.approx(1 / 6)
    assert summary["resolutions"] == {"auto_reply": 2, "need_info": 1, "escalate": 2}
    assert summary["triage"]["category_accuracy"] == 1.0
    assert summary["triage"]["severity_accuracy"] == 1.0
    assert summary["triage"]["severity_within_one_accuracy"] == 1.0
    assert summary["triage"]["entity_accuracy"]["platform_name"] == 1.0
    assert summary["triage"]["error_text_accuracy"] == 1.0
    assert summary["reply"]["answer_rate"] == 0.5
    assert summary["reply"]["citation_hit_rate"] == 0.5
    assert summary["reply"]["correct_refusal_rate"] == pytest.approx(2 / 3)
    assert summary["reply"]["wrong_auto_rate"] == pytest.approx(1 / 6)
    assert summary["latency_ms"]["total_mean"] >= 0.0


def _retrieval_items() -> list[EvalItem]:
    questions = [
        ("r1", "京东店铺授权已失效怎么处理", "店铺授权过期怎么办", "京东"),
        ("r2", "库存同步失败怎么办", "库存同步失败", None),
        ("r3", "电子面单打印提示请先绑定网点", "电子面单打印提示请先绑定网点", None),
        ("r4", "第三方仓库接口调用超时", "第三方仓库接口调用超时", None),
        ("r5", "zzzqqq", "一个根本检索不到的文档", None),  # 有标准答案但检索为空：验证空结果不崩、记 0 分
    ]
    items = []
    for qid, question, title_key, platform in questions:
        items.append(
            EvalItem(
                qid=qid,
                question=question,
                category="other",
                answerable=False,
                gold_doc_ids=["sample_erp_faq" if qid != "r5" else "missing-doc"] if title_key else [],
                # 示例知识库没有 manifest：整个文件是一篇文档，title_key 就是文件名，而不是小节标题
                gold_title_keys=["sample_erp_faq" if qid != "r5" else "missing-doc"] if title_key else [],
                platform_name=platform,
            )
        )
    return items


def test_run_retrieval_on_sample_kb():
    kb_dir = ROOT / "data" / "kb"
    retriever = BM25Retriever(load_kb(kb_dir))
    items = _retrieval_items()

    raw = run_retrieval(items, retriever, k=3, query_mode="raw")
    assert len(raw) == 5
    assert raw[0]["metrics"]["hit@k_strict"] == 1.0
    assert raw[0]["metrics"]["hit@k_lenient"] == 1.0
    assert raw[4]["hits"] == []
    assert raw[4]["metrics"]["hit@k_strict"] == 0.0
    assert all(record["latency_ms"] >= 0.0 for record in raw)

    triaged = run_retrieval(items, retriever, k=3, query_mode="triaged")
    assert triaged[0]["query"].startswith("京东")
    assert triaged[0]["metrics"]["hit@k_strict"] == 1.0

    summary = summarize_retrieval(raw, k=3)
    assert summary["n"] == 5
    assert summary["hit@k_strict"] == pytest.approx(4 / 5)


def test_run_retrieval_script_end_to_end_with_fake_factory(tmp_path: Path, monkeypatch):
    kb_dir = ROOT / "data" / "kb"
    dataset_path = tmp_path / "dataset.jsonl"
    dump_dataset(_retrieval_items(), dataset_path)

    fake_factory = types.ModuleType("carepilot.retrieval.factory")
    fake_factory.build_retriever = lambda kind, kb, **kwargs: BM25Retriever(load_kb(kb))
    monkeypatch.setitem(sys.modules, "carepilot.retrieval.factory", fake_factory)

    spec = importlib.util.spec_from_file_location("eval_run_retrieval", ROOT / "eval" / "run_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    code = module.main(
        [
            "--dataset",
            str(dataset_path),
            "--kb-dir",
            str(kb_dir),
            "--retriever",
            "bm25",
            "--k",
            "3",
            "--name",
            "unit",
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    reports = list(tmp_path.glob("*_unit.json"))
    assert len(reports) == 1
    assert load_report(reports[0])["n_items"] == 5


def test_error_text_rejects_whole_message_as_prediction():
    # 回归：只看覆盖率时，把整条客户消息当报错原文返回也会满分
    message = "库存同步到京东一直失败，日志里提示请求过于频繁，麻烦看下"
    assert m.error_text_match("请求过于频繁", "请求过于频繁", message)
    assert not m.error_text_match(message, "请求过于频繁", message)


def _hit(doc_id: str, title_key: str, i: int = 0) -> Hit:
    return Hit(Chunk(id=f"{doc_id}#{i}", title="t", text="x", doc_id=doc_id, meta={"title_key": title_key}), 1.0)


def test_hit_curve_hand_computed():
    # chunk 顺序：a#0, a#1, b#0, c#0；折叠成文档后是 [a, b, c]
    hits = [_hit("a", "ka", 0), _hit("a", "ka", 1), _hit("b", "kb"), _hit("c", "kc")]
    curve = m.hit_curve(hits, ["c"], ["kb"])
    assert curve["hit@1_strict"] == 0 and curve["hit@3_strict"] == 1  # c 是文档排名第 3
    assert curve["hit@1_lenient"] == 0 and curve["hit@3_lenient"] == 1  # kb 是第 2
    assert curve["hit@5_strict"] == 1 and curve["hit@10_strict"] == 1
    # 前 4 个 chunk 里包含 c#0，所以 chunk_hit@4 命中；前 3 个 chunk 则不会
    assert curve["chunk_hit@4_strict"] == 1
    assert m.hit_curve(hits[:3], ["c"], [], chunk_cutoffs=(4,))["chunk_hit@4_strict"] == 0


def test_aggregate_retrieval_averages_curve():
    rows = [{"curve": {"hit@1_strict": 1.0}}, {"curve": {"hit@1_strict": 0.0}}]
    assert m.aggregate_retrieval(rows)["hit@1_strict"] == 0.5


def test_run_retrieval_skips_unanswerable_items_so_they_do_not_dilute_hit_rates():
    kb_dir = ROOT / "data" / "kb"
    retriever = BM25Retriever(load_kb(kb_dir))
    answerable = _retrieval_items()[:4]
    unanswerable = [
        EvalItem(qid=f"u{i}", question="我的订单为什么不见了", category="data_exception", answerable=False) for i in range(6)
    ]
    only = summarize_retrieval(run_retrieval(answerable, retriever, k=3), k=3)
    mixed = summarize_retrieval(run_retrieval(answerable + unanswerable, retriever, k=3), k=3)
    assert mixed["n"] == only["n"] == 4
    assert mixed["hit@k_strict"] == only["hit@k_strict"]


def test_run_retrieval_both_modes_write_two_distinct_reports(tmp_path: Path, monkeypatch):
    # 回归：--query-mode both 配合 --name 时，两份报告曾写到同一个文件名，后一份覆盖前一份
    dataset_path = tmp_path / "dataset.jsonl"
    dump_dataset(_retrieval_items(), dataset_path)
    fake_factory = types.ModuleType("carepilot.retrieval.factory")
    fake_factory.build_retriever = lambda kind, kb, **kwargs: BM25Retriever(load_kb(kb))
    monkeypatch.setitem(sys.modules, "carepilot.retrieval.factory", fake_factory)
    spec = importlib.util.spec_from_file_location("eval_run_retrieval2", ROOT / "eval" / "run_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    code = module.main(["--dataset", str(dataset_path), "--kb-dir", str(ROOT / "data" / "kb"), "--k", "3",
                        "--query-mode", "both", "--name", "exp", "--report-dir", str(tmp_path)])
    assert code == 0
    assert sorted(p.name.split("_", 2)[2] for p in tmp_path.glob("*_exp_*.json")) == ["exp_raw.json", "exp_triaged.json"]


def test_run_pipeline_follows_the_same_routing_as_the_graph():
    # 评估运行器和建图共用 graph.routing；这里锁住三条路径，防止两边再次分叉
    from fakes import FakeRetriever, ScriptedInvoker, make_draft, make_hit, make_triage

    settings = Settings(llm_api_key="test", llm_base_url="http://localhost:0/v1")

    def run(triage, draft, hits):
        item = EvalItem(qid="q", question="x", category="other", answerable=True)
        models = Models(triage=ScriptedInvoker(triage), draft=ScriptedInvoker(draft))
        return run_pipeline([item], settings, FakeRetriever(hits), models=models, concurrency=1)[0]

    vague = run(make_triage(category="other", is_vague=True), make_draft(), [make_hit()])
    assert vague["resolution"] == "need_info" and "draft_ms" not in vague["timings_ms"]  # 笼统：不检索不生成

    answered = run(make_triage(category="platform_integration"), make_draft(cited_ids=["1"]), [make_hit()])
    assert answered["resolution"] == "auto_reply"  # 缺槽位但可回答：直接回复

    asked = run(make_triage(category="platform_integration"), make_draft(answerable=False), [make_hit()])
    assert asked["resolution"] == "need_info"  # 答不了且缺槽位：追问

    escalated = run(make_triage(category="platform_integration", platform_name="京东"), make_draft(answerable=False), [make_hit()])
    assert escalated["resolution"] == "escalate"  # 答不了且不缺：升级

    direct = run(make_triage(category="data_exception"), make_draft(cited_ids=["1"]), [make_hit()])
    assert direct["resolution"] == "escalate" and direct["n_hits"] == 0  # 需查客户数据：不检索，直接升级
    assert "draft_ms" not in direct["timings_ms"]
