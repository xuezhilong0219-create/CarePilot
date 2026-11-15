"""多轮评估脚本的确定性部分：判分和抽样。不联网、不调模型。"""

import importlib.util
import sys
from pathlib import Path

from carepilot.evaluation.schema import EvalItem
from carepilot.retrieval.base import Chunk

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mt_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mt = _load("run_multiturn")


class _Rec:
    def __init__(self, chunks):
        self.chunks = {c.id: c for c in chunks}


def _chunk(cid: str, doc_id: str, title_key: str) -> Chunk:
    return Chunk(id=cid, title=title_key, text="", doc_id=doc_id, meta={"title_key": title_key})


def _item() -> EvalItem:
    return EvalItem(
        qid="q1",
        question="订单导出只导出第一页",
        category="feature_howto",
        answerable=True,
        gold_doc_ids=["企业版/订单导出#1"],
        gold_title_keys=["订单导出"],
    )


def test_judge_counts_hit_when_second_turn_lands_on_the_same_doc():
    rec = _Rec([_chunk("a", "企业版/订单导出#1", "订单导出")])
    turn = {"error": None, "hits": ["a"], "citations": ["a"], "reply": "在导出设置里改"}
    judged = mt.judge(_item(), turn, rec)
    assert judged["strict_hit"] is True
    assert judged["lenient_hit"] is True
    assert judged["has_citation"] is True


def test_judge_reports_miss_when_second_turn_lands_elsewhere():
    rec = _Rec([_chunk("b", "企业版/库存同步#1", "库存同步")])
    turn = {"error": None, "hits": ["b"], "citations": ["b"], "reply": "看库存同步"}
    judged = mt.judge(_item(), turn, rec)
    assert judged["strict_hit"] is False
    assert judged["lenient_hit"] is False


def test_lenient_hit_survives_a_different_edition_of_the_same_doc():
    """同名文档在知识库里有企业版/极速版多份，只按 doc_id 算会误判（schema.py 里写过）。"""
    rec = _Rec([_chunk("c", "极速版2.0/订单导出#1", "订单导出")])
    turn = {"error": None, "hits": ["c"], "citations": ["c"], "reply": "x"}
    judged = mt.judge(_item(), turn, rec)
    assert judged["strict_hit"] is False
    assert judged["lenient_hit"] is True


def test_judge_flags_reasking_for_information_already_given():
    rec = _Rec([])
    turn = {"error": None, "hits": [], "citations": [], "reply": "为了尽快处理，请提供订单号"}
    assert mt.judge(_item(), turn, rec)["reasks_known"] is True


def test_judge_does_not_flag_a_normal_answer():
    rec = _Rec([])
    turn = {"error": None, "hits": [], "citations": [], "reply": "在「设置 - 导出」里把分页改成全部"}
    assert mt.judge(_item(), turn, rec)["reasks_known"] is False


def test_pairs_sample_only_answerable_items_and_rotate_followups(tmp_path):
    rows = [
        {"qid": "a", "question": "q a", "category": "feature_howto", "answerable": True,
         "gold_doc_ids": ["d1"], "gold_title_keys": ["t1"]},
        {"qid": "b", "question": "q b", "category": "feature_howto", "answerable": False},
        {"qid": "c", "question": "q c", "category": "feature_howto", "answerable": True,
         "gold_doc_ids": ["d2"], "gold_title_keys": ["t2"]},
        {"qid": "d", "question": "q d", "category": "feature_howto", "answerable": True,
         "gold_doc_ids": ["d3"], "gold_title_keys": ["t3"]},
    ]
    path = tmp_path / "ds.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    pairs = mt._pairs(path, 2)
    assert [p[0].qid for p in pairs] == ["a", "c"]  # 3 条里取 2 条：步长 3//2=1
    assert [p[1] for p in pairs] == list(mt.FOLLOWUPS[:2])
