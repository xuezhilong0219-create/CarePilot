"""延迟账脚本的确定性部分：抽样和汇总。不联网、不调模型。

为什么值得测：这两段是「数字对不对」的全部来源——抽错样本、把总耗时相除算比例，
都会让报告里的数看起来很像回事但站不住。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from carepilot.evaluation.schema import EvalItem

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_lat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lat = _load("latency_profile")


def _item(qid: str, *, answerable: bool = True) -> EvalItem:
    return EvalItem(qid=qid, question=f"问题 {qid}", category="feature_howto", answerable=answerable)


def test_pick_only_takes_answerable_items():
    items = [_item(f"q{i}", answerable=(i % 2 == 0)) for i in range(10)]
    picked = lat.pick(items, 0)
    assert [it.qid for it in picked] == ["q0", "q2", "q4", "q6", "q8"]


def test_pick_is_evenly_spaced_and_deterministic():
    items = [_item(f"q{i:02d}") for i in range(20)]
    first = lat.pick(items, 5)
    second = lat.pick(items, 5)
    assert [it.qid for it in first] == [it.qid for it in second]
    # 等距抽取：20 条抽 5 条，步长 4，从第 0 条开始
    assert [it.qid for it in first] == ["q00", "q04", "q08", "q12", "q16"]


def test_pick_returns_everything_when_limit_is_larger_than_pool():
    items = [_item(f"q{i}") for i in range(3)]
    assert len(lat.pick(items, 100)) == 3


def test_summarize_counts_each_node_separately():
    """升级的样本没有检索和起草，它们的样本数必须比整单少，不能当成 0 混进均值。"""
    rows = [
        {"timings_ms": {"classify_ms": 100.0, "retrieve_ms": 10.0, "draft_ms": 900.0, "total_ms": 1010.0}},
        {"timings_ms": {"classify_ms": 200.0, "total_ms": 200.0}},
        {"timings_ms": {"classify_ms": 300.0, "retrieve_ms": 30.0, "draft_ms": 700.0, "total_ms": 1030.0}},
    ]
    summary = lat.summarize(rows)
    assert summary["classify_ms"]["n"] == 3
    assert summary["retrieve_ms"]["n"] == 2
    assert summary["draft_ms"]["mean"] == 800.0
    assert summary[lat.TOTAL]["n"] == 3


def test_summarize_separates_model_time_from_local_time():
    """一条特别慢的样本：本地耗时按它自己算，不能被别的样本平均掉。

    逐条算：第一条模型 100 / 本地 0，第二条模型 1800 / 本地 8200。
    本地耗时的中位数是 4100 ms、最大值 8200 ms——这就是「慢在模型还是慢在别的」的答案。
    """
    rows = [
        {"timings_ms": {"classify_ms": 50.0, "draft_ms": 50.0, "total_ms": 100.0}},
        {"timings_ms": {"classify_ms": 900.0, "draft_ms": 900.0, "total_ms": 10000.0}},
    ]
    summary = lat.summarize(rows)
    assert summary["llm_ms"]["mean"] == pytest.approx(950.0)
    assert summary["local_ms"]["p50"] == pytest.approx(4100.0)
    assert summary["local_ms"]["max"] == pytest.approx(8200.0)


def test_summarize_ignores_rows_without_the_bucket():
    rows: list[dict] = [{}, {"timings_ms": {}}]
    summary = lat.summarize(rows)
    assert summary[lat.TOTAL]["n"] == 0
    assert "classify_ms" not in summary


def test_render_lists_every_node_that_has_samples():
    rows = [{"timings_ms": {"classify_ms": 100.0, "retrieve_ms": 10.0, "draft_ms": 900.0, "total_ms": 1010.0}}]
    text = lat.render(lat.summarize(rows), 1)
    assert "分诊 classify" in text
    assert "检索 retrieve" in text
    assert "起草 draft_reply" in text
    assert "整单 total" in text
    assert "共 1 条工单" in text
