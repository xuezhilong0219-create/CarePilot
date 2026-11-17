"""首字延迟脚本的确定性部分：引用解析和汇总。不联网、不调模型。

为什么值得测：`citations_in` 是「改成自由文本以后，回复还能不能过引用校验」这个判断的唯一依据。
它错了，结论就会反过来——0.000 的可解析率如果是解析写错的产物，那「先不改流式」就是误判。
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HITS = [
    {"id": "企业版/采购入库单#1", "text": "采购入库单 > 保存"},
    {"id": "极速版2.0/换货#0", "text": "换货 > 操作"},
]


def _load():
    path = ROOT / "eval" / "ttft_probe.py"
    spec = importlib.util.spec_from_file_location("ttft_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inline_refs_are_resolved_against_hits():
    mod = _load()
    cited = mod.citations_in("先关纯净模式 [1]，再重装 [2]。", HITS)
    assert cited == ["企业版/采购入库单#1", "极速版2.0/换货#0"]


def test_out_of_range_and_missing_refs_are_dropped():
    mod = _load()
    # [9] 超出资料条数、[0] 不是合法编号：都不猜，直接丢
    assert mod.citations_in("见 [9] 和 [0]", HITS) == []


def test_plain_text_has_no_citations():
    # 实测里模型在自由文本下平均不写 [n]，这就是 0/6 的来源
    mod = _load()
    assert mod.citations_in("是的，您的理解正确。", HITS) == []


def test_trailing_cited_ids_line_is_not_mistaken_for_inline_ref():
    # 模型会把答案写成 JSON 尾巴（cited_ids: ["1", "2"]），但那不是正文引用标记
    mod = _load()
    assert mod.citations_in('操作步骤如上。\n\ncited_ids: ["1", "2"]', HITS) == []


def test_summarize_reports_ttft_total_and_parseable_rate():
    mod = _load()
    rows = [
        {"ttft_ms": 100.0, "total_ms": 900.0, "citations": ["企业版/采购入库单#1"]},
        {"ttft_ms": 300.0, "total_ms": 1100.0, "citations": []},
    ]
    summary = mod.summarize(rows)
    assert summary["n"] == 2
    assert summary["ttft"]["p50"] == 200.0
    assert summary["total"]["p50"] == 1000.0
    assert summary["with_citations"] == 1
    assert summary["parseable_rate"] == 0.5


def test_summarize_ignores_rows_that_never_started_streaming():
    mod = _load()
    rows = [
        {"ttft_ms": None, "total_ms": 50.0, "citations": []},
        {"ttft_ms": 200.0, "total_ms": 400.0, "citations": []},
    ]
    summary = mod.summarize(rows)
    # 首字没有的样本不能当 0 ms 混进 p50，否则首字会被算得比实际快
    assert summary["ttft"]["n"] == 1
    assert summary["ttft"]["p50"] == 200.0
    assert summary["total"]["n"] == 2
    assert summary["parseable_rate"] == 0.0


def test_render_shows_both_ends_and_the_citation_rate():
    mod = _load()
    text = mod.render(mod.summarize([{"ttft_ms": 1270.0, "total_ms": 1670.0, "citations": ["x"]}]))
    assert "首字" in text and "整段" in text
    assert "1.27 s" in text and "1.67 s" in text
    assert "1.000（1/1）" in text
