"""交叉评审脚本的确定性部分：一致率、kappa、稳定抽样。不联网、不调模型。

Judging 本身要调模型，没法进单测；能钉死的是「两个评审人算得多一致」这套算术，
它错了整份报告就在骗人。
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cross_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cross = _load("judge_cross")
LEVELS = cross.LEVELS
quote_missing = sys.modules["judge_replies"].quote_missing


def test_cohen_kappa_is_one_on_perfect_agreement():
    labels = ["none", "minor", "major", "none", "major"]
    assert cross.cohen_kappa(labels, list(labels), LEVELS) == 1.0


def test_cohen_kappa_is_zero_when_agreement_is_exactly_chance():
    # po = 0.5；两个标注者各自各占一半，pe = 0.25 + 0.25 = 0.5 → kappa 0
    a = ["none", "none", "major", "major"]
    b = ["none", "major", "none", "major"]
    assert cross.cohen_kappa(a, b, LEVELS) == 0.0


def test_cohen_kappa_is_negative_when_they_are_systematically_opposed():
    a = ["none", "none", "major", "major"]
    b = ["major", "major", "none", "none"]
    assert cross.cohen_kappa(a, b, LEVELS) == -1.0


def test_cohen_kappa_is_undefined_when_both_stick_to_one_label():
    """两人全判 none 时一致率 100%，但 kappa 无定义 —— 不能拿它当「完全一致」的证据。"""
    assert cross.cohen_kappa(["none"] * 5, ["none"] * 5, LEVELS) is None


def test_cohen_kappa_is_none_for_empty_or_mismatched_inputs():
    assert cross.cohen_kappa([], [], LEVELS) is None
    assert cross.cohen_kappa(["none"], ["none", "major"], LEVELS) is None


def test_agreement_counts_exact_matches():
    assert cross.agreement(["none", "minor"], ["none", "major"]) == 0.5
    assert cross.agreement([], []) is None


def test_high_agreement_can_still_hide_a_low_kappa():
    """这个测试存在的理由：光看一致率会被「都判 none」骗到。

    100 条里 90 条一致（一致率 0.9），但两个模型都偏向 none，扣掉随机一致后
    只剩 0.444 —— 这就是为什么报告里要同时给 kappa。
    """
    a = ["none"] * 85 + ["major"] * 5 + ["none"] * 5 + ["major"] * 5
    b = ["none"] * 85 + ["none"] * 5 + ["major"] * 5 + ["major"] * 5
    assert cross.agreement(a, b) == 0.9
    kappa = cross.cohen_kappa(a, b, LEVELS)
    assert kappa is not None and kappa < 0.5


def test_rubric_is_shared_with_the_single_judge_script():
    """交叉评审必须用同一个评分标准，否则比出来的差异来自 rubric 不是模型。"""
    single = sys.modules["judge_replies"]
    assert cross.SYSTEM is single.SYSTEM
    assert cross.Verdict is single.Verdict


def test_sample_is_stable_and_only_returns_requested_size():
    rows = [{"qid": f"q{i:04d}"} for i in range(50)]
    first = cross._sample(rows, 10)
    second = cross._sample(list(reversed(rows)), 10)
    assert len(first) == 10
    assert [r["qid"] for r in first] == [r["qid"] for r in second]
    assert cross._sample(rows, 0) == rows
    assert cross._sample(rows, 999) == rows


def test_quote_missing_flags_a_bare_level_word():
    """实测出来的毛病：判 major 却把「major」写进摘句字段，人工复核时看到的是废话。"""
    assert quote_missing("major", "major") is True
    assert quote_missing("minor", " Major. ") is True
    assert quote_missing("major", "「major」") is True
    assert quote_missing("major", "  ") is True


def test_quote_missing_accepts_a_real_sentence_and_ignores_none():
    assert quote_missing("major", "菜单路径是「仓储 > 打印」") is False
    assert quote_missing("none", "") is False
    assert quote_missing("none", "major") is False

