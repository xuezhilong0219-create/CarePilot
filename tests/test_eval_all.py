import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from carepilot.evaluation.schema import EvalItem

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_all = _load("eval_all")
export_badcases = _load("export_badcases")


def _item(qid, category, answerable, gold=None):
    return EvalItem(
        qid=qid,
        question=f"问题 {qid}",
        category=category,
        answerable=answerable,
        gold_doc_ids=gold or [],
        gold_title_keys=[],
    )


def _items():
    items = [_item(f"a{i}", "feature_howto", True, ["doc"]) for i in range(6)]
    items += [_item(f"d{i}", "data_exception", False) for i in range(4)]
    items += [_item(f"o{i}", "other", False) for i in range(4)]
    items += [_item(f"v{i}", "other", False) for i in range(4)]
    return items


def test_stratified_subset_keeps_every_group_and_is_reproducible():
    items = _items()
    subset = eval_all.stratified_subset(items, 14, seed=7)
    groups = {}
    for item in subset:
        groups[eval_all.stratum(item)] = groups.get(eval_all.stratum(item), 0) + 1

    assert len(subset) == 14
    assert set(groups) == set(eval_all.GROUPS)
    assert all(count >= 3 for count in groups.values())

    again = eval_all.stratified_subset(items, 14, seed=7)
    assert [i.qid for i in subset] == [i.qid for i in again]


def test_stratified_subset_returns_all_when_n_is_large():
    items = _items()
    assert [i.qid for i in eval_all.stratified_subset(items, 100)] == [i.qid for i in items]


def test_step_retrieval_marks_missing_dataset(tmp_path):
    result = eval_all.step_retrieval(
        {"dataset_path": tmp_path / "nope.jsonl", "quick": False, "kb_dir": tmp_path, "settings": None}
    )
    assert result["status"] == "MISSING"


def test_step_tools_marks_unprovided_runner(tmp_path, monkeypatch):
    dataset = tmp_path / "tool.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(eval_all.importlib.util, "find_spec", lambda name: None)
    result = eval_all.step_tools(
        {"tool_dataset_path": dataset, "quick": False, "kb_dir": tmp_path, "settings": None}
    )
    assert result["status"] == "UNPROVIDED"


def test_run_all_continues_after_failure_and_exits_nonzero(tmp_path):
    called = []

    def make(name):
        def step(ctx):
            called.append(name)
            return {"status": "OK", name: {"value": 0.5, "n": 1}}

        return step

    def boom(ctx):
        called.append("retrieval")
        raise RuntimeError("boom")

    steps = {
        "tests": make("tests"),
        "retrieval": boom,
        "pipeline": make("pipeline"),
        "tools": make("tools"),
    }
    result = eval_all.run_all(
        quick=True,
        report_dir=tmp_path,
        dataset_path=tmp_path / "missing.jsonl",
        kb_dir=tmp_path,
        step_fns=steps,
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert called == ["tests", "retrieval", "pipeline", "tools"]
    assert result["exit_code"] != 0
    assert result["results"]["retrieval"]["status"] == "FAILED"
    assert "FAILED" in result["markdown"] and "boom" in result["markdown"]
    assert result["json_path"].exists()
    payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert payload["results"]["pipeline"]["status"] == "OK"


def test_run_all_skips_requested_steps(tmp_path):
    result = eval_all.run_all(
        quick=True,
        skip=["tests", "tools"],
        report_dir=tmp_path,
        dataset_path=tmp_path / "missing.jsonl",
        kb_dir=tmp_path,
        step_fns={},
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )
    assert result["results"]["tests"]["status"] == "SKIPPED"
    assert result["results"]["tools"]["status"] == "SKIPPED"
    assert result["exit_code"] == 0


def _write_dataset(tmp_path, items):
    path = tmp_path / "dataset.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    return path


def test_expected_outcome_matches_four_groups():
    assert export_badcases.expected_outcome(_item("a1", "feature_howto", True)) == "auto_reply"
    assert export_badcases.expected_outcome(_item("d1", "data_exception", False)) == "escalate"
    assert export_badcases.expected_outcome(_item("o1", "other", False)) == "escalate/need_info"
    assert export_badcases.expected_outcome(_item("v1", "other", False)) == "need_info"
    assert export_badcases.outcome_matches(_item("o1", "other", False), "need_info")


def test_export_badcases_only_writes_mismatches(tmp_path):
    dataset = _write_dataset(
        tmp_path,
        [
            _item("a1", "feature_howto", True, ["doc"]),
            _item("d1", "data_exception", False),
        ],
    )
    report = tmp_path / "run.items.jsonl"
    rows = [
        {"qid": "a1", "resolution": "escalate"},
        {"qid": "d1", "resolution": "escalate"},
    ]
    with open(report, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = tmp_path / "badcases.jsonl"
    candidates = export_badcases.export_candidates(dataset, [report], out)
    assert [c["qid"] for c in candidates] == ["a1"]
    candidate = candidates[0]
    assert candidate["expected"] == "auto_reply"
    assert candidate["actual"] == "escalate"
    assert candidate["source_report"] == report.name
    assert json.loads(out.read_text(encoding="utf-8").strip())["qid"] == "a1"


def test_export_badcases_treats_error_as_mismatch(tmp_path):
    dataset = _write_dataset(tmp_path, [_item("a1", "feature_howto", True, ["doc"])])
    report = tmp_path / "run.items.jsonl"
    report.write_text(json.dumps({"qid": "a1", "error": "boom"}) + "\n", encoding="utf-8")

    candidates = export_badcases.export_candidates(dataset, [report], tmp_path / "out.jsonl")
    assert candidates[0]["actual"] == "ERROR"
