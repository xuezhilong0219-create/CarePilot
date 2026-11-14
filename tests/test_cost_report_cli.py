import importlib.util
import json

from carepilot.config import ROOT
from carepilot.observability.spec import LlmCall, RunTrace, Span
from carepilot.observability.tracer import TraceStore


def _cli():
    spec = importlib.util.spec_from_file_location("cost_report", ROOT / "scripts" / "cost_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed(tmp_path, n=3):
    store = TraceStore(tmp_path / "traces")
    for i in range(n):
        store.save(RunTrace(run_id=f"r{i}", ticket_id=f"T-{i}", category="feature_howto", started_at=float(i), ended_at=i + 1.0, status="resolved",
                            spans=[Span("classify", 0.0, 10.0, llm_calls=[LlmCall("triage", "m1", 1000, 100)]),
                                   Span("draft_reply", 0.1, 20.0, llm_calls=[LlmCall("draft", "m1", 3000, 400)])]))
    return tmp_path / "traces"


def test_without_pricing_reports_tokens_only_and_never_a_zero_cost(tmp_path, capsys):
    assert _cli().main(["--traces", str(_seed(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "未配置单价：仅统计 token" in out
    assert "prompt 12000 tokens" in out  # 3 个 run × (1000+3000)
    assert "completion 1500 tokens" in out  # 3 个 run × (100+400)
    assert "0.0 元" not in out and "¥0" not in out


def test_with_pricing_computes_cost_from_measured_tokens(tmp_path, capsys):
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps({"models": {"m1": {"input_per_mtoken": 2.0, "output_per_mtoken": 8.0}}, "currency": "CNY",
                             "source": "unit test", "as_of": "2026-11-14"}), encoding="utf-8")
    assert _cli().main(["--traces", str(_seed(tmp_path)), "--pricing", str(p), "--out", str(tmp_path / "r.md")]) == 0
    # 每个 run：prompt 4000 × 2/1e6 + completion 500 × 8/1e6 = 0.008 + 0.004 = 0.012；3 个 run = 0.036
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "0.012" in text  # 单个 run 的成本
    assert "0.036" in text  # 3 个 run 的合计


def test_bad_pricing_and_missing_dir_are_clean_errors(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": {"m1": {"input_per_mtoken": 1.0, "output_per_mtoken": 1.0}}, "source": "", "as_of": ""}), encoding="utf-8")
    assert _cli().main(["--traces", str(_seed(tmp_path)), "--pricing", str(bad)]) == 2
    assert _cli().main(["--traces", str(tmp_path / "nope")]) == 2
    (tmp_path / "empty").mkdir()
    assert _cli().main(["--traces", str(tmp_path / "empty")]) == 1  # 目录存在但没有 trace：提示先跑请求，退出码 1
