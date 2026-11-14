import importlib.util
import json
import re
from pathlib import Path

from carepilot.observability.spec import LlmCall, RunTrace, Span

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "trace_viewer.py"
DATA_RE = re.compile(
    r'<script type="application/json" id="trace-data">(.*?)</script>', re.S | re.I
)

_spec = importlib.util.spec_from_file_location("trace_viewer_script", SCRIPT)
assert _spec and _spec.loader
trace_viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_viewer)


def _trace(run_id: str, started_at: float) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        ticket_id="T-" + run_id,
        category="platform",
        started_at=started_at,
        ended_at=started_at + 1.0,
        status="resolved",
        spans=[Span("classify", started_at, 10.0, "in", "out", [LlmCall("triage", "m", 1, 2)])],
    )


def _write(path: Path, trace: RunTrace) -> None:
    path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False), encoding="utf-8")


def _embedded(html: str) -> list[dict]:
    match = DATA_RE.search(html)
    assert match
    return json.loads(match.group(1))


def test_load_skips_corrupt_files_and_main_writes_html(work_tmp, capsys):
    traces = work_tmp / "traces"
    traces.mkdir()
    _write(traces / "good.json", _trace("r1", 1.0))
    (traces / "broken.json").write_text("{not json", encoding="utf-8")
    (traces / "empty.json").write_text("{}", encoding="utf-8")

    out = work_tmp / "viewer" / "trace.html"
    assert trace_viewer.main(["--traces", str(traces), "--out", str(out)]) == 0
    assert out.is_file()
    assert [item["run_id"] for item in _embedded(out.read_text(encoding="utf-8"))] == ["r1"]
    captured = capsys.readouterr()
    assert "broken.json" in captured.err
    assert "empty.json" in captured.err


def test_limit_keeps_most_recent_runs(work_tmp):
    traces = work_tmp / "traces"
    traces.mkdir()
    for index, run_id in enumerate(("r1", "r2", "r3"), 1):
        _write(traces / f"{run_id}.json", _trace(run_id, float(index)))
    out = work_tmp / "trace.html"

    assert trace_viewer.main(["--traces", str(traces), "--out", str(out), "--limit", "2"]) == 0
    ids = [item["run_id"] for item in _embedded(out.read_text(encoding="utf-8"))]
    assert sorted(ids) == ["r2", "r3"]


def test_empty_directory_writes_empty_viewer(work_tmp):
    traces = work_tmp / "traces"
    traces.mkdir()
    out = work_tmp / "trace.html"
    assert trace_viewer.main(["--traces", str(traces), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert _embedded(html) == []
    assert "没有 trace" in html


def test_missing_directory_returns_2_with_clear_error(work_tmp, capsys):
    out = work_tmp / "trace.html"
    assert trace_viewer.main(["--traces", str(work_tmp / "missing"), "--out", str(out)]) == 2
    assert not out.exists()
    assert "不存在" in capsys.readouterr().err


def test_path_that_is_a_file_returns_2(work_tmp, capsys):
    path = work_tmp / "not-a-dir.json"
    path.write_text("{}", encoding="utf-8")
    out = work_tmp / "trace.html"
    assert trace_viewer.main(["--traces", str(path), "--out", str(out)]) == 2
    assert "不是目录" in capsys.readouterr().err
