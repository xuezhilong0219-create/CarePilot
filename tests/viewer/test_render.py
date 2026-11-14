import json
import re
import shutil
import subprocess

import pytest

from carepilot.observability.spec import LlmCall, RunTrace, Span
from carepilot.viewer import render_trace_viewer

DATA_RE = re.compile(
    r'<script type="application/json" id="trace-data">(.*?)</script>', re.S | re.I
)
SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)

DANGEROUS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval",
    "new Function",
)


def _trace(
    run_id: str = "r1",
    *,
    ticket_id: str | None = "T-1",
    category: str | None = "platform",
    status: str = "resolved",
    started_at: float = 1000.0,
    ended_at: float = 1002.0,
    spans: list[Span] | None = None,
) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        ticket_id=ticket_id,
        category=category,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        spans=spans
        if spans is not None
        else [
            Span(
                "classify",
                started_at,
                500.0,
                "客户问店铺授权",
                "category=platform",
                [LlmCall("triage", "qwen-flash", 100, 20, 480.0)],
            ),
            Span(
                "draft_reply",
                started_at + 0.5,
                700.0,
                "3 条资料",
                "auto_reply",
                [LlmCall("draft", "qwen-flash", 900, 120, 690.0, retries=1)],
                retries=1,
            ),
        ],
    )


def _data(html: str):
    match = DATA_RE.search(html)
    assert match, "缺少 trace-data JSON 块"
    return json.loads(match.group(1))


def _logic_scripts(html: str) -> list[str]:
    return [body for attrs, body in SCRIPT_RE.findall(html) if "application/json" not in attrs]


def test_empty_traces_shows_empty_state():
    html = render_trace_viewer([])
    assert _data(html) == []
    assert "没有 trace" in html


def test_self_contained_without_external_resources():
    html = render_trace_viewer([_trace()])
    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()
    assert "cdn" not in html.lower()
    assert "http://" not in html.lower()
    assert "https://" not in html.lower()


def test_multiple_runs_are_all_embedded():
    traces = [_trace("r1"), _trace("r2", status="error"), _trace("r3", ticket_id=None)]
    html = render_trace_viewer(traces)
    assert _data(html) == [trace.to_dict() for trace in traces]


def test_script_has_no_html_injection_api():
    html = render_trace_viewer([_trace()])
    scripts = _logic_scripts(html)
    assert len(scripts) == 1
    source = scripts[0]
    for api in DANGEROUS:
        assert api not in source, f"渲染脚本里出现了危险 API：{api}"


def test_xss_payloads_are_escaped_and_round_trip():
    payloads = [
        "</script><script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "\u2028\u2029",
        "&<>",
    ]
    span = Span(
        name=payloads[4 % len(payloads)],
        started_at=1.0,
        duration_ms=10.0,
        input_summary=payloads[0],
        output_summary=payloads[1],
        error=payloads[2],
        llm_calls=[LlmCall(purpose=payloads[3], model=payloads[0], error=payloads[1])],
    )
    trace = RunTrace(
        run_id=payloads[0],
        ticket_id=payloads[1],
        category=payloads[2],
        started_at=1.0,
        ended_at=2.0,
        status=payloads[3],
        spans=[span],
        meta={"note": payloads[1]},
    )
    html = render_trace_viewer([trace])

    for payload in payloads:
        assert payload not in html, f"未转义的载荷出现在 HTML 里：{payload!r}"
    assert html.count("<script") == 2
    assert html.count("</script>") == 2
    json_block = DATA_RE.search(html)
    assert json_block is not None
    assert "</script>" not in json_block.group(1)
    assert _data(html) == [trace.to_dict()]


def test_node_syntax_check(work_tmp):
    node = shutil.which("node")
    if not node:
        pytest.skip("没有 node，跳过 JS 语法检查")
    html = render_trace_viewer([_trace(), _trace("r2", status="error")])
    scripts = _logic_scripts(html)
    assert scripts
    script_path = work_tmp / "viewer.js"
    script_path.write_text(scripts[0], encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
