import json

import pytest

from carepilot.observability.cost import (
    PricingError,
    load_pricing,
    model_cost,
    summarize_cost,
)
from carepilot.observability.spec import LlmCall, RunTrace, Span


def _pricing_file(tmp_path, input_price, output_price, *, source="", as_of="", currency="CNY"):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "models": {"m": {"input_per_mtoken": input_price, "output_per_mtoken": output_price}},
                "currency": currency,
                "source": source,
                "as_of": as_of,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_missing_file_means_unpriced_not_zero(tmp_path):
    pricing = load_pricing(tmp_path / "nope.json")
    assert model_cost(pricing, "m", 1_000_000, 1_000_000) is None


def test_null_prices_mean_unpriced(tmp_path):
    pricing = load_pricing(_pricing_file(tmp_path, None, None))
    assert model_cost(pricing, "m", 1000, 1000) is None


def test_cost_uses_configured_price_exactly(tmp_path):
    pricing = load_pricing(_pricing_file(tmp_path, 2.0, 8.0, source="gateway", as_of="2026-01-01"))
    assert model_cost(pricing, "m", 1_000_000, 500_000) == pytest.approx(2.0 + 4.0)
    assert model_cost(pricing, "unknown-model", 1_000_000, 0) is None


@pytest.mark.parametrize("kwargs", [{"source": "", "as_of": "2026-01-01"}, {"source": "x", "as_of": ""}])
def test_price_without_source_or_date_raises(tmp_path, kwargs):
    with pytest.raises(PricingError):
        load_pricing(_pricing_file(tmp_path, 2.0, 8.0, **kwargs))


def _trace(run_id, category, calls, *, ticket_id=None):
    spans = [Span(name="draft", started_at=0.0, llm_calls=calls)]
    return RunTrace(run_id=run_id, ticket_id=ticket_id, category=category, started_at=0.0, status="resolved", spans=spans)


def test_summarize_hand_computed(tmp_path):
    pricing = load_pricing(_pricing_file(tmp_path, 2.0, 8.0, source="gateway", as_of="2026-01-01"))
    traces = [
        _trace(
            "r1",
            "cat_a",
            [LlmCall("triage", "m", 100, 50), LlmCall("triage", "m", 200, 100)],
            ticket_id="T-1",
        ),
        _trace("r2", None, [LlmCall("draft", "m", 1000, 500, usage_missing=True)]),
    ]
    report = summarize_cost(traces, pricing)

    assert report.runs == 2
    assert report.calls_by_purpose == {"draft": 1, "triage": 2}
    assert (report.prompt_tokens, report.completion_tokens) == (1300, 650)
    assert report.usage_missing_calls == 1
    assert report.per_ticket[0]["cost"] == pytest.approx((100 * 2 + 50 * 8 + 200 * 2 + 100 * 8) / 1_000_000)
    assert report.per_ticket[1]["cost"] == pytest.approx((1000 * 2 + 500 * 8) / 1_000_000)

    by_cat = {row["category"]: row for row in report.per_category}
    assert by_cat["cat_a"]["runs"] == 1
    assert by_cat["cat_a"]["avg_total_tokens"] == pytest.approx(450)
    assert by_cat["cat_a"]["avg_cost"] == pytest.approx(0.0018)
    assert by_cat["unknown"]["runs"] == 1
    assert by_cat["unknown"]["avg_total_tokens"] == pytest.approx(1500)
    assert by_cat["unknown"]["avg_cost"] == pytest.approx(0.006)

    assert report.per_model == [{"model": "m", "calls": 3, "prompt_tokens": 1300, "completion_tokens": 650}]
    assert json.loads(json.dumps(report.to_dict()))["runs"] == 2


def test_markdown_warns_missing_usage_and_does_not_show_zero_cost(tmp_path):
    pricing = load_pricing(_pricing_file(tmp_path, None, None))
    traces = [_trace("r1", "cat_a", [LlmCall("draft", "m", 10, 5, usage_missing=True)])]
    report = summarize_cost(traces, pricing)
    md = report.to_markdown()

    assert "未配置单价：仅统计 token" in md
    assert "token 统计不完整" in md
    assert "0.0000" not in md
    assert report.per_category[0]["avg_cost"] is None
