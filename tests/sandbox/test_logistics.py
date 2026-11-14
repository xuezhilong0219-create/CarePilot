import time

import pytest

from carepilot.sandbox.logistics import LogisticsClient


class CountingTransport:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0

    def __call__(self, order_id):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_success_returns_ok_and_attempts_once():
    transport = CountingTransport([{"carrier": "顺丰", "latest_node": "运输中"}])
    result = LogisticsClient(transport, timeout_s=0.05).query("A1")
    assert result == {
        "status": "ok",
        "carrier": "顺丰",
        "latest_node": "运输中",
        "attempts": 1,
    }
    assert transport.calls == 1


def test_timeout_then_success_retries_once():
    transport = CountingTransport([TimeoutError("slow"), {"carrier": "中通"}])
    result = LogisticsClient(transport, timeout_s=0.05, sleep=lambda _: None).query("A1")
    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert transport.calls == 2


def test_all_timeouts_degrade_without_raising():
    transport = CountingTransport([TimeoutError("slow")])
    result = LogisticsClient(transport, timeout_s=0.05, sleep=lambda _: None).query("A1")
    assert result["status"] == "unavailable"
    assert result["message"] == "暂时查不到物流信息，请稍后再试"
    assert result["attempts"] == 2
    assert transport.calls == 2


def test_connection_error_is_also_external_failure():
    transport = CountingTransport([ConnectionError("refused")])
    result = LogisticsClient(transport, timeout_s=0.05, sleep=lambda _: None).query("A1")
    assert result["status"] == "unavailable"
    assert result["attempts"] == 2


def test_retries_zero_calls_transport_only_once():
    transport = CountingTransport([TimeoutError("slow")])
    result = LogisticsClient(transport, timeout_s=0.05, retries=0).query("A1")
    assert result["status"] == "unavailable"
    assert result["attempts"] == 1
    assert transport.calls == 1


def test_other_exception_is_not_swallowed():
    transport = CountingTransport([ValueError("bug in parsing")])
    with pytest.raises(ValueError, match="bug in parsing"):
        LogisticsClient(transport, timeout_s=0.05).query("A1")
    assert transport.calls == 1


def test_blocking_transport_is_cut_off_by_timeout():
    def blocking(order_id):
        time.sleep(0.4)
        return {"carrier": "圆通"}

    started = time.monotonic()
    result = LogisticsClient(blocking, timeout_s=0.05, retries=0).query("A1")
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert elapsed < 0.3  # 不能真的等 transport 阻塞结束


def test_sleep_is_called_between_retries():
    transport = CountingTransport([TimeoutError("slow")])
    delays: list[float] = []
    LogisticsClient(transport, timeout_s=0.05, sleep=delays.append).query("A1")
    assert len(delays) == 1
    assert delays[0] >= 0
