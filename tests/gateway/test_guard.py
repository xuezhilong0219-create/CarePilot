import pytest

from carepilot.gateway.guard import BudgetExceeded, RunBudget, RunGuard


def test_same_proposal_model_is_stopped_before_twenty_steps():
    guard = RunGuard(RunBudget())
    def fake_model():
        return "read", {"a": 1}
    attempts = 0
    with pytest.raises(BudgetExceeded) as caught:
        while True:
            attempts += 1
            tool, args = fake_model()
            guard.step("tool", tool=tool, args=args)
    assert caught.value.reason == "loop" and attempts == 3


def test_different_proposals_stop_on_twenty_first_step():
    guard = RunGuard(RunBudget())
    def fake_model(turn):
        return "read", {"a": turn}
    attempts = 0
    with pytest.raises(BudgetExceeded) as caught:
        while True:
            attempts += 1
            tool, args = fake_model(attempts)
            guard.step("tool", tool=tool, args=args)
    assert caught.value.reason == "steps" and attempts == 21


def test_nonconsecutive_loop_uses_canonical_arguments():
    guard = RunGuard(RunBudget())
    for _ in range(3):
        guard.step("tool", tool="read", args={"b": 2, "a": 1})
        guard.step("node")
    with pytest.raises(BudgetExceeded, match="loop"):
        guard.step("tool", tool="read", args={"a": 1, "b": 2})


def test_tokens_time_and_exact_boundaries():
    now = [10.0]
    guard = RunGuard(RunBudget(max_tokens=5, max_seconds=2), clock=lambda: now[0])
    guard.step("node", tokens=5)
    with pytest.raises(BudgetExceeded, match="tokens"):
        guard.step("node", tokens=1)
    guard = RunGuard(RunBudget(max_seconds=2), clock=lambda: now[0])
    now[0] = 12.0
    guard.step("node")
    now[0] = 12.001
    with pytest.raises(BudgetExceeded, match="time"):
        guard.step("node")


def test_budget_exceeded_is_latched_and_tokens_cannot_be_negative():
    guard = RunGuard(RunBudget(max_tokens=1))
    with pytest.raises(ValueError):
        guard.step("node", tokens=-1)
    with pytest.raises(BudgetExceeded, match="tokens"):
        guard.step("node", tokens=2)
    with pytest.raises(BudgetExceeded, match="tokens"):
        guard.step("node")
