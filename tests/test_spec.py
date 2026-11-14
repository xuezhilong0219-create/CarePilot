from carepilot.auth import Principal, Role
from carepilot.gateway.spec import Executed, Gate, NeedsApproval, Rejected, Risk, RuleViolation


def test_risk_levels_are_ordered_by_blast_radius():
    assert Risk.READ < Risk.WRITE_INTERNAL < Risk.STATE_CHANGE < Risk.EXTERNAL_SEND


def test_principal_is_staff():
    assert Principal("a1", Role.AGENT).is_staff and Principal("x", Role.ADMIN).is_staff
    assert not Principal("c1", Role.CUSTOMER, frozenset({"S1"})).is_staff


def test_outcome_types_and_rule_violation_code():
    assert Executed({"ok": 1}).replayed is False
    assert NeedsApproval("ap1", "t", {}, "approve").stage == "approve"
    assert Rejected(Gate.RULE, "illegal_transition", "x").gate is Gate.RULE
    assert RuleViolation("illegal_transition").message == "illegal_transition"
