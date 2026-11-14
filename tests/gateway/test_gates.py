from dataclasses import replace

import pytest
from pydantic import BaseModel

from carepilot.gateway.spec import Executed, Gate, NeedsApproval, Rejected, ToolContext
from conftest import AGENT, CUSTOMER


@pytest.mark.parametrize("tool,args,gate,code", [
    ("missing", {}, Gate.SCHEMA, "unknown_tool"),
    ("read", {"shop_id": "mine", "extra": 1}, Gate.SCHEMA, "invalid_args"),
    ("read", {"shop_id": "mine", "value": "1"}, Gate.SCHEMA, "invalid_args"),
    ("read", {}, Gate.SCHEMA, "invalid_args"),
    ("read", {"shop_id": "mine", "value": -1}, Gate.RULE, "illegal_transition"),
    ("read", {"shop_id": "mine", "value": 101}, Gate.RULE, "too_large"),
    ("internal", {"shop_id": "mine"}, Gate.PERMISSION, "role_denied"),
    ("read", {"shop_id": "theirs"}, Gate.PERMISSION, "not_owner"),
])
def test_invalid_calls_never_reach_tool(setup_gateway, tool, args, gate, code):
    gateway, calls, _, _ = setup_gateway
    outcome = gateway.call(ToolContext(CUSTOMER, "run"), tool, args)
    assert isinstance(outcome, Rejected)
    assert (outcome.gate, outcome.code) == (gate, code)
    assert calls == []
    assert len(gateway.audit(run_id="run")) == 1


def test_risk_requires_approval_and_confirmation(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    ctx = ToolContext(AGENT, "run")
    state = gateway.call(ctx, "state", {"shop_id": "mine"})
    send = gateway.call(ctx, "send", {"shop_id": "mine"})
    assert isinstance(state, NeedsApproval) and state.stage == "approve"
    assert isinstance(send, NeedsApproval) and send.stage == "approve"
    assert gateway.confirm(send.approval_id, AGENT).code == "approval_not_approved"
    assert gateway.approve(state.approval_id, CUSTOMER).code == "approver_role_denied"
    assert gateway.confirm(send.approval_id, CUSTOMER).code == "approver_role_denied"
    approved = gateway.approve(send.approval_id, AGENT)
    assert isinstance(approved, NeedsApproval) and approved.stage == "confirm"
    assert calls == []


def test_gates_short_circuit_and_use_validated_models(setup_gateway):
    gateway, calls, registry, path = setup_gateway
    from carepilot.gateway.gateway import ToolGateway

    seen = []
    def rule(ctx, args):
        assert isinstance(args, BaseModel)
        seen.append("rule")
    def owner(ctx, args):
        assert isinstance(args, BaseModel)
        seen.append("owner")
        return False
    registry["read"] = replace(registry["read"], rules=(rule,), owner_check=owner)
    gateway = ToolGateway(path, registry)
    gateway.call(ToolContext(CUSTOMER, "run"), "read", {"shop_id": "mine", "extra": 1})
    assert seen == []
    outcome = gateway.call(ToolContext(CUSTOMER, "run"), "read", {"shop_id": "mine"})
    assert outcome.gate == Gate.PERMISSION
    assert seen == ["rule", "owner"] and calls == []


@pytest.mark.parametrize("field,code", [("rules", "rule_error"), ("owner_check", "not_owner")])
def test_callback_errors_fail_closed(setup_gateway, field, code):
    _, calls, registry, path = setup_gateway
    from carepilot.gateway.gateway import ToolGateway

    def broken(ctx, args):
        raise RuntimeError("broken")
    registry["read"] = replace(registry["read"], **{field: (broken,) if field == "rules" else broken})
    outcome = ToolGateway(path, registry).call(ToolContext(CUSTOMER, "run"), "read", {"shop_id": "mine"})
    assert outcome.code == code and calls == []


def test_malicious_model_ten_proposals_have_no_effect(setup_gateway):
    gateway, calls, _, _ = setup_gateway
    proposals = [
        ("missing", {}), ("read", {}),
        ("read", {"shop_id": "mine", "extra": 1}),
        ("read", {"shop_id": "mine", "value": "1"}),
        ("read", {"shop_id": "mine", "value": -1}),
        ("read", {"shop_id": "mine", "value": 101}),
        ("internal", {"shop_id": "mine"}),
        ("read", {"shop_id": "theirs"}),
        ("state", {"shop_id": "mine"}), ("send", {"shop_id": "mine"}),
    ]
    fake_model = iter(proposals)
    for tool, args in fake_model:
        assert not isinstance(gateway.call(ToolContext(CUSTOMER, "attack"), tool, args), Executed)
    assert calls == []
    assert len(gateway.audit(run_id="attack")) == 10
