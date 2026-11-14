"""工具"副作用已发生、之后才抛异常"：网关不能把它当成"没执行"而允许重试，否则会重复执行（回归：原规格的缺陷）。"""

from pathlib import Path
from typing import Any

from carepilot.auth import Role
from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import Executed, NeedsApproval, Rejected, Risk, ToolContext, ToolSpec

from conftest import ADMIN, AGENT, Args


def _gateway(tmp_path: Path, *, effect_happens: bool, verify_mode: str, risk: Risk = Risk.STATE_CHANGE):
    """effect_happens: 抛异常前副作用是否已经发生。verify_mode: found / raises / none。"""
    done: list[str] = []
    attempts = {"n": 0}

    def fn(ctx: ToolContext, args: Args) -> dict[str, Any]:
        attempts["n"] += 1
        if effect_happens:
            done.append(args.shop_id)
        raise ConnectionError("downstream connection reset after the request was sent")

    def verify(ctx: ToolContext, args: Args):
        if verify_mode == "raises":
            raise RuntimeError("verify backend down")
        return {"value": args.value, "phone": "", "adopted": True} if done else None

    spec = ToolSpec(
        "state", Args, fn, risk, frozenset(Role),
        owner_check=lambda ctx, a: True,
        verify=None if verify_mode == "none" else verify,
        summarize_args=lambda a: f"shop={a.shop_id}",
    )
    return ToolGateway(tmp_path / "g.sqlite", {"state": spec}), done, attempts


def _propose_and_approve(gw: ToolGateway, run_id: str = "r1"):
    out = gw.call(ToolContext(AGENT, run_id, "T-1"), "state", {"shop_id": "S1"})
    assert isinstance(out, NeedsApproval)
    return out.approval_id, gw.approve(out.approval_id, ADMIN)


def test_effect_happened_and_verify_confirms_it_adopts_result_instead_of_retrying(tmp_path):
    gw, done, attempts = _gateway(tmp_path, effect_happens=True, verify_mode="found")
    _, out = _propose_and_approve(gw)
    assert isinstance(out, Executed) and out.result["adopted"] is True and not out.replayed
    assert done == ["S1"] and attempts["n"] == 1
    # 再来一次（同 run 同参数）只会回放，不会再执行
    again = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    assert isinstance(again, Executed) and again.replayed and attempts["n"] == 1
    codes = [e["code"] for e in gw.audit(run_id="r1") if e["decision"] == "executed"]
    assert "adopted_after_error" in codes


def test_effect_did_not_happen_verify_says_none_allows_a_clean_retry(tmp_path):
    gw, done, attempts = _gateway(tmp_path, effect_happens=False, verify_mode="found")
    approval_id, out = _propose_and_approve(gw)
    assert isinstance(out, Rejected) and out.code == "tool_error" and done == []
    # 副作用确实没发生：审批回到 pending，可以重新批准重试
    retry = gw.approve(approval_id, ADMIN)
    assert isinstance(retry, Rejected) and attempts["n"] == 2 and done == []


def test_no_verify_on_a_state_change_tool_is_never_retried_automatically(tmp_path):
    gw, done, attempts = _gateway(tmp_path, effect_happens=True, verify_mode="none")
    approval_id, out = _propose_and_approve(gw)
    assert isinstance(out, Rejected) and out.code == "tool_error_unverified"
    again = gw.approve(approval_id, ADMIN)
    assert isinstance(again, Rejected)
    again2 = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    assert isinstance(again2, (Rejected, NeedsApproval))
    assert attempts["n"] == 1 and done == ["S1"]  # 全程只执行过一次
    assert gw.reconcile()[0]["status"] == "needs_manual"


def test_verify_itself_failing_keeps_the_action_unresolved_for_reconcile(tmp_path):
    gw, done, attempts = _gateway(tmp_path, effect_happens=True, verify_mode="raises")
    approval_id, out = _propose_and_approve(gw)
    assert isinstance(out, Rejected) and out.code == "tool_error_unverified"
    again = gw.approve(approval_id, ADMIN)
    assert isinstance(again, Rejected) and attempts["n"] == 1


def test_read_tool_error_is_retryable_without_verify(tmp_path):
    gw, done, attempts = _gateway(tmp_path, effect_happens=False, verify_mode="none", risk=Risk.READ)
    out1 = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    out2 = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    assert isinstance(out1, Rejected) and out1.code == "tool_error"
    assert isinstance(out2, Rejected) and attempts["n"] == 2  # 只读工具允许直接重试
