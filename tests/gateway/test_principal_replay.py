"""回放不能跨主体：同一个 (run_id, 工具, 参数) 第二次被"另一个调用者"调用时，不能拿到第一个调用者的缓存结果。

回归：集成测试里发现，同一 run 内坐席查工单会拿到客户那次的缓存结果（少了 SLA 和处理人）。
幂等键仍然是 sha1(run_id + tool + 规范化参数)，所以不改键，而是在回放时校验调用者一致。
"""

from carepilot.gateway.spec import Executed, Gate, Rejected, ToolContext

from conftest import ADMIN, AGENT, CUSTOMER


def test_replay_by_the_same_principal_is_still_allowed(setup_gateway):
    gw, calls, _, _ = setup_gateway
    ctx = ToolContext(AGENT, "r1", "T-1")
    a = gw.call(ctx, "read", {"shop_id": "mine"})
    b = gw.call(ctx, "read", {"shop_id": "mine"})
    assert isinstance(a, Executed) and isinstance(b, Executed) and b.replayed and len(calls) == 1


def test_replay_by_a_different_principal_is_rejected_and_audited(setup_gateway):
    gw, calls, _, _ = setup_gateway
    first = gw.call(ToolContext(AGENT, "r1", "T-1"), "read", {"shop_id": "mine"})
    assert isinstance(first, Executed)
    second = gw.call(ToolContext(CUSTOMER, "r1", "T-1"), "read", {"shop_id": "mine"})
    assert isinstance(second, Rejected) and second.gate is Gate.RISK and second.code == "principal_mismatch"
    assert len(calls) == 1  # 没有再次执行，也没有回放第一个调用者的结果
    assert any(e["code"] == "principal_mismatch" and e["actor_id"] == CUSTOMER.user_id for e in gw.audit(run_id="r1"))


def test_different_roles_with_same_user_id_are_different_principals(setup_gateway):
    from carepilot.auth import Principal, Role

    gw, calls, _, _ = setup_gateway
    gw.call(ToolContext(Principal("u1", Role.AGENT), "r1", None), "read", {"shop_id": "mine"})
    out = gw.call(ToolContext(Principal("u1", Role.ADMIN), "r1", None), "read", {"shop_id": "mine"})
    assert isinstance(out, Rejected) and out.code == "principal_mismatch" and len(calls) == 1


def test_a_different_run_id_is_unaffected(setup_gateway):
    gw, calls, _, _ = setup_gateway
    gw.call(ToolContext(AGENT, "r1", "T-1"), "read", {"shop_id": "mine"})
    out = gw.call(ToolContext(CUSTOMER, "r2", "T-1"), "read", {"shop_id": "mine"})
    assert isinstance(out, Executed) and not out.replayed and len(calls) == 2
