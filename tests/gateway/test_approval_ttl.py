"""审批过期：待批准和"已批准待二次确认"的审批超过 TTL 就不能再执行，避免拿陈旧上下文做决定。"""

from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import Executed, Gate, NeedsApproval, Rejected, ToolContext

from conftest import ADMIN, AGENT


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _gw(setup_gateway, tmp_path, ttl):
    _, _, registry, _ = setup_gateway
    clock = Clock()
    return ToolGateway(tmp_path / "ttl.sqlite", registry, clock=clock, approval_ttl_s=ttl), clock


def test_pending_approval_expires(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, 100)
    out = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    assert isinstance(out, NeedsApproval)
    clock.t += 101
    res = gw.approve(out.approval_id, ADMIN)
    assert isinstance(res, Rejected) and res.gate is Gate.RISK and res.code == "approval_expired"
    assert any(e["code"] == "approval_expired" for e in gw.audit(run_id="r1"))


def test_approval_within_ttl_still_works(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, 100)
    out = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    clock.t += 99
    assert isinstance(gw.approve(out.approval_id, ADMIN), Executed)


def test_approved_send_must_be_confirmed_within_ttl_of_the_approval(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, 100)
    out = gw.call(ToolContext(AGENT, "r1", "T-1"), "send", {"shop_id": "S1"})
    clock.t += 60
    assert gw.approve(out.approval_id, ADMIN).stage == "confirm"
    clock.t += 60  # 距创建 120 秒，但距批准只有 60 秒：仍然有效
    assert isinstance(gw.confirm(out.approval_id, ADMIN), Executed)


def test_unconfirmed_send_expires_and_expired_proposal_can_be_reopened(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, 100)
    ctx = ToolContext(AGENT, "r1", "T-1")
    out = gw.call(ctx, "send", {"shop_id": "S1"})
    gw.approve(out.approval_id, ADMIN)
    clock.t += 101
    res = gw.confirm(out.approval_id, ADMIN)
    assert isinstance(res, Rejected) and res.code == "approval_expired"
    # 同一个 run 里重新提议同样的动作：重新打开成待批准，而不是被永久卡死
    reopened = gw.call(ctx, "send", {"shop_id": "S1"})
    assert isinstance(reopened, NeedsApproval) and reopened.stage == "approve"
    gw.approve(reopened.approval_id, ADMIN)
    assert isinstance(gw.confirm(reopened.approval_id, ADMIN), Executed)


def test_replayed_confirmation_is_not_subject_to_expiry(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, 100)
    out = gw.call(ToolContext(AGENT, "r1", "T-1"), "send", {"shop_id": "S1"})
    gw.approve(out.approval_id, ADMIN)
    first = gw.confirm(out.approval_id, ADMIN)
    clock.t += 10_000
    again = gw.confirm(out.approval_id, ADMIN)
    assert isinstance(first, Executed) and isinstance(again, Executed) and again.replayed


def test_ttl_none_never_expires(setup_gateway, tmp_path):
    gw, clock = _gw(setup_gateway, tmp_path, None)
    out = gw.call(ToolContext(AGENT, "r1", "T-1"), "state", {"shop_id": "S1"})
    clock.t += 10**9
    assert isinstance(gw.approve(out.approval_id, ADMIN), Executed)
