from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from carepilot.auth import Principal, Role
from carepilot.gateway.spec import Risk, RuleViolation, ToolContext, ToolSpec


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    shop_id: str
    value: int = 1
    phone: str = ""


AGENT = Principal("agent-1", Role.AGENT)
ADMIN = Principal("admin-1", Role.ADMIN)
CUSTOMER = Principal("customer-1", Role.CUSTOMER, frozenset({"mine"}))


@pytest.fixture
def setup_gateway(tmp_path: Path):
    from carepilot.gateway.gateway import ToolGateway

    calls: list[tuple[ToolContext, Args]] = []

    def effect(ctx: ToolContext, args: Args) -> dict[str, Any]:
        calls.append((ctx, args))
        return {"value": args.value, "phone": args.phone}

    def rule(ctx: ToolContext, args: Args) -> None:
        if args.value < 0:
            raise RuleViolation("illegal_transition", "cannot reopen")
        if args.value > 100:
            raise RuleViolation("too_large")

    base = ToolSpec(
        "read", Args, effect, Risk.READ, frozenset(Role),
        rules=(rule,),
        owner_check=lambda ctx, args: ctx.principal.is_staff or args.shop_id in ctx.principal.shop_ids,
        summarize_args=lambda args: f"shop={args.shop_id}, value={args.value}, phone=***",
    )
    registry = {
        "read": base,
        "internal": replace(base, name="internal", risk=Risk.WRITE_INTERNAL, roles=frozenset({Role.AGENT, Role.ADMIN})),
        "state": replace(base, name="state", risk=Risk.STATE_CHANGE),
        "send": replace(base, name="send", risk=Risk.EXTERNAL_SEND),
    }
    path = tmp_path / "gateway.sqlite"
    gateway = ToolGateway(path, registry)
    return gateway, calls, registry, path
