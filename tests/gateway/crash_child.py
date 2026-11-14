import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from carepilot.auth import Principal, Role
from carepilot.gateway.gateway import ToolGateway
from carepilot.gateway.spec import Executed, Risk, ToolContext, ToolSpec


class FileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("call", "recover", "concurrent"))
    parser.add_argument("db", type=Path)
    parser.add_argument("effects", type=Path)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--risk", type=int, default=int(Risk.WRITE_INTERNAL))
    options = parser.parse_args()

    def effect(ctx: ToolContext, args: FileArgs) -> dict[str, Any]:
        with options.effects.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": ctx.idem_key, "text": args.text}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"sent": args.text}

    def verify(ctx: ToolContext, args: FileArgs) -> dict[str, Any] | None:
        if options.effects.exists():
            for line in options.effects.read_text(encoding="utf-8").splitlines():
                if json.loads(line)["key"] == ctx.idem_key:
                    return {"sent": args.text}
        return None

    spec = ToolSpec("append", FileArgs, effect, Risk(options.risk), frozenset({Role.AGENT}), verify=None if options.no_verify else verify)
    if options.mode == "concurrent":
        options.ready.write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 20
        while not options.release.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("release marker")
            time.sleep(0.01)
    gateway = ToolGateway(options.db, {"append": spec})
    principal = Principal("agent", Role.AGENT)
    ctx = ToolContext(principal, "crash-run", "ticket")
    recovered = gateway.reconcile() if options.mode == "recover" else []
    outcome = gateway.call(ctx, "append", {"text": "one"})
    if not isinstance(outcome, Executed) and options.risk == int(Risk.STATE_CHANGE):
        outcome = gateway.approve(outcome.approval_id, principal)
    elif not isinstance(outcome, Executed) and options.risk == int(Risk.EXTERNAL_SEND):
        gateway.approve(outcome.approval_id, principal)
        outcome = gateway.confirm(outcome.approval_id, principal)
    print(json.dumps({"outcome": asdict(outcome), "reconciled": recovered}, ensure_ascii=False), flush=True)
    if options.mode != "recover" and not isinstance(outcome, Executed):
        sys.exit(1)


if __name__ == "__main__":
    main()
