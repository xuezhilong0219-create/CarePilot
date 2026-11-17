"""一键质量门禁：ruff 静态检查 + pytest 全量测试。

为什么是一个脚本而不是 Makefile：本机没有 `make`（Windows 自带），一个跑不起来的
Makefile 只是装饰。CI 里要跑的也是这条命令，所以两边不会漂。

    uv run python scripts/check.py
    uv run python scripts/check.py --skip tests    # 只跑 lint
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = (
    ("lint", [sys.executable, "-m", "ruff", "check", "."]),
    ("tests", [sys.executable, "-m", "pytest", "-q", "--no-header"]),
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    skip = set()
    if "--skip" in argv:
        skip = {s.strip() for s in argv[argv.index("--skip") + 1].split(",") if s.strip()}

    failed: list[str] = []
    ran: list[str] = []
    for name, cmd in STEPS:
        if name in skip:
            print(f"[{name}] 跳过")
            continue
        ran.append(name)
        started = time.monotonic()
        print(f"[{name}] {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        took = time.monotonic() - started
        print(f"[{name}] {'PASS' if code == 0 else 'FAIL'}（{took:.1f}s）\n", flush=True)
        if code != 0:
            failed.append(name)

    if failed:
        print(f"门禁失败：{', '.join(failed)}")
        return 1
    if not ran:
        print("门禁未运行：没有执行任何检查")
        return 2
    print(f"门禁通过：{' + '.join(ran)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
