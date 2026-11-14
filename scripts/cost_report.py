"""成本报表（D2）：实测 token × 你配置的单价。

单价必须由你填写（pricing.json），并写明来源和取价日期；没配置就只报 token，**不会编价格**。
网关的价格接口（/api/pricing）返回的是模糊值（model_ratio=0），读不到真实单价，所以不能自动取。

    uv run python scripts/cost_report.py --traces data/traces [--pricing pricing.json] [--out report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from carepilot.observability.cost import PricingError, load_pricing, summarize_cost
from carepilot.observability.tracer import TraceStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, required=True, help="trace 目录（每次 run 一份 JSON）")
    ap.add_argument("--pricing", type=Path, default=None, help="单价配置，见 pricing.example.json；不给则只统计 token")
    ap.add_argument("--out", type=Path, default=None, help="同时写入 markdown 文件")
    args = ap.parse_args(argv)

    if not args.traces.is_dir():
        print(f"错误：trace 目录不存在：{args.traces}", file=sys.stderr)
        return 2
    try:
        pricing = load_pricing(args.pricing) if args.pricing else load_pricing(Path("__no_pricing__.json"))
    except PricingError as exc:
        print(f"错误：单价配置不合法：{exc}", file=sys.stderr)
        return 2
    traces = TraceStore(args.traces).load_all()
    if not traces:
        print("没有 trace：先跑几次请求（治理模式的 API 会自动落盘）", file=sys.stderr)
        return 1
    text = summarize_cost(traces, pricing).to_markdown()
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
