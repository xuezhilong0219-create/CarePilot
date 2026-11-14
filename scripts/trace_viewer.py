"""把一个目录下的 RunTrace JSON 渲染成单文件 HTML。

这里直接读目录里的 `*.json`，不依赖 TraceStore。读不了的文件跳过并提示，
不让一个坏文件毁掉整次导出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许 `python scripts/trace_viewer.py` 直接从仓库根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.observability.spec import RunTrace  # noqa: E402
from carepilot.viewer import render_trace_viewer  # noqa: E402


def load_traces(directory: Path, limit: int | None = None) -> list[RunTrace]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"trace 目录不存在：{directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    traces: list[RunTrace] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("顶层不是 JSON 对象")
            traces.append(RunTrace.from_dict(data))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"跳过无法读取的 trace：{path.name}（{exc}）", file=sys.stderr)

    traces.sort(key=lambda trace: trace.started_at)
    if limit is not None:
        traces = traces[-limit:] if limit > 0 else []
    return traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 trace JSON 目录渲染成单文件 HTML 查看器")
    parser.add_argument("--traces", type=Path, required=True, help="包含 *.json trace 的目录")
    parser.add_argument("--out", type=Path, required=True, help="输出的 HTML 路径")
    parser.add_argument("--limit", type=int, default=None, help="只取最近 N 个 run（按 started_at）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能为负数")
    try:
        traces = load_traces(args.traces, args.limit)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    html = render_trace_viewer(traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"已写入 {len(traces)} 个 run 到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
