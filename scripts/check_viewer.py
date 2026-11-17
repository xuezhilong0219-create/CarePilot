"""用真实浏览器验证 trace 查看器：渲染、跑脚本、点一个 run、看详情。

为什么单独一个脚本：查看器的单测只能保证「HTML 里的字符串是对的」，保证不了
「浏览器愿意执行它」——CSP、`<script>` 里被转义的 `</script>`、`file://` 下的
限制，都要真跑一次才知道。README 里承认过这一点，这里把它变成可重复的检查。

找不到浏览器就明确跳过（退出码 2），不假装通过：CI 的 ubuntu 上没有 Edge，
这条检查本来就不该在那里跑。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.observability.spec import RunTrace  # noqa: E402
from carepilot.viewer import render_trace_viewer  # noqa: E402

# Windows 上默认装了 Edge；Linux 上试 chrome/chromium。
CANDIDATES = (
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    "msedge",
    "google-chrome",
    "chromium",
    "chromium-browser",
)
# 详情面板里一定会出现的东西。少了任何一个，说明脚本没跑起来、或者数据没进去。
EXPECT = ("classify", "human_review", "输入摘要")


def find_browser() -> str | None:
    for cand in CANDIDATES:
        path = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if path:
            return path
    return None


def load_traces(directory: Path) -> list[RunTrace]:
    traces = []
    for path in sorted(directory.glob("*.json")):
        traces.append(RunTrace.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return traces


def dump_dom(browser: str, html: Path) -> str:
    """无头跑一次，把执行完 JS 的 DOM 打出来。"""
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--virtual-time-budget=5000",
        "--dump-dom",
        html.resolve().as_uri(),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"浏览器退出码 {out.returncode}：{out.stderr.strip()[:300]}")
    return out.stdout


def check(dom: str) -> list[str]:
    """返回缺失的标记。空列表 = 通过。"""
    return [token for token in EXPECT if token not in dom]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实浏览器里验证 trace 查看器。")
    parser.add_argument("--traces", type=Path, default=Path("data/traces"))
    parser.add_argument("--keep", action="store_true", help="保留生成的 HTML 便于手工查看")
    args = parser.parse_args(argv)

    traces = load_traces(args.traces)
    if not traces:
        print(f"{args.traces} 里没有 trace。")
        return 2

    browser = find_browser()
    if browser is None:
        print("没找到 Edge/Chrome，跳过（这不是通过）。")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix=".tmp-viewer-", dir=Path(__file__).resolve().parents[1]))
    try:
        html = workdir / "viewer.html"
        html.write_text(render_trace_viewer(traces), encoding="utf-8")
        dom = dump_dom(browser, html)
        missing = check(dom)
        size = len(html.read_text(encoding="utf-8"))
        rendered = len(dom)
        if missing:
            print(f"渲染后缺少标记：{missing}（HTML {size} 字节，DOM {rendered} 字节）")
            return 1
        print(f"通过：{len(traces)} 个 run，HTML {size} 字节，执行后的 DOM {rendered} 字节。")
        print(f"浏览器：{browser}")
        if args.keep:
            print(f"产物：{html}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
