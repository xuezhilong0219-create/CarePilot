"""一键评测（D3）：测试 / 检索 / 流水线 / 工具选择，跨平台，不依赖 make。

用法：
    python eval/eval_all.py --quick
    python eval/eval_all.py --full --report-dir eval/real/reports
    python eval/eval_all.py --quick --skip tests,tools

每一步是一个只接收 ctx 字典的函数，测试里可以整体替换，不跑真模型、不联网。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from carepilot.evaluation.metrics import citation_hit  # noqa: E402
from carepilot.evaluation.report import markdown_table  # noqa: E402
from carepilot.evaluation.runners import run_pipeline, run_retrieval, summarize_retrieval  # noqa: E402
from carepilot.evaluation.schema import EvalItem, load_dataset  # noqa: E402

STEPS = ("tests", "retrieval", "pipeline", "tools")
GROUPS = ("可答", "需查数据", "无关", "含糊")
QUICK_N = 60
QUICK_SEED = 42
DEFAULT_DATASET = ROOT / "eval" / "real" / "dataset_full.jsonl"
DEFAULT_TOOL_DATASET = ROOT / "eval" / "real" / "dataset_tools.jsonl"
DEFAULT_REPORT_DIR = ROOT / "eval" / "real" / "reports"

StepFn = Callable[[dict[str, Any]], dict[str, Any]]


def stratum(item: EvalItem) -> str:
    """和 eval/outcomes.py 的 group_of 同一口径：四组互斥。"""
    if item.qid[:1] in "vw":
        return "含糊"
    if item.answerable:
        return "可答"
    return "需查数据" if item.category == "data_exception" else "无关"


def stratified_subset(items: Sequence[EvalItem], n: int, seed: int = QUICK_SEED) -> list[EvalItem]:
    """按四组分层抽样，每组至少 3 条（组本身够大、n 也够时），同 seed 可复现。"""
    items = list(items)
    if n <= 0 or n >= len(items):
        return items

    groups: dict[str, list[EvalItem]] = OrderedDict((g, []) for g in GROUPS)
    for item in items:
        groups[stratum(item)].append(item)
    nonempty = [g for g in GROUPS if groups[g]]
    if not nonempty:
        return []

    rng = random.Random(seed)
    for g in nonempty:
        rng.shuffle(groups[g])

    min_each = min(3, n // len(nonempty))
    quotas = {g: min(min_each, len(groups[g])) for g in nonempty}
    remaining = n - sum(quotas.values())

    # 按各组剩余容量比例分配（最大余数法），保证不超组大小。
    avail = {g: len(groups[g]) - quotas[g] for g in nonempty}
    pool = sum(avail.values())
    extra_total = min(remaining, pool)
    if extra_total > 0:
        raw = {g: avail[g] / pool * extra_total for g in nonempty}
        extra = {g: int(raw[g]) for g in nonempty}
        left = extra_total - sum(extra.values())
        for g in sorted(nonempty, key=lambda k: (raw[k] - extra[k], -GROUPS.index(k)), reverse=True):
            if left <= 0:
                break
            if extra[g] < avail[g]:
                extra[g] += 1
                left -= 1
        for g in nonempty:
            quotas[g] += extra[g]

    selected = [item for g in nonempty for item in groups[g][: quotas[g]]]
    if len(selected) < n:
        chosen = {id(item) for item in selected}
        rest = [item for item in items if id(item) not in chosen]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    return selected


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _items_for(ctx: Mapping[str, Any]) -> list[EvalItem] | None:
    path = Path(ctx["dataset_path"])
    if not path.exists():
        return None
    items = load_dataset(path)
    if ctx.get("quick"):
        items = stratified_subset(items, QUICK_N, QUICK_SEED)
    return items


def _missing_dataset(ctx: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "MISSING", "reason": f"数据集缺失：{ctx['dataset_path']}"}


def _build_bm25(ctx: Mapping[str, Any]):
    from carepilot.retrieval.factory import build_retriever

    return build_retriever("bm25", Path(ctx["kb_dir"]), settings=ctx.get("settings"))


def _count(summary: str, word: str) -> int:
    match = re.search(rf"(\d+) {word}\b", summary)
    return int(match.group(1)) if match else 0


def step_tests(ctx: Mapping[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = next((line for line in reversed(output.splitlines()) if line.strip()), "")
    return {
        "status": "OK" if proc.returncode == 0 else "FAILED",
        "passed": _count(output, "passed"),
        "failed": _count(output, "failed"),
        "errors": _count(output, "error") + _count(output, "errors"),
        "skipped": _count(output, "skipped"),
        "returncode": proc.returncode,
        "reason": tail[:300],
    }


def step_retrieval(ctx: Mapping[str, Any]) -> dict[str, Any]:
    items = _items_for(ctx)
    if items is None:
        return _missing_dataset(ctx)
    records = run_retrieval(items, _build_bm25(ctx), k=10)
    summary = summarize_retrieval(records, 10)
    return {
        "status": "OK",
        "n": summary["n"],
        "recall@5": summary.get("hit@5_strict", 0.0),
        "hit@10": summary.get("hit@10_strict", 0.0),
        "chunk_hit@4": summary.get("chunk_hit@4_strict", 0.0),
        "mrr": summary.get("mrr_strict", 0.0),
    }


def step_pipeline(ctx: Mapping[str, Any]) -> dict[str, Any]:
    items = _items_for(ctx)
    if items is None:
        return _missing_dataset(ctx)
    settings = ctx.get("settings")
    if settings is None:
        from carepilot.config import get_settings

        settings = get_settings()
    records = run_pipeline(items, settings, _build_bm25({**ctx, "settings": settings}), concurrency=4)

    ok = [(item, rec) for item, rec in zip(items, records, strict=True) if not rec.get("error")]
    answerable = [(item, rec) for item, rec in ok if item.answerable]
    unanswerable = [(item, rec) for item, rec in ok if not item.answerable]
    vague = [(item, rec) for item, rec in ok if stratum(item) == "含糊"]

    solved = sum(
        1
        for item, rec in answerable
        if rec.get("resolution") == "auto_reply"
        and citation_hit(
            rec.get("cited_doc_ids") or [],
            rec.get("cited_title_keys") or [],
            item.gold_doc_ids,
            item.gold_title_keys,
        )
    )
    wrong_refused = sum(1 for _, rec in answerable if rec.get("resolution") != "auto_reply")
    right_refused = sum(1 for _, rec in unanswerable if rec.get("resolution") != "auto_reply")
    vague_asked = sum(1 for _, rec in vague if rec.get("resolution") == "need_info")

    return {
        "status": "OK",
        "n": len(records),
        "errors": len(records) - len(ok),
        "答对率": {"value": _ratio(solved, len(answerable)), "n": len(answerable)},
        "拒答准确率": {"value": _ratio(right_refused, len(unanswerable)), "n": len(unanswerable)},
        "误拒率": {"value": _ratio(wrong_refused, len(answerable)), "n": len(answerable)},
        "笼统消息追问率": {"value": _ratio(vague_asked, len(vague)), "n": len(vague)},
    }


def step_tools(ctx: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(ctx["tool_dataset_path"])
    if not path.exists():
        return {"status": "MISSING", "reason": f"数据集缺失：{path}"}
    try:
        spec = importlib.util.find_spec("carepilot.evaluation.tool_eval")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return {"status": "UNPROVIDED", "reason": "runner 未提供：carepilot.evaluation.tool_eval"}
    module = importlib.import_module("carepilot.evaluation.tool_eval")
    runner = getattr(module, "run_tool_selection", None)
    if runner is None:
        return {"status": "UNPROVIDED", "reason": "runner 未提供：run_tool_selection"}
    settings = ctx.get("settings")
    if settings is None:
        from carepilot.config import get_settings

        settings = get_settings()
    result = runner(path, settings)
    return {"status": "OK", **(result or {})}


DEFAULT_STEPS: dict[str, StepFn] = {
    "tests": step_tests,
    "retrieval": step_retrieval,
    "pipeline": step_pipeline,
    "tools": step_tools,
}


def _render_markdown(results: Mapping[str, Mapping[str, Any]], *, quick: bool, dataset_path: Path, kb_dir: Path) -> str:
    rows: list[list[Any]] = []
    for name in STEPS:
        res = results.get(name) or {}
        status = res.get("status", "UNKNOWN")
        if status != "OK":
            rows.append([name, status, res.get("reason", ""), ""])
            continue
        for key, value in res.items():
            if key == "status":
                continue
            if isinstance(value, Mapping) and "value" in value:
                rows.append([name, key, value["value"], value.get("n", "")])
            elif key in ("n", "passed", "failed", "errors", "skipped", "returncode"):
                rows.append([name, key, value, ""])
            elif key == "reason":
                continue
            else:
                rows.append([name, key, value, ""])

    lines = [
        f"# 一键评测（{'quick' if quick else 'full'}）",
        "",
        f"- 数据集：{dataset_path}",
        f"- 知识库：{kb_dir}",
        "",
        markdown_table(["步骤", "指标", "值", "样本数"], rows),
    ]
    return "\n".join(lines)


def run_all(
    *,
    quick: bool,
    skip: Sequence[str] = (),
    report_dir: Path = DEFAULT_REPORT_DIR,
    dataset_path: Path = DEFAULT_DATASET,
    kb_dir: Path | None = None,
    tool_dataset_path: Path = DEFAULT_TOOL_DATASET,
    step_fns: Mapping[str, StepFn] | None = None,
    settings: Any = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    steps = dict(DEFAULT_STEPS if step_fns is None else step_fns)
    kb_dir = Path(kb_dir) if kb_dir is not None else _default_kb_dir()
    ctx: dict[str, Any] = {
        "quick": quick,
        "dataset_path": Path(dataset_path),
        "tool_dataset_path": Path(tool_dataset_path),
        "kb_dir": Path(kb_dir),
        "settings": settings,
    }

    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in STEPS:
        if name in skip:
            results[name] = {"status": "SKIPPED"}
            continue
        fn = steps.get(name)
        if fn is None:
            results[name] = {"status": "UNPROVIDED", "reason": "step not registered"}
            continue
        try:
            res = fn(ctx) or {}
        except Exception as exc:  # 单步失败不能拖垮后面
            res = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        res.setdefault("status", "OK")
        if res["status"] == "FAILED":
            failures.append(name)
            res.setdefault("reason", "step failed")
        results[name] = res

    markdown = _render_markdown(results, quick=quick, dataset_path=Path(dataset_path), kb_dir=Path(kb_dir))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    now = timestamp or datetime.now()
    payload = {
        "created_at": now.isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full",
        "skip": list(skip),
        "dataset": str(dataset_path),
        "kb_dir": str(kb_dir),
        "results": results,
    }
    json_path = report_dir / f"eval_all_{now:%Y%m%d_%H%M%S}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(markdown)
    print(f"\n报告已写入 {json_path}")
    return {
        "markdown": markdown,
        "json_path": json_path,
        "results": results,
        "failures": failures,
        "exit_code": 1 if failures else 0,
    }


def _default_kb_dir() -> Path:
    real = ROOT / "data" / "kb_real"
    return real if real.exists() else ROOT / "data" / "kb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键评测：测试 / 检索 / 流水线 / 工具选择。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="分层抽样约 60 条")
    group.add_argument("--full", action="store_true", help="跑全部样本")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--skip", default="", help="逗号分隔，可选 tests,retrieval,pipeline,tools")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tool-dataset", type=Path, default=DEFAULT_TOOL_DATASET)
    parser.add_argument("--kb-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_all(
        quick=args.quick,
        skip=_split(args.skip),
        report_dir=args.report_dir,
        dataset_path=args.dataset,
        kb_dir=args.kb_dir,
        tool_dataset_path=args.tool_dataset,
    )
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
