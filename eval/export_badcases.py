"""badcase 回流（D4，骨架）：把「实际去向与期望不符」的样本导出为候选集。

候选集**不会**自动并入评估集，需要人工逐条确认后才可加入。默认写
eval/real/badcase_candidates.jsonl（gitignore）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from carepilot.evaluation.schema import EvalItem, load_dataset  # noqa: E402

DEFAULT_OUT = ROOT / "eval" / "real" / "badcase_candidates.jsonl"


def stratum(item: EvalItem) -> str:
    if item.qid[:1] in "vw":
        return "含糊"
    if item.answerable:
        return "可答"
    return "需查数据" if item.category == "data_exception" else "无关"


def expected_outcome(item: EvalItem) -> str:
    """期望去向：可答→auto_reply；需查数据→escalate；无关→escalate/need_info；含糊→need_info。"""
    group = stratum(item)
    if group == "可答":
        return "auto_reply"
    if group == "需查数据":
        return "escalate"
    if group == "含糊":
        return "need_info"
    return "escalate/need_info"


def outcome_matches(item: EvalItem, actual: str) -> bool:
    expected = expected_outcome(item)
    if expected == "escalate/need_info":
        return actual in ("escalate", "need_info")
    return actual == expected


def actual_outcome(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "ERROR"
    return row.get("resolution") or "ERROR"


def _report_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.items.jsonl")))
        else:
            files.append(path)
    return [p for p in files if p.exists()]


def export_candidates(
    dataset_path: Path, report_paths: Sequence[Path], out_path: Path = DEFAULT_OUT
) -> list[dict[str, Any]]:
    items = {item.qid: item for item in load_dataset(Path(dataset_path))}
    candidates: list[dict[str, Any]] = []
    for report in _report_files(report_paths):
        with open(report, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                item = items.get(row.get("qid"))
                if item is None:
                    continue
                actual = actual_outcome(row)
                if outcome_matches(item, actual):
                    continue
                expected = expected_outcome(item)
                candidates.append(
                    {
                        "qid": item.qid,
                        "question": item.question,
                        "expected": expected,
                        "actual": actual,
                        "reason": f"{stratum(item)}：期望 {expected}，实际 {actual}",
                        "source_report": Path(report).name,
                    }
                )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 badcase 候选集（需人工确认，不自动并入评估集）。")
    parser.add_argument("--dataset", type=Path, required=True, help="评估集 jsonl")
    parser.add_argument("reports", nargs="+", type=Path, help="*.items.jsonl 或包含它的目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = export_candidates(args.dataset, args.reports, args.out)
    print(f"导出 {len(candidates)} 条 badcase 候选到 {args.out}")
    print("这些只是候选，必须人工确认后再加入评估集；脚本不会自动合并。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
