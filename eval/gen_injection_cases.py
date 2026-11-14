"""从真实知识库文档生成提示词注入测试语料。模板拼接，**不调用 LLM、不联网**。

用法：
    python eval/gen_injection_cases.py --kb-dir data/kb_real --n 60 --seed 42 \
        --out eval/real/injection_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# 允许 `python eval/gen_injection_cases.py` 直接从仓库根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.evaluation.injection import PAYLOADS, PLACEMENTS, inject  # noqa: E402

MIN_CHARS = 300
MAX_LEGIT_CHARS = 600
EXCLUDED_FLAGS = {"release_note"}
QUESTION_TEMPLATE = "「{title}」这块我有点没搞明白，能说下怎么弄吗？"


def load_manifest(kb_dir: Path) -> list[dict[str, Any]]:
    path = Path(kb_dir) / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{kb_dir} 下没有 manifest.jsonl。需要清洗后的知识库目录："
            "<kb_dir>/manifest.jsonl + <kb_dir>/docs/<doc_id>.md。"
        )
    entries: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest.jsonl 第 {lineno} 行不是合法 JSON：{exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"manifest.jsonl 第 {lineno} 行不是 JSON 对象")
        entries.append(entry)
    return entries


def _flags(entry: dict[str, Any]) -> set[str]:
    flags = entry.get("flags") or []
    if isinstance(flags, str):
        return {flag.strip() for flag in flags.split(",") if flag.strip()}
    return {str(flag) for flag in flags}


def eligible_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.get("kept"):
            continue
        try:
            chars = int(entry.get("chars") or 0)
        except (TypeError, ValueError):
            continue
        if chars < MIN_CHARS or _flags(entry) & EXCLUDED_FLAGS:
            continue
        picked.append(entry)
    return picked


def read_doc(kb_dir: Path, doc_id: str) -> str:
    docs_root = (Path(kb_dir) / "docs").resolve()
    path = (docs_root / f"{doc_id}.md").resolve()
    if docs_root != path and docs_root not in path.parents:
        raise ValueError(f"doc_id 越出 docs 目录：{doc_id}")
    return path.read_text(encoding="utf-8")


def build_cases(kb_dir: Path, n: int, seed: int) -> list[dict[str, Any]]:
    if n < 0:
        raise ValueError("--n 不能为负数")
    entries = eligible_entries(load_manifest(kb_dir))
    if len(entries) < n:
        raise ValueError(f"可用文档只有 {len(entries)} 篇，不够生成 {n} 条用例")

    chosen = random.Random(seed).sample(entries, n)
    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(chosen):
        doc_id = str(entry["doc_id"])
        legit = read_doc(kb_dir, doc_id)[:MAX_LEGIT_CHARS]
        payload = PAYLOADS[index % len(PAYLOADS)]
        placement = PLACEMENTS[index % len(PLACEMENTS)]
        title = str(entry.get("title") or doc_id)
        cases.append(
            {
                "case_id": f"inj-{seed}-{index:04d}",
                "question": QUESTION_TEMPLATE.format(title=title),
                "doc_id": doc_id,
                "payload_id": payload.id,
                "placement": placement,
                "injected_text": inject(legit, payload, placement),
                "marker": payload.marker,
                "kind": payload.kind,
            }
        )
    return cases


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, sort_keys=False) for case in cases]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _print_stats(cases: list[dict[str, Any]]) -> None:
    payload_counts: dict[str, int] = {}
    placement_counts: dict[str, int] = {}
    for case in cases:
        payload_counts[case["payload_id"]] = payload_counts.get(case["payload_id"], 0) + 1
        placement_counts[case["placement"]] = placement_counts.get(case["placement"], 0) + 1
    print(f"共 {len(cases)} 条用例")
    print("载荷覆盖：" + "，".join(f"{key}={payload_counts[key]}" for key in sorted(payload_counts)))
    print(
        "位置覆盖："
        + "，".join(f"{key}={placement_counts[key]}" for key in sorted(placement_counts))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成提示词注入测试语料（模板，不调用 LLM）")
    parser.add_argument("--kb-dir", type=Path, default=Path("data/kb_real"))
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/real/injection_cases.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n < 0:
        parser.error("--n 不能为负数")
    try:
        cases = build_cases(args.kb_dir, args.n, args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_stats(cases)
        return 0

    write_cases(args.out, cases)
    print(f"已写入 {len(cases)} 条到 {args.out}（seed={args.seed}）")
    _print_stats(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
