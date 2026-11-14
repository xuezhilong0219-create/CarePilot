"""评估报告的读写和终端打表。报告写在 eval/real/reports/（已 gitignore）。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE) or "run"


def save_report(
    report_dir: Path,
    name: str,
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    timestamp: datetime | None = None,
) -> tuple[Path, Path]:
    """写 <时间戳>_<name>.json（汇总+配置）和同名的 .items.jsonl（逐条明细）。"""
    ts = timestamp or datetime.now()
    stem = f"{ts:%Y%m%d_%H%M%S}_{_safe_name(name)}"
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": name,
        "created_at": ts.isoformat(timespec="seconds"),
        "n_items": len(items),
        "config": dict(config),
        "summary": dict(summary),
    }
    json_path = report_dir / f"{stem}.json"
    items_path = report_dir / f"{stem}.items.jsonl"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(items_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return json_path, items_path


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_items(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
