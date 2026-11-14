from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from carepilot.ingest.clean import (
    HTML_STRIP_TAGS,
    CleanedDoc,
    Prepared,
    exact_dup_of,
    finalize_document,
    html_tag_counts,
    is_release_note,
    prepare_text,
    title_key,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "慧策wiki-2"
DEFAULT_OUT = ROOT / "data" / "kb_real"

TINY_CHARS = 80
SHORT_CHARS = 200


@dataclass
class BuildResult:
    manifest: list[dict]
    stats: dict
    errors: list[dict]


@dataclass
class _Acc:
    total: int = 0
    kept: int = 0
    tiny: int = 0
    exact_dup: int = 0
    release_note: int = 0
    short: int = 0


def _long_path(path: Path) -> Path:
    # Windows 默认 260 字符上限；\\?\ 前缀绕过它
    if os.name != "nt":
        return path
    absolute = os.path.abspath(str(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    return Path("\\\\?\\" + absolute)


def _read_text(path: Path) -> str:
    with open(_long_path(path), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    _long_path(path.parent).mkdir(parents=True, exist_ok=True)
    with open(_long_path(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _iter_raw(raw: Path) -> list[tuple[str, str, Path]]:
    # 用带 \\?\ 前缀的根做 os.walk，否则超过 260 字符的目录遍历不到
    root = _long_path(raw)
    docs: list[tuple[str, str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(".md"):
                continue
            path = Path(dirpath) / filename
            rel = Path(os.path.relpath(str(path), str(root)))
            docs.append((rel.with_suffix("").as_posix(), Path(filename).stem, path))
    return sorted(docs)


def _top_dir(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else ""


def build(raw: Path, out: Path) -> BuildResult:
    raw = raw.resolve()
    out = out.resolve()
    if out == raw or raw.is_relative_to(out) or out.is_relative_to(raw):
        raise ValueError(f"--out {out} 不能是 --raw {raw} 本身、父目录或子目录")
    if out == Path.cwd().resolve() or out == Path(out.anchor):
        raise ValueError(f"拒绝清空 {out}：看起来是工作目录或盘符根目录")

    if out.exists():
        shutil.rmtree(out)

    raw_docs = _iter_raw(raw)
    errors: list[dict] = []
    prepared_docs: dict[str, Prepared] = {}
    titles: dict[str, str] = {}
    top_dirs: dict[str, str] = {}
    chars_before = 0
    prepared_texts: list[str] = []

    for doc_id, title, path in raw_docs:
        try:
            raw_text = _read_text(path)
        except OSError as exc:
            errors.append({"doc_id": doc_id, "error": f"read: {exc}"})
            continue
        chars_before += len(raw_text)
        prepared = prepare_text(raw_text)
        prepared_docs[doc_id] = prepared
        prepared_texts.append(prepared.text)
        titles[doc_id] = title
        top_dirs[doc_id] = _top_dir(doc_id)

    tag_counter = html_tag_counts(prepared_texts)

    cleaned: dict[str, CleanedDoc] = {}
    for doc_id, prepared in prepared_docs.items():
        cleaned[doc_id] = finalize_document(prepared, titles[doc_id])

    kept_after_tiny = {d: doc for d, doc in cleaned.items() if doc.chars >= TINY_CHARS}
    dup_map = exact_dup_of(kept_after_tiny, top_dirs)

    kept_ids = {d for d in kept_after_tiny if d not in dup_map}
    title_counts = Counter(title_key(doc.title) for d, doc in cleaned.items() if d in kept_ids)

    manifest: list[dict] = []
    for doc_id in sorted(cleaned):
        doc = cleaned[doc_id]
        key = title_key(doc.title)
        release_note = is_release_note(doc.title)
        if doc_id in dup_map:
            kept, drop_reason, dup_of = False, "exact_dup", dup_map[doc_id]
        elif doc.chars < TINY_CHARS:
            kept, drop_reason, dup_of = False, "tiny", None
        else:
            kept, drop_reason, dup_of = True, None, None

        flags: list[str] = []
        if release_note:
            flags.append("release_note")
        if kept and TINY_CHARS <= doc.chars < SHORT_CHARS:
            flags.append("short")
        if kept and title_counts[key] >= 2:
            flags.append("dup_title")

        manifest.append(
            {
                "doc_id": doc_id,
                "title": doc.title,
                "title_key": key,
                "top_dir": top_dirs[doc_id],
                "source_url": doc.source_url,
                "chars": doc.chars,
                "kept": kept,
                "drop_reason": drop_reason,
                "dup_of": dup_of,
                "flags": flags,
            }
        )

    for entry in manifest:
        if not entry["kept"]:
            continue
        doc = cleaned[entry["doc_id"]]
        try:
            _write_text(out / "docs" / f"{entry['doc_id']}.md", doc.text + "\n")
        except OSError as exc:
            errors.append({"doc_id": entry["doc_id"], "error": f"write: {exc}"})

    by_top_dir: dict[str, _Acc] = defaultdict(_Acc)
    for entry in manifest:
        acc = by_top_dir[entry["top_dir"]]
        acc.total += 1
        if entry["kept"]:
            acc.kept += 1
        if entry["drop_reason"] == "tiny":
            acc.tiny += 1
        if entry["drop_reason"] == "exact_dup":
            acc.exact_dup += 1
        if "release_note" in entry["flags"]:
            acc.release_note += 1
        if "short" in entry["flags"]:
            acc.short += 1

    chars_after = sum(len(doc.text) for doc in cleaned.values())
    chars_after_kept = sum(len(cleaned[e["doc_id"]].text) for e in manifest if e["kept"])
    residual = html_tag_counts(cleaned[d].text for d in cleaned)

    stats = {
        "total": len(manifest),
        "kept_total": sum(1 for e in manifest if e["kept"]),
        "tiny_total": sum(1 for e in manifest if e["drop_reason"] == "tiny"),
        "exact_dup_total": sum(1 for e in manifest if e["drop_reason"] == "exact_dup"),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "chars_after_kept": chars_after_kept,
        "html_tags_top20": [{"tag": tag, "count": n} for tag, n in tag_counter.most_common(20)],
        "html_tags_residual_top20": [
            {"tag": tag, "count": n} for tag, n in residual.most_common(20)
        ],
        "known_html_tags_residual": sum(
            n for tag, n in residual.items() if tag in HTML_STRIP_TAGS
        ),
        "by_top_dir": {
            key: {
                "total": acc.total,
                "kept": acc.kept,
                "tiny": acc.tiny,
                "exact_dup": acc.exact_dup,
                "release_note": acc.release_note,
                "short": acc.short,
            }
            for key, acc in sorted(by_top_dir.items())
        },
        "errors": errors,
    }

    _long_path(out).mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=False) + "\n" for entry in manifest
    )
    _write_text(out / "manifest.jsonl", manifest_text)
    _write_text(out / "stats.json", json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return BuildResult(manifest, stats, errors)


def print_summary(stats: dict) -> None:
    print(f"{'目录':<34}{'总数':>7}{'保留':>7}{'tiny':>7}{'重复':>7}{'更新日志':>9}{'短文档':>7}")
    for name, acc in stats["by_top_dir"].items():
        print(
            f"{name or '(根目录)':<34}{acc['total']:>7}{acc['kept']:>7}{acc['tiny']:>7}"
            f"{acc['exact_dup']:>7}{acc['release_note']:>9}{acc['short']:>7}"
        )
    print(
        f"合计: {stats['total']} 篇，保留 {stats['kept_total']}，"
        f"tiny {stats['tiny_total']}，完全重复 {stats['exact_dup_total']}；"
        f"字符 {stats['chars_before']} -> {stats['chars_after']}"
    )
    if stats["errors"]:
        print(f"写入失败 {len(stats['errors'])} 篇，详见 stats.json 的 errors")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清洗语雀导出的知识库并生成 manifest")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    try:
        result = build(args.raw, args.out)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print_summary(result.stats)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
