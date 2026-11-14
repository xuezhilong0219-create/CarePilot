import json
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from .base import Chunk

_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _blocks(text: str) -> Iterator[tuple[int, str]]:
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        fence = _FENCE.match(line)
        heading = _HEADING.match(line)
        if fence:
            start = i
            marker = fence[1]
            i += 1
            while i < len(lines):
                closing = lines[i].strip()
                i += 1
                if len(closing) >= len(marker) and set(closing) == {marker[0]}:
                    break
            yield 0, "\n".join(lines[start:i])
        elif heading:
            yield len(heading[1]), heading[2]
            i += 1
        elif i + 1 < len(lines) and "|" in line and _TABLE_SEPARATOR.match(lines[i + 1]):
            start = i
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                i += 1
            yield 0, "\n".join(lines[start:i])
        elif line.startswith(("    ", "\t")):
            start = i
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(("    ", "\t"))):
                i += 1
            yield 0, "\n".join(lines[start:i]).rstrip()
        else:
            start = i
            i += 1
            while i < len(lines) and lines[i].strip() and not _HEADING.match(lines[i]) and not _FENCE.match(lines[i]):
                if i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEPARATOR.match(lines[i + 1]):
                    break
                i += 1
            yield 0, "\n".join(lines[start:i])


def split_markdown(
    doc_name: str, text: str, max_chars: int = 600, *,
    doc_title: str | None = None, meta: Mapping[str, str] | None = None,
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sections: list[tuple[str, str, list[str]]] = []
    headings: list[tuple[int, str]] = []
    for level, content in _blocks(text):
        if level:
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, content))
            continue
        titles = [title for _, title in headings]
        if doc_title and (not titles or titles[0] != doc_title):
            titles.insert(0, doc_title)
        if not titles:
            titles = [doc_title or doc_name]
        breadcrumb = " > ".join(titles)
        title = headings[-1][1] if headings else (doc_title or doc_name)
        if sections and sections[-1][0] == breadcrumb:
            sections[-1][2].append(content)
        else:
            sections.append((breadcrumb, title, [content]))

    chunks: list[Chunk] = []
    min_chars = min(80, max_chars // 4)
    for breadcrumb, title, blocks in sections:
        pieces: list[str] = []
        buffer = ""
        for block in blocks:
            if buffer and len(breadcrumb) + len(buffer) + len(block) + 3 > max_chars:
                pieces.append(buffer)
                buffer = ""
            buffer = f"{buffer}\n\n{block}" if buffer else block
        if buffer:
            if pieces and len(buffer) < min_chars:
                pieces[-1] += "\n\n" + buffer
            else:
                pieces.append(buffer)
        # 表格、代码和无空行的长段保持完整，因此 max_chars 是软上限。
        if len(pieces) > 1 and len(pieces[0]) < min_chars:
            pieces[:2] = ["\n\n".join(pieces[:2])]
        for piece in pieces:
            chunks.append(Chunk(f"{doc_name}#{len(chunks)}", title, f"{breadcrumb}\n{piece}", doc_name, dict(meta or {})))
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged:
            previous = merged[-1]
            previous_heading, _, previous_body = previous.text.partition("\n")
            heading, _, body = chunk.text.partition("\n")
            siblings = " > " in heading and " > " in previous_heading and heading.split(" > ")[0] == previous_heading.split(" > ")[0]
            if siblings and min(len(previous_body), len(body)) < min_chars and len(previous.text) + len(chunk.text) + 2 <= max_chars:
                merged[-1] = replace(previous, text=previous.text + "\n\n" + chunk.text)
                continue
        merged.append(chunk)
    return [replace(chunk, id=f"{doc_name}#{i}") for i, chunk in enumerate(merged)]


def _title_key(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title)
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    return re.sub(r"\s+", "", normalized).lower()


def load_kb(
    kb_dir: Path, exclude_dirs: Sequence[str] = (), exclude_flags: Sequence[str] = (),
) -> list[Chunk]:
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {kb_dir}")
    chunks: list[Chunk] = []
    manifest = kb_dir / "manifest.jsonl"
    if not manifest.exists():
        for path in sorted(kb_dir.rglob("*.md")):
            relative = path.relative_to(kb_dir)
            top_dir = relative.parts[0] if len(relative.parts) > 1 else ""
            if top_dir in exclude_dirs:
                continue
            meta = dict(top_dir=top_dir, title_key=_title_key(path.stem), flags="", source_url="")
            chunks.extend(split_markdown(relative.with_suffix("").as_posix(), path.read_text(encoding="utf-8"), meta=meta))
        return chunks

    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("kept") is not True:
                continue
            doc_id = entry["doc_id"]
            relative = PurePosixPath(doc_id)
            if not doc_id or relative.is_absolute() or ".." in relative.parts or ":" in doc_id or "\\" in doc_id:
                raise ValueError(f"Invalid manifest doc_id: {doc_id!r}")
            if doc_id in seen:
                raise ValueError(f"Duplicate manifest doc_id: {doc_id!r}")
            seen.add(doc_id)
            flags = entry.get("flags", [])
            if entry.get("top_dir", "") in exclude_dirs or set(flags).intersection(exclude_flags):
                continue
            path = kb_dir / "docs" / f"{doc_id}.md"
            if not path.resolve().is_relative_to((kb_dir / "docs").resolve()):
                raise ValueError(f"Manifest doc_id escapes docs directory: {doc_id!r}")
            title = entry.get("title", relative.name)
            meta = dict(top_dir=entry.get("top_dir", ""), title_key=entry.get("title_key", _title_key(title)), flags=",".join(flags), source_url=entry.get("source_url") or "")
            chunks.extend(split_markdown(doc_id, path.read_text(encoding="utf-8"), doc_title=title, meta=meta))
    return chunks
