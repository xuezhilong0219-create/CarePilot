import re
from pathlib import Path

from .base import Chunk

_HEADING = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)


def split_markdown(doc_name: str, text: str, max_chars: int = 600) -> list[Chunk]:
    """按 markdown 标题切分，过长的段落再按空行切。
    标题会拼进 chunk 正文，保证每个 chunk 脱离上下文也有语义。"""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(text))
    if not matches:
        sections.append((doc_name, text.strip()))
    else:
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((m.group(1).strip(), text[m.end():end].strip()))

    chunks: list[Chunk] = []
    for title, body in sections:
        if not body:
            continue
        pieces, buf = [], ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > max_chars:
                pieces.append(buf)
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
        if buf:
            pieces.append(buf)
        for piece in pieces:
            chunks.append(
                Chunk(id=f"{doc_name}#{len(chunks)}", title=title, text=f"{title}\n{piece}", doc_id=doc_name)
            )
    return chunks


def load_kb(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("**/*.md")):
        chunks.extend(split_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks
