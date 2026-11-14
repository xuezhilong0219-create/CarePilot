from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# 图片：`![alt](url)`，url 里允许一层括号（nlark 的 url 偶尔带括号）
_MD_IMAGE = re.compile(r"!\[[^\]\n]*\]\((?:[^()\n]|\([^()\n]*\))*\)")
_IMG_TAG = re.compile(r"<img\b[^<>]*>", re.IGNORECASE)

# 只匹配真正的标签名；`<https://...>` 这类自动链接因为名字后面紧跟 `:` 不会命中
_HTML_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)(\s[^<>]*?)?\s*/?>")

_SOURCE = re.compile(r"^>\s*来源[:：]\s*(\S+)[^\n]*\n?", re.MULTILINE)

# 语雀导出里真实出现过的 HTML 残留标签。code / response / flag 等是 XML
# 报文示例里的业务字段，不属于 HTML，必须原样保留。
HTML_STRIP_TAGS = {
    "a", "b", "body", "br", "color", "del", "div", "em", "font", "h1", "h2",
    "h3", "h4", "h5", "h6", "head", "html", "i", "ins", "li", "ol", "p", "s",
    "section", "span", "strike", "strong", "sub", "sup", "table", "tbody",
    "td", "tfoot", "th", "thead", "tr", "u", "ul",
}
_BLOCK_TAGS = {
    "body", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "head", "html",
    "li", "ol", "p", "section", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "ul",
}
_NAMED_ANCHOR = re.compile(r"<a\s+name=[^<>]*>\s*</a>", re.IGNORECASE)
_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)

_COPY_SUFFIX = re.compile(r"(?:\s*[\(（]\d+[\)）])+\s*$")
_RELEASE_TITLE = re.compile(r"^\d+(\.\d+){2,}")
_RELEASE_WORDS = ("更新日志", "版本更新", "发布说明", "升级说明")

# exact_dup 保留优先级：数字越小越优先；其余目录按名称排在中间。
_TOP_DIR_ORDER = {
    "企业版": 0,
    "极速版2.0": 1,
    "平台对接_店铺授权": 2,
    "物流对接": 3,
    "第三方仓对接": 4,
    "仓储对接": 5,
    "供销平台": 6,
}
_LOW_PRIORITY = {"原掌上百科（Wiki）内部使用": 8, "慧策价值传递素材仓库": 9}
_OTHER_TIER = 7


@dataclass(frozen=True)
class Prepared:
    text: str
    source_url: str | None


@dataclass(frozen=True)
class CleanedDoc:
    title: str
    text: str
    source_url: str | None
    chars: int
    body_md5: str


def normalize_newlines(text: str) -> str:
    text = text.removeprefix("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_source_url(text: str) -> tuple[str | None, str]:
    match = _SOURCE.search(text)
    if not match:
        return None, text
    return match.group(1), text[: match.start()] + text[match.end() :]


def strip_images(text: str) -> str:
    return _IMG_TAG.sub("", _MD_IMAGE.sub("", text))


def html_tag_counts(texts: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(m.group(1).lower() for m in _HTML_TAG.finditer(text))
    return counter


def _strip_html_segment(segment: str) -> str:
    segment = _NAMED_ANCHOR.sub("", segment)

    def repl(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        if name not in HTML_STRIP_TAGS:
            return match.group(0)
        if name == "br" or name in _BLOCK_TAGS:
            return "\n"
        return ""

    return _HTML_TAG.sub(repl, segment)


def strip_html(text: str) -> str:
    # 代码块里的尖括号是内容（如 XML 示例），不能当标签删掉
    parts: list[str] = []
    pos = 0
    for fence in _FENCE.finditer(text):
        parts.append(_strip_html_segment(text[pos : fence.start()]))
        parts.append(fence.group(0))
        pos = fence.end()
    parts.append(_strip_html_segment(text[pos:]))
    return "".join(parts)


def normalize_layout(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 连续 3 个以上换行（2 个以上空行）压成一个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def title_key(title: str) -> str:
    text = unicodedata.normalize("NFKC", title)
    text = _COPY_SUFFIX.sub("", text)
    return re.sub(r"\s+", "", text).lower()


def body_char_count(text: str) -> int:
    lines = text.split("\n", 1)
    body = lines[1] if len(lines) > 1 else ""
    return len(re.sub(r"\s+", "", body))


def ensure_title(text: str, title: str) -> str:
    first = text.split("\n", 1)[0].strip()
    if first == f"# {title}":
        return text
    return f"# {title}\n\n{text}" if text else f"# {title}"


def is_release_note(title: str) -> bool:
    return bool(_RELEASE_TITLE.match(title)) or any(word in title for word in _RELEASE_WORDS)


def prepare_text(raw: str) -> Prepared:
    text = normalize_newlines(raw)
    source_url, text = extract_source_url(text)
    return Prepared(strip_images(text), source_url)


def finalize_document(prepared: Prepared, title: str) -> CleanedDoc:
    text = normalize_layout(strip_html(prepared.text))
    text = ensure_title(text, title)
    body = text.partition("\n")[2]
    digest = hashlib.md5(re.sub(r"\s+", "", body).encode("utf-8")).hexdigest()
    return CleanedDoc(title, text, prepared.source_url, body_char_count(text), digest)


def clean_markdown(raw: str, title: str) -> CleanedDoc:
    return finalize_document(prepare_text(raw), title)


def top_dir_priority(top_dir: str) -> tuple[int, str]:
    if top_dir in _TOP_DIR_ORDER:
        return _TOP_DIR_ORDER[top_dir], ""
    if top_dir in _LOW_PRIORITY:
        return _LOW_PRIORITY[top_dir], ""
    return _OTHER_TIER, top_dir


def exact_dup_of(
    candidates: Mapping[str, CleanedDoc], top_dirs: Mapping[str, str]
) -> dict[str, str]:
    """返回 {被丢弃的 doc_id: 保留的 doc_id}，只包含完全重复的文档。"""
    groups: dict[str, list[str]] = {}
    for doc_id, doc in candidates.items():
        groups.setdefault(doc.body_md5, []).append(doc_id)
    result: dict[str, str] = {}
    for ids in groups.values():
        if len(ids) < 2:
            continue
        ids.sort(key=lambda d: (top_dir_priority(top_dirs[d]), d))
        keeper = ids[0]
        for doc_id in ids[1:]:
            result[doc_id] = keeper
    return result
