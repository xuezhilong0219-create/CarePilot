"""产品线陷阱评测集：企业版 / 极速版同名文档的配对、特有内容与生成流程。

纯函数只依赖 manifest 条目和文档正文，生成入口用 ``CachedGenerator`` 注入，
单测可换成假 LLM。这里不做真实模型调用。
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .gen_core import (
    CachedGenerator,
    GeneratedTicket,
    ManifestEntry,
    REJECT_DUP,
    RETRY_TEMPERATURES,
    check_answerable,
    read_doc,
)

ENTERPRISE_DIR = "企业版"
EXPRESS_DIR = "极速版2.0"
EDITION_DIRS = (ENTERPRISE_DIR, EXPRESS_DIR)
EXCLUDED_FLAGS = {"release_note", "short"}
MIN_DOC_CHARS = 300

REJECT_EDITION = "问题里没有自然出现版本名"
REJECT_FORBIDDEN_WORD = "问题里出现文档/资料/知识库"
REJECT_DOC = "文档读取失败"
REJECT_LLM = "LLM 调用失败"

EDITION_LABEL = {ENTERPRISE_DIR: "企业版", EXPRESS_DIR: "极速版"}

SYSTEM_EDITION_TRAP = (
    "你是电商 ERP 的真实商家用户，不是客服。按用户给的要求写一条发给售后的消息，"
    "并抽取消息里真实出现的实体；不要解释，不要复述要求。"
)


def _eligible(entry: ManifestEntry) -> bool:
    return (
        entry.kept
        and entry.chars >= MIN_DOC_CHARS
        and not (set(entry.flags) & EXCLUDED_FLAGS)
    )


def find_edition_pairs(
    entries: Iterable[ManifestEntry],
) -> list[tuple[ManifestEntry, ManifestEntry]]:
    """按 title_key 配对；任一侧不是恰好一篇就跳过，避免同名歧义。"""
    grouped: dict[str, dict[str, list[ManifestEntry]]] = {}
    for entry in entries:
        if entry.top_dir not in EDITION_DIRS or not _eligible(entry):
            continue
        grouped.setdefault(entry.title_key, {}).setdefault(entry.top_dir, []).append(entry)

    pairs: list[tuple[ManifestEntry, ManifestEntry]] = []
    for title_key in sorted(grouped):
        sides = grouped[title_key]
        enterprise = sides.get(ENTERPRISE_DIR, [])
        express = sides.get(EXPRESS_DIR, [])
        if len(enterprise) != 1 or len(express) != 1:
            continue
        pairs.append((enterprise[0], express[0]))
    return pairs


def _is_content_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return any(ch.isalnum() for ch in stripped)


def unique_lines(
    doc_a: str, doc_b: str, *, min_len: int = 10, cap: int = 15
) -> list[str]:
    """doc_a 里有、doc_b 里没有的内容行；去首尾空白后比较，保持原顺序。"""
    other = {line.strip() for line in doc_b.splitlines()}
    out: list[str] = []
    seen: set[str] = set()
    for line in doc_a.splitlines():
        stripped = line.strip()
        if stripped in seen or stripped in other:
            continue
        if len(stripped) < min_len or not _is_content_line(stripped):
            continue
        seen.add(stripped)
        out.append(stripped)
        if len(out) >= cap:
            break
    return out


def is_discriminative(a: str, b: str, *, min_unique: int = 3) -> bool:
    return (
        len(unique_lines(a, b, cap=min_unique)) >= min_unique
        and len(unique_lines(b, a, cap=min_unique)) >= min_unique
    )


def edition_mentioned(question: str, label: str) -> bool:
    """版本名必须出现在正文里；只放在括号里不算"自然说出"。"""
    outside = re.sub(r"[（(][^）)]*[）)]", "", question)
    return label in outside


def check_trap_question(
    question: str,
    doc_text: str,
    label: str,
    existing: Iterable[str] = (),
) -> str | None:
    reason = check_answerable(question, doc_text, None, existing)
    if reason:
        return reason
    if not edition_mentioned(question, label):
        return REJECT_EDITION
    if re.search(r"文档|资料|知识库", question):
        return REJECT_FORBIDDEN_WORD
    return None


def build_trap_prompt(entry: ManifestEntry, edition_label: str, lines: Sequence[str],
                      doc_text: str) -> str:
    body = doc_text[:2500]
    special = "\n".join(f"- {line}" for line in lines[:15])
    return f"""请扮演一位使用旺店通 {edition_label} 的商家，给售后客服写一条报障或咨询消息。

要求：
1. 必须在消息里自然地说出自己用的是 {edition_label}，版本名要出现在句子里，不能只放在括号里。
2. 问题必须只能靠下面【本版本特有内容】回答，答案要取自这些行。
3. 15 到 80 个字，口语化，像真实商家在聊天窗口里打的字。
4. 不能照抄任何一行原文；与原文连续相同达到 12 个字符会被判为照抄。
5. 不要出现“文档”“资料”“知识库”这些词。
6. severity 按影响选择：critical=业务中断；high=主要功能不可用；medium=局部异常；low=咨询。

【本版本特有内容】
{special}

【文档标题】
{entry.title}

【文档正文（只给前 2500 字）】
{body}
"""


@dataclass(frozen=True)
class EditionTrap:
    qid: str
    question: str
    edition: str
    gold_doc_id: str
    forbidden_doc_ids: tuple[str, ...]
    title_key: str
    unique_lines_used: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "edition": self.edition,
            "gold_doc_id": self.gold_doc_id,
            "forbidden_doc_ids": list(self.forbidden_doc_ids),
            "title_key": self.title_key,
            "unique_lines_used": list(self.unique_lines_used),
        }


def _attempt_side(
    entry: ManifestEntry,
    other: ManifestEntry,
    lines: Sequence[str],
    doc_text: str,
    generator: CachedGenerator,
    model: str,
    seen: set[str],
    lock: threading.Lock,
) -> tuple[tuple[ManifestEntry, ManifestEntry, GeneratedTicket, list[str]] | None,
           list[str]]:
    label = EDITION_LABEL[entry.top_dir]
    rejections: list[str] = []
    prompt = build_trap_prompt(entry, label, lines, doc_text)
    for temperature in RETRY_TEMPERATURES:
        try:
            ticket = generator.generate(
                GeneratedTicket, model, temperature, SYSTEM_EDITION_TRAP, prompt
            )
        except Exception as exc:  # 单条失败不该拖垮整批
            rejections.append(f"{REJECT_LLM}: {type(exc).__name__}")
            continue
        if not isinstance(ticket, GeneratedTicket):
            rejections.append(REJECT_LLM)
            continue
        reason = check_trap_question(ticket.question, doc_text, label, seen)
        if reason is None:
            question = ticket.question.strip()
            with lock:
                if question in seen:
                    reason = REJECT_DUP
                else:
                    seen.add(question)
                    return (entry, other, ticket, list(lines[:5])), rejections
        rejections.append(reason)
    return None, rejections


def run_edition_traps(
    pairs: Sequence[tuple[ManifestEntry, ManifestEntry]],
    kb_dir: Path,
    generator: CachedGenerator,
    model: str,
    workers: int = 1,
) -> tuple[list[EditionTrap], Counter]:
    """每个配对两侧各一条；任一侧失败就丢掉整个配对，保证正负例成对。"""
    seen: set[str] = set()
    lock = threading.Lock()
    stats: Counter = Counter()

    def work(pair: tuple[ManifestEntry, ManifestEntry]):
        enterprise, express = pair
        try:
            enterprise_text = read_doc(kb_dir, enterprise)
            express_text = read_doc(kb_dir, express)
        except OSError as exc:
            return pair, [], [f"{REJECT_DOC}: {type(exc).__name__}"]
        jobs = [
            (enterprise, express, enterprise_text, express_text),
            (express, enterprise, express_text, enterprise_text),
        ]
        calls = [
            (entry, other, unique_lines(text, other_text), text)
            for entry, other, text, other_text in jobs
        ]
        results = []
        rejections: list[str] = []
        if workers > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        _attempt_side, entry, other, lines, text,
                        generator, model, seen, lock,
                    )
                    for entry, other, lines, text in calls
                ]
                for future in futures:
                    results.append(future.result())
        else:
            for entry, other, lines, text in calls:
                results.append(
                    _attempt_side(
                        entry, other, lines, text, generator, model, seen, lock
                    )
                )
        accepted = []
        for result, reasons in results:
            rejections.extend(reasons)
            if result is not None:
                accepted.append(result)
        if len(accepted) != 2:
            return pair, [], rejections
        return pair, accepted, rejections

    ordered_pairs = list(pairs)
    if workers > 1 and ordered_pairs:
        with ThreadPoolExecutor(max_workers=min(workers, len(ordered_pairs))) as pool:
            outcomes = list(pool.map(work, ordered_pairs))
    else:
        outcomes = [work(pair) for pair in ordered_pairs]

    traps: list[EditionTrap] = []
    for _, items, rejections in outcomes:
        stats.update(rejections)
        for entry, other, ticket, used in items:
            traps.append(
                EditionTrap(
                    qid=f"e{len(traps) + 1:04d}",
                    question=ticket.question.strip(),
                    edition=entry.top_dir,
                    gold_doc_id=entry.doc_id,
                    forbidden_doc_ids=(other.doc_id,),
                    title_key=entry.title_key,
                    unique_lines_used=tuple(used),
                )
            )
    return traps, stats


def discriminator_pairs(
    pairs: Sequence[tuple[ManifestEntry, ManifestEntry]], kb_dir: Path
) -> list[tuple[ManifestEntry, ManifestEntry, list[str], list[str]]]:
    """过滤出两个方向都有足够特有行的配对，并带上各自的特有行。"""
    out = []
    for enterprise, express in pairs:
        try:
            enterprise_text = read_doc(kb_dir, enterprise)
            express_text = read_doc(kb_dir, express)
        except OSError:
            continue
        enterprise_lines = unique_lines(enterprise_text, express_text)
        express_lines = unique_lines(express_text, enterprise_text)
        if len(enterprise_lines) >= 3 and len(express_lines) >= 3:
            out.append((enterprise, express, enterprise_lines, express_lines))
    return out


def sample_pairs(
    candidates: Sequence[tuple[ManifestEntry, ManifestEntry, list[str], list[str]]],
    n_pairs: int,
    seed: int,
) -> list[tuple[ManifestEntry, ManifestEntry, list[str], list[str]]]:
    import random

    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    return ordered[: max(0, n_pairs)]


def dump_traps(path: Path, traps: Sequence[EditionTrap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for trap in traps:
            f.write(json.dumps(trap.to_json(), ensure_ascii=False) + "\n")


def write_review(path: Path, traps: Sequence[EditionTrap], seed: int, n: int = 12) -> None:
    import random

    picked = random.Random(seed).sample(list(traps), min(n, len(traps)))
    lines = ["# 产品线陷阱评测集人工抽查", "", f"共 {len(traps)} 条，随机抽 {len(picked)} 条。", ""]
    for trap in picked:
        lines.append(f"## {trap.qid}")
        lines.append(f"- 问题：{trap.question}")
        lines.append(f"- 版本：{trap.edition}")
        lines.append(f"- 金标文档：docs/{trap.gold_doc_id}.md")
        lines.append("- 禁止引用：" + "、".join(f"docs/{d}.md" for d in trap.forbidden_doc_ids))
        lines.append("- 用到的特有行：")
        for line in trap.unique_lines_used:
            lines.append(f"  - {line}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
