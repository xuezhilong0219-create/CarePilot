"""评估集生成的核心逻辑：抽样、类别、提示词、防照抄校验、缓存与拼装。

这里不直接调用真实 LLM：生成入口是一个 ``invoke`` 回调，命令行注入真实的
``StructuredLLM``，单测注入返回预设结果的假实现。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..graph.state import Category, Severity
from .schema import EvalItem

MIN_DOC_CHARS = 300
MIN_QUESTION_CHARS = 12
MAX_QUESTION_CHARS = 100
COPY_WINDOW = 12
PROMPT_DOC_CHARS = 2500
# 第一次用低温保证稳定，拒收后逐步升温，逼模型换一种说法
RETRY_TEMPERATURES = (0.0, 0.5, 1.0)

REJECT_LENGTH = "长度不在 12~100"
REJECT_COPY = "与原文连续相同 >= 12 字"
REJECT_ERROR_TEXT = "error_text 未在问题里逐字出现"
REJECT_ERROR_MISMATCH = "error_text 与给定的报错串不一致"
REJECT_DUP = "与已有问题重复"
REJECT_LLM = "LLM 调用失败"
REJECT_DOC = "文档读取失败"

SPECIAL_DIRS = {"慧策价值传递素材仓库", "原掌上百科（Wiki）内部使用"}
WAREHOUSE_DIRS = {"物流对接", "第三方仓对接", "仓储对接"}
EXCLUDED_FLAGS = {"release_note", "short"}

# 分层名额上限：单目录 <= 35%，两个历史素材目录合计 <= 15%
SINGLE_DIR_CAP = 0.35
SPECIAL_GROUP_CAP = 0.15


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    title: str
    title_key: str
    top_dir: str
    source_url: str = ""
    chars: int = 0
    kept: bool = True
    drop_reason: str | None = None
    dup_of: str | None = None
    flags: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, obj: dict) -> "ManifestEntry":
        title = str(obj.get("title") or "")
        return cls(
            doc_id=str(obj["doc_id"]),
            title=title,
            title_key=str(obj.get("title_key") or title_key_for(title)),
            top_dir=str(obj.get("top_dir") or ""),
            source_url=str(obj.get("source_url") or ""),
            chars=int(obj.get("chars") or 0),
            kept=bool(obj.get("kept", True)),
            drop_reason=obj.get("drop_reason"),
            dup_of=obj.get("dup_of"),
            flags=tuple(str(f) for f in (obj.get("flags") or ())),
        )


def title_key_for(title: str) -> str:
    """与 TEAM.md 的 title_key 规则一致：NFKC → 去副本后缀 → 去空白 → 小写。"""
    text = unicodedata.normalize("NFKC", title).strip()
    text = re.sub(r"\s*[（(]\d+[)）]\s*$", "", text)
    return re.sub(r"\s+", "", text).lower()


def load_manifest(kb_dir: Path) -> list[ManifestEntry]:
    path = Path(kb_dir) / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{kb_dir} 下没有 manifest.jsonl。生成脚本需要清洗后的知识库目录："
            "<kb_dir>/manifest.jsonl + <kb_dir>/docs/<doc_id>.md（格式见 .team/TEAM.md）。"
        )
    entries: list[ManifestEntry] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entries.append(ManifestEntry.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"manifest.jsonl 第 {lineno} 行格式错误：{exc}") from exc
    return entries


def candidate_docs(entries: Iterable[ManifestEntry]) -> list[ManifestEntry]:
    return [
        e
        for e in entries
        if e.kept
        and e.chars >= MIN_DOC_CHARS
        and not (set(e.flags) & EXCLUDED_FLAGS)
        and "导航" not in e.title  # 导航页不是支持文档，拿它当标准答案只会生成怪问题
    ]


def category_for(top_dir: str, title: str) -> Category:
    """类别标签必须确定性，不能交给 LLM 判断。顺序即优先级。"""
    if top_dir == "平台对接_店铺授权":
        return "platform_integration"
    if top_dir in WAREHOUSE_DIRS:
        return "warehouse_logistics"
    if re.search(r"规则|政策|公告|通知|变更", title):
        return "rule_change"
    return "feature_howto"


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """按权重把 total 拆成整数，余数按小数部分降序分配，保证结果确定。"""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    weight_sum = sum(weights.values())
    raw = {k: total * w / weight_sum for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(out.values())
    order = sorted(weights, key=lambda k: (-(raw[k] - out[k]), k))
    for key in order[:left]:
        out[key] += 1
    return out


def allocate_quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    """按 sqrt(候选数) 分层分配名额，并满足单目录 / 特殊目录组合的上限。"""
    dirs = sorted(counts)
    if n < 0:
        raise ValueError("名额不能为负数")
    if n == 0:
        return {d: 0 for d in dirs}
    if sum(counts.values()) < n:
        raise ValueError(f"候选文档只有 {sum(counts.values())} 篇，不够抽 {n} 篇")

    single_cap = {d: int(SINGLE_DIR_CAP * n) for d in dirs}
    group_cap = int(SPECIAL_GROUP_CAP * n)
    quotas = {d: 0 for d in dirs}
    remaining = n

    while remaining > 0:
        active = [
            d for d in dirs if quotas[d] < single_cap[d] and quotas[d] < counts[d]
        ]
        group_used = sum(quotas[d] for d in dirs if d in SPECIAL_DIRS)
        group_room = max(0, group_cap - group_used)
        if group_room <= 0:
            # 素材目录合计上限已满：必须把它们移出候选，否则它们会继续按权重分到名额、
            # 又被清零，剩下的零头永远分不出去（真实分布下会"还差 1 篇"）
            active = [d for d in active if d not in SPECIAL_DIRS]
        if not active:
            break
        weights = {d: math.sqrt(counts[d]) for d in active}
        alloc = _largest_remainder(remaining, weights)
        for d in active:
            alloc[d] = min(alloc[d], single_cap[d] - quotas[d], counts[d] - quotas[d])

        group_dirs = [d for d in active if d in SPECIAL_DIRS]
        if sum(alloc[d] for d in group_dirs) > group_room:
            if group_room <= 0:
                for d in group_dirs:
                    alloc[d] = 0
            else:
                scaled = _largest_remainder(group_room, {d: weights[d] for d in group_dirs})
                for d in group_dirs:
                    alloc[d] = min(alloc[d], scaled[d])

        added = sum(alloc.values())
        if added <= 0:
            break
        for d in active:
            quotas[d] += alloc[d]
        remaining -= added

    if remaining:
        raise ValueError(
            f"候选文档不足或名额上限过严：还差 {remaining} 篇。"
            f"各目录候选数={counts}，单目录上限={int(SINGLE_DIR_CAP * n)}，"
            f"特殊目录组合上限={group_cap}"
        )
    return quotas


@dataclass
class SamplePlan:
    quotas: dict[str, int]
    selected: list[ManifestEntry]
    backups: list[ManifestEntry]


def plan_samples(entries: Sequence[ManifestEntry], n: int, seed: int) -> SamplePlan:
    """分层抽样。每个目录用独立的种子，目录之间互不影响，同 seed 结果完全一致。"""
    candidates = candidate_docs(entries)
    counts = Counter(e.top_dir for e in candidates)
    quotas = allocate_quotas(dict(counts), n)

    selected: list[ManifestEntry] = []
    backups: list[ManifestEntry] = []
    for top_dir in sorted(counts):
        docs = sorted(
            (e for e in candidates if e.top_dir == top_dir), key=lambda e: e.doc_id
        )
        random.Random(f"{seed}:{top_dir}").shuffle(docs)
        quota = quotas[top_dir]
        selected.extend(docs[:quota])
        backups.extend(docs[quota:])
    return SamplePlan(quotas=quotas, selected=selected, backups=backups)


@dataclass
class ErrorSamplePlan:
    quotas: dict[str, int]
    selected: list[tuple[ManifestEntry, str]]


def plan_error_samples(
    entries: Sequence[ManifestEntry],
    kb_dir: Path,
    n: int,
    seed: int,
    exclude_doc_ids: Iterable[str] = (),
) -> ErrorSamplePlan:
    """抽「带报错原文」的文档：同一套候选过滤与 sqrt 分层，再排除已被可答样本选中的文档。"""
    if n < 0:
        raise ValueError("名额不能为负数")
    if n == 0:
        return ErrorSamplePlan(quotas={}, selected=[])
    excluded = set(exclude_doc_ids)
    pool: dict[str, list[tuple[ManifestEntry, list[str]]]] = {}
    for entry in candidate_docs(entries):
        if entry.doc_id in excluded:
            continue
        try:
            text = read_doc(kb_dir, entry)
        except OSError:
            continue
        errors = extract_error_candidates(text)
        if errors:
            pool.setdefault(entry.top_dir, []).append((entry, errors))

    counts = {top_dir: len(docs) for top_dir, docs in pool.items()}
    quotas = allocate_quotas(counts, n)
    selected: list[tuple[ManifestEntry, str]] = []
    for top_dir in sorted(pool):
        docs = sorted(pool[top_dir], key=lambda pair: pair[0].doc_id)
        random.Random(f"{seed}:error:{top_dir}").shuffle(docs)
        for entry, errors in docs[: quotas[top_dir]]:
            error = random.Random(f"{seed}:error:{entry.doc_id}").choice(errors)
            selected.append((entry, error))
    return ErrorSamplePlan(quotas=quotas, selected=selected)


def doc_path(kb_dir: Path, entry: ManifestEntry) -> Path:
    return Path(kb_dir) / "docs" / f"{entry.doc_id}.md"


def read_doc(kb_dir: Path, entry: ManifestEntry) -> str:
    return doc_path(kb_dir, entry).read_text(encoding="utf-8")


# ---------------------------------------------------------------- 报错原文抽取

_QUOTED_ERROR_RE = re.compile(r'[“"]([^“”"\n]{6,80})[”"]')
# 方括号内允许空格，这样 oauth-error-code[30111]description[access token invalid] 才能完整抽出
_CODE_ERROR_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.#:-]*(?:\[[^\]\n]{1,60}\][A-Za-z0-9_.#:-]*)*")
_ERROR_CONTEXT_RE = re.compile(r"报错|提示|失败|异常|错误|返回")
_ERROR_CONTEXT_WINDOW = 10  # 引号前这么近才算"提示“…”"；30 字时会混进大量界面文案
# 引号内容本身必须带失败类词，否则 `success！`、`开启声音提示` 这类界面文案会被当成报错。
# 真实知识库上对比：窗口 30 无内容要求 436 条、噪声大；窗口 10 + 内容要求 166 条、基本都是真报错。
_ERROR_CUE_RE = re.compile(
    r"失败|错误|异常|超时|无效|失效|不存在|不足|过期|频繁|禁止|拒绝|无权|权限|未[^\s,，。]{1,3}|无法|不能|不允许|不支持|重复"
    r"|error|fail|invalid|exception|denied|timeout|not found|null",
    re.IGNORECASE,
)
# 不含单独的 `code`：`…sourceOrderCode]` 这类 XML 字段路径不是报错
_ERROR_WORD_RE = re.compile(r"error|fail|invalid|exception", re.IGNORECASE)


def _meaningful_error(text: str) -> bool:
    if not text or "\n" in text or "\r" in text:
        return False
    if text.isdigit():  # 纯数字（含全角）不是报错原文
        return False
    return any(ch.isalnum() for ch in text)  # 纯标点也排除


def extract_error_candidates(doc_text: str) -> list[str]:
    """抽出像报错提示的原文串，结果保证是文档的逐字子串，按出现顺序去重。"""
    found: list[tuple[int, str]] = []
    for match in _QUOTED_ERROR_RE.finditer(doc_text):
        inner = match.group(1).strip()
        if not (6 <= len(inner) <= 80):
            continue
        prefix = doc_text[max(0, match.start() - _ERROR_CONTEXT_WINDOW) : match.start()]
        if not _ERROR_CONTEXT_RE.search(prefix) or not _ERROR_CUE_RE.search(inner):
            continue
        if _meaningful_error(inner):
            found.append((match.start(), inner))
    for match in _CODE_ERROR_RE.finditer(doc_text):
        token = match.group(0)
        if len(token) < 12 or not _ERROR_WORD_RE.search(token):
            continue
        if _meaningful_error(token):
            found.append((match.start(), token))

    found.sort(key=lambda pair: pair[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, text in found:
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


# ---------------------------------------------------------------- 生成模型


class GeneratedTicket(BaseModel):
    question: str = Field(description="商家发给售后客服的消息原文")
    severity: Severity = Field(description="严重度")
    platform_name: str | None = Field(None, description="消息里出现的平台名称原文")
    feature_name: str | None = Field(None, description="消息里出现的功能/菜单名称原文")
    partner_name: str | None = Field(None, description="消息里出现的仓库或物流商名称原文")
    error_text: str | None = Field(None, description="消息里逐字出现的报错原文")


class UnanswerableItem(BaseModel):
    question: str = Field(description="商家发给售后客服的消息原文")


class UnanswerableBatch(BaseModel):
    items: list[UnanswerableItem] = Field(description="生成的消息列表")


SYSTEM_ANSWERABLE = (
    "你是电商 ERP 的真实商家用户，不是客服。按用户给的要求写一条发给售后的消息，"
    "并抽取消息里真实出现的实体；不要解释，不要复述要求。"
)
SYSTEM_UNANSWERABLE = (
    "你在帮团队构造评估集里的负样本。你扮演不同行业的电商商家，写真实、口语化的消息。"
)


def build_answerable_prompt(entry: ManifestEntry, doc_text: str) -> str:
    body = doc_text[:PROMPT_DOC_CHARS]
    return f"""请扮演一位使用旺店通 ERP 的商家，给售后客服写一条报障或咨询消息。

要求：
1. 口语化，像真实商家在聊天窗口里打的字，15 到 80 个字（按汉字个数算）。
2. 只问一个具体问题，而且仅凭下面的文档就能回答。
3. 不能照抄文档标题；不能照抄文档里的完整句子。与原文连续相同达到 12 个字符就会被判为照抄。
4. 如果文档里有典型报错提示，可以把报错原文逐字贴进消息；此时 error_text 必须填这句原文，并保证它逐字出现在 question 里。
5. severity 按影响选择：critical=无法发货/订单下不来/登录不了等业务中断；high=主要功能不可用；medium=局部功能异常；low=咨询。
6. platform_name / feature_name / partner_name 只能填消息里逐字出现的原文，没提到就填 null。
7. 消息里不要出现“文档”“资料”“知识库”这些词。

【文档标题】
{entry.title}

【文档正文（只给前 {PROMPT_DOC_CHARS} 字）】
{body}
"""


def build_error_prompt(entry: ManifestEntry, doc_text: str, error_text: str) -> str:
    body = doc_text[:PROMPT_DOC_CHARS]
    return f"""请扮演一位使用旺店通 ERP 的商家，给售后客服写一条报障消息。

这位商家在操作时遇到了下面这条报错，请把报错原文逐字贴进消息里：
{error_text}

要求：
1. 必须把上面这句报错原文一个字不改地贴进消息里；error_text 字段填同一句原文，不要改写、不要删减。
2. 再用一两句口语描述当时在做什么、看到什么现象。
3. 整条消息 20 到 100 个字（按汉字个数算），像真实商家在聊天窗口里打的字。
4. 不要照抄文档里的其他句子，尤其不要大段复制解决方案；与原文连续相同达到 12 个字符会被判为照抄。
5. severity 按影响选择：critical=无法发货/订单下不来/登录不了等业务中断；high=主要功能不可用；medium=局部功能异常；low=咨询。
6. platform_name / feature_name / partner_name 只能填消息里逐字出现的原文，没提到就填 null。
7. 消息里不要出现“文档”“资料”“知识库”这些词。

【文档标题】
{entry.title}

【文档正文（只给前 {PROMPT_DOC_CHARS} 字）】
{body}
"""


def build_unanswerable_prompt(
    category: Literal["data_exception", "other"], n: int
) -> str:
    if category == "data_exception":
        kind = "需要查客户自己的订单 / 库存 / 账单数据才能回答"
        detail = (
            "每条都要指向具体的某笔订单、某个库存数字或某张账单（例如“昨天那 3 笔淘宝订单去哪了”），"
            "不能是操作步骤咨询；覆盖不同平台（淘宝、京东、抖店、拼多多等）和不同功能（订单、库存、财务、售后）。"
        )
    else:
        kind = "与 ERP 无关、闲聊，或要求办理产品之外的事项"
        detail = "覆盖不同话题、不同语气，彼此差异要大，不要都写成同一类无关问题。"
    return f"""请生成 {n} 条彼此差异很大的商家消息，类型是：{kind}。

要求：
1. {detail}
2. 每条 15 到 80 字，口语化，像真实商家的原话。
3. 不能和类型定义里已经给出的示例重复。
4. 只输出 items 列表，每条一个 question 字段。
"""


# ---------------------------------------------------------------- 防照抄校验


def _squash(text: str) -> str:
    """去掉所有空白后再比对，防止模型把换行/空格换个位置就绕过照抄检查。"""
    return "".join(text.split())


def longest_common_substring_length(a: str, b: str) -> int:
    """最长公共子串长度（字符级）。短串做内层循环，内存 O(较短串长度)。"""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if ch == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def shares_window(text: str, body: str, window: int) -> bool:
    """等价于 longest_common_substring_length(text, body) >= window，但用 C 级的 `in` 查找，
    对上万字的文档也是毫秒级（DP 版本要秒级）。"""
    return any(text[i : i + window] in body for i in range(len(text) - window + 1))


def check_answerable(
    question: str,
    doc_text: str,
    error_text: str | None,
    existing: Iterable[str] = (),
) -> str | None:
    """返回拒收原因；None 表示通过。顺序有讲究：先长度，再报错原文，最后才是照抄。"""
    text = question.strip()
    if not (MIN_QUESTION_CHARS <= len(text) <= MAX_QUESTION_CHARS):
        return REJECT_LENGTH
    error = (error_text or "").strip()
    if error and error not in question:
        return REJECT_ERROR_TEXT
    body = doc_text.replace(error, "") if error else doc_text
    if shares_window(_squash(text), _squash(body), COPY_WINDOW):
        return REJECT_COPY
    if text in set(existing):
        return REJECT_DUP
    return None


def sanitize_ticket(ticket: GeneratedTicket) -> GeneratedTicket:
    """提示词要求实体必须逐字出现在消息里；模型没做到就清空，避免黄金标签污染。"""

    def keep(value: str | None) -> str | None:
        return value if value and value in ticket.question else None

    return ticket.model_copy(
        update={
            "platform_name": keep(ticket.platform_name),
            "feature_name": keep(ticket.feature_name),
            "partner_name": keep(ticket.partner_name),
        }
    )


# ---------------------------------------------------------------- 缓存与生成


def request_cache_key(
    model: str, temperature: float, schema_name: str, system: str, prompt: str
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "schema": schema_name,
            "system": system,
            "prompt": prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Invoke = Callable[[type[BaseModel], float, list[BaseMessage]], BaseModel]


class CachedGenerator:
    """一次结构化生成 + 磁盘缓存。同一个输入重跑不会重复花钱。"""

    def __init__(self, cache_dir: Path, invoke: Invoke):
        self.cache_dir = Path(cache_dir)
        self._invoke = invoke
        self._lock = threading.Lock()
        self.calls = 0

    def generate(
        self,
        schema: type[BaseModel],
        model: str,
        temperature: float,
        system: str,
        prompt: str,
    ) -> BaseModel:
        key = request_cache_key(model, temperature, schema.__name__, system, prompt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return schema.model_validate_json(path.read_text(encoding="utf-8"))

        result = self._invoke(
            schema,
            temperature,
            [SystemMessage(content=system), HumanMessage(content=prompt)],
        )
        data = result.model_dump()
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{key}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            self.calls += 1
        return schema.model_validate(data)


@dataclass
class DocAttempt:
    entry: ManifestEntry
    ticket: GeneratedTicket | None
    rejections: list[str]


def attempt_doc(
    entry: ManifestEntry,
    doc_text: str,
    generator: CachedGenerator,
    model: str,
    seen: set[str],
    lock: threading.Lock,
) -> DocAttempt:
    rejections: list[str] = []
    for temperature in RETRY_TEMPERATURES:
        try:
            ticket = generator.generate(
                GeneratedTicket,
                model,
                temperature,
                SYSTEM_ANSWERABLE,
                build_answerable_prompt(entry, doc_text),
            )
        except Exception as exc:  # 单篇失败不该拖垮整批，换备选文档
            rejections.append(f"{REJECT_LLM}: {type(exc).__name__}")
            continue
        assert isinstance(ticket, GeneratedTicket)
        reason = check_answerable(ticket.question, doc_text, ticket.error_text, seen)
        if reason is None:
            question = ticket.question.strip()
            with lock:
                if question in seen:
                    reason = REJECT_DUP
                else:
                    seen.add(question)
                    return DocAttempt(entry, sanitize_ticket(ticket), rejections)
        rejections.append(reason)
    return DocAttempt(entry, None, rejections)


def run_answerables(
    plan: SamplePlan,
    kb_dir: Path,
    generator: CachedGenerator,
    model: str,
    n: int,
    workers: int = 1,
) -> tuple[list[DocAttempt], Counter]:
    """先跑抽中的文档，不够再用备选补；所有拒收原因累计返回。"""
    seen: set[str] = set()
    lock = threading.Lock()
    stats: Counter = Counter()

    def work(entry: ManifestEntry) -> DocAttempt:
        try:
            text = read_doc(kb_dir, entry)
        except OSError as exc:
            return DocAttempt(entry, None, [f"{REJECT_DOC}: {type(exc).__name__}"])
        return attempt_doc(entry, text, generator, model, seen, lock)

    def run_batch(entries: Sequence[ManifestEntry]) -> list[DocAttempt]:
        if not entries:
            return []
        if workers > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as pool:
                return list(pool.map(work, entries))
        return [work(e) for e in entries]

    attempts = run_batch(plan.selected)
    accepted = [a for a in attempts if a.ticket is not None]
    for attempt in attempts:
        stats.update(attempt.rejections)

    index = 0
    while len(accepted) < n and index < len(plan.backups):
        size = max(workers, n - len(accepted), 1)
        batch = plan.backups[index : index + size]
        index += len(batch)
        for attempt in run_batch(batch):
            stats.update(attempt.rejections)
            if attempt.ticket is not None:
                accepted.append(attempt)

    if len(accepted) < n:
        raise ValueError(f"只生成出 {len(accepted)} 条可答样本，少于要求的 {n} 条；拒收统计={dict(stats)}")
    return accepted[:n], stats


def attempt_error(
    entry: ManifestEntry,
    doc_text: str,
    error_text: str,
    generator: CachedGenerator,
    model: str,
    seen: set[str],
    lock: threading.Lock,
) -> DocAttempt:
    """报错样本：除了沿用 check_answerable，还强制 error_text 与给定的报错串完全一致。"""
    rejections: list[str] = []
    for temperature in RETRY_TEMPERATURES:
        try:
            ticket = generator.generate(
                GeneratedTicket,
                model,
                temperature,
                SYSTEM_ANSWERABLE,
                build_error_prompt(entry, doc_text, error_text),
            )
        except Exception as exc:  # 单篇失败不该拖垮整批
            rejections.append(f"{REJECT_LLM}: {type(exc).__name__}")
            continue
        assert isinstance(ticket, GeneratedTicket)
        if (ticket.error_text or "").strip() != error_text:
            rejections.append(REJECT_ERROR_MISMATCH)
            continue
        reason = check_answerable(ticket.question, doc_text, error_text, seen)
        if reason is None:
            question = ticket.question.strip()
            with lock:
                if question in seen:
                    reason = REJECT_DUP
                else:
                    seen.add(question)
                    fixed = sanitize_ticket(ticket).model_copy(
                        update={"error_text": error_text}
                    )
                    return DocAttempt(entry, fixed, rejections)
        rejections.append(reason)
    return DocAttempt(entry, None, rejections)


def run_errors(
    plan: ErrorSamplePlan,
    kb_dir: Path,
    generator: CachedGenerator,
    model: str,
    workers: int = 1,
) -> tuple[list[DocAttempt], Counter]:
    """每篇文档只出一条报错样本；拒收就跳过，不用备选文档补位。"""
    seen: set[str] = set()
    lock = threading.Lock()
    stats: Counter = Counter()

    def work(pair: tuple[ManifestEntry, str]) -> DocAttempt:
        entry, error_text = pair
        try:
            text = read_doc(kb_dir, entry)
        except OSError as exc:
            return DocAttempt(entry, None, [f"{REJECT_DOC}: {type(exc).__name__}"])
        return attempt_error(entry, text, error_text, generator, model, seen, lock)

    if not plan.selected:
        return [], stats
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(plan.selected))) as pool:
            attempts = list(pool.map(work, plan.selected))
    else:
        attempts = [work(pair) for pair in plan.selected]
    for attempt in attempts:
        stats.update(attempt.rejections)
    return attempts, stats


def run_unanswerables(
    generator: CachedGenerator, model: str, n: int, workers: int = 1
) -> list[tuple[Category, str]]:
    """负样本：2/3 需要查客户数据，1/3 闲聊或产品外事项。"""
    if n <= 0:
        return []
    n_data = (2 * n + 2) // 3  # ceil(2n/3)
    jobs = [("data_exception", n_data), ("other", n - n_data)]
    jobs = [(c, cnt) for c, cnt in jobs if cnt > 0]

    def work(job: tuple[Literal["data_exception", "other"], int]) -> tuple[str, UnanswerableBatch]:
        category, count = job
        try:
            batch = generator.generate(
                UnanswerableBatch,
                model,
                0.8,
                SYSTEM_UNANSWERABLE,
                build_unanswerable_prompt(category, count),
            )
        except Exception:
            batch = UnanswerableBatch(items=[])
        assert isinstance(batch, UnanswerableBatch)
        return category, batch

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            batches = list(pool.map(work, jobs))
    else:
        batches = [work(job) for job in jobs]

    out: list[tuple[Category, str]] = []
    seen: set[str] = set()
    limits = dict(jobs)
    taken: Counter = Counter()
    for category, batch in batches:
        for item in batch.items:
            if taken[category] >= limits[category]:
                break
            question = item.question.strip()
            if not question or question in seen or not (8 <= len(question) <= 200):
                continue
            seen.add(question)
            taken[category] += 1
            out.append((category, question))  # type: ignore[arg-type]
    return out


def assemble_dataset(
    answerables: Sequence[tuple[ManifestEntry, GeneratedTicket]],
    unanswerables: Sequence[tuple[Category, str]],
    error_samples: Sequence[tuple[ManifestEntry, GeneratedTicket]] = (),
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for entry, ticket in answerables:
        question = ticket.question.strip()
        items.append(
            EvalItem(
                qid=f"q{len(items) + 1:04d}",
                question=question,
                category=category_for(entry.top_dir, entry.title),
                answerable=True,
                gold_doc_ids=[entry.doc_id],
                gold_title_keys=[entry.title_key],
                severity=ticket.severity,
                platform_name=ticket.platform_name,
                feature_name=ticket.feature_name,
                partner_name=ticket.partner_name,
                error_text=ticket.error_text,
                source="generated",
            )
        )
    for category, question in unanswerables:
        items.append(
            EvalItem(
                qid=f"q{len(items) + 1:04d}",
                question=question,
                category=category,
                answerable=False,
                gold_doc_ids=[],
                gold_title_keys=[],
                source="generated",
            )
        )
    for entry, ticket in error_samples:
        items.append(
            EvalItem(
                qid=f"q{len(items) + 1:04d}",
                question=ticket.question.strip(),
                category=category_for(entry.top_dir, entry.title),
                answerable=True,
                gold_doc_ids=[entry.doc_id],
                gold_title_keys=[entry.title_key],
                severity=ticket.severity,
                platform_name=ticket.platform_name,
                feature_name=ticket.feature_name,
                partner_name=ticket.partner_name,
                error_text=ticket.error_text,
                source="generated",
            )
        )
    return items


def _append_review_item(lines: list[str], item: EvalItem, kb_dir: Path) -> None:
    lines.append(f"## {item.qid}")
    lines.append(f"- 问题：{item.question}")
    if item.error_text:
        lines.append(f"- 报错原文：{item.error_text}")
    if item.gold_doc_ids:
        doc_id = item.gold_doc_ids[0]
        relative = f"docs/{doc_id}.md"
        try:
            text = (Path(kb_dir) / relative).read_text(encoding="utf-8")
        except OSError:
            text = "（读取文档失败）"
        preview = " ".join(text.split())[:200]
        lines.append(f"- 标准答案文档：{relative}")
        lines.append(f"- 文档开头：{preview}")
    else:
        lines.append("- 标准答案文档：无（期望拒答）")
    lines.append("")


def write_review(
    path: Path, items: Sequence[EvalItem], kb_dir: Path, seed: int, n: int = 25
) -> None:
    error_items = [it for it in items if it.error_text]
    if not error_items:
        # n_error=0 时保持原有格式和抽样方式不变
        picked = random.Random(seed).sample(items, min(n, len(items)))
        lines = ["# 评估集人工抽查", "", f"共 {len(items)} 条，随机抽 {len(picked)} 条。", ""]
        for item in picked:
            _append_review_item(lines, item, kb_dir)
    else:
        normal = [it for it in items if not it.error_text]
        picked = random.Random(seed).sample(normal, min(n, len(normal)))
        errors = random.Random(f"{seed}:error").sample(error_items, min(10, len(error_items)))
        lines = [
            "# 评估集人工抽查",
            "",
            f"共 {len(items)} 条（其中报错原文样本 {len(error_items)} 条），"
            f"可答/不可答抽 {len(picked)} 条，报错样本抽 {len(errors)} 条。",
            "",
            "## 可答 / 不可答样本",
            "",
        ]
        for item in picked:
            _append_review_item(lines, item, kb_dir)
        lines.extend(["## 报错原文样本", ""])
        for item in errors:
            _append_review_item(lines, item, kb_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
