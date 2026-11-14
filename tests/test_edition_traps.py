import importlib.util
import json
import re
import shutil
import threading
import uuid
from pathlib import Path

import pytest

from carepilot.evaluation import edition_traps as et
from carepilot.evaluation import gen_core as core


@pytest.fixture
def work_tmp():
    path = Path(__file__).resolve().parents[1] / ".tmp_test" / f"w-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def build_kb(work_tmp: Path, specs: list[dict]) -> Path:
    root = work_tmp / "kb"
    (root / "docs").mkdir(parents=True, exist_ok=True)
    lines = []
    for spec in specs:
        doc_id = spec["doc_id"]
        title = spec.get("title") or doc_id.rsplit("/", 1)[-1]
        text = spec.get("text") or f"# {title}\n\n这是 {title} 的正文。"
        path = root / "docs" / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        lines.append(
            {
                "doc_id": doc_id,
                "title": title,
                "title_key": spec.get("title_key") or core.title_key_for(title),
                "top_dir": spec["top_dir"],
                "source_url": "",
                "chars": spec.get("chars", 1000),
                "kept": spec.get("kept", True),
                "drop_reason": None,
                "dup_of": None,
                "flags": list(spec.get("flags", [])),
            }
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8"
    )
    return root


ENTERPRISE_TEXT = """# 订单审核

企业版订单审核支持批量审核，快捷键是 Ctrl+Shift+E。
企业版可以按店铺设置独立的审核规则。
企业版审核完成后自动打印快递单。
"""

EXPRESS_TEXT = """# 订单审核

极速版订单审核一次最多选择五十单操作。
极速版审核结果会同步到手机端消息中心。
极速版不支持批量修改收件人地址。
"""


def pair_specs(
    title_key: str = "订单审核",
    *,
    enterprise_text: str = ENTERPRISE_TEXT,
    express_text: str = EXPRESS_TEXT,
    enterprise_extra: dict | None = None,
    express_extra: dict | None = None,
) -> list[dict]:
    enterprise = {
        "doc_id": f"企业版/{title_key}",
        "top_dir": "企业版",
        "title": title_key,
        "title_key": title_key,
        "text": enterprise_text,
    }
    express = {
        "doc_id": f"极速版2.0/{title_key}",
        "top_dir": "极速版2.0",
        "title": title_key,
        "title_key": title_key,
        "text": express_text,
    }
    enterprise.update(enterprise_extra or {})
    express.update(express_extra or {})
    return [enterprise, express]


# ---------------------------------------------------------------- 配对


def test_find_edition_pairs_basic(work_tmp):
    kb = build_kb(work_tmp, pair_specs())
    pairs = et.find_edition_pairs(core.load_manifest(kb))
    assert len(pairs) == 1
    enterprise, express = pairs[0]
    assert enterprise.top_dir == et.ENTERPRISE_DIR
    assert express.top_dir == et.EXPRESS_DIR
    assert enterprise.title_key == express.title_key == "订单审核"


def test_find_edition_pairs_skips_missing_side(work_tmp):
    specs = [pair_specs()[0]]
    kb = build_kb(work_tmp, specs)
    assert et.find_edition_pairs(core.load_manifest(kb)) == []


def test_find_edition_pairs_skips_ambiguous_side(work_tmp):
    specs = pair_specs()
    specs.append(
        {
            "doc_id": "极速版2.0/订单审核副本",
            "top_dir": "极速版2.0",
            "title": "订单审核(2)",
            "title_key": "订单审核",
            "text": EXPRESS_TEXT,
        }
    )
    kb = build_kb(work_tmp, specs)
    assert et.find_edition_pairs(core.load_manifest(kb)) == []


def test_find_edition_pairs_excludes_release_short_unkept_and_tiny(work_tmp):
    specs = []
    for i, flags in enumerate([["release_note"], ["short"], ["release_note", "short"]]):
        specs += pair_specs(title_key=f"通知{i}", enterprise_extra={"flags": flags})
    specs += pair_specs(title_key="太小", enterprise_extra={"chars": 299})
    specs += pair_specs(title_key="丢了", express_extra={"kept": False})
    kb = build_kb(work_tmp, specs)
    assert et.find_edition_pairs(core.load_manifest(kb)) == []


# ---------------------------------------------------------------- 特有行


def test_unique_lines_order_dedup_and_filters():
    doc_a = (
        "# 标题行不算\n"
        "\n"
        "两边都有的这一行内容\n"
        "只有 A 有的第一行内容\n"
        "只有 A 有的第一行内容\n"
        "！！！……？？？\n"
        "太短\n"
        "只有 A 有的第二行内容\n"
    )
    doc_b = "# 标题行不算\n两边都有的这一行内容\n"
    assert et.unique_lines(doc_a, doc_b) == [
        "只有 A 有的第一行内容",
        "只有 A 有的第二行内容",
    ]


def test_unique_lines_identical_docs_is_empty():
    assert et.unique_lines(ENTERPRISE_TEXT, ENTERPRISE_TEXT) == []


def test_unique_lines_cap():
    lines = "\n".join(f"这是第{i}条只在A出现的内容行" for i in range(20))
    assert len(et.unique_lines(lines, "")) == 15
    assert et.unique_lines(lines, "", cap=3) == [
        "这是第0条只在A出现的内容行",
        "这是第1条只在A出现的内容行",
        "这是第2条只在A出现的内容行",
    ]


def test_unique_lines_uses_trimmed_comparison():
    doc_a = "  前后有空格的这一行内容  \n"
    doc_b = "前后有空格的这一行内容\n"
    assert et.unique_lines(doc_a, doc_b) == []


def test_unique_lines_min_len():
    assert et.unique_lines("一二三四五六七八九\n", "", min_len=10) == []
    assert et.unique_lines("一二三四五六七八九十\n", "", min_len=10) == [
        "一二三四五六七八九十"
    ]


def test_is_discriminative():
    assert et.is_discriminative(ENTERPRISE_TEXT, EXPRESS_TEXT)
    similar = "# 订单审核\n\n企业版订单审核支持批量操作。\n共有的一行内容在这里。\n"
    other = "# 订单审核\n\n极速版订单审核支持批量操作。\n共有的一行内容在这里。\n"
    assert not et.is_discriminative(similar, other)


# ---------------------------------------------------------------- 生成


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[float] = []
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, schema, temperature, messages):
        with self._lock:
            self.calls.append(temperature)
            self.prompts.append(messages[-1].content)
            call = len(self.calls)
        return self.responses(messages[-1].content, call)


def _label_from_prompt(prompt: str) -> str:
    match = re.search(r"使用旺店通 (.+?) 的商家", prompt)
    assert match, prompt
    return match.group(1)


def _ticket(question: str) -> core.GeneratedTicket:
    return core.GeneratedTicket(question=question, severity="medium")


def _run(kb: Path, fake, work_tmp: Path, n_pairs: int = 10, seed: int = 42, workers: int = 2):
    entries = core.load_manifest(kb)
    candidates = et.discriminator_pairs(et.find_edition_pairs(entries), kb)
    selected = [(e, x) for e, x, _, _ in et.sample_pairs(candidates, n_pairs, seed)]
    generator = core.CachedGenerator(work_tmp / "cache", fake)
    return et.run_edition_traps(selected, kb, generator, "fake-model", workers=workers), generator


def _valid_responses(prompt: str, call: int) -> core.GeneratedTicket:
    label = _label_from_prompt(prompt)
    return _ticket(f"我们店里一直用{label}，想咨询订单审核的一个设置问题")


def test_each_pair_generates_two_directions(work_tmp):
    kb = build_kb(work_tmp, pair_specs())
    fake = FakeLLM(_valid_responses)
    (traps, stats), _ = _run(kb, fake, work_tmp)

    assert len(traps) == 2
    by_edition = {trap.edition: trap for trap in traps}
    assert set(by_edition) == {"企业版", "极速版2.0"}
    enterprise = by_edition["企业版"]
    express = by_edition["极速版2.0"]
    assert enterprise.gold_doc_id == "企业版/订单审核"
    assert enterprise.forbidden_doc_ids == ("极速版2.0/订单审核",)
    assert express.gold_doc_id == "极速版2.0/订单审核"
    assert express.forbidden_doc_ids == ("企业版/订单审核",)
    assert "企业版" in enterprise.question
    assert "极速版" in express.question
    assert [t.qid for t in traps] == ["e0001", "e0002"]
    assert all(t.unique_lines_used for t in traps)
    assert max(len(t.unique_lines_used) for t in traps) <= 5
    assert not stats


def test_question_without_edition_is_rejected_then_retried(work_tmp):
    kb = build_kb(work_tmp, pair_specs())

    def responses(prompt, call):
        # 每一侧的前两次都不带版本名；缓存键含温度，因此每次重试都会真调用
        if call in (1, 2):
            return _ticket("订单审核的时候批量操作有个设置想确认一下")
        return _valid_responses(prompt, call)

    fake = FakeLLM(responses)
    (traps, stats), _ = _run(kb, fake, work_tmp, workers=1)
    assert len(traps) == 2
    assert stats[et.REJECT_EDITION] == 2
    assert fake.calls == [0.0, 0.5, 1.0, 0.0]


def test_edition_only_in_parentheses_is_rejected(work_tmp):
    kb = build_kb(work_tmp, pair_specs())

    def responses(prompt, call):
        label = _label_from_prompt(prompt)
        return _ticket(f"订单审核的时候批量操作有个设置想确认一下（{label}）")

    fake = FakeLLM(responses)
    (traps, stats), _ = _run(kb, fake, work_tmp, workers=1)
    assert traps == []
    assert stats[et.REJECT_EDITION] >= 4


def test_copy_from_doc_is_rejected(work_tmp):
    kb = build_kb(work_tmp, pair_specs())

    def responses(prompt, call):
        label = _label_from_prompt(prompt)
        line = (
            "企业版可以按店铺设置独立的审核规则"
            if label == "企业版"
            else "极速版不支持批量修改收件人地址"
        )
        return _ticket(f"我们用的是{label}，{line}")

    fake = FakeLLM(responses)
    (traps, stats), _ = _run(kb, fake, work_tmp, workers=1)
    assert traps == []
    assert stats[core.REJECT_COPY] >= 2


def test_pair_dropped_when_one_side_always_invalid(work_tmp):
    kb = build_kb(work_tmp, pair_specs())

    def responses(prompt, call):
        label = _label_from_prompt(prompt)
        if label == "极速版":
            return _ticket("批量操作的时候有个设置想确认一下")
        return _valid_responses(prompt, call)

    fake = FakeLLM(responses)
    (traps, stats), _ = _run(kb, fake, work_tmp, workers=1)
    assert traps == []
    assert stats[et.REJECT_EDITION] == 3


def test_same_seed_and_cache_hit(work_tmp):
    kb = build_kb(work_tmp, pair_specs())
    fake = FakeLLM(_valid_responses)
    (first, generator), _ = _run(kb, fake, work_tmp, seed=7)
    calls_after_first = len(fake.calls)

    (second, _), _ = _run(kb, fake, work_tmp, seed=7)
    assert len(fake.calls) == calls_after_first  # 第二次全部命中缓存
    assert [t.to_json() for t in first] == [t.to_json() for t in second]
    assert first[0].question == second[0].question


def test_dump_and_review(work_tmp):
    kb = build_kb(work_tmp, pair_specs())
    (traps, _), _ = _run(kb, FakeLLM(_valid_responses), work_tmp)
    out = work_tmp / "out" / "traps.jsonl"
    et.dump_traps(out, traps)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["qid"] for r in rows] == ["e0001", "e0002"]
    assert rows[0]["forbidden_doc_ids"] == ["极速版2.0/订单审核"]

    review = work_tmp / "review.md"
    et.write_review(review, traps, seed=1)
    text = review.read_text(encoding="utf-8")
    assert "e0001" in text and "docs/企业版/订单审核.md" in text
    assert "被禁止" in text or "禁止引用" in text


# ---------------------------------------------------------------- 命令行


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "gen_edition_traps.py"
    spec = importlib.util.spec_from_file_location("gen_edition_traps", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_prints_stats_and_prompt(work_tmp, capsys):
    kb = build_kb(work_tmp, pair_specs())
    cli = _load_cli_module()
    code = cli.main(["--kb-dir", str(kb), "--n-pairs", "5", "--seed", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "同名配对数" in out
    assert "可区分配对数：1" in out
    assert "订单审核" in out
    assert "提示词样例" in out
    assert "本版本特有内容" in out


def test_cli_missing_manifest_clear_error(work_tmp, capsys):
    cli = _load_cli_module()
    code = cli.main(["--kb-dir", str(work_tmp / "missing"), "--dry-run"])
    assert code == 2
    assert "manifest.jsonl" in capsys.readouterr().err
