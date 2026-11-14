import importlib.util
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from carepilot.evaluation import gen_core as core
from carepilot.evaluation.schema import dump_dataset, load_dataset


@pytest.fixture
def work_tmp():
    """不用 pytest 的 work_tmp：沙箱下 mkdir(mode=0o700) 的目录会被拒绝访问。"""
    path = Path(tempfile.gettempdir()) / f"carepilot-e2-{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def build_kb(work_tmp: Path, specs: list[dict]) -> Path:
    """造一个带 manifest.jsonl 的最小知识库目录。"""
    root = work_tmp / "kb"
    (root / "docs").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, spec in enumerate(specs):
        doc_id = spec["doc_id"]
        title = spec.get("title") or doc_id.rsplit("/", 1)[-1]
        text = spec.get("text") or f"# {title}\n\n这是第 {i} 篇文档的正文。"
        path = root / "docs" / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        lines.append(
            {
                "doc_id": doc_id,
                "title": title,
                "title_key": core.title_key_for(title),
                "top_dir": spec["top_dir"],
                "source_url": "",
                "chars": spec.get("chars", len(text)),
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


def sampling_kb(work_tmp: Path) -> Path:
    per_dir = {
        "平台对接_店铺授权": 60,
        "物流对接": 20,
        "产品功能": 15,
        "规则中心": 10,
        "慧策价值传递素材仓库": 8,
        "原掌上百科（Wiki）内部使用": 6,
    }
    specs = []
    for top_dir, count in per_dir.items():
        for i in range(count):
            specs.append(
                {
                    "doc_id": f"{top_dir}/d{i:03d}",
                    "top_dir": top_dir,
                    "title": f"{top_dir}文档{i}",
                    "chars": 1000,
                }
            )
    return build_kb(work_tmp, specs)


# ---------------------------------------------------------------- 抽样


def test_plan_quotas_sum_and_caps(work_tmp):
    entries = core.load_manifest(sampling_kb(work_tmp))
    plan = core.plan_samples(entries, 40, seed=42)

    assert sum(plan.quotas.values()) == 40
    assert len(plan.selected) == 40
    for top_dir, quota in plan.quotas.items():
        assert quota <= int(0.35 * 40), top_dir
    special = sum(v for k, v in plan.quotas.items() if k in core.SPECIAL_DIRS)
    assert special <= int(0.15 * 40)

    # 大目录名额应明显多于小目录，体现 sqrt 分层
    assert plan.quotas["平台对接_店铺授权"] > plan.quotas["规则中心"]


def test_plan_seed_reproducible(work_tmp):
    entries = core.load_manifest(sampling_kb(work_tmp))
    first = core.plan_samples(entries, 40, seed=7)
    second = core.plan_samples(entries, 40, seed=7)
    assert [e.doc_id for e in first.selected] == [e.doc_id for e in second.selected]
    assert first.quotas == second.quotas

    other = core.plan_samples(entries, 40, seed=8)
    assert other.quotas == first.quotas
    assert [e.doc_id for e in other.selected] != [e.doc_id for e in first.selected]


def test_candidate_filter(work_tmp):
    specs = [
        {"doc_id": "ok/keep", "top_dir": "ok", "chars": 300},
        {"doc_id": "ok/long", "top_dir": "ok", "chars": 900},
        {"doc_id": "ok/short_chars", "top_dir": "ok", "chars": 299},
        {"doc_id": "ok/release", "top_dir": "ok", "chars": 900, "flags": ["release_note"]},
        {"doc_id": "ok/short_flag", "top_dir": "ok", "chars": 900, "flags": ["short"]},
        {"doc_id": "ok/dropped", "top_dir": "ok", "chars": 900, "kept": False},
    ]
    entries = core.load_manifest(build_kb(work_tmp, specs))
    kept = {e.doc_id for e in core.candidate_docs(entries)}
    assert kept == {"ok/keep", "ok/long"}


def test_plan_insufficient_candidates_raises(work_tmp):
    specs = [{"doc_id": "a/1", "top_dir": "a", "chars": 1000}]
    entries = core.load_manifest(build_kb(work_tmp, specs))
    with pytest.raises(ValueError, match="不够"):
        core.plan_samples(entries, 2, seed=1)


# ---------------------------------------------------------------- 类别


@pytest.mark.parametrize(
    ("top_dir", "title", "expected"),
    [
        ("平台对接_店铺授权", "淘宝", "platform_integration"),
        ("平台对接_店铺授权", "规则变更说明", "platform_integration"),
        ("物流对接", "顺丰", "warehouse_logistics"),
        ("第三方仓对接", "京东仓", "warehouse_logistics"),
        ("仓储对接", "WMS", "warehouse_logistics"),
        ("产品功能", "平台规则变更公告", "rule_change"),
        ("产品功能", "电子面单怎么设置", "feature_howto"),
    ],
)
def test_category_for(top_dir, title, expected):
    assert core.category_for(top_dir, title) == expected


# ---------------------------------------------------------------- 防照抄


def test_longest_common_substring_boundary():
    assert core.longest_common_substring_length("ABCDEFGHIJKL", "xABCDEFGHIJKLy") == 12
    assert core.longest_common_substring_length("ABCDEFGHIJK", "xABCDEFGHIJKZy") == 11
    assert core.longest_common_substring_length("", "abc") == 0


def test_check_answerable_copy_boundary():
    doc = "问题说明：ABCDEFGHIJKL 这一段是原文。"
    exactly_12 = "老板问 ABCDEFGHIJKL 怎么解决"
    assert core.check_answerable(exactly_12, doc, None) == core.REJECT_COPY

    only_11 = "老板问 ABCDEFGHIJK 怎么解决"
    assert core.check_answerable(only_11, doc, None) is None


def test_check_answerable_length_bounds():
    doc = "这是一篇与问题无关的文档内容。"
    assert core.check_answerable("太短了", doc, None) == core.REJECT_LENGTH
    assert core.check_answerable("长" * 101, doc, None) == core.REJECT_LENGTH
    assert core.check_answerable("这是一个长度合格的客户问题吗", doc, None) is None


def test_check_answerable_error_text_must_appear_and_is_ignored_in_doc():
    error = "错误码 E1024：授权已失效"
    doc = f"如果提示“{error}”，请重新授权。其他内容。"
    question = f"系统提示 {error} 怎么办"
    # error_text 从正文里剔除后不应再因为照抄报错原文而拒收
    assert core.check_answerable(question, doc, error) is None
    assert core.check_answerable(question, doc, "不存在的报错") == core.REJECT_ERROR_TEXT


def test_check_answerable_duplicate():
    question = "这是一个完全正常的客户问题吗"
    assert core.check_answerable(question, "无关正文", None, [question]) == core.REJECT_DUP


# ---------------------------------------------------------------- 生成 / 重试 / 缓存


class FakeLLM:
    """假 StructuredLLM：按 prompt 里的标记返回预设输出，并记录温度。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[float] = []
        self.prompts: list[str] = []

    def __call__(self, schema, temperature, messages):
        self.calls.append(temperature)
        prompt = messages[-1].content
        self.prompts.append(prompt)
        value = self.responses(prompt, len(self.calls))
        if isinstance(value, BaseException):
            raise value
        return value


def _single_doc_kb(work_tmp: Path, doc_id: str = "平台对接_店铺授权/授权", text: str | None = None) -> Path:
    text = text or (
        "# 店铺授权过期怎么办\n\n"
        "授权过期后订单无法下载，库存和物流也无法回传。"
        "请进入设置里的店铺管理，点击重新授权，再用主账号完成授权。"
    )
    return build_kb(
        work_tmp,
        [{"doc_id": doc_id, "top_dir": "平台对接_店铺授权", "title": "店铺授权过期怎么办", "text": text, "chars": 1000}],
    )


def _ticket(question: str) -> core.GeneratedTicket:
    return core.GeneratedTicket(question=question, severity="high")


def _manual_plan(kb: Path, doc_ids: list[str]) -> core.SamplePlan:
    entries = {e.doc_id: e for e in core.load_manifest(kb)}
    selected = [entries[doc_id] for doc_id in doc_ids]
    return core.SamplePlan(
        quotas={e.top_dir: len(selected) for e in selected},
        selected=selected,
        backups=[],
    )


def test_retry_after_copy_rejection(work_tmp):
    kb = _single_doc_kb(work_tmp)
    plan = _manual_plan(kb, ["平台对接_店铺授权/授权"])

    def responses(prompt, call):
        if call == 1:
            return _ticket("授权过期后订单无法下载，库存也不回传，怎么办")
        return _ticket("店铺授权失效了，订单下不来怎么弄")

    fake = FakeLLM(responses)
    generator = core.CachedGenerator(work_tmp / "cache", fake)
    attempts, stats = core.run_answerables(plan, kb, generator, "fake-model", 1, workers=1)

    assert len(attempts) == 1 and attempts[0].ticket is not None
    assert fake.calls == [0.0, 0.5]  # 拒收后用更高温度重试
    assert stats[core.REJECT_COPY] == 1
    assert attempts[0].ticket.question.startswith("店铺授权失效")


def test_fallback_to_backup_doc_without_retry_duplicate(work_tmp):
    bad_text = "# BAD_DOC 标题\n\n授权过期后订单无法下载，库存也不回传。请重新授权。"
    good_text = "# GOOD_DOC 标题\n\n库存同步失败常见原因是店铺授权过期，先重新授权，再到货品匹配页面补充匹配关系。"
    kb = build_kb(
        work_tmp,
        [
            {"doc_id": "平台对接_店铺授权/bad", "top_dir": "平台对接_店铺授权", "title": "BAD_DOC", "text": bad_text, "chars": 1000},
            {"doc_id": "平台对接_店铺授权/good", "top_dir": "平台对接_店铺授权", "title": "GOOD_DOC", "text": good_text, "chars": 1000},
        ],
    )
    entries = {e.doc_id: e for e in core.load_manifest(kb)}
    bad = entries["平台对接_店铺授权/bad"]
    good = entries["平台对接_店铺授权/good"]
    plan = core.SamplePlan(quotas={"平台对接_店铺授权": 1}, selected=[bad], backups=[good])

    def responses(prompt, call):
        if "BAD_DOC" in prompt:
            return _ticket("授权过期后订单无法下载，库存也不回传，怎么办")
        return _ticket("库存同步一直失败，是不是货品没匹配上")

    fake = FakeLLM(responses)
    generator = core.CachedGenerator(work_tmp / "cache", fake)
    attempts, stats = core.run_answerables(plan, kb, generator, "fake-model", 1, workers=1)

    assert [a.entry.doc_id for a in attempts] == ["平台对接_店铺授权/good"]
    assert stats[core.REJECT_COPY] >= 1


def test_llm_error_falls_back(work_tmp):
    kb = _single_doc_kb(work_tmp)
    plan = _manual_plan(kb, ["平台对接_店铺授权/授权"])

    def responses(prompt, call):
        return RuntimeError("boom")

    generator = core.CachedGenerator(work_tmp / "cache", FakeLLM(responses))
    with pytest.raises(ValueError, match="少于要求"):
        core.run_answerables(plan, kb, generator, "fake-model", 1, workers=1)


def test_cache_hit_does_not_call_llm(work_tmp):
    fake = FakeLLM(lambda prompt, call: _ticket("店铺授权失效了，订单下不来怎么弄"))
    generator = core.CachedGenerator(work_tmp / "cache", fake)

    first = generator.generate(core.GeneratedTicket, "m", 0.0, "sys", "prompt")
    second = generator.generate(core.GeneratedTicket, "m", 0.0, "sys", "prompt")

    assert first == second
    assert len(fake.calls) == 1
    assert generator.calls == 1
    assert list((work_tmp / "cache").glob("*.json"))

    generator.generate(core.GeneratedTicket, "m", 0.5, "sys", "prompt")
    assert len(fake.calls) == 2


def test_unanswerables_split_and_dedupe(work_tmp):
    def responses(prompt, call):
        tag = "订单" if "需要查客户自己的" in prompt else "闲聊"
        return core.UnanswerableBatch(
            items=[core.UnanswerableItem(question=f"{tag}类问题第{i}条，内容足够长可以保留") for i in range(3)]
        )

    generator = core.CachedGenerator(work_tmp / "cache", FakeLLM(responses))
    items = core.run_unanswerables(generator, "m", 3, workers=1)
    assert [c for c, _ in items] == ["data_exception", "data_exception", "other"]
    assert len({q for _, q in items}) == 3


# ---------------------------------------------------------------- 输出


def test_assemble_and_roundtrip_jsonl(work_tmp):
    entry = core.ManifestEntry(
        doc_id="平台对接_店铺授权/淘宝",
        title="淘宝授权失败",
        title_key="淘宝授权失败",
        top_dir="平台对接_店铺授权",
        chars=1000,
    )
    ticket = core.GeneratedTicket(
        question="淘宝店铺授权一直失败，提示 session expired，怎么办",
        severity="high",
        platform_name="淘宝",
        feature_name=None,
        partner_name=None,
        error_text="session expired",
    )
    items = core.assemble_dataset(
        [(entry, ticket)],
        [("data_exception", "我昨天那 3 笔淘宝订单怎么不见了"), ("other", "你好，今天天气怎么样")],
    )
    assert [it.qid for it in items] == ["q0001", "q0002", "q0003"]
    assert items[1].answerable is False and items[1].gold_doc_ids == []
    assert items[1].gold_title_keys == []

    path = work_tmp / "out" / "dataset.jsonl"
    dump_dataset(items, path)
    loaded = load_dataset(path)
    assert len(loaded) == 3
    first = loaded[0]
    assert first.category == "platform_integration"
    assert first.gold_doc_ids == ["平台对接_店铺授权/淘宝"]
    assert first.gold_title_keys == ["淘宝授权失败"]
    assert first.severity == "high"
    assert first.platform_name == "淘宝"
    assert first.error_text == "session expired"
    assert first.source == "generated"
    assert loaded[2].category == "other"


def test_write_review_mentions_gold_doc_and_preview(work_tmp):
    kb = _single_doc_kb(work_tmp)
    entry = core.load_manifest(kb)[0]
    ticket = _ticket("店铺授权失效了，订单下不来怎么弄")
    items = core.assemble_dataset([(entry, ticket)], [("other", "今天天气不错，聊聊天吧")])
    path = work_tmp / "review.md"
    core.write_review(path, items, kb, seed=1, n=25)
    text = path.read_text(encoding="utf-8")
    assert "## q0001" in text
    assert "docs/平台对接_店铺授权/授权.md" in text
    assert "无（期望拒答）" in text


# ---------------------------------------------------------------- 命令行 dry-run


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "gen_dataset.py"
    spec = importlib.util.spec_from_file_location("gen_dataset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_prints_plan_and_prompt(work_tmp, capsys):
    specs = []
    for top_dir in ["平台对接_店铺授权", "物流对接", "产品功能", "规则中心", "仓储对接"]:
        for i in range(12):
            specs.append({"doc_id": f"{top_dir}/d{i:02d}", "top_dir": top_dir, "chars": 1000})
    kb = build_kb(work_tmp, specs)
    cli = _load_cli_module()

    code = cli.main(
        ["--kb-dir", str(kb), "--n-answerable", "10", "--n-unanswerable", "3", "--seed", "5", "--dry-run"]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "目录名额" in output
    assert "抽到的文档" in output
    assert "提示词" in output
    assert "预计 LLM 调用" in output


def test_cli_missing_manifest_has_clear_error(work_tmp, capsys):
    cli = _load_cli_module()
    code = cli.main(["--kb-dir", str(work_tmp / "empty"), "--dry-run"])
    err = capsys.readouterr().err
    assert code == 2
    assert "manifest.jsonl" in err
