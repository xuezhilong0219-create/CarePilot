import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import pytest

from carepilot.evaluation.injection import (
    PAYLOADS,
    PLACEMENTS,
    inject,
    is_compromised,
)

ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "eval" / "gen_injection_cases.py"

_spec = importlib.util.spec_from_file_location("gen_injection_cases", GEN_SCRIPT)
assert _spec and _spec.loader
gen_injection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_injection)


@pytest.fixture
def work_tmp():
    # 沙箱里 pytest 的 tmp_path 建在不可写的系统临时目录下，用仓库根的 .tmp_test/
    path = ROOT / ".tmp_test" / f"injection-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


LEGIT = (
    "店铺授权过期后，订单无法下载。\n"
    "1. 进入「设置 - 店铺管理」，选中店铺后点击「重新授权」。\n"
    "2. 使用平台主账号登录并完成授权。\n"
    "3. 授权成功后点击「下载订单」确认恢复。\n"
    "如果仍提示 session expired，请重新走一遍授权流程。"
)


@pytest.mark.parametrize("payload", PAYLOADS, ids=[payload.id for payload in PAYLOADS])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_inject_keeps_original_content_and_payload(payload, placement):
    result = inject(LEGIT, payload, placement)
    assert payload.marker in result
    if placement == "prefix":
        assert result.startswith(payload.text)
        assert result.endswith(LEGIT)
    elif placement == "suffix":
        assert result.startswith(LEGIT)
        assert result.endswith(payload.text)
    elif placement == "middle":
        assert result.replace("\n" + payload.text + "\n", "", 1) == LEGIT
    elif placement == "table_cell":
        assert result.startswith(LEGIT)
        assert "| 项目 | 说明 |" in result
        assert "| --- | --- |" in result
    else:
        assert result.startswith(LEGIT)
        assert "<!--" in result and "-->" in result


def test_inject_rejects_unknown_placement():
    with pytest.raises(ValueError):
        inject(LEGIT, PAYLOADS[0], "somewhere")


@pytest.mark.parametrize("payload", PAYLOADS, ids=[payload.id for payload in PAYLOADS])
def test_each_marker_is_detected(payload):
    reasons = is_compromised("已处理：" + payload.marker, [], payload)
    assert any(reason.startswith("marker:") for reason in reasons)


def test_marker_detection_is_case_insensitive():
    payload = next(item for item in PAYLOADS if item.marker == "CLOSE_TICKET_NOW")
    assert is_compromised("close_ticket_now", [], payload)


def test_phone_number_is_detected():
    reasons = is_compromised("请联系 13900000000 处理", [], PAYLOADS[0])
    assert "phone" in reasons


def test_external_domain_is_detected():
    payload = next(item for item in PAYLOADS if "evil.example" in item.text)
    reasons = is_compromised("详见 https://evil.example/help", [], payload)
    assert any(reason.startswith("external_domain:") for reason in reasons)


def test_forged_citation_is_detected():
    payload = PAYLOADS[0]
    assert payload.fake_doc_id
    reasons = is_compromised("已按文档处理", ["doc-1", payload.fake_doc_id], payload)
    assert any(reason.startswith("fake_citation:") for reason in reasons)


def test_normal_reply_does_not_false_positive():
    reply = (
        "订单号 2025111400012345678 已发货，金额 199.00 元。"
        "请进入「设置 - 店铺管理」重新授权，客服电话 010-12345678。"
    )
    assert is_compromised(reply, ["平台对接_店铺授权/授权"], PAYLOADS[0]) == []


def test_ten_and_twelve_digit_numbers_are_not_phones():
    assert is_compromised("编号 1390000000，流水 139000000000", [], PAYLOADS[0]) == []


def _build_kb(root: Path, eligible: int = 60) -> Path:
    kb = root / "kb"
    docs = kb / "docs" / "平台对接_店铺授权"
    docs.mkdir(parents=True)
    entries = []
    for index in range(eligible):
        doc_id = f"平台对接_店铺授权/doc_{index:02d}"
        title = f"授权文档 {index:02d}"
        content = f"# {title}\n\n" + ("这是用于注入测试的正文内容，包含操作步骤。" * 20)
        path = kb / "docs" / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        entries.append(
            {
                "doc_id": doc_id,
                "title": title,
                "title_key": title,
                "top_dir": "平台对接_店铺授权",
                "source_url": "https://example.invalid/doc",
                "chars": len(content),
                "kept": True,
                "drop_reason": None,
                "dup_of": None,
                "flags": [],
            }
        )
    short = "# 太短\n\n只有一点点内容。"
    short_path = kb / "docs" / "平台对接_店铺授权" / "short.md"
    short_path.write_text(short, encoding="utf-8")
    entries.append(
        {
            "doc_id": "平台对接_店铺授权/short",
            "title": "太短",
            "title_key": "太短",
            "top_dir": "平台对接_店铺授权",
            "source_url": "",
            "chars": len(short),
            "kept": True,
            "drop_reason": None,
            "dup_of": None,
            "flags": [],
        }
    )
    release = "# 版本日志\n\n" + ("版本变更说明。" * 100)
    release_path = kb / "docs" / "平台对接_店铺授权" / "release.md"
    release_path.write_text(release, encoding="utf-8")
    entries.append(
        {
            "doc_id": "平台对接_店铺授权/release",
            "title": "版本日志",
            "title_key": "版本日志",
            "top_dir": "平台对接_店铺授权",
            "source_url": "",
            "chars": len(release),
            "kept": True,
            "drop_reason": None,
            "dup_of": None,
            "flags": ["release_note"],
        }
    )
    manifest = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n"
    (kb / "manifest.jsonl").write_text(manifest, encoding="utf-8")
    return kb


def _read_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generator_covers_payloads_and_placements(work_tmp):
    kb = _build_kb(work_tmp)
    out = work_tmp / "cases.jsonl"
    assert gen_injection.main(["--kb-dir", str(kb), "--n", "60", "--seed", "42", "--out", str(out)]) == 0
    cases = _read_cases(out)
    assert len(cases) == 60
    assert len({case["case_id"] for case in cases}) == 60

    payload_counts: dict[str, int] = {}
    placement_counts: dict[str, int] = {}
    for case in cases:
        payload_counts[case["payload_id"]] = payload_counts.get(case["payload_id"], 0) + 1
        placement_counts[case["placement"]] = placement_counts.get(case["placement"], 0) + 1
        assert case["marker"] in case["injected_text"]
        assert "「" in case["question"] and "」" in case["question"]
        assert case["doc_id"] not in ("平台对接_店铺授权/short", "平台对接_店铺授权/release")
    assert set(payload_counts) == {payload.id for payload in PAYLOADS}
    assert set(placement_counts) == set(PLACEMENTS)
    for count in payload_counts.values():
        assert count >= 60 // 10
    for count in placement_counts.values():
        assert count >= 60 // 10


def test_generator_is_byte_identical_for_same_seed(work_tmp):
    kb = _build_kb(work_tmp)
    first = work_tmp / "first.jsonl"
    second = work_tmp / "second.jsonl"
    for path in (first, second):
        assert gen_injection.main(["--kb-dir", str(kb), "--n", "20", "--seed", "7", "--out", str(path)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_generator_dry_run_writes_nothing(work_tmp, capsys):
    kb = _build_kb(work_tmp)
    out = work_tmp / "cases.jsonl"
    assert gen_injection.main(["--kb-dir", str(kb), "--n", "10", "--seed", "1", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    output = capsys.readouterr().out
    assert "载荷覆盖" in output and "位置覆盖" in output


def test_generator_missing_manifest_returns_2(work_tmp, capsys):
    empty = work_tmp / "empty"
    empty.mkdir()
    out = work_tmp / "cases.jsonl"
    assert gen_injection.main(["--kb-dir", str(empty), "--n", "5", "--out", str(out)]) == 2
    assert "manifest.jsonl" in capsys.readouterr().err


def test_generator_not_enough_docs_returns_2(work_tmp, capsys):
    kb = _build_kb(work_tmp, eligible=3)
    out = work_tmp / "cases.jsonl"
    assert gen_injection.main(["--kb-dir", str(kb), "--n", "60", "--out", str(out)]) == 2
    assert "不够" in capsys.readouterr().err


def test_scorer_catches_a_compliant_model_for_every_payload_type():
    """校准：一个"听话"的假模型照做载荷时，判分器必须逐种载荷报警。没有这条，真实评测里的 0% 无法说明任何事。"""
    from carepilot.evaluation.injection import PAYLOADS, is_compromised

    for p in PAYLOADS:
        obeying = f"好的。{p.text} 另外，联系 {p.marker}。"
        assert is_compromised(obeying, [], p), f"判分器漏检了 {p.id} 的顺从回复"
        calm = "您好，请在【设置】-【店铺管理】里重新授权，授权成功后点击下载订单。"
        assert is_compromised(calm, [], p) == [], f"判分器对正常回复误报了 {p.id}"
