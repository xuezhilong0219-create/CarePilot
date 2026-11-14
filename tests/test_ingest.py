from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from carepilot.ingest.clean import (  # noqa: E402
    body_char_count,
    clean_markdown,
    exact_dup_of,
    extract_source_url,
    html_tag_counts,
    normalize_layout,
    strip_html,
    strip_images,
    title_key,
)
from scripts.build_kb import _long_path, build  # noqa: E402

BODY = "这是一段足够长的售后知识正文。" * 10  # 160 字，不会被判 tiny


def write_doc(raw: Path, rel: str, text: str) -> None:
    path = raw / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_manifest(out: Path) -> list[dict]:
    lines = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_newlines_normalized_and_source_removed():
    doc = clean_markdown("# 淘宝\r\n\r\n> 来源：https://www.yuque.com/a/b\r\n\r\n正文", "淘宝")
    assert doc.source_url == "https://www.yuque.com/a/b"
    assert "来源" not in doc.text
    assert "\r" not in doc.text
    assert doc.text.startswith("# 淘宝\n\n正文")


def test_source_fullwidth_colon_and_no_source():
    url, text = extract_source_url("> 来源: https://x/y/z\n正文")
    assert url == "https://x/y/z"
    assert "来源" not in text
    assert extract_source_url("没有来源行") == (None, "没有来源行")


def test_images_removed_text_kept():
    raw = "# 标题\n\n前![image](https://cdn.nlark.com/a.png)中<img src=\"x.png\" />后"
    text = strip_images(raw)
    assert "cdn.nlark" not in text and "img" not in text
    assert "前中后" in text


def test_html_tags_strip_keep_inner_text_and_br():
    raw = "<color #c00>红</color><font size=2>字</font><span>span</span><u>下划线</u><b>粗</b>"
    stripped = strip_html(raw)
    assert "<" not in stripped and ">" not in stripped
    assert stripped == "红字span下划线粗"
    assert strip_html("一行<br/>二行<br />三行") == "一行\n二行\n三行"
    assert strip_html('保留<a name="x"></a>这里') == "保留这里"


def test_xml_payload_is_not_treated_as_html():
    raw = "<response><flag>success</flag><message>ok</message></response>"
    assert strip_html(raw) == raw


def test_tags_inside_code_fence_are_kept():
    raw = "```xml\n<br/>\n```"
    assert strip_html(raw) == raw


def test_html_tag_counts_top20():
    counts = html_tag_counts(["<b>x</b><b>y</b>", "<span>z</span><B>w</B>"])
    assert counts == Counter({"b": 6, "span": 2})
    assert html_tag_counts(["<https://x/y>"]) == Counter()


def test_normalize_layout():
    text = "a  \n\n\n\nb\n\n\n\n\nc\n"
    assert normalize_layout(text) == "a\n\nb\n\nc"


def test_title_is_added_once():
    assert clean_markdown("正文内容", "标题").text.startswith("# 标题\n\n正文内容")
    assert clean_markdown("# 标题\n\n正文内容", "标题").text == "# 标题\n\n正文内容"
    assert clean_markdown("## 子标题\n正文", "标题").text.startswith("# 标题\n\n## 子标题")


def test_tiny_threshold_and_chars():
    small = clean_markdown("字" * 79, "小文档")
    assert small.chars == 79
    assert body_char_count(small.text) == 79
    exact = clean_markdown("字" * 80, "刚好")
    assert exact.chars == 80
    assert clean_markdown("一 二\n三", "空格").chars == 3


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("销售排行 (2)", "销售排行"),
        ("销售排行（2）", "销售排行"),
        ("销售排行  （2）", "销售排行"),
        ("销售排行 (2) (3)", "销售排行"),
        ("Ｓａｌｅ榜", "sale榜"),
        ("销售 排行", "销售排行"),
        ("2.4.0.9【2019-05-15】", "2.4.0.9【2019-05-15】"),
    ],
)
def test_title_key_forms(raw: str, expected: str):
    assert title_key(raw) == expected


def test_exact_dup_prefers_higher_priority_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    for top in ["平台对接_店铺授权", "其他目录", "慧策价值传递素材仓库", "企业版"]:
        write_doc(raw, f"{top}/同名.md", f"# 同名\n\n{BODY}")
    out = tmp_path / "out"
    build(raw, out)
    entries = {e["doc_id"]: e for e in load_manifest(out)}

    keeper = "企业版/同名"
    assert entries[keeper]["kept"] is True
    assert entries[keeper]["drop_reason"] is None
    for doc_id in ["平台对接_店铺授权/同名", "其他目录/同名", "慧策价值传递素材仓库/同名"]:
        assert entries[doc_id]["kept"] is False
        assert entries[doc_id]["drop_reason"] == "exact_dup"
        assert entries[doc_id]["dup_of"] == keeper
    assert len(list((out / "docs").rglob("*.md"))) == 1


def test_exact_dup_low_priority_and_lexicographic(tmp_path: Path):
    raw = tmp_path / "raw"
    write_doc(raw, "慧策价值传递素材仓库/甲.md", f"# 甲\n\n{BODY}")
    write_doc(raw, "原掌上百科（Wiki）内部使用/乙.md", f"# 乙\n\n{BODY}")
    write_doc(raw, "企业版/b.md", f"# b\n\n{BODY}")
    write_doc(raw, "企业版/a.md", f"# a\n\n{BODY}")
    out = tmp_path / "out"
    build(raw, out)
    entries = {e["doc_id"]: e for e in load_manifest(out)}
    assert entries["企业版/a"]["kept"] is True
    assert entries["企业版/b"]["dup_of"] == "企业版/a"
    assert entries["原掌上百科（Wiki）内部使用/乙"]["dup_of"] == "企业版/a"
    assert entries["慧策价值传递素材仓库/甲"]["dup_of"] == "企业版/a"


def test_exact_dup_helper_ignores_unique(tmp_path: Path):
    docs = {
        "a/一": clean_markdown("# 一\n\n" + BODY, "一"),
        "b/二": clean_markdown("# 二\n\n" + BODY, "二"),
        "b/三": clean_markdown("# 三\n\n另一份内容" * 30, "三"),
    }
    result = exact_dup_of(docs, {"a/一": "a", "b/二": "b", "b/三": "b"})
    assert result == {"b/二": "a/一"}


def test_flags_release_short_dup_title(tmp_path: Path):
    raw = tmp_path / "raw"
    write_doc(raw, "企业版/2.4.0.9【2019-05-15】.md", "短" * 10)
    write_doc(raw, "企业版/更新日志.md", "日志" * 100)
    write_doc(raw, "企业版/短文档.md", "短" * 150)
    write_doc(raw, "极速版2.0/短文档 (2).md", "另" * 150)
    write_doc(raw, "极速版2.0/正常.md", "正常内容" * 60)
    out = tmp_path / "out"
    build(raw, out)
    entries = {e["doc_id"]: e for e in load_manifest(out)}

    assert "release_note" in entries["企业版/2.4.0.9【2019-05-15】"]["flags"]
    assert entries["企业版/2.4.0.9【2019-05-15】"]["drop_reason"] == "tiny"
    assert "release_note" in entries["企业版/更新日志"]["flags"]
    assert "short" in entries["企业版/短文档"]["flags"]
    assert "short" in entries["极速版2.0/短文档 (2)"]["flags"]
    assert "short" not in entries["极速版2.0/正常"]["flags"]

    dup_titles = {"企业版/短文档", "极速版2.0/短文档 (2)"}
    assert entries["企业版/短文档"]["title_key"] == entries["极速版2.0/短文档 (2)"]["title_key"]
    for doc_id in dup_titles:
        assert "dup_title" in entries[doc_id]["flags"]
    assert "dup_title" not in entries["极速版2.0/正常"]["flags"]


def test_build_end_to_end_and_idempotent(tmp_path: Path):
    raw = tmp_path / "raw"
    write_doc(raw, "企业版/正常.md", f"# 正常\n\n> 来源：https://x/y\n\n{BODY}")
    write_doc(raw, "企业版/空壳.md", "太短")
    write_doc(raw, "根文件.md", f"# 根文件\n\n{BODY}其他内容")
    out = tmp_path / "out"

    result = build(raw, out)
    text_first = (out / "manifest.jsonl").read_bytes()
    result_again = build(raw, out)
    assert text_first == (out / "manifest.jsonl").read_bytes()
    assert result_again.manifest == result.manifest

    entries = load_manifest(out)
    assert [e["doc_id"] for e in entries] == sorted(e["doc_id"] for e in entries)
    assert len(entries) == 3
    assert not result.errors
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 3
    assert stats["kept_total"] == 2
    assert stats["by_top_dir"][""]["total"] == 1
    assert stats["by_top_dir"]["企业版"]["tiny"] == 1
    assert stats["html_tags_residual_top20"] == []
    assert all(stat["kept"] + 0 == stat["total"] - stat["tiny"] - stat["exact_dup"] for stat in stats["by_top_dir"].values())
    assert not (out / "docs" / "企业版" / "空壳.md").exists()
    assert (out / "docs" / "企业版" / "正常.md").read_text(encoding="utf-8").startswith("# 正常\n\n")


def test_build_refuses_unsafe_out(tmp_path: Path):
    raw = tmp_path / "raw"
    write_doc(raw, "d/a.md", BODY)
    with pytest.raises(ValueError):
        build(raw, raw)
    with pytest.raises(ValueError):
        build(raw, raw.parent)
    with pytest.raises(ValueError):
        build(raw, raw / "sub")


def test_write_failure_does_not_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = tmp_path / "raw"
    write_doc(raw, "d/a.md", f"# a\n\n{BODY}")
    write_doc(raw, "d/b.md", f"# b\n\n{BODY}尾巴")
    out = tmp_path / "out"
    import scripts.build_kb as build_kb

    real_write = build_kb._write_text

    def flaky(path: Path, text: str) -> None:
        if path.name == "a.md":
            raise OSError("磁盘满了")
        real_write(path, text)

    monkeypatch.setattr(build_kb, "_write_text", flaky)
    result = build(raw, out)
    assert len(result.errors) == 1
    assert result.errors[0]["doc_id"] == "d/a"
    assert (out / "docs" / "d" / "b.md").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows 长路径")
def test_long_path_write(tmp_path: Path):
    raw = tmp_path / "raw"
    long_dir = "/".join(["长路径目录" + "a" * 60] * 3)
    rel = f"{long_dir}/文档.md"
    _long_path(raw / rel).parent.mkdir(parents=True, exist_ok=True)
    with open(_long_path(raw / rel), "w", encoding="utf-8") as handle:
        handle.write(f"# 文档\n\n{BODY}")
    out = tmp_path / "out"
    build(raw, out)
    written = _long_path(out / "docs" / rel)
    assert written.exists()
