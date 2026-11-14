import json
from pathlib import Path

import pytest

from carepilot.retrieval.chunking import load_kb, split_markdown


def test_nested_headings_include_document_breadcrumb_and_keep_preamble():
    chunks = split_markdown(
        "guide", "前言说明。\n\n# 授权指南\n## 淘宝\n" + "处理步骤。" * 30 + "\n### 重新授权\n" + "登录主账号。" * 30
    )
    assert chunks[0].text.startswith("guide\n前言说明。")
    assert any(c.text.startswith("授权指南 > 淘宝\n") for c in chunks)
    assert chunks[-1].text.startswith("授权指南 > 淘宝 > 重新授权\n")
    assert [c.id for c in chunks] == [f"guide#{i}" for i in range(len(chunks))]


def test_tables_and_fenced_code_are_atomic_even_above_target_size():
    table = "| 商品 | 数量 |\n| --- | --- |\n" + "| 苹果 | 12 |\n" * 30
    code = "```python\n# not a heading\n" + "print('库存')\n\n" * 30 + "```"
    chunks = split_markdown("guide", f"# 使用说明\n前言\n\n{table}\n{code}\n\n结束", max_chars=100)
    assert sum(table.strip() in c.text for c in chunks) == 1
    assert sum(code in c.text for c in chunks) == 1
    assert all(c.title == "使用说明" for c in chunks)


def test_short_tail_is_merged_and_paragraphs_are_not_lost():
    paragraphs = ["甲" * 65, "乙" * 65, "短尾"]
    chunks = split_markdown("doc", "# 标题\n" + "\n\n".join(paragraphs), max_chars=90)
    assert len(chunks) == 2
    assert chunks[-1].text.endswith("短尾")
    for paragraph in paragraphs:
        assert sum(paragraph in c.text for c in chunks) == 1


def test_manifest_filters_and_metadata(tmp_path: Path):
    entries = [
        dict(doc_id="平台/淘宝", title="淘宝指南", title_key="淘宝", top_dir="平台", kept=True, flags=[], source_url="https://example.test/doc"),
        dict(doc_id="发布/更新", title="更新", title_key="更新", top_dir="发布", kept=True, flags=["release_note"], source_url=""),
        dict(doc_id="丢弃", title="丢弃", title_key="丢弃", top_dir="", kept=False, flags=[], source_url=""),
    ]
    for entry in entries[:2]:
        path = tmp_path / "docs" / f"{entry['doc_id']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 操作步骤\n登录店铺重新授权。", encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in entries), encoding="utf-8")
    chunks = load_kb(tmp_path, exclude_flags=["release_note"])
    assert len(chunks) == 1
    assert chunks[0].doc_id == "平台/淘宝"
    assert chunks[0].text.startswith("淘宝指南 > 操作步骤\n")
    assert chunks[0].meta == dict(top_dir="平台", title_key="淘宝", flags="", source_url="https://example.test/doc")
    assert load_kb(tmp_path, exclude_dirs=["平台"])[0].meta["flags"] == "release_note"


def test_fallback_uses_relative_paths_to_avoid_duplicate_ids(tmp_path: Path):
    for directory in ["a", "b"]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "same.md").write_text("# 标题\n正文", encoding="utf-8")
    chunks = load_kb(tmp_path)
    assert [c.doc_id for c in chunks] == ["a/same", "b/same"]
    assert len({c.id for c in chunks}) == 2
    assert load_kb(tmp_path, exclude_dirs=["a"])[0].doc_id == "b/same"


def test_manifest_cannot_escape_docs_directory(tmp_path: Path):
    (tmp_path / "manifest.jsonl").write_text(json.dumps(dict(doc_id="../escape", kept=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="doc_id"):
        load_kb(tmp_path)


def test_empty_markdown_and_invalid_target():
    assert split_markdown("empty", "   ") == []
    with pytest.raises(ValueError, match="max_chars"):
        split_markdown("doc", "text", max_chars=0)


def test_short_subsection_merges_with_neighbor_without_losing_breadcrumb():
    text = "# 店铺指南\n## 提醒\n先备份。\n## 处理步骤\n" + "重新授权并检查订单。" * 20
    chunks = split_markdown("doc", text)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("店铺指南 > 提醒\n先备份。")
    assert "店铺指南 > 处理步骤" in chunks[0].text


def test_headings_in_indented_code_do_not_change_breadcrumb():
    code = "    # example\n\n    print('example')"
    chunks = split_markdown("doc", "# 使用说明\n" + code + "\n\n正文", max_chars=20)
    assert sum(code in c.text for c in chunks) == 1
    assert all(c.title == "使用说明" for c in chunks)
