"""评估集的数据格式（冻结）。生成脚本和评估脚本都只依赖这里，互不依赖。

一行 JSON 一个样本。答案的对齐方式有两档：
- 严格：检索到的 chunk 的 doc_id 在 gold_doc_ids 里。
- 宽松：chunk.meta["title_key"] 在 gold_title_keys 里。知识库里同名文档有很多份（企业版 / 极速版 / 内部版），
  内容几乎一样，只按 doc_id 算会把"检索到了同名的另一份"误判为没命中，所以两档指标都要报。
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..graph.state import Category, Severity


class EvalItem(BaseModel):
    qid: str
    question: str = Field(description="客户原话")
    category: Category
    answerable: bool = Field(description="知识库里是否有答案；False 的样本期望系统拒答并升级")
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_title_keys: list[str] = Field(default_factory=list)
    # 期望抽取结果，用来评估分诊的实体抽取；没有就是 None
    severity: Severity | None = None
    platform_name: str | None = None
    feature_name: str | None = None
    partner_name: str | None = None
    error_text: str | None = None
    source: Literal["generated", "manual"] = "generated"


def load_dataset(path: Path) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        return [EvalItem.model_validate_json(line) for line in f if line.strip()]


def dump_dataset(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.model_dump(), ensure_ascii=False) + "\n")
