from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    id: str  # 形如 "平台对接_店铺授权/淘宝#3"，回复里的引用就指向它
    title: str
    text: str
    doc_id: str = ""  # 来源文档，= 原始文件相对路径去掉 .md；评估的"正确答案"用它对齐
    meta: Mapping[str, str] = field(default_factory=dict)  # top_dir / title_key / flags 等，来自 manifest


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """检索器接口（冻结）。BM25 / 向量 / 混合 / 重排 / ima 适配器都实现这一个方法，
    这样评测脚本可以在同一份数据集上横向对比不同检索策略。
    返回按相关度降序、已按 chunk 级别排好的结果；同一文档可以出现多个 chunk。"""

    def search(self, query: str, k: int = 4) -> list[Hit]: ...
