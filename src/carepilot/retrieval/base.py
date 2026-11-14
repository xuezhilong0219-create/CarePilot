from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    id: str  # 形如 "doc-name#3"，回复里的引用就指向它
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """检索器接口。BM25 / 向量 / 混合 / ima 适配器都实现这一个方法，
    这样评测脚本可以在同一份数据集上横向对比不同检索策略。"""

    def search(self, query: str, k: int = 4) -> list[Hit]: ...
