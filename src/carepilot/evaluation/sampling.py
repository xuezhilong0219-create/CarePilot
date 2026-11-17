"""抽样：评估脚本共用，保证同一份数据集抽到同一批样本。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def even_sample(pool: Sequence[T], limit: int) -> list[T]:
    """等距抽 limit 条。等距而不是随机：报告要可复现，也不用管随机种子。"""
    if limit <= 0 or limit >= len(pool):
        return list(pool)
    step = len(pool) / limit
    return [pool[int(i * step)] for i in range(limit)]
