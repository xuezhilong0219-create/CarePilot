"""长期记忆的 Store 接口与读写约定（冻结）。**只留接口，本项目不实现真正的记忆。**

读写约定（写进接口是为了让以后接入时有规矩可循）：
1. 写入时机：只在工单 resolved、且人工审核通过之后，由确定性代码写入；模型不能自己决定写什么。
2. 写什么：已确认的用户偏好（如「习惯用企业版」）、已验证有效的处理方案摘要。**不写**原始消息、手机号等敏感信息、未经审核的模型回复。
3. 读取：只作为"不可信的背景上下文"放进提示词，和检索到的知识库内容同等对待，不能改变工具调用策略。
4. 命名空间：每条记录归属一个 user_id，读取时必须带 user_id，不能跨用户读。
5. 过期：每条记录必须有 ttl_seconds，或显式声明 permanent。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryRecord:
    user_id: str
    key: str
    value: Any
    written_at: float
    ttl_seconds: float | None  # None 表示 permanent，必须由写入方显式声明
    source_run_id: str  # 记录来自哪次 run，便于审计和撤销


class MemoryStore(Protocol):
    def get(self, user_id: str, key: str) -> MemoryRecord | None: ...
    def put(self, user_id: str, key: str, value: Any, *, source_run_id: str, ttl_seconds: float | None) -> MemoryRecord: ...
    def search(self, user_id: str, query: str, *, k: int = 5) -> list[MemoryRecord]: ...
    def delete(self, user_id: str, key: str) -> bool: ...


class NullMemoryStore:
    """什么都不存的实现：接口已就位，但不产生任何记忆行为。"""

    def get(self, user_id: str, key: str) -> MemoryRecord | None:
        return None

    def put(self, user_id: str, key: str, value: Any, *, source_run_id: str, ttl_seconds: float | None) -> MemoryRecord:
        return MemoryRecord(user_id, key, value, 0.0, ttl_seconds, source_run_id)

    def search(self, user_id: str, query: str, *, k: int = 5) -> list[MemoryRecord]:
        return []

    def delete(self, user_id: str, key: str) -> bool:
        return False
