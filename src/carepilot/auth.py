"""角色与身份（冻结接口，第二阶段所有模块共用）。

注意：角色靠请求头模拟（X-Role / X-User-Id），**不是真正的认证**。这里只定义"调用者是谁、能碰哪些店铺"，
授权判定在网关和工单存储里做，不在 API 层零散地写 if。
"""

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Role
    # 客户可访问的店铺；坐席和管理员不靠它判定，留空即可
    shop_ids: frozenset[str] = frozenset()

    @property
    def is_staff(self) -> bool:
        return self.role in (Role.AGENT, Role.ADMIN)
