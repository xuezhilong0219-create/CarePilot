"""Mock 的工单 / 店铺系统。真实环境里这里会换成公司内部系统的 HTTP 客户端，
图里的节点只依赖这几个方法的签名，所以替换时不用动工作流。"""

import itertools
from dataclasses import dataclass, field

_ids = itertools.count(1)


@dataclass
class TicketSystem:
    replies: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)

    def check_shop_auth(self, platform_name: str, shop_name: str) -> dict:
        # 演示用：店铺名长度为奇数视为授权已过期。真实系统里这里查平台授权表。
        expired = len(shop_name.strip()) % 2 == 1
        return {"platform_name": platform_name, "shop_name": shop_name, "authorized": not expired}

    def send_reply(self, ticket_id: str, text: str) -> dict:
        rec = {"ticket_id": ticket_id, "text": text}
        self.replies.append(rec)
        return {"ok": True, **rec}

    def create_escalation(self, ticket_id: str, team: str, note: str) -> dict:
        rec = {"escalation_id": f"ESC-{next(_ids):04d}", "ticket_id": ticket_id, "team": team, "note": note}
        self.escalations.append(rec)
        return {"ok": True, **rec}
