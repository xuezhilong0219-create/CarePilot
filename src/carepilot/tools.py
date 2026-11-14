"""Mock 的工单 / 订单系统。真实环境里这里会换成公司内部系统的 HTTP 客户端，
图里的节点只依赖这几个函数的签名，所以替换时不用动工作流。"""

import itertools
from dataclasses import dataclass, field

_ids = itertools.count(1)


@dataclass
class TicketSystem:
    replies: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)

    def lookup_warranty(self, serial_number: str) -> dict:
        # 约定：序列号以偶数结尾视为在保，仅用于演示
        in_warranty = serial_number.strip()[-1:].isdigit() and int(serial_number.strip()[-1]) % 2 == 0
        return {"serial_number": serial_number, "in_warranty": in_warranty}

    def send_reply(self, ticket_id: str, text: str) -> dict:
        rec = {"ticket_id": ticket_id, "text": text}
        self.replies.append(rec)
        return {"ok": True, **rec}

    def create_escalation(self, ticket_id: str, team: str, note: str) -> dict:
        rec = {"escalation_id": f"ESC-{next(_ids):04d}", "ticket_id": ticket_id, "team": team, "note": note}
        self.escalations.append(rec)
        return {"ok": True, **rec}
