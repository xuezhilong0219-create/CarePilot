from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["hardware_fault", "software_issue", "warranty_repair", "logistics", "billing", "other"]
Severity = Literal["low", "medium", "high", "critical"]


class Triage(BaseModel):
    """分类节点的结构化输出。抽不到的字段填 null，不要编造。"""

    category: Category = Field(description="工单类型")
    severity: Severity = Field(description="严重度：critical=业务中断/安全问题，high=主要功能不可用")
    product_model: str | None = Field(None, description="产品型号")
    serial_number: str | None = Field(None, description="设备序列号")
    order_id: str | None = Field(None, description="订单号")
    fault_description: str | None = Field(None, description="故障现象的简述")


class DraftResult(BaseModel):
    """回复节点的结构化输出。"""

    answerable: bool = Field(description="仅凭给定资料能否可靠地回答；资料不相关或不足就填 false")
    reply: str = Field(description="给客户的回复；answerable=false 时留空")
    cited_ids: list[str] = Field(default_factory=list, description="回复所依据的资料编号，如 ['faq#2']")


class TicketState(TypedDict, total=False):
    ticket_id: str
    customer_message: str
    # 分类
    triage: dict
    missing_fields: list[str]
    # 检索 / 生成
    hits: list[dict]
    warranty: dict
    draft: str
    citations: list[str]
    resolution: Literal["auto_reply", "escalate", "need_info"]
    escalate_reason: str
    route_to: list[str]
    # 人工审核与执行
    review: dict
    actions: list[dict]
    status: Literal["need_info", "resolved", "escalated"]
    followup_question: str
