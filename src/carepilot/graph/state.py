from typing import Literal, TypedDict

from pydantic import BaseModel, Field

# 分类体系对应知识库的四类内容：功能怎么用 / 平台对接 / 仓库物流对接 / 平台规则变更，
# 外加两类知识库回答不了的：需要查客户数据的异常、闲聊或无关。
Category = Literal[
    "feature_howto",
    "platform_integration",
    "warehouse_logistics",
    "rule_change",
    "data_exception",
    "other",
]
Severity = Literal["low", "medium", "high", "critical"]


class Triage(BaseModel):
    """分类节点的结构化输出。客户没提到的字段填 null，禁止编造。"""

    category: Category = Field(
        description=(
            "工单类型。feature_howto=产品功能怎么用/怎么配置；"
            "platform_integration=电商平台对接、店铺授权、订单/库存/物流回传同步问题；"
            "warehouse_logistics=第三方仓库(WMS)或物流/电子面单对接；"
            "rule_change=平台规则或政策变更、版本更新带来的影响；"
            "data_exception=客户自己的订单/库存/账务数据对不上，需要查数据才能回答；"
            "other=闲聊、与产品无关"
        )
    )
    severity: Severity = Field(
        description="严重度：critical=无法发货/订单下不来/系统登录不了等业务中断；high=主要功能不可用；medium=局部功能异常；low=咨询"
    )
    platform_name: str | None = Field(None, description="电商平台名称，如 淘宝、京东、抖店")
    shop_name: str | None = Field(None, description="客户的店铺名称")
    feature_name: str | None = Field(None, description="涉及的产品功能或菜单名称")
    partner_name: str | None = Field(None, description="第三方仓库或物流商名称")
    product_edition: str | None = Field(None, description="产品版本，如 企业版、极速版")
    error_text: str | None = Field(None, description="客户贴出的报错原文，逐字保留，不要改写")
    fault_description: str | None = Field(None, description="问题现象的一句话概括")


class DraftResult(BaseModel):
    """回复节点的结构化输出。"""

    answerable: bool = Field(description="仅凭给定资料能否可靠地回答；资料不相关或不足就填 false")
    reply: str = Field(description="给客户的回复；answerable=false 时留空")
    cited_ids: list[str] = Field(default_factory=list, description="回复所依据的资料编号")


class TicketState(TypedDict, total=False):
    ticket_id: str
    customer_message: str
    # 分类
    triage: dict
    missing_fields: list[str]
    # 检索 / 生成
    query: str
    hits: list[dict]
    shop_status: dict
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
