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
            "data_exception=客户要求查看或核对自己的某一笔具体订单/库存数/账单（会给出订单尾号、金额、日期、SKU 等具体记录），必须查客户数据才能回答；仅仅贴了弹窗报错、问某个提示的原因或处理办法、问计算口径，不属于此类；"
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
    standalone_question: str | None = Field(
        None,
        description=(
            "把客户本句话改写成脱离上下文也能看懂的完整问题：对话历史里已经出现过的对象，要把「它」「这个」「刚才那个」换成具体名称。"
            "没有对话历史，或本句本身已经完整时，原样复述客户消息"
        ),
    )
    is_vague: bool = Field(
        False,
        description=(
            "消息是否笼统到无法判断要查什么：没有说明是哪个功能/页面、什么现象、什么报错，"
            "例如「系统用不了」「订单有问题」「帮我看一下」。只要提到了具体的功能、操作、现象或报错，就填 false"
        ),
    )


class DraftResult(BaseModel):
    """回复节点的结构化输出。"""

    answerable: bool = Field(description="仅凭给定资料能否可靠地回答；资料不相关或不足就填 false")
    reply: str = Field(description="给客户的回复；answerable=false 时留空")
    cited_ids: list[str] = Field(default_factory=list, description="回复所依据资料的编号：资料前方括号里的数字，如 [\"1\", \"3\"]")


class ToolPlan(BaseModel):
    """工具规划节点的结构化输出：从客户消息里决定调用哪个只读查询工具。只填客户消息里逐字出现的标识符。"""

    tool: Literal["query_order", "query_shipment", "query_shop_auth", "query_ticket", "none"] = Field(
        description=(
            "query_order=按订单号查订单状态/金额；query_shipment=按订单号查物流到哪了；"
            "query_shop_auth=查某个店铺的授权是否过期（店铺编号形如 S1）；query_ticket=按工单号（形如 T-000123）查工单进度；"
            "none=这些工具都帮不上（比如问账单为什么少了钱、库存数字对不上）"
        )
    )
    order_id: str | None = Field(None, description="客户消息里逐字出现的完整订单号；没有就填 null，不要猜、不要补全尾号")
    shop_id: str | None = Field(None, description="客户消息里逐字出现的店铺编号（如 S1）；没有就填 null")
    ticket_id: str | None = Field(None, description="客户消息里逐字出现的工单号（如 T-000123）；没有就填 null")


class TicketState(TypedDict, total=False):
    ticket_id: str
    customer_message: str
    # 运行上下文（接入工具箱时由 API 提供）
    run_id: str
    principal: dict  # {"user_id", "role", "shop_ids"}；发起这次运行的调用者，只读工具按它的店铺范围查询
    history: list[dict]  # 本工单之前的公开对话 [{"role": "customer"|"agent", "text": ...}]，最近几条
    slots: dict  # 之前几轮累积的槽位（平台/店铺/功能/仓库/版本），本轮分诊后会合并
    # 分类
    triage: dict
    missing_fields: list[str]
    # 检索 / 生成
    query: str
    hits: list[dict]
    shop_status: dict
    tool_run: dict
    draft: str
    citations: list[str]
    citation_details: list[dict]  # 结构化引用：文档名/章节/产品线/片段，供审核人点开核对
    pending_confirmation: dict  # 对外发送已批准、等待二次确认：{"approval_id", "tool"}
    resolution: Literal["auto_reply", "escalate", "need_info"]
    escalate_reason: str
    route_to: list[str]
    # 人工审核与执行
    review: dict
    actions: list[dict]
    status: Literal["need_info", "resolved", "escalated", "awaiting_confirmation", "action_failed", "needs_human"]
    followup_question: str
