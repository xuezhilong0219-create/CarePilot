"""图里的条件路由，单独成文件：build.py 建图和 evaluation/runners.py 评估都调用同一份，
避免评估器照抄一份逻辑、改图后悄悄分叉（评估的就不是线上实际的流程了）。"""

from typing import Literal

from .state import TicketState


def after_classify(state: TicketState) -> Literal["ask_for_info", "route_escalation", "retrieve"]:
    """优先级：笼统 → 追问；需要查客户具体数据 → 直接升级；其余 → 去检索。
    - 不用"缺必填槽位"做门槛：槽位抽取不可靠，旧版因此把 20% 本可回答的工单拦下来追问。
    - data_exception 不检索也不生成：知识库回答不了「帮我查下这笔订单」，让模型硬答只会编。
      实测修掉引用 bug 后，不可答样本被自动放行 5/21，全是这一类，且分诊都判对了类别。
      直接升级还省掉最贵的一次回复调用。"""
    triage = state["triage"]
    if triage.get("is_vague"):
        return "ask_for_info"
    if triage["category"] == "data_exception":
        return "route_escalation"
    return "retrieve"


def after_draft(state: TicketState) -> Literal["human_review", "ask_for_info", "route_escalation"]:
    """先尝试回答。回答失败且确实缺关键信息 → 追问；不缺 → 升级。"""
    if state["resolution"] == "auto_reply":
        return "human_review"
    return "ask_for_info" if state["missing_fields"] else "route_escalation"
