"""图里的条件路由，单独成文件：build.py 建图和 evaluation/runners.py 评估都调用同一份，
避免评估器照抄一份逻辑、改图后悄悄分叉（评估的就不是线上实际的流程了）。"""

from typing import Literal

from .state import TicketState


def after_classify(state: TicketState) -> Literal["ask_for_info", "retrieve"]:
    """消息笼统（分诊模型的语义判断 is_vague）→ 检索前直接追问；否则去检索。
    不用"缺必填槽位"做门槛：槽位抽取不可靠，旧版因此把 20% 本可回答的工单拦下来追问。"""
    return "ask_for_info" if state["triage"].get("is_vague") else "retrieve"


def after_draft(state: TicketState) -> Literal["human_review", "ask_for_info", "route_escalation"]:
    """先尝试回答。回答失败且确实缺关键信息 → 追问；不缺 → 升级。"""
    if state["resolution"] == "auto_reply":
        return "human_review"
    return "ask_for_info" if state["missing_fields"] else "route_escalation"
