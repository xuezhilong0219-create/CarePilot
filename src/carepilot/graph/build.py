from collections.abc import Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..retrieval.base import Retriever
from ..tools import TicketSystem
from ..toolbox import Toolbox
from .nodes import Models, make_nodes
from .routing import after_classify, after_draft, after_tools
from .state import TicketState


def build_graph(
    settings: Settings,
    retriever: Retriever,
    system: TicketSystem,
    checkpointer: BaseCheckpointSaver,
    models: Models | None = None,
    *,
    toolbox: Toolbox | None = None,
    edition_index: Mapping[str, frozenset[str]] | None = None,
):
    """toolbox 为空时和接入工具之前完全一致（旧的 mock 系统、没有规划节点）；有工具箱时多出：
    需查数据的工单先规划并查询只读工具、草稿写内部备注、写操作全部经网关执行。"""
    tools_on = toolbox is not None
    n = make_nodes(settings, retriever, system, models, toolbox=toolbox, edition_index=edition_index)
    g = StateGraph(TicketState)
    for name, fn in n.items():
        if name == "plan_tools" and not tools_on:
            continue
        if name == "save_draft" and not tools_on:
            continue
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        lambda s: after_classify(s, tools_enabled=tools_on),
        {"ask_for_info": "ask_for_info", "route_escalation": "route_escalation", "retrieve": "retrieve",
         **({"plan_tools": "plan_tools"} if tools_on else {})},
    )
    g.add_edge("ask_for_info", END)  # 追问后流程结束；客户补充信息后作为新一轮请求进入
    g.add_edge("retrieve", "draft_reply")
    review_entry = "save_draft" if tools_on else "human_review"  # 进入人工审核前，先把草稿存进内部备注
    g.add_conditional_edges(
        "draft_reply",
        after_draft,
        {"human_review": review_entry, "ask_for_info": "ask_for_info", "route_escalation": "route_escalation"},
    )
    if tools_on:
        g.add_conditional_edges(
            "plan_tools",
            after_tools,
            {"draft_reply": "draft_reply", "human_review": review_entry, "route_escalation": "route_escalation", "end": END},
        )
        g.add_edge("save_draft", "human_review")
    g.add_edge("route_escalation", "human_review")
    g.add_edge("human_review", "execute_actions")
    g.add_edge("execute_actions", END)
    return g.compile(checkpointer=checkpointer)
