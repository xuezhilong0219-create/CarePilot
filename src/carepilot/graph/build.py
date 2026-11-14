from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..retrieval.base import Retriever
from ..tools import TicketSystem
from .nodes import make_nodes
from .state import TicketState


def build_graph(
    settings: Settings,
    retriever: Retriever,
    system: TicketSystem,
    checkpointer: BaseCheckpointSaver,
):
    n = make_nodes(settings, retriever, system)
    g = StateGraph(TicketState)
    for name, fn in n.items():
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        lambda s: "ask_for_info" if s["missing_fields"] else "retrieve",
        {"ask_for_info": "ask_for_info", "retrieve": "retrieve"},
    )
    g.add_edge("ask_for_info", END)  # 追问后流程结束；客户补充信息后作为新一轮请求进入
    g.add_edge("retrieve", "draft_reply")
    g.add_conditional_edges(
        "draft_reply",
        lambda s: "human_review" if s["resolution"] == "auto_reply" else "route_escalation",
        {"human_review": "human_review", "route_escalation": "route_escalation"},
    )
    g.add_edge("route_escalation", "human_review")
    g.add_edge("human_review", "execute_actions")
    g.add_edge("execute_actions", END)
    return g.compile(checkpointer=checkpointer)
