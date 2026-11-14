from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from ..config import Settings
from ..llm import StructuredLLM, get_llm
from ..retrieval.base import Retriever
from ..tools import TicketSystem
from .state import DraftResult, TicketState, Triage

# 每类工单必须具备的信息；缺了就追问，而不是让模型硬猜。放在代码里是为了行为可预期、可测试。
REQUIRED_FIELDS: dict[str, list[str]] = {
    "hardware_fault": ["product_model", "fault_description"],
    "software_issue": ["product_model", "fault_description"],
    "warranty_repair": ["product_model", "serial_number"],
    "logistics": ["order_id"],
    "billing": ["order_id"],
    "other": [],
}
FIELD_LABELS = {
    "product_model": "产品型号",
    "fault_description": "故障现象",
    "serial_number": "设备序列号",
    "order_id": "订单号",
}
TEAM_BY_CATEGORY = {
    "hardware_fault": "hardware_team",
    "software_issue": "software_team",
    "warranty_repair": "rma_team",
    "logistics": "logistics_team",
    "billing": "finance_team",
    "other": "tier2_support",
}

TRIAGE_PROMPT = "你是售后工单分诊员。阅读客户消息，抽取结构化信息。客户没提到的字段必须填 null，禁止猜测。"
DRAFT_PROMPT = """你是售后客服。只能依据【资料】回答客户，禁止使用资料以外的知识。
- 资料没有覆盖客户的问题时，answerable 填 false，reply 留空。
- 回复要具体、可操作、语气礼貌；用到哪条资料就把它的编号写进 cited_ids。"""


def make_nodes(settings: Settings, retriever: Retriever, system: TicketSystem):
    fast = get_llm(settings, "fast")
    main = get_llm(settings, "main")
    triage_llm = StructuredLLM(fast, Triage)
    draft_llm = StructuredLLM(main, DraftResult)

    def classify(state: TicketState) -> dict:
        triage: Triage = triage_llm.invoke(
            [SystemMessage(TRIAGE_PROMPT), HumanMessage(state["customer_message"])]
        )
        data = triage.model_dump()
        missing = [f for f in REQUIRED_FIELDS[triage.category] if not data.get(f)]
        return {"triage": data, "missing_fields": missing}

    def ask_for_info(state: TicketState) -> dict:
        labels = "、".join(FIELD_LABELS[f] for f in state["missing_fields"])
        question = f"为了尽快帮您处理，请补充以下信息：{labels}。"
        return {"followup_question": question, "resolution": "need_info", "status": "need_info"}

    def retrieve(state: TicketState) -> dict:
        t = state["triage"]
        query = " ".join(filter(None, [t.get("product_model"), t.get("fault_description"), state["customer_message"]]))
        hits = retriever.search(query, k=settings.retrieval_top_k)
        out: dict = {
            "hits": [{"id": h.chunk.id, "title": h.chunk.title, "text": h.chunk.text, "score": round(h.score, 3)} for h in hits]
        }
        if t.get("serial_number"):
            out["warranty"] = system.lookup_warranty(t["serial_number"])
        return out

    def draft_reply(state: TicketState) -> dict:
        if not state["hits"]:
            return {"resolution": "escalate", "escalate_reason": "知识库无相关资料"}
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in state["hits"])
        extra = f"\n\n【保修查询结果】{state['warranty']}" if state.get("warranty") else ""
        result: DraftResult = draft_llm.invoke(
            [
                SystemMessage(DRAFT_PROMPT),
                HumanMessage(f"【资料】\n{context}{extra}\n\n【客户消息】\n{state['customer_message']}"),
            ]
        )
        # 引用必须真实存在于检索结果里，模型编造的编号一律丢弃
        valid_ids = {h["id"] for h in state["hits"]}
        cited = [c for c in result.cited_ids if c in valid_ids]
        if not result.answerable or not result.reply.strip():
            return {"resolution": "escalate", "escalate_reason": "资料不足以可靠回答"}
        if not cited:
            return {"resolution": "escalate", "escalate_reason": "回复没有可校验的引用，拒绝放行"}
        return {"draft": result.reply, "citations": cited, "resolution": "auto_reply"}

    def route_escalation(state: TicketState) -> dict:
        t = state["triage"]
        teams = [TEAM_BY_CATEGORY[t["category"]]]
        if t["severity"] == "critical":
            teams.append("oncall")
        return {"route_to": teams}

    def human_review(state: TicketState) -> dict:
        # interrupt 之前不能有副作用：恢复执行时本节点会从头重跑
        decision = interrupt(
            {
                "resolution": state["resolution"],
                "triage": state["triage"],
                "draft": state.get("draft"),
                "citations": state.get("citations"),
                "escalate_reason": state.get("escalate_reason"),
                "route_to": state.get("route_to"),
            }
        )
        return {"review": decision}

    def execute_actions(state: TicketState) -> dict:
        review = state["review"]
        action = review.get("action", "reject")
        if state["resolution"] == "auto_reply" and action in ("approve", "edit"):
            text = review.get("reply") if action == "edit" and review.get("reply") else state["draft"]
            result = system.send_reply(state["ticket_id"], text)
            return {"actions": [{"tool": "send_reply", **result}], "status": "resolved"}
        # 无法自动解决，或审核人驳回了草稿 → 升级
        teams = state.get("route_to") or route_escalation(state)["route_to"]
        note = review.get("note") or state.get("escalate_reason") or "审核人驳回自动回复"
        results = [system.create_escalation(state["ticket_id"], team, note) for team in teams]
        return {"actions": [{"tool": "create_escalation", **r} for r in results], "status": "escalated"}

    return {
        "classify": classify,
        "ask_for_info": ask_for_info,
        "retrieve": retrieve,
        "draft_reply": draft_reply,
        "route_escalation": route_escalation,
        "human_review": human_review,
        "execute_actions": execute_actions,
    }
