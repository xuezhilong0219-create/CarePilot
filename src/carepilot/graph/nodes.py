import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from ..config import Settings
from ..llm import StructuredLLM, get_llm
from ..retrieval.base import Retriever
from ..tools import TicketSystem
from .state import DraftResult, TicketState, Triage

# 每类工单必须具备的信息；缺了就追问，而不是让模型硬猜。放在代码里是为了行为可预期、可测试。
REQUIRED_FIELDS: dict[str, list[str]] = {
    "feature_howto": ["feature_name"],
    "platform_integration": ["platform_name"],
    "warehouse_logistics": ["partner_name"],
    "rule_change": ["platform_name"],
    "data_exception": ["fault_description"],
    "other": [],
}
FIELD_LABELS = {
    "feature_name": "具体是哪个功能或菜单",
    "platform_name": "电商平台名称",
    "partner_name": "仓库或物流商名称",
    "fault_description": "问题现象",
}
TEAM_BY_CATEGORY = {
    "feature_howto": "product_support_team",
    "platform_integration": "platform_integration_team",
    "warehouse_logistics": "warehouse_logistics_team",
    "rule_change": "platform_integration_team",
    "data_exception": "data_support_team",
    "other": "tier2_support",
}

TRIAGE_PROMPT = """你是旺店通 ERP 售后技术支持的工单分诊员。阅读客户消息，抽取结构化信息。
- 客户没提到的字段必须填 null，禁止猜测；error_text 必须逐字摘录客户贴的报错原文。
- is_vague：只要客户提到了具体的功能、操作、现象或报错，就填 false；只有「系统用不了」「订单有问题」「帮我看一下」这种说不出要查什么的才填 true。
- 分类看客户真正要解决的问题：问"怎么操作/怎么设置"是 feature_howto；平台授权、订单下载不了、库存或物流回传失败是 platform_integration；
  第三方仓或物流商接口问题是 warehouse_logistics；问平台新规则/政策影响是 rule_change；
  问"我的某笔订单/某个库存数字为什么不对"这类要看客户自己数据的是 data_exception。"""

DRAFT_PROMPT = """你是旺店通 ERP 的售后客服。只能依据【资料】回答客户，禁止使用资料以外的知识。
- 资料没有覆盖客户的问题时，answerable 填 false，reply 留空。
- 资料可能来自不同产品版本（企业版/极速版等）或较早的版本；若资料适用的版本与客户不一致，要在回复里说明。
- 回复要具体、可操作、分步骤，语气礼貌。
- 资料前面有方括号编号，如 [1]、[2]。用到哪条资料，就把它的数字编号写进 cited_ids，例如 ["1", "3"]；只写数字，不要写别的。
- answerable 为 true 时 cited_ids 不能为空。"""


class StructuredInvoker(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> BaseModel: ...


@dataclass
class Models:
    """图里用到的两个结构化模型。单测里换成假实现，评估时可以单独替换某一个。"""

    triage: StructuredInvoker
    draft: StructuredInvoker


def default_models(settings: Settings) -> Models:
    return Models(
        triage=StructuredLLM(get_llm(settings, "fast"), Triage),
        draft=StructuredLLM(get_llm(settings, "main"), DraftResult),
    )


_REF_NUMBER = re.compile(r"^\D{0,4}(\d{1,2})\D{0,2}$")


def resolve_citations(cited: list[str], chunk_ids: list[str]) -> list[str]:
    """把模型给出的引用还原成真实 chunk id，去重并保持顺序。
    接受两种写法：短编号（"2"、"[2]"、"资料2"，对应上下文里的 [2]）和完整的 chunk id。
    编号只认 1..N；其余（"#0"、"0"、编号越界、查无此 id）一律丢弃，不做猜测。"""
    valid = set(chunk_ids)
    out: list[str] = []
    for raw in cited:
        text = raw.strip()
        if text in valid:
            resolved = text
        else:
            m = _REF_NUMBER.match(text)
            n = int(m.group(1)) if m else 0
            if not 1 <= n <= len(chunk_ids):
                continue
            resolved = chunk_ids[n - 1]
        if resolved not in out:
            out.append(resolved)
    return out


def build_query(triage: dict, customer_message: str) -> str:
    """拼检索词。客户习惯直接拿报错原文、平台名、功能名搜，所以把抽取出的实体放在最前面，
    再附上客户原话兜底。是纯函数，评估脚本复用它，保证评估的检索词和线上一致。"""
    parts = [triage.get(k) for k in ("platform_name", "feature_name", "partner_name", "error_text")]
    return " ".join([p for p in parts if p] + [customer_message])


def make_nodes(settings: Settings, retriever: Retriever, system: TicketSystem, models: Models | None = None):
    models = models or default_models(settings)

    def classify(state: TicketState) -> dict:
        triage: Triage = models.triage.invoke(
            [SystemMessage(TRIAGE_PROMPT), HumanMessage(state["customer_message"])]
        )
        data = triage.model_dump()
        missing = [f for f in REQUIRED_FIELDS[triage.category] if not data.get(f)]
        return {"triage": data, "missing_fields": missing}

    def ask_for_info(state: TicketState) -> dict:
        if state["missing_fields"]:
            labels = "、".join(FIELD_LABELS[f] for f in state["missing_fields"])
            question = f"为了尽快帮您处理，请补充以下信息：{labels}。"
        else:  # 消息笼统、没有具体缺哪个槽位
            question = "为了尽快帮您处理，请描述一下具体遇到的问题：在哪个功能或页面操作，出现了什么现象或报错提示。"
        return {"followup_question": question, "resolution": "need_info", "status": "need_info"}

    def retrieve(state: TicketState) -> dict:
        t = state["triage"]
        query = build_query(t, state["customer_message"])
        hits = retriever.search(query, k=settings.retrieval_top_k)
        out: dict = {
            "query": query,
            "hits": [
                {"id": h.chunk.id, "title": h.chunk.title, "text": h.chunk.text, "score": round(h.score, 3)}
                for h in hits
            ],
        }
        if t["category"] == "platform_integration" and t.get("platform_name") and t.get("shop_name"):
            out["shop_status"] = system.check_shop_auth(t["platform_name"], t["shop_name"])
        return out

    def draft_reply(state: TicketState) -> dict:
        if not state["hits"]:
            return {"resolution": "escalate", "escalate_reason": "知识库无相关资料"}
        # 给模型看 [1]~[N] 的短编号，不给 chunk 的长路径 id：真实 id 形如「企业版/采购入库单#1」，
        # 实测模型会缩写成「资料#1」「#0」「10」甚至留空，导致 203 条可答样本里 78 条（38%）被引用校验误杀
        context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(state["hits"], 1))
        extra = f"\n\n【店铺授权状态】{state['shop_status']}" if state.get("shop_status") else ""
        result: DraftResult = models.draft.invoke(
            [
                SystemMessage(DRAFT_PROMPT),
                HumanMessage(f"【资料】\n{context}{extra}\n\n【客户消息】\n{state['customer_message']}"),
            ]
        )
        # 引用必须能还原成检索结果里真实存在的 chunk，还原不了的（模型编造的）一律丢弃
        cited = resolve_citations(result.cited_ids, [h["id"] for h in state["hits"]])
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
