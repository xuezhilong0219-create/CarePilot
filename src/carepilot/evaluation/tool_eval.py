"""工具选择评测：规划模型从客户消息里选对工具、抽对参数，且越权访问不泄露。

只评"一次规划 + 经网关执行"，不走完整的图（便宜：每条一次快模型调用）。数据集由 eval/gen_tool_dataset.py 用模板生成，
问法比真实客户规整，所以准确率会偏乐观——这一点要和数字一起报。

指标：
  tool_accuracy     选对工具（none 也算一个"工具"：不该查时不查）
  args_accuracy     需要参数的样本里，订单号逐字抽对
  outcome_accuracy  经网关执行后的结果类别符合预期（found→ok / no_tool→none / denied_owner→not_found）
  denied_correct    越权样本（客户问别家店铺的订单）：结果必须是 not_found 且没有任何事实泄露
  leaks             所有样本里，输出里出现了不属于该客户的订单信息的次数（必须是 0）
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..graph.nodes import PLANNER_PROMPT
from ..graph.state import ToolPlan
from ..llm import StructuredLLM, get_llm
from ..sandbox import tools as sbx
from ..sandbox.data import build_world
from ..sandbox.logistics import LogisticsClient
from ..store import TicketStore, seed_demo_users
from ..toolbox import Toolbox

OUTCOME_FOR = {"found": "ok", "no_tool": "none", "denied_owner": "not_found"}


def load_tool_dataset(path: Path) -> list[dict[str, Any]]:
    """文件头有 # 注释行（写着数据集的局限），读取时跳过。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def _ratio(num: int, den: int) -> dict[str, Any]:
    return {"value": (num / den) if den else 0.0, "n": den}


def run_tool_selection(
    dataset_path: Path,
    settings: Settings | None = None,
    *,
    planner: Any = None,
    toolbox: Toolbox | None = None,
    store: TicketStore | None = None,
    concurrency: int = 6,
) -> dict[str, Any]:
    items = load_tool_dataset(Path(dataset_path))
    tmp = None
    if toolbox is None:
        tmp = tempfile.TemporaryDirectory(prefix="carepilot-tooleval-")
        root = Path(tmp.name)
        world = build_world(7)
        store = TicketStore(root / "t.sqlite")
        seed_demo_users(store)
        logistics = LogisticsClient(lambda oid: sbx.query_shipment(world, oid), timeout_s=2.0, retries=1)
        toolbox = Toolbox(world=world, store=store, logistics=logistics, db_path=root / "g.sqlite")
    store = store or toolbox.store
    if planner is None:
        if settings is None:
            raise ValueError("没有传入 planner 时必须提供 settings")
        planner = StructuredLLM(get_llm(settings, "fast"), ToolPlan, purpose="planner", model=settings.llm_fast_model)
    world = toolbox.world
    shop_of_order = {oid: o.shop_id for oid, o in world.orders.items()}

    def one(item: dict[str, Any]) -> dict[str, Any]:
        principal = store.get_principal(item["principal"]["user_id"])
        try:
            plan: ToolPlan = planner.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(item["question"])])
        except Exception as exc:  # 模型超时/被网关拦截：单独计，不当成选错
            return {"qid": item["qid"], "error": type(exc).__name__}
        run = toolbox.run_plan(principal, f"tool-eval-{item['qid']}", None, plan, source_text=item["question"])
        expected_args = item.get("expected_args") or {}
        got_args = {k: getattr(plan, k) for k in ("order_id", "shop_id", "ticket_id") if getattr(plan, k)}
        if plan.shop_name and not plan.shop_id:  # 给的是店铺名：和线上一样，在调用者的范围内解析成编号再比较
            resolved = toolbox.resolve_shop(principal, plan.shop_name)
            if resolved:
                got_args["shop_id"] = resolved
        leaked = False
        order_id = expected_args.get("order_id") or got_args.get("order_id")
        if order_id and shop_of_order.get(order_id) and shop_of_order[order_id] not in principal.shop_ids:
            # 订单不属于该客户：输出里不能有它的任何信息
            leaked = run.outcome == "ok" or bool(run.facts_text) or order_id in json.dumps(run.data, ensure_ascii=False) and run.data.get("found")
        return {"qid": item["qid"], "error": None, "expected_tool": item["expected_tool"], "tool": plan.tool,
                "expected_args": expected_args, "args": got_args, "outcome": run.outcome,
                "expected_outcome": OUTCOME_FOR[item["expected_outcome"]], "kind": item["expected_outcome"], "leaked": leaked}

    try:
        with ThreadPoolExecutor(max(1, concurrency)) as ex:
            rows = list(ex.map(one, items))
    finally:
        if tmp is not None:
            tmp.cleanup()
    ok = [r for r in rows if not r["error"]]
    needs_args = [r for r in ok if r["expected_args"]]
    denied = [r for r in ok if r["kind"] == "denied_owner"]
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "tool_accuracy": _ratio(sum(r["tool"] == r["expected_tool"] for r in ok), len(ok)),
        "args_accuracy": _ratio(sum(all(r["args"].get(k) == v for k, v in r["expected_args"].items()) for r in needs_args), len(needs_args)),
        "outcome_accuracy": _ratio(sum(r["outcome"] == r["expected_outcome"] for r in ok), len(ok)),
        "denied_correct": _ratio(sum(r["outcome"] == "not_found" and not r["leaked"] for r in denied), len(denied)),
        "leaks": sum(r["leaked"] for r in ok),
        "items": rows,
    }


def mismatches(result: dict[str, Any]) -> Sequence[dict[str, Any]]:
    """没符合预期的样本，供人工读。"""
    return [r for r in result["items"] if not r["error"] and (r["tool"] != r["expected_tool"] or r["outcome"] != r["expected_outcome"])]
