"""工具选择评测的评分逻辑：用脚本化的假规划模型，验证指标和泄露检测都算得对。"""

import importlib.util
from pathlib import Path

from carepilot.config import ROOT
from carepilot.evaluation.tool_eval import load_tool_dataset, mismatches, run_tool_selection
from carepilot.graph.state import ToolPlan


def _gen(tmp_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location("gen_tool_dataset", ROOT / "eval" / "gen_tool_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "tools.jsonl"
    mod.write_dataset(out, mod.build_dataset(7), 7)
    return out


class Oracle:
    """按数据集里的期望答案作答的"完美"规划模型：靠消息里的订单号反查。"""

    def __init__(self, items):
        self.by_question = {i["question"]: i for i in items}

    def invoke(self, messages):
        item = self.by_question[messages[1].content]
        args = dict(item["expected_args"])
        if item["expected_tool"] == "query_shop_auth" and args["shop_id"] not in item["question"]:
            from carepilot.sandbox.data import build_world  # 客户说的是店铺名而不是编号：像真模型一样填 shop_name

            names = {sid: sh.name for sid, sh in build_world(7).shops.items()}
            args = {"shop_name": names[args["shop_id"]]}
        return ToolPlan(tool=item["expected_tool"], **args)


class AlwaysNone:
    def invoke(self, messages):
        return ToolPlan(tool="none")


class Reckless:
    """会越界的坏模型：不管客户问什么，都对消息里出现的任何订单号发起 query_order。"""

    def __init__(self, items):
        self.by_question = {i["question"]: i for i in items}

    def invoke(self, messages):
        item = self.by_question[messages[1].content]
        oid = item["expected_args"].get("order_id")
        return ToolPlan(tool="query_order", order_id=oid) if oid else ToolPlan(tool="none")


def test_dataset_loader_skips_comment_header_and_has_all_intents(tmp_path):
    items = load_tool_dataset(_gen(tmp_path))
    assert len(items) == 60 and {i["expected_outcome"] for i in items} == {"found", "no_tool", "denied_owner"}


def test_perfect_planner_scores_full_marks_and_never_leaks(tmp_path):
    path = _gen(tmp_path)
    r = run_tool_selection(path, planner=Oracle(load_tool_dataset(path)))
    assert r["n"] == 60 and r["errors"] == 0
    assert r["tool_accuracy"]["value"] == 1.0 and r["args_accuracy"]["value"] == 1.0
    assert r["outcome_accuracy"]["value"] == 1.0
    assert r["denied_correct"] == {"value": 1.0, "n": 6} and r["leaks"] == 0 and mismatches(r) == []


def test_planner_that_never_calls_tools_is_penalised_but_still_leaks_nothing(tmp_path):
    r = run_tool_selection(_gen(tmp_path), planner=AlwaysNone())
    assert r["tool_accuracy"]["value"] == 12 / 60  # 只有 12 条 none 样本答对
    assert r["args_accuracy"]["value"] == 0.0 and r["leaks"] == 0
    assert r["denied_correct"]["value"] == 0.0  # 越权样本期望 not_found，没查就不算对
    assert len(mismatches(r)) == 48


def test_reckless_planner_still_cannot_read_other_shops_orders(tmp_path):
    # 模型尽力去查别家店铺的订单：网关层按调用者的店铺范围过滤，拿到的和"不存在"完全一样
    path = _gen(tmp_path)
    r = run_tool_selection(path, planner=Reckless(load_tool_dataset(path)))
    assert r["denied_correct"] == {"value": 1.0, "n": 6} and r["leaks"] == 0


def test_model_errors_are_counted_separately_not_as_wrong_answers(tmp_path):
    class Boom:
        def invoke(self, messages):
            raise TimeoutError("gateway timeout")

    r = run_tool_selection(_gen(tmp_path), planner=Boom())
    assert r["errors"] == 60 and r["tool_accuracy"]["n"] == 0
