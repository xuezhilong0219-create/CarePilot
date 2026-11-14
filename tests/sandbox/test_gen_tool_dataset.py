import importlib.util
import json
from collections import Counter
from pathlib import Path

from carepilot.sandbox.data import build_world


def _load_module():
    path = Path(__file__).resolve().parents[2] / "eval" / "gen_tool_dataset.py"
    spec = importlib.util.spec_from_file_location("gen_tool_dataset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_size_and_intent_distribution():
    rows = _load_module().build_dataset(seed=7)
    assert len(rows) == 60
    assert len({row["qid"] for row in rows}) == 60

    by_outcome = Counter(row["expected_outcome"] for row in rows)
    assert by_outcome["no_tool"] == 12  # 20%
    assert by_outcome["denied_owner"] == 6  # 10%
    assert by_outcome["found"] == 42

    by_tool = Counter(row["expected_tool"] for row in rows)
    assert by_tool["query_order"] == 26
    assert by_tool["query_shipment"] == 12
    assert by_tool["query_shop_auth"] == 10
    assert by_tool["none"] == 12

    # 每种意图至少 5 种不同说法
    for tool in ("query_order", "query_shipment", "query_shop_auth", "none"):
        questions = {row["question"] for row in rows if row["expected_tool"] == tool}
        assert len(questions) >= 5, tool


def test_questions_are_unique_and_principals_valid():
    rows = _load_module().build_dataset(seed=7)
    assert len({row["question"] for row in rows}) == len(rows)
    for row in rows:
        assert row["principal"]["role"] == "customer"
        assert row["principal"]["user_id"] in {"c1", "c2", "c3"}
        assert set(row) == {
            "qid",
            "question",
            "principal",
            "expected_tool",
            "expected_args",
            "expected_outcome",
        }


def test_all_order_ids_exist_and_ownership_matches_outcome():
    module = _load_module()
    world = build_world(seed=7)
    rows = module.build_dataset(seed=7)

    for row in rows:
        args = row["expected_args"]
        if row["expected_tool"] == "query_order":
            order = world.orders[args["order_id"]]
            allowed = set(world.customer_shops[row["principal"]["user_id"]])
            if row["expected_outcome"] == "denied_owner":
                assert order.shop_id not in allowed
            else:
                assert row["expected_outcome"] == "found"
                assert order.shop_id in allowed
        elif row["expected_tool"] == "query_shipment":
            order = world.orders[args["order_id"]]
            assert args["order_id"] in world.shipments
            assert order.shop_id in world.customer_shops[row["principal"]["user_id"]]
        elif row["expected_tool"] == "query_shop_auth":
            assert args["shop_id"] in world.shops
            assert args["shop_id"] in world.customer_shops[row["principal"]["user_id"]]
        else:
            assert row["expected_tool"] == "none"
            assert args == {}


def test_same_seed_is_reproducible():
    module = _load_module()
    assert module.build_dataset(7) == module.build_dataset(7)
    assert module.build_dataset(7) != module.build_dataset(8)


def test_write_and_load_roundtrip_with_limit_header(work_dir):
    module = _load_module()
    rows = module.build_dataset(7)
    path = work_dir / "dataset_tools.jsonl"
    module.write_dataset(path, rows)

    header = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert any("模板" in line and "偏乐观" in line for line in header)
    assert module.load_dataset(path) == rows
    # 跳过注释后每行都是合法 JSON
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            assert json.loads(line)["expected_tool"] in {
                "query_order",
                "query_shipment",
                "query_shop_auth",
                "none",
            }
