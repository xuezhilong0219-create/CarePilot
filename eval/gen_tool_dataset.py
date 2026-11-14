"""工具选择评测集生成器：模板 + 合成种子数据，**不调用 LLM、不联网**。

每条样本的字段：
    qid, question, principal{role,user_id}, expected_tool, expected_args, expected_outcome

expected_outcome 取值：
    found         工具应成功返回（订单/物流/授权存在且属于该客户）
    denied_owner  工具应被调用，但越权，网关应拒绝为 denied_owner
    no_tool       知识库类问题，不应调用任何工具

局限（同样写进输出文件头）：模板生成的问法比真实客户规整，工具选择准确率会偏乐观。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 允许 `python eval/gen_tool_dataset.py` 直接从仓库根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carepilot.sandbox.data import World, build_world  # noqa: E402

CUSTOMERS = ("c1", "c2", "c3")

# 这三种问法既可能是查订单状态，也可能是查物流进度，语义上就是歧义的。真实模型选 query_shipment 是合理的，
# 所以这些样本两个工具都算对，并且单独统计（见 tool_eval），不和"明确问法"的准确率混在一起。
AMBIGUOUS_ORDER_TEMPLATES = (
    "订单号 {order_id}，麻烦看下发货了没",
    "客户说没收到货，订单 {order_id} 是什么情况",
    "订单 {order_id} 为什么还没发货呀",
)

ORDER_TEMPLATES = (
    "帮我查一下订单 {order_id} 现在什么状态",
    "订单号 {order_id}，麻烦看下发货了没",
    "客户说没收到货，订单 {order_id} 是什么情况",
    "{order_id} 这笔订单的金额和状态帮我确认一下",
    "我这边有个淘宝订单，尾号 {tail}，单号 {order_id}，能查下吗？",
    "查订单 {order_id}，急！",
    "麻烦帮忙看下 {order_id} 这个订单到哪一步了",
    "订单 {order_id} 为什么还没发货呀",
)

SHIPMENT_TEMPLATES = (
    "订单 {order_id} 的快递到哪了",
    "帮我查下 {order_id} 的物流",
    "{order_id} 发的是哪个快递，什么时候能到",
    "客户问订单 {order_id} 这个件走到哪了",
    "订单 {order_id} 的物流一直没更新，麻烦看下",
    "麻烦看下订单 {order_id} 的物流进度",
)

AUTH_TEMPLATES = (
    "我的店铺 {name}（{shop_id}）授权过期了吗",
    "{shop_id} 这个店还能正常同步订单吗，授权有没有到期",
    "帮我查下店铺 {name} 的授权状态",
    "店铺 {shop_id} 的授权什么时候到期",
    "我怀疑 {name} 授权失效了，帮我看下",
)

NONE_QUESTIONS = (
    "库存同步失败怎么办",
    "电子面单怎么设置",
    "淘宝店铺如何重新授权",
    "订单审核流程怎么配置",
    "为什么订单下载不下来",
    "如何设置仓库和货品匹配",
    "发货单打印失败怎么处理",
    "商品匹配不上怎么办",
    "怎么批量导入商品资料",
    "平台账单对不上怎么查",
    "怎么设置订单自动审核",
    "客服子账号权限怎么配置",
)

DENIED_TEMPLATES = (
    "帮我查一下订单 {order_id}",
    "订单 {order_id} 是什么状态",
    "麻烦看下 {order_id} 这个订单的金额",
)


def _visible_orders(world: World, customer: str) -> list[Any]:
    shops = set(world.customer_shops[customer])
    return sorted(
        (o for o in world.orders.values() if o.shop_id in shops),
        key=lambda o: o.order_id,
    )


def _visible_shops(world: World, customer: str) -> list[Any]:
    return [world.shops[sid] for sid in sorted(world.customer_shops[customer])]


def _principal(customer: str) -> dict[str, str]:
    return {"role": "customer", "user_id": customer}


def _item(
    qid: str,
    question: str,
    customer: str,
    tool: str,
    args: dict[str, str],
    outcome: str,
) -> dict[str, Any]:
    return {
        "qid": qid,
        "question": question,
        "principal": _principal(customer),
        "expected_tool": tool,
        "expected_args": args,
        "expected_outcome": outcome,
        "acceptable_tools": [tool],
        "ambiguous": False,
    }


def build_dataset(seed: int = 7) -> list[dict[str, Any]]:
    """60 条：订单 20、物流 12、授权 10、不该调工具 12（20%）、越权 6（10%）。"""
    world = build_world(seed)
    rows: list[dict[str, Any]] = []

    def add(question: str, customer: str, tool: str, args: dict[str, str], outcome: str) -> None:
        rows.append(_item(f"t{len(rows) + 1:04d}", question, customer, tool, args, outcome))

    # 订单查询：按客户轮转，保证订单真实存在且属于该客户
    for i in range(20):
        customer = CUSTOMERS[i % len(CUSTOMERS)]
        orders = _visible_orders(world, customer)
        order = orders[(i // len(CUSTOMERS)) % len(orders)]
        template = ORDER_TEMPLATES[i % len(ORDER_TEMPLATES)]
        question = template.format(order_id=order.order_id, tail=order.order_id[-4:])
        add(question, customer, "query_order", {"order_id": order.order_id}, "found")
        if template in AMBIGUOUS_ORDER_TEMPLATES:
            rows[-1]["acceptable_tools"] = ["query_order", "query_shipment"]
            rows[-1]["ambiguous"] = True

    # 物流查询：只取有物流记录的订单
    shipment_pairs = [
        (customer, order)
        for customer in CUSTOMERS
        for order in _visible_orders(world, customer)
        if order.order_id in world.shipments
    ]
    for i in range(12):
        customer, order = shipment_pairs[i % len(shipment_pairs)]
        template = SHIPMENT_TEMPLATES[i % len(SHIPMENT_TEMPLATES)]
        question = template.format(order_id=order.order_id)
        add(question, customer, "query_shipment", {"order_id": order.order_id}, "found")

    # 店铺授权查询
    for i in range(10):
        customer = CUSTOMERS[i % len(CUSTOMERS)]
        shops = _visible_shops(world, customer)
        shop = shops[(i // len(CUSTOMERS)) % len(shops)]
        template = AUTH_TEMPLATES[i % len(AUTH_TEMPLATES)]
        question = template.format(shop_id=shop.shop_id, name=shop.name)
        add(question, customer, "query_shop_auth", {"shop_id": shop.shop_id}, "found")

    # 知识库类问题：不该调用任何工具
    for question in NONE_QUESTIONS:
        add(question, "c1", "none", {}, "no_tool")

    # 越权：客户问别的店铺的订单，工具会被调用但网关应拒绝。
    # 每个客户取不同偏移的订单，保证 6 条问题不重复。
    denied_rows: list[tuple[str, Any]] = []
    for customer_index, customer in enumerate(CUSTOMERS):
        allowed = set(world.customer_shops[customer])
        others = sorted(
            (o for o in world.orders.values() if o.shop_id not in allowed),
            key=lambda o: o.order_id,
        )
        for j in range(2):
            denied_rows.append((customer, others[customer_index * 2 + j]))
    for i, (customer, order) in enumerate(denied_rows):
        question = DENIED_TEMPLATES[i % len(DENIED_TEMPLATES)].format(order_id=order.order_id)
        add(question, customer, "query_order", {"order_id": order.order_id}, "denied_owner")

    return rows


HEADER = (
    "# 工具选择评测集：合成数据 + 模板生成，不调用 LLM。",
    "# 局限：模板问法比真实客户规整，工具选择准确率会偏乐观；越权样本的比例也只有 10%。",
)


def write_dataset(path: Path, rows: list[dict[str, Any]], seed: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = list(HEADER)
    lines.extend(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """跳过文件头注释行。"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成工具选择评测集（模板，不调用 LLM）")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("eval/real/dataset_tools.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_dataset(args.seed)
    write_dataset(args.out, rows, args.seed)
    print(f"已写入 {len(rows)} 条到 {args.out}（seed={args.seed}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
