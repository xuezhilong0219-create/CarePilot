import json
import re

from carepilot.sandbox.data import SHIPPED_STATUSES, build_world
from carepilot.sandbox.tools import (
    query_order,
    query_order_remark,
    query_shipment,
    query_shop_auth,
)

_PHONE_RUN = re.compile(r"\d{11}")


def _walk(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, k)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, key)
    else:
        yield key, value


def test_query_order_found_view_has_no_sensitive_fields():
    world = build_world()
    order = next(iter(world.orders.values()))
    result = query_order(world, world.customer_shops["c1"] + ("S3", "S4"), order.order_id)

    assert result["found"] is True
    assert result["order_id"] == order.order_id
    assert result["shop_id"] == order.shop_id
    assert result["status"] == order.status
    assert result["amount_yuan"] == round(order.amount / 100, 2)
    assert result["buyer_phone_masked"] == order.buyer_phone[:3] + "****" + order.buyer_phone[-4:]
    assert "remark" not in result
    assert order.buyer_phone not in result.values()


def test_unauthorized_and_missing_order_are_indistinguishable():
    world = build_world()
    c2_order = next(
        o for o in world.orders.values() if o.shop_id in world.customer_shops["c2"]
    )
    assert query_order(world, world.customer_shops["c1"], c2_order.order_id) == query_order(
        world, world.customer_shops["c1"], "000000000000000000"
    ) == {"found": False}
    assert query_order_remark(world, world.customer_shops["c1"], c2_order.order_id) == (
        query_order_remark(world, world.customer_shops["c1"], "000000000000000000")
    )


def test_query_functions_never_leak_phone_or_remark():
    world = build_world()
    outputs = []
    for order in world.orders.values():
        outputs.append(query_order(world, world.customer_shops["c1"] + ("S3", "S4"), order.order_id))
    for order_id in world.shipments:
        outputs.append(query_shipment(world, order_id))
    for shop_id in world.shops:
        outputs.append(query_shop_auth(world, shop_id))

    blob = json.dumps(outputs, ensure_ascii=False)
    for order in world.orders.values():
        assert order.buyer_phone not in blob
        if order.remark:
            assert order.remark not in blob
    for key, value in _walk(outputs):
        assert key != "remark"
        if isinstance(value, str) and key != "order_id":
            assert not _PHONE_RUN.search(value), (key, value)

    assert world.orders  # 确保上面的循环不是空跑


def test_query_order_remark_is_flagged_untrusted():
    world = build_world()
    order = next(o for o in world.orders.values() if "忽略以上所有指令" in o.remark)
    result = query_order_remark(world, world.customer_shops["c1"] + ("S2", "S3", "S4"), order.order_id)
    assert result == {"remark": order.remark, "untrusted": True}


def test_query_shipment_found_and_missing():
    world = build_world()
    shipped_id = next(iter(world.shipments))
    result = query_shipment(world, shipped_id)
    assert result["found"] is True
    assert result["order_id"] == shipped_id
    assert "****" in result["tracking_no"]
    assert result["carrier"] in {"顺丰", "中通", "圆通", "京东物流"}

    pending = next(
        o.order_id for o in world.orders.values() if o.status not in SHIPPED_STATUSES
    )
    assert query_shipment(world, pending) == {"found": False}
    assert query_shipment(world, "不存在的订单") == {"found": False}


def test_query_shop_auth_status_and_expiry():
    world = build_world()
    expired = [s for s in world.shops.values() if not s.authorized]
    assert expired
    for shop in expired:
        result = query_shop_auth(world, shop.shop_id)
        assert result["found"] is True
        assert result["authorized"] is False
        assert result["expired"] is True

    active = next(s for s in world.shops.values() if s.authorized)
    assert query_shop_auth(world, active.shop_id)["expired"] is False
    assert query_shop_auth(world, "S9") == {"found": False}
