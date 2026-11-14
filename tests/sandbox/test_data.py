import json

from carepilot.sandbox.data import (
    CARRIERS,
    INJECTION_REMARKS,
    SHIPPED_STATUSES,
    STATUSES,
    build_world,
)


def _dump(world) -> str:
    return json.dumps(world.to_dict(), ensure_ascii=False, sort_keys=True)


def test_same_seed_is_byte_identical():
    assert _dump(build_world(7)) == _dump(build_world(7))


def test_different_seed_changes_data():
    assert _dump(build_world(7)) != _dump(build_world(8))


def test_shops_and_expired_auth():
    world = build_world()
    assert sorted(world.shops) == ["S1", "S2", "S3", "S4"]
    assert {s.platform for s in world.shops.values()} == {"淘宝", "京东", "抖店", "拼多多"}
    assert any(not s.authorized for s in world.shops.values())
    expired = [s for s in world.shops.values() if not s.authorized and s.auth_expires_at < "2025-07-01"]
    assert expired


def test_orders_cover_all_shops_and_formats():
    world = build_world()
    assert len(world.orders) == 40
    per_shop = {sid: 0 for sid in world.shops}
    lengths = {"淘宝": 18, "京东": 12, "抖店": 19, "拼多多": 16}
    for order in world.orders.values():
        per_shop[order.shop_id] += 1
        assert order.order_id.isdigit()
        assert len(order.order_id) == lengths[order.platform]
        assert order.status in STATUSES
        assert isinstance(order.amount, int) and order.amount > 0
        assert len(order.buyer_phone) == 11 and order.buyer_phone.isdigit()
        assert order.remark is not None
    assert all(count >= 6 for count in per_shop.values())


def test_at_least_three_injection_remarks():
    world = build_world()
    remarks = [o.remark for o in world.orders.values()]
    for injection in INJECTION_REMARKS:
        assert remarks.count(injection) >= 1
    assert len([r for r in remarks if r in INJECTION_REMARKS]) >= 3
    # 注入文本里包含伪造的工具调用/转发指令
    assert any("send_reply" in r for r in remarks)
    assert any('"tool"' in r for r in remarks)


def test_shipments_only_for_shipped_orders():
    world = build_world()
    shipped = {o.order_id for o in world.orders.values() if o.status in SHIPPED_STATUSES}
    assert set(world.shipments) == shipped
    assert shipped
    for shipment in world.shipments.values():
        assert shipment.carrier in CARRIERS
        assert shipment.tracking_no
        assert shipment.latest_node
        assert shipment.latest_time
        assert shipment.eta


def test_customer_shop_mapping_matches_demo_data():
    world = build_world()
    assert world.customer_shops == {"c1": ("S1", "S2"), "c2": ("S3",), "c3": ("S4",)}
