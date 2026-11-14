"""三个只读查询纯函数。

不依赖网关、不调用模型。权限只做「不属于我的店铺就当不存在」，返回结构里永远不含
订单留言（remark）和原始手机号。**所有数据都是合成的。**
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date

from .data import Order, Shipment, World
from .mask import mask_address, mask_phone, mask_tracking_no

# 沙箱里的“今天”固定，保证授权是否过期可复现
SANDBOX_TODAY = date(2025, 7, 1)


def _visible_order(world: World, shop_ids: Collection[str], order_id: str) -> Order | None:
    order = world.orders.get(order_id)
    if order is None or order.shop_id not in set(shop_ids):
        return None
    return order


def _order_view(order: Order) -> dict:
    return {
        "found": True,
        "order_id": order.order_id,
        "shop_id": order.shop_id,
        "platform": order.platform,
        "status": order.status,
        "amount_yuan": round(order.amount / 100, 2),
        "buyer_nick": order.buyer_nick,
        "buyer_phone_masked": mask_phone(order.buyer_phone),
        "receiver_address_masked": mask_address(order.receiver_address),
        "created_at": order.created_at,
    }


def query_order(world: World, shop_ids: Collection[str], order_id: str) -> dict:
    """按订单号查订单。不存在和无权访问返回一模一样的 `{"found": False}`，不泄露订单是否存在。"""
    order = _visible_order(world, shop_ids, order_id)
    if order is None:
        return {"found": False}
    return _order_view(order)


def query_order_remark(world: World, shop_ids: Collection[str], order_id: str) -> dict:
    """单独读买家留言，并显式标记为不可信文本；无权访问同样不泄露。"""
    order = _visible_order(world, shop_ids, order_id)
    if order is None:
        return {"remark": None, "untrusted": True}
    return {"remark": order.remark, "untrusted": True}


def _shipment_view(shipment: Shipment) -> dict:
    return {
        "found": True,
        "order_id": shipment.order_id,
        "carrier": shipment.carrier,
        "tracking_no": mask_tracking_no(shipment.tracking_no),
        "latest_node": shipment.latest_node,
        "latest_time": shipment.latest_time,
        "eta": shipment.eta,
    }


def query_shipment(world: World, order_id: str) -> dict:
    """查物流。没有物流记录（未发货）时返回 `{"found": False}`。"""
    shipment = world.shipments.get(order_id)
    if shipment is None:
        return {"found": False}
    return _shipment_view(shipment)


def query_shop_auth(world: World, shop_id: str) -> dict:
    """查店铺授权状态。expired 以沙箱固定的“今天”判断，避免测试随系统日期漂移。"""
    shop = world.shops.get(shop_id)
    if shop is None:
        return {"found": False}
    expires_at = date.fromisoformat(shop.auth_expires_at)
    return {
        "found": True,
        "shop_id": shop.shop_id,
        "name": shop.name,
        "platform": shop.platform,
        "authorized": shop.authorized,
        "auth_expires_at": shop.auth_expires_at,
        "expired": (not shop.authorized) or expires_at < SANDBOX_TODAY,
    }
