"""limit_maker 开仓主单测试：postOnly 限价 payload、未成交撤单转市价、拒单直转市价、
market 默认模式行为不变。全部网络层均为 Fake，不发起真实请求。"""

import json

from app.market.futures import BinanceApiError
from app.models import Order
from app.state import set_state
from app.trading import live, maker

from fakes import FakeFuturesClient, market_orders, patch_live, signal


def test_open_live_limit_maker_places_post_only_order_at_mark_price(db, monkeypatch):
    set_state(db, "execution_mode", "live")
    set_state(db, "order_type", "limit_maker")
    monkeypatch.setattr(maker, "MAKER_POLL_INTERVAL_SECONDS", 0)
    fake = FakeFuturesClient(step="0.1", maker_status="FILLED", maker_avg_price="100.4")
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None and order.source == "live"
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    limit = orders[0]
    assert limit["side"] == "BUY" and limit["timeInForce"] == "GTX"
    assert limit["price"] == "101"  # 标记价（premium_index），按 tickSize 取整
    assert limit["quantity"] == "46.3"  # 与市价单相同的步长取整
    assert order.entry_price == 100.4  # 用 maker 成交均价入账
    assert json.loads(order.note)["order_type"] == "limit_maker"
    assert fake.cancel_calls == []  # 已成交，不撤单


def test_open_live_limit_maker_cancels_and_falls_back_to_market_when_unfilled(db, monkeypatch):
    set_state(db, "execution_mode", "live")
    set_state(db, "order_type", "limit_maker")
    monkeypatch.setattr(maker, "MAKER_POLL_INTERVAL_SECONDS", 0)
    fake = FakeFuturesClient(step="0.1", maker_status="NEW")  # 轮询始终未成交
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None and order.source == "live"
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert fake.cancel_calls == [("BTCUSDT", fake.limit_order_id)]  # 未成交先撤单
    market = orders[1]
    assert market["type"] == "MARKET" and market["quantity"] == "46.3"  # 保证成交确定性
    assert order.entry_price == 100.5  # 市价补足成交均价


def test_open_live_limit_maker_rejection_goes_straight_to_market(db, monkeypatch):
    set_state(db, "execution_mode", "live")
    set_state(db, "order_type", "limit_maker")
    fake = FakeFuturesClient(step="0.1", limit_reject=BinanceApiError("would immediately match", -5022))
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None and order.source == "live"
    orders = market_orders(fake)
    # 限价单被交易所拒绝（订单未成立）：不撤单，直接市价保证成交
    assert [item["type"] for item in orders] == ["LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert fake.cancel_calls == []
    assert order.entry_price == 100.5


def test_open_live_market_mode_payload_unchanged(db, monkeypatch):
    set_state(db, "execution_mode", "live")
    set_state(db, "order_type", "market")  # 默认模式：不下 LIMIT/GTX，不轮询
    fake = FakeFuturesClient(step="0.1")
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert all(item.get("timeInForce") is None for item in orders)
    assert json.loads(order.note)["order_type"] == "market"


def test_maker_poll_constants_bounded():
    # 轮询预算有界：默认约 10s 内必须给出成交/未成交结论，不允许无限挂单
    assert maker.MAKER_POLL_MAX_TIMES * maker.MAKER_POLL_INTERVAL_SECONDS <= 10.0


def test_limit_maker_cancel_unknown_order_checks_final_state(db, monkeypatch):
    # 竞态：轮询结束后订单恰好成交，撤单返回 -2011（不带成交量）。
    # 必须再查最终状态；若直接转市价会按满量重复建仓。
    set_state(db, "execution_mode", "live")
    set_state(db, "order_type", "limit_maker")
    monkeypatch.setattr(maker, "MAKER_POLL_INTERVAL_SECONDS", 0)
    fake = FakeFuturesClient(step="0.1", maker_status="NEW")
    patch_live(db, monkeypatch, fake)
    original_get_order = fake.get_order
    state = {"canceled": False}

    def cancel_order(symbol, order_id):
        state["canceled"] = True
        raise BinanceApiError("Unknown order sent", -2011)

    def get_order(symbol, order_id):
        if state["canceled"] and order_id == fake.limit_order_id:
            return {"orderId": order_id, "status": "FILLED",
                    "avgPrice": "100.4", "executedQty": "46.3"}
        return original_get_order(symbol, order_id)

    fake.cancel_order = cancel_order
    fake.get_order = get_order
    order = live.open_live(db, signal())
    assert order is not None and order.source == "live"
    orders = market_orders(fake)
    # 已成交：不得再挂市价单补量（否则仓位翻倍）
    assert [item["type"] for item in orders] == ["LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert state["canceled"] is True
    assert order.entry_price == 100.4
