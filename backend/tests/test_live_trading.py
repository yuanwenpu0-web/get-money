"""实盘（合约）执行链路测试：开仓取整/保护单/回滚、平仓结算、改单、周期同步、分发路由。
全部网络层均为 Fake，不发起真实请求。limit_maker 专项测试见 test_maker_orders.py。"""

import json

from app.market.futures import BinanceApiError
from app.models import Order
from app.state import set_state
from app.trading import dispatch, live, live_sync

from fakes import FakeFuturesClient, live_order, market_orders, patch_live, signal


def test_open_live_floors_qty_to_step_and_builds_protection_orders(db, monkeypatch):
    fake = FakeFuturesClient(step="0.1")
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None and order.source == "live"
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
    entry = orders[0]
    assert entry["side"] == "BUY" and entry["quantity"] == "46.3"  # 46.3134 向下取整到 0.1 步长
    sl, tp = orders[1], orders[2]
    assert sl["side"] == "SELL" and sl["closePosition"] == "true" and sl["stopPrice"] == "99"
    assert tp["closePosition"] == "true" and tp["stopPrice"] == "103"
    note = json.loads(order.note)
    assert note["exchange_order_id"] and note["sl_order_id"] and note["tp_order_id"]
    assert order.entry_price == 100.5  # 用交易所成交均价入账
    assert order.sl == 99.0 and order.tp == 103.0


def test_open_live_rejects_below_min_notional(db, monkeypatch):
    fake = FakeFuturesClient(notional="1000000")
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal()) is None
    assert market_orders(fake) == []
    assert db.query(Order).count() == 0


def test_open_live_rejects_without_credentials(db, monkeypatch):
    monkeypatch.setattr(live, "load_credentials", lambda session: ("", ""))
    assert live.open_live(db, signal()) is None
    assert db.query(Order).count() == 0


def test_open_live_aborts_when_leverage_rejected(db, monkeypatch):
    fake = FakeFuturesClient(fail_leverage=True)
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal()) is None
    # 杠杆失败必须在下单前中断，不允许产生任何交易所订单
    assert market_orders(fake) == []
    assert db.query(Order).count() == 0


def test_protection_failure_rolls_back_open_position(db, monkeypatch):
    fake = FakeFuturesClient(fail_protection=True)
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal()) is None
    assert db.query(Order).count() == 0  # 不留脏订单
    orders = market_orders(fake)
    assert orders[0]["type"] == "MARKET"  # 先开仓
    rollback = orders[-1]
    assert rollback["type"] == "MARKET" and rollback["reduceOnly"] == "true" and rollback["side"] == "SELL"
    # 两个保护单都未挂成功，回滚不应产生撤单请求
    assert fake.cancel_calls == []


def test_tp_failure_rolls_back_and_cancels_placed_sl_order(db, monkeypatch):
    # 止盈挂失败时，已挂成功的止损保护单必须被撤销，否则交易所残留 closePosition=true 脏单
    fake = FakeFuturesClient(fail_tp=True)
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal()) is None
    assert db.query(Order).count() == 0  # 不留脏订单
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET", "MARKET"]
    rollback = orders[-1]
    assert rollback["reduceOnly"] == "true" and rollback["side"] == "SELL"
    # 止损保护单的 orderId（fake 按调用序号从 101 递增，含账户准备调用）
    assert fake.cancel_calls == [("BTCUSDT", 105)]


def test_tp_failure_rollback_cancels_sl_even_if_close_fails(db, monkeypatch):
    # 撤单/平仓任一失败都不得互相阻塞：先平仓，撤单失败也要尝试，全部留痕
    fake = FakeFuturesClient(fail_tp=True)
    fake.new_order = _partial_fail_close(fake.new_order)
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal()) is None
    assert db.query(Order).count() == 0
    # 即使回滚平仓请求也失败了，止损保护单的撤销请求仍然发出
    assert ("BTCUSDT", 105) in fake.cancel_calls


def _partial_fail_close(original_new_order):
    def wrapped(**params):
        if params.get("reduceOnly") == "true":
            raise BinanceApiError("close rejected", -2019)
        return original_new_order(**params)
    return wrapped


def test_close_live_cancels_protections_and_settles_at_fill_price(db, monkeypatch):
    fake = FakeFuturesClient(avg_price="101.0")
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    closed = live.close_live(db, order, "manual")
    assert closed.status == "CLOSED" and closed.close_reason == "manual"
    assert closed.close_price == 101.0  # 真实成交均价，无本地滑点
    assert closed.realized_pnl == round((101.0 - 100.0) * 1.0 - 101.0 * 1.0 * 0.0004, 6)
    assert sorted(fake.cancel_calls) == [("BTCUSDT", 101), ("BTCUSDT", 102)]
    close_order = [item for item in market_orders(fake) if item.get("reduceOnly") == "true"][0]
    assert close_order["side"] == "SELL" and close_order["quantity"] == "1"


def test_close_live_ignores_unknown_order_cancel_error(db, monkeypatch):
    fake = FakeFuturesClient(cancel_error=BinanceApiError("Unknown order sent", -2011))
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    closed = live.close_live(db, order, "manual")
    assert closed.status == "CLOSED"


def test_close_live_settles_when_reduce_only_rejected_position_gone(db, monkeypatch):
    # 保护单刚触发导致仓位消失：reduceOnly 被拒(-2022)，按交易所已平仓结算而非报错
    fake = FakeFuturesClient(reject_reduce_only=True, mark_price="102.5")
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    closed = live.close_live(db, order, "manual")
    assert closed.status == "CLOSED"
    assert closed.close_price == 102.5  # 用交易所标记价结算
    assert market_orders(fake)[-1].get("reduceOnly") == "true"


def test_modify_live_replaces_protection_orders(db, monkeypatch):
    fake = FakeFuturesClient()
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    live.modify_live(db, order, sl=100.2)
    assert order.sl == 100.2
    note = json.loads(order.note)
    assert note["sl_order_id"] == 100 + len(fake.calls)  # 新保护单号已回写
    assert ("BTCUSDT", 101) in fake.cancel_calls  # 旧止损保护单已撤销
    replaced = market_orders(fake)[-1]
    assert replaced["type"] == "STOP_MARKET" and replaced["stopPrice"] == "100.2"


def test_sync_live_closes_when_exchange_position_gone(db, monkeypatch):
    fake = FakeFuturesClient(position_amt="0", sl_status="FILLED", mark_price="101.5")
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    result = live_sync.sync_live(db, order)
    assert result is not None
    assert result.status == "CLOSED" and result.close_reason == "exchange_sl"
    assert result.close_price == 101.5


def test_sync_live_keeps_order_open_while_position_exists(db, monkeypatch):
    fake = FakeFuturesClient(position_amt="0.5", mark_price="101.7")
    patch_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    assert live_sync.sync_live(db, order) is None
    assert order.status == "OPEN"
    assert order.mark_price == 101.7  # 标记价格已同步


def test_dispatch_routes_open_by_mode_and_close_by_source(db, monkeypatch):
    # paper 模式
    set_state(db, "execution_mode", "paper")
    paper_order = dispatch.open_position(db, signal())
    assert paper_order is not None and paper_order.source == "paper"
    # live 模式（换币种避免与 paper 持仓互斥）
    fake = FakeFuturesClient()
    patch_live(db, monkeypatch, fake)
    set_state(db, "execution_mode", "live")
    order = dispatch.open_position(db, signal(symbol="ETHUSDT"))
    assert order is not None and order.source == "live"
    assert market_orders(fake), "live 模式必须触达交易所下单"
    # 平仓按订单 source 路由：live 订单走交易所，paper 订单走本地
    assert dispatch.close_position(db, paper_order, 100.0, "manual").status == "CLOSED"
    assert market_orders(fake)[-1].get("reduceOnly") is None  # paper 平仓不触达交易所
    dispatch.close_position(db, order, 100.5, "manual")
    assert any(item.get("reduceOnly") == "true" for item in market_orders(fake))
