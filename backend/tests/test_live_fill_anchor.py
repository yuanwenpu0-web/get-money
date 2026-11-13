"""成交价锚定：保护单必须把「信号价 → sl/tp」的绝对距离平移到实际成交价，
从而在追价/有利成交后仍保持 1.5×ATR 风险与 target_r 盈亏比。网络层全部为 Fake。"""

import pytest

from app.models import Order
from app.trading import live

from fakes import FakeFuturesClient, market_orders, patch_live, signal


def test_reanchor_keeps_stop_distance_when_long_fill_chased():
    sl, tp = live.reanchor_exits(101.0, 100.0, 99.0, 103.0)
    assert (sl, tp) == (100.0, 104.0)
    assert 101.0 - sl == pytest.approx(100.0 - 99.0)
    assert tp - 101.0 == pytest.approx(103.0 - 100.0)


def test_reanchor_keeps_stop_distance_when_short_fill_chased():
    sl, tp = live.reanchor_exits(99.0, 100.0, 101.0, 97.0)
    assert (sl, tp) == (100.0, 96.0)
    assert sl - 99.0 == pytest.approx(101.0 - 100.0)
    assert 99.0 - tp == pytest.approx(100.0 - 97.0)


def test_reanchor_is_identity_when_fill_equals_signal_price():
    assert live.reanchor_exits(100.0, 100.0, 99.0, 103.0) == (99.0, 103.0)


def test_protection_triggers_follow_fill_price_for_long(db, monkeypatch):
    fake = FakeFuturesClient(avg_price="100.5")
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal())
    assert order is not None
    sl, tp = market_orders(fake)[1:3]
    assert sl["triggerPrice"] == "99.5"
    assert tp["triggerPrice"] == "103.5"
    assert order.entry_price == 100.5
    assert order.sl == 99.5 and order.tp == 103.5


def test_protection_triggers_follow_fill_price_for_short(db, monkeypatch):
    fake = FakeFuturesClient(avg_price="99.5")
    patch_live(db, monkeypatch, fake)
    order = live.open_live(db, signal(direction="SHORT", sl=101.0, tp=97.0))
    assert order is not None
    sl, tp = market_orders(fake)[1:3]
    assert sl["triggerPrice"] == "100.5"
    assert tp["triggerPrice"] == "96.5"
    assert order.entry_price == 99.5
    assert order.sl == 100.5 and order.tp == 96.5


def test_risk_usdt_and_reward_risk_are_independent_of_fill_price(db, monkeypatch):
    risks, ratios = [], []
    for index, avg_price in enumerate(("100.0", "100.5", "102.0", "98.0")):
        fake = FakeFuturesClient(avg_price=avg_price)
        patch_live(db, monkeypatch, fake)
        order = live.open_live(db, signal(symbol=f"ANCHOR{index}USDT"))
        assert order is not None
        risks.append(order.qty * abs(order.entry_price - order.sl))
        ratios.append(abs(order.tp - order.entry_price) / abs(order.entry_price - order.sl))
        # 清掉上一笔，保持权益/可用保证金不变，否则仓位缩放会掩盖成交价的影响
        db.query(Order).delete()
        db.commit()
    assert max(risks) - min(risks) == pytest.approx(0.0, abs=1e-9)
    assert ratios == [pytest.approx(3.0)] * len(ratios)


class CoarseTickClient(FakeFuturesClient):
    """tick 粗到会把重锚后的止损推到成交价上（99.9 → 100.0）。"""

    def exchange_info(self, symbol):
        info = super().exchange_info(symbol)
        for item in info["symbols"][0]["filters"]:
            if item["filterType"] == "PRICE_FILTER":
                item["tickSize"] = "5"
        return info


def test_open_rolls_back_when_tick_rounding_puts_stop_at_fill(db, monkeypatch):
    fake = CoarseTickClient(avg_price="100.0")
    patch_live(db, monkeypatch, fake)
    assert live.open_live(db, signal(sl=99.9, tp=100.3)) is None
    assert db.query(Order).count() == 0
    orders = market_orders(fake)
    assert [item["type"] for item in orders] == ["MARKET", "MARKET"]
    assert orders[-1]["reduceOnly"] == "true" and orders[-1]["side"] == "SELL"
    assert fake.cancel_calls == []