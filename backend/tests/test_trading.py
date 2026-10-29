from types import SimpleNamespace

import pandas as pd
import pytest
from app.models import Order
from app.trading import order_manager, paper
from app.trading.risk import position_size


def signal(**overrides):
    return SimpleNamespace(
        **(
            dict(
                symbol="BTCUSDT",
                interval="1m",
                direction="LONG",
                price=100.0,
                sl=99.0,
                tp=103.0,
                leverage=10,
                position_pct=0.25,
                strategy="absorption_reversal",
                score=1.0,
                prob_up=0.8,
            )
            | overrides
        )
    )


def test_paper_stop_loss_respects_fee_inclusive_risk(db):
    from app.models import StrategyConfig

    # 吸收反转默认无冷却（脚本没有该闸门）；本用例单独配置冷却来验证撮合侧闸门
    db.add(StrategyConfig(code="absorption_reversal", name="吸收反转", enabled=1,
                          params='{"cooldown_bars": 10}'))
    db.commit()
    order = paper.open_position(db, signal())
    assert order is not None
    closed = paper.close_position(db, order, order.sl, "stop_loss")
    assert (
        closed.realized_pnl >= -50.01
    )  # default 0.5% of 10,000, including both fees/slips
    assert paper.open_position(db, signal()) is None  # persistent cooldown


def test_sizing_cannot_overspend_cash_at_high_leverage():
    margin, qty = position_size(
        100.0, 100.0, 1.0, 125, 100.0, 99.0, 1, 0.01, 0.001, 0.05
    )
    assert margin + qty * 100 * 0.01 <= 100.0 + 1e-9


def test_stale_signal_cannot_open(db, monkeypatch):
    monkeypatch.setattr(paper, "now_ms", lambda: 1_000_000)
    assert paper.open_position(db, signal(ts=1)) is None


def test_paper_allows_stop_only_signal_without_take_profit(db):
    # ZigZag 反手型：反向通道为止损，无止盈（tp=0）必须能开仓
    order = paper.open_position(db, signal(
        strategy="simple_zigzag_double", tp=0.0, sl=105.0,
        direction="SHORT", leverage=1, position_pct=0.5,
    ))
    assert order is not None
    assert order.tp == 0 and order.sl == 105.0 and order.side == "SHORT"


def test_zigzag_order_follows_reverse_level_and_skips_generic_stops(db, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")

    def make_order(sl):
        return Order(
            symbol="BTCUSDT", interval="1m", side="LONG", qty=1.0,
            entry_price=110.0, mark_price=110.0, margin=110.0, leverage=1,
            sl=sl, tp=0.0, strategy="simple_zigzag_double",
            status="OPEN", created_at=start.to_pydatetime(),
        )

    frame = pd.DataFrame(
        dict(open=[110.0], high=[110.5], low=[109.5], close=[110.0], volume=[1.0]),
        index=pd.date_range(start, periods=1, freq="1min"),
    )
    monkeypatch.setattr(order_manager.MarketDataService, "load_frame", lambda *a, **k: frame)
    monkeypatch.setattr(
        order_manager, "now_ms",
        lambda: int((start + pd.Timedelta(minutes=1)).timestamp() * 1000),
    )

    def zigzag_signal(short_level):
        return SimpleNamespace(
            ts=int(start.timestamp() * 1000), direction="FLAT",
            reason={"zz_levels": {"long": 118.0, "short": short_level}},
        )

    # 多头止损跟随最新空头通道：pivot 下移 → 止损只下调
    order = make_order(105.0)
    db.add(order)
    db.commit()
    result = order_manager.manage_order(db, order, zigzag_signal(102.0))
    assert order.sl == 102.0 and "zz_stop=102.0" in result["actions"]
    assert order.status == "OPEN"  # 未触及，正常持仓

    # 止损只单向松动：新的通道价位更高时不回收紧
    db.add(make_order(102.0))
    db.commit()
    tighter = db.query(Order).filter(Order.id != order.id).first()
    order_manager.manage_order(db, tighter, zigzag_signal(106.0))
    assert tighter.sl == 102.0


def test_order_does_not_replay_pre_entry_candle_or_same_candle(db, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    order = Order(
        symbol="BTCUSDT",
        interval="1m",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        mark_price=100.0,
        margin=100.0,
        leverage=1,
        sl=99.0,
        tp=105.0,
        strategy="absorption_reversal",
        status="OPEN",
        created_at=start.to_pydatetime(),
    )
    db.add(order)
    db.commit()
    frame = pd.DataFrame(
        dict(
            open=[100.0, 100.0],
            high=[102.0, 100.2],
            low=[90.0, 99.8],
            close=[100.0, 100.1],
            volume=[1.0, 1.0],
        ),
        index=pd.date_range(start - pd.Timedelta(minutes=1), periods=2, freq="1min"),
    )
    monkeypatch.setattr(
        order_manager.MarketDataService, "load_frame", lambda *a, **k: frame
    )
    monkeypatch.setattr(
        order_manager,
        "now_ms",
        lambda: int((start + pd.Timedelta(minutes=1)).timestamp() * 1000),
    )
    order_manager.manage_order(db, order)
    assert order.status == "OPEN"
    order.sl = (
        100.05  # manual tightening must not apply retroactively to the same candle
    )
    db.commit()
    assert order_manager.manage_order(db, order)["actions"] == []
    assert order.status == "OPEN"


def test_signal_engine_matches_shared_rules_and_blocks_stale(db, monkeypatch):
    import numpy as np
    from app.features.indicators import build_features
    from app.strategy import engine
    from app.strategy.rules import evaluate_signals, resolve_params

    index = pd.date_range("2026-01-01", periods=200, freq="1min", tz="UTC")
    close = 100 + np.arange(200) * 0.04
    frame = pd.DataFrame(
        dict(open=close, high=close + 0.1, low=close - 0.1, close=close, volume=100.0),
        index=index,
    )
    monkeypatch.setattr(engine.MarketDataService, "load_frame", lambda *a, **k: frame)
    timestamp = int(index[-1].timestamp() * 1000) + 60000
    monkeypatch.setattr(engine, "now_ms", lambda: timestamp)
    result = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "absorption_reversal")
    shared = evaluate_signals(
        build_features(frame).iloc[-1:],
        "absorption_reversal",
        resolve_params("absorption_reversal", {}),
    ).iloc[0]
    assert result.direction == {0: "FLAT", 1: "LONG", -1: "SHORT"}[shared.direction]
    assert result.position_pct == pytest.approx(shared.position_pct)
    monkeypatch.setattr(engine, "now_ms", lambda: timestamp + 300000)
    stale = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "absorption_reversal")
    assert stale.direction == "FLAT"
    assert stale.reason["block_reason"] == "stale_data"
