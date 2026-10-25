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
                strategy="trend_follow",
                score=1.0,
                prob_up=0.8,
            )
            | overrides
        )
    )


def test_paper_stop_loss_respects_fee_inclusive_risk(db):
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
        strategy="trend_follow",
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
    result = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "trend_follow")
    shared = evaluate_signals(
        build_features(frame).iloc[-1:],
        "trend_follow",
        resolve_params("trend_follow", {}),
    ).iloc[0]
    assert result.direction == {0: "FLAT", 1: "LONG", -1: "SHORT"}[shared.direction]
    assert result.position_pct == pytest.approx(shared.position_pct)
    monkeypatch.setattr(engine, "now_ms", lambda: timestamp + 300000)
    stale = engine.StrategyEngine(db).evaluate("BTCUSDT", "1m", "trend_follow")
    assert stale.direction == "FLAT"
    assert stale.reason["block_reason"] == "stale_data"
