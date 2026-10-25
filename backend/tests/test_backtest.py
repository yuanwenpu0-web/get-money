import numpy as np
import pandas as pd
import pytest
from app.backtest.execution import simulate
from app.backtest.runner import _downsample, _metrics_from_equity
from app.ml.dataset import purged_split
from app.strategy.rules import resolve_params


def fixture_bars(n=5):
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    data = pd.DataFrame(
        dict(open=100.0, high=100.1, low=99.9, close=100.0, atr_pct=0.002), index=index
    )
    signals = pd.DataFrame(
        dict(
            direction=1,
            score=1.0,
            leverage=2,
            position_pct=0.1,
            sl_pct=0.01,
            tp_pct=0.02,
        ),
        index=index,
    )
    return data, signals


def params(**kwargs):
    return resolve_params(
        "trend_follow",
        {},
        {"risk_per_trade_pct": 0.02, "trailing_enabled": False, **kwargs},
    )


def test_next_open_fill_and_total_fee_accounting():
    data, signals = fixture_bars()
    signals.loc[:, "direction"] = 0
    signals.iloc[0, 0] = 1
    equity, trades = simulate(data, signals, params(), 10000.0, 0.001, 0.0, "1m")
    assert trades[0]["entry_time"] == str(data.index[1])
    assert trades[0]["pnl"] == pytest.approx(-4.0)
    assert equity.iloc[-1] == pytest.approx(9996.0)
    assert sum(t["pnl"] for t in trades) == pytest.approx(equity.iloc[-1] - 10000.0)


def test_stop_priority_and_post_exit_cooldown():
    data, signals = fixture_bars()
    data.loc[data.index[1], ["high", "low"]] = [103.0, 98.0]
    equity, trades = simulate(
        data, signals, params(cooldown_bars=10), 10000.0, 0.0, 0.0, "1m"
    )
    assert len(trades) == 1
    assert trades[0]["status"] == "stop_loss"
    assert trades[0]["exit_price"] == 99.0
    assert equity.iloc[-1] == 9980.0


def test_gap_stop_fills_at_open_not_stale_stop():
    data, signals = fixture_bars()
    data.loc[data.index[2], ["open", "high", "low", "close"]] = [95.0, 96.0, 94.0, 95.0]
    _, trades = simulate(data, signals, params(), 10000.0, 0.0, 0.0, "1m")
    assert trades[0]["exit_price"] == 95.0


def test_no_entry_across_missing_candles():
    data, signals = fixture_bars()
    data.index = data.index[:2].append(data.index[2:] + pd.Timedelta(days=1))
    signals.index = data.index
    signals["direction"] = [0, 1, 0, 0, 0]
    _, trades = simulate(data, signals, params(), 10000.0, 0.0, 0.0, "1m")
    assert trades == []


def test_purge_removes_labels_reaching_test_period():
    data = pd.DataFrame({"close": np.arange(100.0)})
    train, test = purged_split(data, 75, 5)
    assert train.index[-1] == 69
    assert test.index[0] == 75
    assert train.index[-1] + 5 < test.index[0]


def test_metrics_include_initial_loss_and_undefined_profit_factor():
    index = pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC")
    metrics = _metrics_from_equity(
        pd.Series([90.0, 90.0], index=index), [], 100.0, "1m"
    )
    assert metrics["max_drawdown_pct"] == -10.0
    metrics = _metrics_from_equity(
        pd.Series([100.0, 110.0], index=index),
        [{"pnl": 10.0, "fees": 0.0}],
        100.0,
        "1m",
    )
    assert metrics["profit_factor"] is None
    assert metrics["profit_factor_infinite"] is True


def test_downsample_includes_exact_last_point():
    equity = pd.Series(
        np.arange(3100.0), index=pd.date_range("2026-01-01", periods=3100, freq="1min")
    )
    points = _downsample(equity, limit=100)
    assert points[-1]["equity"] == 3099.0
    assert len(points) <= 100
