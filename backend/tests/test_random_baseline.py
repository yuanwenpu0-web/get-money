import json

import numpy as np
import pandas as pd
import pytest
from app.backtest import runner
from app.backtest.execution import (
    RANDOM_BASELINE_SEED,
    exposure_metrics,
    randomize_directions,
    simulate,
)
from app.backtest.metrics import metrics_from_equity
from app.features import indicators
from app.strategy.rules import resolve_params


def params(**kwargs):
    return resolve_params(
        "trend_follow",
        {},
        {"risk_per_trade_pct": 0.02, "trailing_enabled": False, **kwargs},
    )


def prices(n=1200):
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 + 0.01 * np.arange(n) + 0.8 * np.sin(np.arange(n) / 20)
    return pd.DataFrame(
        dict(open=close, high=close + 0.1, low=close - 0.1, close=close, volume=100.0),
        index=index,
    )


def bars(n=1200, seed=42):
    """合成 1m K 线：开盘=前收，当根 high/low 覆盖 ±2%，
    任意方向都在入场当根触发 SL/TP，保证持仓不跨到后续信号。"""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        dict(
            open=open_,
            high=np.maximum(open_, close) * 1.02,
            low=np.minimum(open_, close) * 0.98,
            close=close,
            atr_pct=0.02,
        ),
        index=index,
    )


def signals_of(data, every=20):
    signals = pd.DataFrame(
        dict(
            direction=0,
            score=1.0,
            leverage=2,
            position_pct=0.1,
            sl_pct=0.005,
            tp_pct=0.01,
        ),
        index=data.index,
    )
    for i in range(every, len(data), every):
        signals.iloc[i, 0] = 1 if (i // every) % 2 == 0 else -1
    return signals


def test_random_baseline_is_deterministic_for_fixed_seed():
    data, signals = bars(), signals_of(bars())
    first = simulate(
        data, randomize_directions(signals), params(), 10000.0, 0.001, 0.0, "1m"
    )
    second = simulate(
        data, randomize_directions(signals), params(), 10000.0, 0.001, 0.0, "1m"
    )
    pd.testing.assert_series_equal(first[0], second[0])
    assert first[1] == second[1]
    assert randomize_directions(signals, RANDOM_BASELINE_SEED).equals(
        randomize_directions(signals)
    )


def test_flip_ratio_is_balanced_and_only_direction_changes():
    data, signals = bars(), signals_of(bars())
    randomized = randomize_directions(signals)
    live = signals["direction"] != 0
    flipped = (randomized["direction"] != signals["direction"]) & live
    ratio = flipped.sum() / live.sum()
    assert 0.3 < ratio < 0.7
    assert (randomized.loc[~live, "direction"] == 0).all()
    assert randomized["direction"].isin([-1, 0, 1]).all()
    pd.testing.assert_frame_equal(
        signals.drop(columns=["direction"]), randomized.drop(columns=["direction"])
    )


def test_random_baseline_keeps_entry_timing_and_trade_count():
    data, signals = bars(), signals_of(bars())
    _, baseline_trades = simulate(data, signals, params(), 10000.0, 0.001, 0.0, "1m")
    _, random_trades = simulate(
        data, randomize_directions(signals), params(), 10000.0, 0.001, 0.0, "1m"
    )
    assert len(baseline_trades) == len(random_trades) == int((signals["direction"] != 0).sum())
    assert [t["entry_time"] for t in baseline_trades] == [
        t["entry_time"] for t in random_trades
    ]


def test_direction_flip_mirrors_stop_and_target_distances():
    index = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
    data = pd.DataFrame(
        dict(
            open=[100.0, 100.0, 100.0],
            high=[100.1, 100.1, 103.0],
            low=[99.9, 99.9, 99.5],
            close=[100.0, 100.0, 100.5],
            atr_pct=[0.002, 0.002, 0.002],
        ),
        index=index,
    )
    template = pd.DataFrame(
        dict(
            direction=[0, 1, 0],
            score=[0.0, 1.0, 0.0],
            leverage=[1, 2, 1],
            position_pct=[0.0, 0.1, 0.0],
            sl_pct=[0.0, 0.01, 0.0],
            tp_pct=[0.0, 0.02, 0.0],
        ),
        index=index,
    )
    _, long_trades = simulate(data, template, params(), 10000.0, 0.0, 0.0, "1m")
    short_signals = template.copy()
    short_signals.iloc[1, 0] = -1
    _, short_trades = simulate(data, short_signals, params(), 10000.0, 0.0, 0.0, "1m")
    long_trade, short_trade = long_trades[0], short_trades[0]
    assert long_trade["entry_price"] == short_trade["entry_price"] == pytest.approx(100.0)
    assert long_trade["status"] == "take_profit"
    assert long_trade["exit_price"] == pytest.approx(102.0)
    assert short_trade["status"] == "stop_loss"
    assert short_trade["exit_price"] == pytest.approx(101.0)
    assert long_trade["exit_price"] - long_trade["entry_price"] == pytest.approx(
        100.0 * 0.02
    )
    assert short_trade["exit_price"] - short_trade["entry_price"] == pytest.approx(
        100.0 * 0.01
    )


def test_exposure_pct_math_for_partial_and_full_occupation():
    index = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    equity = pd.Series(np.arange(10.0), index=index)
    one_trade = [
        {"entry_time": str(index[0]), "exit_time": str(index[4])}
    ]
    assert exposure_metrics(equity, one_trade, "1m") == {
        "exposure_pct": 50.0,
        "avg_bars_held": 5.0,
    }
    two_trades = one_trade + [{"entry_time": str(index[5]), "exit_time": str(index[7])}]
    assert exposure_metrics(equity, two_trades, "1m") == {
        "exposure_pct": 80.0,
        "avg_bars_held": 4.0,
    }


def test_exposure_metrics_zero_on_no_trades_or_empty_equity():
    index = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    equity = pd.Series(np.arange(10.0), index=index)
    assert exposure_metrics(equity, [], "1m") == {
        "exposure_pct": 0.0,
        "avg_bars_held": 0.0,
    }
    empty = exposure_metrics(pd.Series(dtype=float), [], "1m")
    assert empty == {"exposure_pct": 0.0, "avg_bars_held": 0.0}


def test_zero_trade_boundary_does_not_break_metrics():
    data, signals = bars(), signals_of(bars())
    signals = signals.copy()
    signals["direction"] = 0
    equity, trades = simulate(data, signals, params(), 10000.0, 0.001, 0.0, "1m")
    assert trades == []
    assert bool((equity == 10000.0).all())
    metrics = metrics_from_equity(equity, trades, 10000.0, "1m")
    metrics.update(exposure_metrics(equity, trades, "1m"))
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["expectancy"] == 0.0
    assert metrics["exposure_pct"] == 0.0
    assert metrics["avg_bars_held"] == 0.0


def test_run_backtest_exposes_random_baseline_and_exposure(db, monkeypatch):
    data = indicators.build_features(prices(), with_label=True).dropna()
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **kw: data)
    result = runner.run_backtest(db, "BTCUSDT", "1m", "trend_follow")
    metrics = result["metrics"]
    for group in ("baseline", "random_baseline"):
        assert group in metrics
        for key in (
            "total_return_pct",
            "max_drawdown_pct",
            "total_trades",
            "exposure_pct",
            "avg_bars_held",
        ):
            assert key in metrics[group]
    for key in ("exposure_pct", "avg_bars_held"):
        assert key in metrics
    assert isinstance(metrics["warnings"], list)
    json.dumps(result, allow_nan=False)
