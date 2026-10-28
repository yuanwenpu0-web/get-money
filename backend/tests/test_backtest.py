import numpy as np
import pandas as pd
import pytest
from app.backtest.execution import simulate
from app.backtest.metrics import downsample, metrics_from_equity
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
        "absorption_reversal",
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
    metrics = metrics_from_equity(
        pd.Series([90.0, 90.0], index=index), [], 100.0, "1m"
    )
    assert metrics["max_drawdown_pct"] == -10.0
    metrics = metrics_from_equity(
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
    points = downsample(equity, limit=100)
    assert points[-1]["equity"] == 3099.0
    assert len(points) <= 100


def test_minimum_sample_trades_scales_with_validation_size():
    from app.backtest.runner import minimum_sample_trades

    assert minimum_sample_trades(2880) == 57  # 2880 根验证集 → 57 笔门槛
    assert minimum_sample_trades(1000) == 30  # 20 被 30 的保底门槛取代
    assert minimum_sample_trades(100) == 30  # 短验证集退回 30 笔保底
    assert minimum_sample_trades(0) == 30


def _selection(monkeypatch, validation_bars, trades):
    """用受控成交笔数跑 select_parameters，隔离信号与撮合细节。"""
    from app.backtest import runner

    validation = pd.DataFrame(
        index=pd.date_range("2026-01-01", periods=validation_bars, freq="1min", tz="UTC")
    )
    equity = pd.Series(10000.0, index=validation.index)
    monkeypatch.setattr(runner, "evaluate_signals", lambda *a, **k: None)
    monkeypatch.setattr(runner, "simulate", lambda *a, **k: (equity, trades))
    base = resolve_params("absorption_reversal", {})
    return runner.select_parameters(
        validation, "absorption_reversal", base, 10000.0, 0.0004, 0.0002, "1m"
    )


def test_select_parameters_rejects_small_sample_candidates(monkeypatch):
    # 1000 根验证集 → 30 笔门槛；12 笔成交的小样本幸运儿必须被拒
    trades = [{"pnl": 5.0, "fees": 0.1}] * 12
    best, selection = _selection(monkeypatch, 1000, trades)
    assert selection["status"] == "insufficient_trades"
    assert selection["minimum_trades"] == 30
    assert all(not candidate["eligible"] for candidate in selection["candidates"])
    assert best == resolve_params("absorption_reversal", {})  # 保留当前参数


def test_select_parameters_accepts_large_sample_candidates(monkeypatch):
    trades = [{"pnl": 5.0, "fees": 0.1}] * 40  # 超过 30 笔门槛，正常参与选择
    best, selection = _selection(monkeypatch, 1000, trades)
    assert selection["status"] == "selected"
    assert selection["selected"] == "current"
    assert all(candidate["eligible"] for candidate in selection["candidates"])
    assert best["min_score"] == resolve_params("absorption_reversal", {})["min_score"]


def test_select_parameters_zero_trade_variants_do_not_error(monkeypatch):
    best, selection = _selection(monkeypatch, 1000, [])
    assert selection["status"] == "insufficient_trades"
    assert all(candidate["total_trades"] == 0 for candidate in selection["candidates"])
    assert best == resolve_params("absorption_reversal", {})
