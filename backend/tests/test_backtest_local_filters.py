import numpy as np
import pandas as pd
import pytest

from app.backtest import runner
from app.backtest.execution import randomize_directions, simulate
from app.backtest.tradingview_native import tradingview_signals
from app.features.indicators import build_features
from app.ml import dataset
from app.strategy.rules import resolve_params


def prices(n=1200):
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 + 0.8 * np.sin(np.arange(n) / 15)
    return pd.DataFrame(dict(open=close, high=close + 0.1, low=close - 0.1,
                             close=close, volume=100.0), index=index)


def test_filter_blocks_entry_but_not_reverse_exit_and_applies_to_random_direction():
    data = prices(5).assign(atr_pct=0.002)
    signals = pd.DataFrame(dict(direction=[1, -1, 0, 0, 0], position_pct=0.1,
                                leverage=2, sl_pct=0.1, tp_pct=0.1,
                                allow_long_entry=[True, False, False, False, False],
                                allow_short_entry=[False] * 5), index=data.index)
    params = resolve_params("trend_follow", {}, {"min_hold_bars": 0, "trailing_enabled": False})
    _, trades = simulate(data, signals, params, 10000, 0, 0, "1m")
    assert len(trades) == 1
    assert trades[0]["status"] == "signal_reverse"
    assert trades[0]["exit_time"] == str(data.index[2])
    # Seed 1 flips the first long into a short; the original short permission
    # must still prohibit that randomized entry.
    flipped = randomize_directions(signals, seed=1)
    assert flipped.direction.iloc[0] == -1
    _, trades = simulate(data, flipped, params, 10000, 0, 0, "1m")
    assert trades == []


def test_dataset_filter_uses_raw_contiguous_candles_and_is_causal(monkeypatch):
    frame = prices(500).drop(prices(500).index[230:240])
    as_of = int(frame.index[-1].timestamp() * 1000) + 60_000
    monkeypatch.setattr(dataset, "now_ms", lambda: as_of)
    monkeypatch.setattr(dataset.MarketDataService, "load_frame", lambda *a, **kw: frame.copy())
    filters = [{"code": "sma_crossover", "params": {"fast_length": 5, "slow_length": 25}}]
    result = dataset.build_dataset(None, "BTCUSDT", "1m", entry_filters=filters)
    for segment in (frame.iloc[:230], frame.iloc[230:]):
        raw = tradingview_signals(segment[["open", "high", "low", "close"]].to_dict("records"),
                                  "sma_crossover", filters[0]["params"])
        expected = pd.Series([s["long"] for s in raw], index=segment.index)
        common = result.index.intersection(segment.index)
        assert len(common) > 0
        pd.testing.assert_series_equal(result.loc[common, "allow_long_entry"], expected.loc[common],
                                       check_names=False)
    original = result.loc[result.index < frame.index[-100], ["allow_long_entry", "allow_short_entry"]]
    frame.iloc[-100:, frame.columns.get_indexer(["open", "high", "low", "close"])] *= 2
    changed = dataset.build_dataset(None, "BTCUSDT", "1m", entry_filters=filters)
    pd.testing.assert_frame_equal(original, changed.loc[original.index, original.columns])


def test_validation_baseline_and_random_paths_receive_filter_permissions(db, monkeypatch):
    data = build_features(prices(), with_label=True).dropna()
    data["allow_long_entry"] = False
    data["allow_short_entry"] = False
    selected = [{"code": "sma_crossover", "params": {}, "enabled": True,
                 "bound_strategies": ["trend_follow"]}]
    monkeypatch.setattr(runner, "list_filters", lambda db: selected)
    def fake_data(*args, **kwargs):
        assert kwargs["entry_filters"] == selected
        return data
    monkeypatch.setattr(runner, "build_dataset", fake_data)
    result = runner.run_backtest(db, "BTCUSDT", "1m", "trend_follow", algo="rule", optimize=True)
    assert result["metrics"]["local_entry_filters"] == [{"code": "sma_crossover", "params": {}}]
    assert result["metrics"]["total_trades"] == 0
    assert result["metrics"]["baseline"]["total_trades"] == 0
    assert result["metrics"]["random_baseline"]["total_trades"] == 0
    assert result["metrics"]["significance"]["random_distribution"]["max_pct"] == 0
    assert all(candidate["total_trades"] == 0 for candidate in result["metrics"]["validation"]["candidates"])
