import math

import pytest

from app.backtest.tv_more_trend_ports import SOURCES, _donchian_midpoint, tradingview_signals
from app.backtest import tv_more_trend_ports


def candles(values):
    return [{"open": x, "high": x + 1, "low": x - 1, "close": x} for x in values]


VALUES = [103, 102, 107, 106, 111, 116, 121, 124, 119, 120, 117, 122, 117, 114, 109, 108, 109, 106, 107, 110, 105, 108, 105, 100, 105, 102, 103, 102, 99, 100, 97, 92, 89, 92, 95, 96, 93, 90, 85, 80]


def test_ichimoku_uses_displaced_cloud_and_unshifted_exit_cross():
    params = {"conversion_period": 2, "base_period": 4, "lagging_span2_period": 8, "displacement": 2}
    signals = tradingview_signals(candles(VALUES), "ichimoku_cloud", params)
    assert [i for i, s in enumerate(signals) if s["short"]] == [18, 21, 27, 37]
    shifted = tradingview_signals(candles(VALUES), "ichimoku_cloud", {**params, "displacement": 8})
    assert [i for i, s in enumerate(shifted) if s["short"]] == [21, 27, 37]
    assert [s["close_long"] for s in signals] == [s["close_long"] for s in shifted]
    reverse = tradingview_signals(candles([200-x for x in VALUES]), "ichimoku_cloud", params)
    assert [i for i, s in enumerate(reverse) if s["long"]] == [18, 21, 27, 37]
    assert reverse[18]["close_short"]
    assert _donchian_midpoint(candles([10, 12, 11]), 3) == [None, None, 11]


def test_chandelier_uses_atr_stop_cross_and_trend_filter_only_for_entry():
    params = {"atr_length": 2, "atr_multiplier": .5, "trend_ema": 6, "use_trend_filter": False}
    bars = candles(VALUES)
    raw = tradingview_signals(bars, "chandelier_exit", params)
    filtered = tradingview_signals(bars, "chandelier_exit", {**params, "use_trend_filter": True})
    assert not any(raw[0].values()) and not any(raw[1].values())
    assert raw[2]["long"] and raw[2]["close_short"]
    assert raw[3]["short"] and raw[3]["close_long"]
    assert not filtered[3]["short"] and filtered[3]["close_long"]
    assert [s["close_short"] for s in raw] == [s["close_short"] for s in filtered]


def test_ichimoku_cloud_direction_uses_pine_comparison_precision(monkeypatch):
    # A crossover with the displaced clouds different by less than one Pine
    # comparison unit must not count as a bullish cloud.
    lines = {2: [1., 2., 3.], 4: [2., 2., 2.], 8: [2., 2. - 0.0000000004, 2.]}
    monkeypatch.setattr(tv_more_trend_ports, "_donchian_midpoint", lambda bars, length: lines[length])
    params = {"conversion_period": 2, "base_period": 4, "lagging_span2_period": 8, "displacement": 1}
    assert not tradingview_signals(candles([10, 10, 10]), "ichimoku_cloud", params)[2]["long"]
    lines[8][1] = 2. - 0.00000001
    assert tradingview_signals(candles([10, 10, 10]), "ichimoku_cloud", params)[2]["long"]


def test_chandelier_trend_filter_uses_pine_comparison_precision(monkeypatch):
    # Force a stop crossover and isolate the original trend-filter comparison.
    monkeypatch.setattr(tv_more_trend_ports, "_cross", lambda *args: True)
    monkeypatch.setattr(tv_more_trend_ports, "_ema", lambda values, length: [10., 10. + 0.0000000004])
    bars = candles([10, 10])
    params = {"atr_length": 1, "atr_multiplier": .5, "trend_ema": 2, "use_trend_filter": True}
    assert not tradingview_signals(bars, "chandelier_exit", params)[1]["short"]
    monkeypatch.setattr(tv_more_trend_ports, "_ema", lambda values, length: [10., 10. + 0.00000001])
    assert tradingview_signals(bars, "chandelier_exit", params)[1]["short"]


@pytest.mark.parametrize("strategy", SOURCES)
def test_additional_bars_do_not_change_historical_decisions(strategy):
    data = candles([100 + i * .1 + 3 * math.sin(i / 4) for i in range(180)])
    assert tradingview_signals(data[:120], strategy) == tradingview_signals(data, strategy)[:120]


@pytest.mark.parametrize("strategy,params", [
    ("ichimoku_cloud", {"conversion_period": 0}),
    ("ichimoku_cloud", {"displacement": True}),
    ("ichimoku_cloud", {"lagging_span2_period": 0}),
    ("chandelier_exit", {"atr_multiplier": .4}),
    ("chandelier_exit", {"use_trend_filter": 1}),
    ("chandelier_exit", {"trend_ema": float("nan")}),
    ("chandelier_exit", {"unknown": 1}),
])
def test_params_checked_even_when_empty(strategy, params):
    with pytest.raises(ValueError):
        tradingview_signals([], strategy, params)
