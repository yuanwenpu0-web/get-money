import math

import pytest

from app.backtest.tv_trend_ports import SOURCES, _atr, _ema, _linear_regression, tradingview_signals
from app.backtest.pine_compare import gt, ge, lt, le


def test_trend_alignment_and_thresholds_use_pine_nine_decimal_comparison():
    assert not gt(1.0000000004, 1.0)
    assert ge(1.0000000004, 1.0)
    assert not lt(.9999999996, 1.0)
    assert le(.9999999996, 1.0)
    assert not gt(None, 1.0)
    assert not le(1.0, None)
    bars = [{"open": p, "high": p, "low": p, "close": p} for p in (1.0, 1.0000000004)]
    result = tradingview_signals(bars, "ema_ribbon", {"ema_fast": 2, "ema_mid": 3, "ema_slow": 5})
    assert not result[-1]["long"]


def candles(values):
    return [{"open": x, "high": x + 1, "low": x - 1, "close": x} for x in values]


def test_pine_ema_ribbon_alignment_crosses_and_exits():
    values = [10, 11, 12, 13, 12, 11, 13, 14, 12, 10, 9, 11, 13, 14, 12, 15]
    bars = candles(values)
    signals = tradingview_signals(bars, "ema_ribbon", {"ema_fast": 2, "ema_mid": 3, "ema_slow": 5})
    assert [i for i, s in enumerate(signals) if s["long"]] == [1, 6, 15]
    assert [i for i, s in enumerate(signals) if s["close_long"]] == [5, 8]
    assert [i for i, s in enumerate(signals) if s["close_short"]] == [1, 6, 12]
    inverted = tradingview_signals(candles([100 - x for x in values]), "ema_ribbon", {"ema_fast": 2, "ema_mid": 3, "ema_slow": 5})
    assert [i for i, s in enumerate(inverted) if s["short"]] == [1, 6, 15]
    assert [i for i, s in enumerate(inverted) if s["close_short"]] == [5, 8]
    assert _ema([10, 12, 11], 2) == pytest.approx([10, 11.333333333333334, 11.11111111111111])


def test_keltner_rma_warmup_and_long_short_bounces():
    long_bars = candles([100, 100, 90, 100])
    long_bars[-1]["low"] = 95
    short_bars = candles([100, 100, 110, 100])
    short_bars[-1]["high"] = 105
    params = {"ema_length": 3, "atr_length": 1, "atr_multiplier": .2, "trend_filter": 20}
    assert tradingview_signals(long_bars, "keltner_channel", params)[-1]["long"]
    assert tradingview_signals(short_bars, "keltner_channel", params)[-1]["short"]
    # First true range uses high-low, and the RMA seed is an SMA, not EMA.
    assert _atr(long_bars, 2) == pytest.approx([None, 2, 6.5, 8.75], nan_ok=True)
    assert not any(s["long"] or s["short"] for s in tradingview_signals(long_bars[:2], "keltner_channel", {**params, "atr_length": 3}))


def test_regression_channel_uses_current_window_and_causal_slope():
    values = [10, 11, 12, 13, 12, 11, 13, 14, 12, 10, 9, 11, 13, 14, 12, 15]
    bars = candles(values)
    params = {"length_lr": 10, "deviation_multiplier": .5, "trend_ema": 5}
    signals = tradingview_signals(bars, "linear_regression_channel", params)
    assert [i for i, s in enumerate(signals) if s["short"]] == [14]
    assert signals[14]["close_long"]
    assert not any(s["long"] or s["short"] for s in signals[:14])
    inverted = tradingview_signals(candles([100 - x for x in values]), "linear_regression_channel", params)
    assert [i for i, s in enumerate(inverted) if s["long"]] == [14]
    assert inverted[14]["close_short"]
    line, dev = _linear_regression(list(range(1, 11)), 10)
    assert line[-1] == pytest.approx(10)
    assert dev[-1] == pytest.approx(0)


@pytest.mark.parametrize("strategy", SOURCES)
def test_signals_do_not_rewrite_history_when_future_bars_arrive(strategy):
    data = candles([100 + i * .1 + 3 * math.sin(i / 4) for i in range(160)])
    prefix = tradingview_signals(data[:120], strategy)
    assert tradingview_signals(data, strategy)[:120] == prefix


@pytest.mark.parametrize("strategy,params", [
    ("ema_ribbon", {"ema_fast": 0}),
    ("ema_ribbon", {"ema_mid": True}),
    ("keltner_channel", {"atr_multiplier": .01}),
    ("keltner_channel", {"atr_length": 0}),
    ("linear_regression_channel", {"length_lr": 9}),
    ("linear_regression_channel", {"deviation_multiplier": float("nan")}),
    ("ema_ribbon", {"unknown": 1}),
])
def test_invalid_params_fail_even_with_no_candles(strategy, params):
    with pytest.raises(ValueError):
        tradingview_signals([], strategy, params)
