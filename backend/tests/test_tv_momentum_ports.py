import pytest

from app.backtest.tv_momentum_ports import DEFAULTS, SOURCES, _cross_down, _cross_up, tradingview_signals


def bars(prices):
    return [{"open": p, "high": p + 1, "low": p - 1, "close": p} for p in prices]


def test_cci_uses_close_and_signal_crosses_with_opposite_close():
    prices = [100, 102, 101, 98, 99, 102, 101, 98, 97, 101, 103, 99]
    result = tradingview_signals(bars(prices), "cci_strategy", {
        "cci_length": 3, "oversold_level": -50, "overbought_level": 50,
    })
    # First usable CCI is at index 2. At 3 it is -100; at 4 it is -20,
    # crossing -50. At 6 it drops from +100 to +25, crossing +50.
    assert [i for i, s in enumerate(result) if s["long"]] == [4, 9]
    assert [i for i, s in enumerate(result) if s["short"]] == [6, 11]
    assert all(s["close_short"] == s["long"] and s["close_long"] == s["short"] for s in result)
    assert not any(result[i]["long"] or result[i]["short"] for i in range(3))


def test_awesome_zero_cross_and_source_contradictory_patterns():
    prices = [100, 102, 101, 98, 99, 102, 101, 98, 97, 101, 103, 99]
    result = tradingview_signals(bars(prices), "awesome_oscillator", {"fast_length": 1, "slow_length": 3})
    # AO is the median price less its 3-bar average. A cross requires two
    # available AO values, not just the first valid oscillator value.
    assert [i for i, s in enumerate(result) if s["long"]] == [5, 9]
    assert [i for i, s in enumerate(result) if s["short"]] == [3, 7, 11]
    assert all(s["close_short"] == s["long"] and s["close_long"] == s["short"] for s in result)


def test_cci_extreme_reversal_can_fire_before_standard_threshold_cross():
    # After an extreme move below -200, recovery to roughly -155 remains
    # below the ordinary -100 threshold but satisfies the source's second path.
    long = tradingview_signals(bars([110] * 19 + [90, 103]), "cci_strategy")[-1]
    short = tradingview_signals(bars([110] * 19 + [130, 117]), "cci_strategy")[-1]
    assert long == {"long": True, "short": False, "close_long": False, "close_short": True}
    assert short == {"long": False, "short": True, "close_long": True, "close_short": False}


def test_williams_source_contradiction_never_enters_or_exits():
    prices = [100 + (i % 11) * (-1 if i % 3 == 0 else 1) for i in range(100)]
    result = tradingview_signals(bars(prices), "williams_r_reversal", {"wr_length": 2})
    assert all(not any(row.values()) for row in result)


@pytest.mark.parametrize("code", SOURCES)
def test_prefix_is_causal_and_defaults_match_strategy(code):
    prices = [100 + (i % 13) * (-1 if i % 4 == 0 else 1) for i in range(100)]
    candles = bars(prices)
    assert tradingview_signals(candles, code)[:75] == tradingview_signals(candles[:75], code)
    assert DEFAULTS[code] == {"initial_capital": 1000.0, "position_pct": 1.0,
                              "commission_rate": 0.0, "entry_mode": "flat_only"}


def test_validates_source_parameters_and_prices_even_on_empty_input():
    with pytest.raises(ValueError):
        tradingview_signals([], "cci_strategy", {"cci_length": 0})
    with pytest.raises(ValueError):
        tradingview_signals([], "williams_r_reversal", {"confirm_bars": 6})
    with pytest.raises(ValueError):
        tradingview_signals([], "awesome_oscillator", {"fast_length": True})
    with pytest.raises(ValueError):
        tradingview_signals([], "awesome_oscillator", {"unknown": 2})
    with pytest.raises(ValueError):
        tradingview_signals([{"open": 2, "high": 1, "low": 1, "close": 2}], "cci_strategy")


def test_pine_float_comparisons_use_nine_fractional_digits():
    assert not _cross_up(0.0000000004, 0)
    assert not _cross_down(-0.0000000004, 0)
    assert _cross_up(0.0000000006, 0)
    assert _cross_down(-0.0000000006, 0)
