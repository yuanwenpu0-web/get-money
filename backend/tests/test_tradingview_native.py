import pytest

from app.backtest.tradingview_native import _cross_up, _cross_down, run_tradingview_backtest, tradingview_signals


def test_pine_nine_decimal_comparisons_for_cross_and_donchian_breakout():
    assert not _cross_up(1.0000000004, 1.0, .9, 1.0)
    assert _cross_up(1.0000000006, 1.0, 1.0000000004, 1.0)
    assert not _cross_down(.9999999996, 1.0, 1.1, 1.0)
    assert _cross_down(.9999999994, 1.0, .9999999996, 1.0)
    def tiny(last):
        return [{"open": x, "high": x, "low": x, "close": x} for x in (1.0, 1.0, last)]
    assert not tradingview_signals(tiny(1.0000000004), "donchian_channel", {"channel_length": 1})[-1]["long"]
    assert tradingview_signals(tiny(1.0000000006), "donchian_channel", {"channel_length": 1})[-1]["long"]


def bars(closes, opens=None):
    opens = closes if opens is None else opens
    return [{"time": i, "open": o, "high": max(c, o) + 0.5,
             "low": min(c, o) - 0.5, "close": c}
            for i, (c, o) in enumerate(zip(closes, opens))]


def test_sma_cross_fills_next_open_reverses_and_charges_each_side():
    candles = bars([3, 2, 1, 2, 3, 2, 1, 2], [3, 2, 1, 2, 4, 2, 2, 2])
    result = run_tradingview_backtest(candles, "sma_crossover",
                                      {"fast_length": 1, "slow_length": 2},
                                      initial_capital=1000, position_pct=0.1,
                                      commission_rate=0.001)
    assert result["trades"][0]["entry_time"] == 4
    assert result["trades"][0]["entry_price"] == 4
    assert result["trades"][0]["exit_time"] == 6
    assert result["trades"][0]["exit_price"] == 2
    # Order quantity uses the signal-bar close of 2, not the next open of 4.
    assert result["trades"][0]["quantity"] == pytest.approx(50)
    assert result["trades"][0]["pnl"] == pytest.approx(-100.3)
    assert result["open_position"]["direction"] == -1
    assert result["open_position"]["quantity"] == pytest.approx(44.99)
    assert result["final_equity"] == pytest.approx(899.61002)
    # Last-bar signal is not filled at a nonexistent next open.
    assert result["unfilled_final_orders"] == ["long"]


def test_donchian_uses_shifted_channel_and_requires_new_breakout():
    candles = bars([10, 10, 10, 12, 13, 9, 8, 11])
    signals = tradingview_signals(candles, "donchian_channel", {"channel_length": 2})
    assert [i for i, x in enumerate(signals) if x["long"]] == [3, 7]
    assert [i for i, x in enumerate(signals) if x["short"]] == [5]
    assert signals[5]["close_long"]
    result = run_tradingview_backtest(candles, "donchian_channel", {"channel_length": 2})
    assert result["trades"][0]["entry_time"] == 4
    assert result["trades"][0]["exit_time"] == 6
    assert result["open_position"]["direction"] == -1


def test_donchian_zero_confirmation_and_invalid_parameters():
    candles = bars([10, 10, 10, 12, 13, 9])
    signals = tradingview_signals(candles, "donchian_channel", {"channel_length": 2, "breakout_confirm": 0})
    assert not any(s["long"] or s["short"] for s in signals)
    with pytest.raises(ValueError):
        tradingview_signals(candles, "sma_crossover", {"fast_length": 0})
    with pytest.raises(ValueError):
        tradingview_signals(candles, "donchian_channel", {"breakout_confirm": 6})
    with pytest.raises(ValueError):
        tradingview_signals(candles, "sma_crossover", {"typo": 2})


def test_donchian_middle_cross_closes_without_reversal():
    candles = bars([10, 10, 10, 12, 13, 11, 10])
    candles[4]["low"] = 9
    signals = tradingview_signals(candles, "donchian_channel", {"channel_length": 2})
    assert signals[5]["close_long"] and not signals[5]["short"]
    result = run_tradingview_backtest(candles, "donchian_channel", {"channel_length": 2})
    assert result["trades"][0]["entry_time"] == 4
    assert result["trades"][0]["exit_time"] == 6
    assert result["open_position"] is None


def test_negative_equity_entry_fails_instead_of_silently_skipping():
    candles = bars([3, 2, 1, 2, 3, 2, 1], [3, 2, 1, 2, 100, 2, 1])
    with pytest.raises(ValueError, match="negative equity"):
        run_tradingview_backtest(candles, "sma_crossover", {"fast_length": 1, "slow_length": 2},
                                 initial_capital=1000, position_pct=1, commission_rate=0)
