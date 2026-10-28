import pytest

from app.backtest.tv_external_ports import DEFAULTS, SOURCES, tradingview_signals


def candles(times):
    prices = [10, 10, 12, 10]
    return [{"time": t, "open": p, "high": p, "low": p, "close": p} for t, p in zip(times, prices)]


def test_bb_ema_follows_market_entry_exit_and_does_not_open_short():
    bars = candles(["2020-01-01T00:00:00+00:00", "2020-01-01T04:00:00+00:00",
                    "2020-01-01T08:00:00+00:00", "2020-01-01T12:00:00+00:00"])
    signals = tradingview_signals(bars, "bb_ema_eth", {"ema_length": 3, "bb_length": 2, "bb_multiplier": 0.5})
    assert signals[2] == {"long": True, "short": False, "close_long": False, "close_short": False}
    assert signals[3] == {"long": False, "short": False, "close_long": True, "close_short": False}
    assert DEFAULTS["bb_ema_eth"] == {"initial_capital": 1000.0, "position_pct": 1.0, "commission_rate": 0.0}
    assert SOURCES["bb_ema_eth"].endswith("BB_EMA_ETHUSDT.pine")


def test_bb_ema_keeps_default_date_range_strict_at_start_and_end():
    params = {"ema_length": 3, "bb_length": 2, "bb_multiplier": 0.5}
    after_window = candles(["2026-01-01T00:00:00+00:00", "2026-01-01T04:00:00+00:00",
                            "2026-01-01T08:00:00+00:00", "2026-01-01T12:00:00+00:00"])
    assert not any(any(s.values()) for s in tradingview_signals(after_window, "bb_ema_eth", params))
    assert tradingview_signals(after_window, "bb_ema_eth", {**params, "segment": "All"})[2]["long"]
    start = candles(["2017-12-31T16:00:00+00:00", "2017-12-31T20:00:00+00:00",
                     "2018-01-01T00:00:00+00:00", "2018-01-01T04:00:00+00:00"])
    assert not tradingview_signals(start, "bb_ema_eth", params)[2]["long"]
    end = candles(["2022-12-30T16:00:00+00:00", "2022-12-30T20:00:00+00:00",
                   "2022-12-31T00:00:00+00:00", "2022-12-31T04:00:00+00:00"])
    assert not tradingview_signals(end, "bb_ema_eth", params)[2]["long"]


def test_bb_ema_fails_closed_for_missing_or_naive_time_and_bad_parameters():
    bar = {"open": 1, "high": 1, "low": 1, "close": 1}
    with pytest.raises(ValueError, match="time is required"):
        tradingview_signals([bar], "bb_ema_eth")
    with pytest.raises(ValueError, match="time zone"):
        tradingview_signals([{**bar, "time": "2020-01-01"}], "bb_ema_eth")
    with pytest.raises(ValueError, match="segment"):
        tradingview_signals([], "bb_ema_eth", {"segment": "Current"})


def test_bb_ema_does_not_trigger_on_sub_nine_decimal_cross():
    prices = [1.0, 1.0, 1.0000000004]
    bars = [{"time": 1_600_000_000_000 + i * 60_000, "open": p, "high": p, "low": p, "close": p}
            for i, p in enumerate(prices)]
    params = {"ema_length": 3, "bb_length": 2, "bb_multiplier": .5, "segment": "All"}
    assert not tradingview_signals(bars, "bb_ema_eth", params)[-1]["long"]
