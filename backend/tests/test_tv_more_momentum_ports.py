import math
import random

import pytest

from app.backtest.tv_more_momentum_ports import DEFAULTS, SOURCES, _cross, tradingview_signals


def _candles(prices):
    return [{"open": p, "high": p + 0.1, "low": p - 0.1, "close": p} for p in prices]


def _reference_prices(n):
    random_source = random.Random(6213)
    prices = [100.0]
    for i in range(1, n):
        prices.append(max(20, prices[-1] + random_source.uniform(-1.5, 1.5) + 0.1 * math.sin(i / 15)))
    return prices


def test_stoch_rsi_source_signals_on_reference_ohlc():
    # Signal bar indices were independently compared with execution of the
    # unchanged source in PineTS; that engine is not a TradingView oracle.
    result = tradingview_signals(_candles(_reference_prices(200)), "stoch_rsi_crossover")
    assert [i for i, row in enumerate(result) if row["long"]] == [22, 42, 49, 71, 76, 132, 161, 167, 187]
    assert [i for i, row in enumerate(result) if row["short"]] == [32, 55, 92, 106, 138, 140, 151, 195]
    assert all(row["close_long"] == row["short"] and row["close_short"] == row["long"] for row in result)


def test_stoch_rsi_constant_data_has_no_valid_crossover_or_lookahead():
    constant = _candles([100] * 80)
    assert all(not any(row.values()) for row in tradingview_signals(constant, "stoch_rsi_crossover"))
    prices = _reference_prices(250)
    prefix = tradingview_signals(_candles(prices[:170]), "stoch_rsi_crossover")
    assert prefix == tradingview_signals(_candles(prices), "stoch_rsi_crossover")[:170]


def test_stoch_rsi_pine_comparison_precision():
    assert not _cross(50.0000000004, 50, 49, 50, True)
    assert _cross(50.0000000006, 50, 49, 50, True)
    assert not _cross(49.9999999996, 50, 51, 50, False)
    assert _cross(49.9999999994, 50, 51, 50, False)


def test_stoch_rsi_parameter_validation_and_source_defaults():
    assert list(SOURCES) == ["stoch_rsi_crossover"]
    assert DEFAULTS["stoch_rsi_crossover"] == {
        "initial_capital": 1000.0, "position_pct": 1.0,
        "commission_rate": 0.0, "entry_mode": "flat_only",
    }
    for params in ({"rsi_length": 0}, {"smooth_k": True}, {"lower_level": 51}, {"upper_level": 49}, {"x": 1}):
        with pytest.raises(ValueError):
            tradingview_signals([], "stoch_rsi_crossover", params)
    with pytest.raises(ValueError):
        tradingview_signals([{"open": 2, "close": 2, "high": 1, "low": 1}], "stoch_rsi_crossover")
