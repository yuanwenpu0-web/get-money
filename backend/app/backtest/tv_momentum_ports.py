"""Causal ports of three Eterna momentum strategies' order conditions.

All three source scripts place entries only while flat and close the opposite
position on a signal; callers must honor that position-dependent rule. Their
Pine defaults are initial_capital=1000, percent_of_equity=100 and no commission.
The two sizing/leverage inputs in the scripts are alert fields, not order sizing.
This module emits conditions, not fills, broker emulation, alerts or plots.
"""
from __future__ import annotations

import math
from collections import deque

SOURCES = {
    "cci_strategy": "data/pine-crypto-collection/eterna/strategies/momentum/cci_strategy.pine",
    "awesome_oscillator": "data/pine-crypto-collection/eterna/strategies/momentum/awesome_oscillator.pine",
    "williams_r_reversal": "data/pine-crypto-collection/eterna/strategies/momentum/williams_r_reversal.pine",
}
DEFAULTS = {code: {"initial_capital": 1000.0, "position_pct": 1.0,
                   "commission_rate": 0.0, "entry_mode": "flat_only"}
            for code in SOURCES}


def _int(params, key, default, minimum, maximum=None):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{key} must be an integer in [{minimum}, {maximum if maximum is not None else '∞'}]")
    return value


def _sma(values, length):
    rolling, total, result = deque(), 0.0, []
    for value in values:
        rolling.append(value)
        total += value
        if len(rolling) > length:
            total -= rolling.popleft()
        result.append(total / length if len(rolling) == length else None)
    return result


def _cross_up(a, previous_a, b=0.0):
    return a is not None and previous_a is not None and round(a, 9) > round(b, 9) and round(previous_a, 9) <= round(b, 9)


def _cross_down(a, previous_a, b=0.0):
    return a is not None and previous_a is not None and round(a, 9) < round(b, 9) and round(previous_a, 9) >= round(b, 9)


def tradingview_signals(candles, code, params=None):
    """Generate per-bar signal conditions without lookahead.

    ``long``/``short`` are only eligible for orders while flat; ``close_long``
    and ``close_short`` are evaluated against the pre-fill position. Input bars
    must be in time order with gaps handled by the caller.
    """
    if code not in SOURCES:
        raise ValueError(f"unsupported TradingView strategy: {code}")
    params = dict(params or {})
    allowed = {
        "cci_strategy": {"cci_length", "overbought_level", "oversold_level", "extreme_ob", "extreme_os"},
        "awesome_oscillator": {"fast_length", "slow_length"},
        "williams_r_reversal": {"wr_length", "overbought_level", "oversold_level", "confirm_bars"},
    }[code]
    unknown = params.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")

    bars = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(x) and x > 0 for x in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        bars.append(bar)

    result = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    if code == "cci_strategy":
        length = _int(params, "cci_length", 20, 1)
        overbought = _int(params, "overbought_level", 100, 50)
        oversold = _int(params, "oversold_level", -100, -2**63, -50)
        extreme_ob = _int(params, "extreme_ob", 200, 150)
        extreme_os = _int(params, "extreme_os", -200, -2**63, -150)
        closes = [b["close"] for b in bars]
        cci = []
        for i in range(len(closes)):
            window = closes[max(0, i - length + 1):i + 1]
            if len(window) < length:
                cci.append(None)
                continue
            mean = sum(window) / length
            deviation = sum(abs(x - mean) for x in window) / length
            cci.append((closes[i] - mean) / (0.015 * deviation) if deviation else None)
        for i in range(1, len(cci)):
            cur, prev = cci[i], cci[i - 1]
            if cur is None or prev is None:
                continue
            rounded_cur, rounded_prev = round(cur, 9), round(prev, 9)
            buy = _cross_up(cur, prev, oversold) or (rounded_prev < extreme_os and rounded_cur > extreme_os and rounded_cur < oversold)
            sell = _cross_down(cur, prev, overbought) or (rounded_prev > extreme_ob and rounded_cur < extreme_ob and rounded_cur > overbought)
            result[i] = {"long": buy, "short": sell, "close_long": sell, "close_short": buy}
    elif code == "awesome_oscillator":
        fast = _int(params, "fast_length", 5, 1)
        slow = _int(params, "slow_length", 34, 1)
        median = [(b["high"] + b["low"]) / 2 for b in bars]
        fast_sma, slow_sma = _sma(median, fast), _sma(median, slow)
        ao = [a - b if a is not None and b is not None else None for a, b in zip(fast_sma, slow_sma)]
        for i in range(1, len(ao)):
            # The source's twin-peak and saucer clauses are contradictory:
            # each requires ao both above and below the same value. Therefore
            # only the source's zero-line crossover can produce an entry.
            buy, sell = _cross_up(ao[i], ao[i - 1]), _cross_down(ao[i], ao[i - 1])
            result[i] = {"long": buy, "short": sell, "close_long": sell, "close_short": buy}
    else:
        _int(params, "wr_length", 14, 1)
        _int(params, "overbought_level", -20, -50, 0)
        _int(params, "oversold_level", -80, -100, -50)
        _int(params, "confirm_bars", 2, 1, 5)
        # Pine's strongBullish requires crossover(wr, oversold) AND wr below
        # oversold on the same bar, which is impossible; strongBearish has the
        # analogous contradiction. No entry or close order can be generated.
    return result
