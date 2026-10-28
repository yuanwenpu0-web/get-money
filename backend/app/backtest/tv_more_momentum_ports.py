"""Further source-faithful Eterna momentum signal conditions.

These strategies submit market entries only when flat; an opposite signal closes
the open position. Returned conditions are not fills, alerts or chart graphics.
Input candles must already be in order and callers must handle data gaps.
"""
from __future__ import annotations

import math
from collections import deque

SOURCES = {
    "stoch_rsi_crossover": "data/pine-crypto-collection/eterna/strategies/momentum/stoch_rsi_crossover.pine",
}
DEFAULTS = {code: {"initial_capital": 1000.0, "position_pct": 1.0,
                   "commission_rate": 0.0, "entry_mode": "flat_only"}
            for code in SOURCES}


def _int(params, key, default, minimum, maximum=None):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {key}")
    return value


def _sma_non_na(values, length):
    queue, total, out = deque(), 0.0, []
    for value in values:
        if value is not None:
            queue.append(value)
            total += value
            if len(queue) > length:
                total -= queue.popleft()
        out.append(total / length if len(queue) == length else None)
    return out


def _rsi(prices, length):
    gains, losses, avg_gain, avg_loss, out = [], [], None, None, [None]
    for prev, current in zip(prices, prices[1:]):
        diff = current - prev
        gain, loss = max(diff, 0), max(-diff, 0)
        if avg_gain is None:
            gains.append(gain)
            losses.append(loss)
            if len(gains) == length:
                avg_gain, avg_loss = sum(gains) / length, sum(losses) / length
        else:
            avg_gain = (avg_gain * (length - 1) + gain) / length
            avg_loss = (avg_loss * (length - 1) + loss) / length
        if avg_gain is None or (avg_gain == 0 and avg_loss == 0):
            out.append(None)
        elif avg_loss == 0:
            out.append(100.0)
        elif avg_gain == 0:
            out.append(0.0)
        else:
            out.append(100 - 100 / (1 + avg_gain / avg_loss))
    return out if prices else []


def _cross(a, b, old_a, old_b, up):
    if any(x is None for x in (a, b, old_a, old_b)):
        return False
    # Pine comparisons round floating operands to nine fractional digits.
    a, b, old_a, old_b = (round(x, 9) for x in (a, b, old_a, old_b))
    return (a > b and old_a <= old_b) if up else (a < b and old_a >= old_b)


def tradingview_signals(candles, code, params=None):
    if code not in SOURCES:
        raise ValueError(f"unsupported TradingView strategy: {code}")
    params = dict(params or {})
    allowed = {"rsi_length", "stoch_length", "smooth_k", "smooth_d", "upper_level", "lower_level"}
    if params.keys() - allowed:
        raise ValueError(f"unsupported parameters: {sorted(params.keys() - allowed)}")
    closes = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(v) and v > 0 for v in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        closes.append(bar["close"])
    result = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in closes]
    rsi_length = _int(params, "rsi_length", 14, 1)
    stoch_length = _int(params, "stoch_length", 14, 1)
    smooth_k = _int(params, "smooth_k", 3, 1)
    smooth_d = _int(params, "smooth_d", 3, 1)
    upper = _int(params, "upper_level", 80, 50, 100)
    lower = _int(params, "lower_level", 20, 0, 50)
    rsi = _rsi(closes, rsi_length)
    stoch = []
    for i, value in enumerate(rsi):
        window = [v for v in rsi[max(0, i - stoch_length + 1):i + 1] if v is not None]
        if value is None or not window or max(window) == min(window):
            stoch.append(None)
        else:
            stoch.append(100 * (value - min(window)) / (max(window) - min(window)))
    k = _sma_non_na(stoch, smooth_k)
    d = _sma_non_na(k, smooth_d)
    for i in range(1, len(closes)):
        buy = bool(_cross(k[i], d[i], k[i - 1], d[i - 1], True) and round(k[i], 9) < lower)
        sell = bool(_cross(k[i], d[i], k[i - 1], d[i - 1], False) and round(k[i], 9) > upper)
        result[i] = {"long": buy, "short": sell, "close_long": sell, "close_short": buy}
    return result
