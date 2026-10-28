"""Closed-bar signal ports of three Eterna Pine v5 market-order strategies.

Only the Pine entry and close conditions are computed here. The caller is
responsible for chronological, gap-free OHLC candles, Pine order sizing and
broker execution. These functions do not simulate intrabar fills, margin calls,
symbol lot sizes or real exchange execution.
"""
from __future__ import annotations

import math
from numbers import Real
from .pine_compare import gt, ge, lt, le

SOURCES = {
    "ema_ribbon": "data/pine-crypto-collection/eterna/strategies/trend-following/ema-ribbon/strategy.pine",
    "keltner_channel": "data/pine-crypto-collection/eterna/strategies/trend-following/keltner-channel/strategy.pine",
    "linear_regression_channel": "data/pine-crypto-collection/eterna/strategies/trend-following/linear-regression-channel/strategy.pine",
}


def _int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def _ema(values, length):
    alpha = 2.0 / (length + 1.0)
    result = []
    for value in values:
        result.append(value if not result else alpha * value + (1 - alpha) * result[-1])
    return result


def _atr(bars, length):
    # ta.tr(true) uses high-low on the first bar; ta.rma seeds with an SMA.
    trs = []
    for i, bar in enumerate(bars):
        high, low = bar["high"], bar["low"]
        previous = bars[i - 1]["close"] if i else None
        trs.append(high - low if previous is None else max(high - low, abs(high - previous), abs(low - previous)))
    result = [None] * len(trs)
    if len(trs) >= length:
        result[length - 1] = sum(trs[:length]) / length
        for i in range(length, len(trs)):
            result[i] = (trs[i] + (length - 1) * result[i - 1]) / length
    return result


def _cross(a, b, old_a, old_b, up):
    if any(value is None for value in (a, b, old_a, old_b)):
        return False
    return (gt(a, b) and le(old_a, old_b)) if up else (lt(a, b) and ge(old_a, old_b))


def _linear_regression(values, length):
    # Pine linreg = intercept + slope * (length - 1 - offset).
    # calcDev() uses the *same* current window for each offset, not linreg[i].
    result, dev = [None] * len(values), [None] * len(values)
    xmean = (length - 1) / 2
    denom = length * (length * length - 1) / 12
    for i in range(length - 1, len(values)):
        window = values[i - length + 1:i + 1]
        mean = sum(window) / length
        slope = sum((x - xmean) * (value - mean) for x, value in enumerate(window)) / denom
        intercept = mean - slope * xmean
        result[i] = intercept + slope * (length - 1)
        dev[i] = math.sqrt(sum((value - (intercept + slope * x)) ** 2 for x, value in enumerate(window)) / length)
    return result, dev


def tradingview_signals(candles, strategy, params=None):
    """Calculate entry and close booleans from completed candles without lookahead."""
    allowed = {
        "ema_ribbon": {"ema_fast", "ema_mid", "ema_slow"},
        "keltner_channel": {"ema_length", "atr_length", "atr_multiplier", "trend_filter"},
        "linear_regression_channel": {"length_lr", "deviation_multiplier", "trend_ema"},
    }
    if strategy not in allowed:
        raise ValueError(f"unsupported TradingView strategy: {strategy}")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    params = dict(params or {})
    unknown = params.keys() - allowed[strategy]
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    if strategy == "ema_ribbon":
        fast = _int(params.get("ema_fast", 9), "ema_fast")
        mid = _int(params.get("ema_mid", 21), "ema_mid")
        slow = _int(params.get("ema_slow", 55), "ema_slow")
    elif strategy == "keltner_channel":
        ema_length = _int(params.get("ema_length", 20), "ema_length")
        atr_length = _int(params.get("atr_length", 10), "atr_length")
        atr_multiplier = _number(params.get("atr_multiplier", 2.0), "atr_multiplier", 0.1)
        trend_filter = _int(params.get("trend_filter", 50), "trend_filter")
    else:
        length_lr = _int(params.get("length_lr", 100), "length_lr", 10)
        deviation_multiplier = _number(params.get("deviation_multiplier", 2.0), "deviation_multiplier", 0.5)
        trend_ema = _int(params.get("trend_ema", 50), "trend_ema")
    bars = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(x) and x > 0 for x in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        bars.append(bar)
    closes = [bar["close"] for bar in bars]
    result = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    if strategy == "ema_ribbon":
        f, m, s = _ema(closes, fast), _ema(closes, mid), _ema(closes, slow)
        for i in range(1, len(bars)):
            result[i]["long"] = gt(f[i], m[i]) and gt(m[i], s[i]) and _cross(closes[i], f[i], closes[i-1], f[i-1], True)
            result[i]["short"] = lt(f[i], m[i]) and lt(m[i], s[i]) and _cross(closes[i], f[i], closes[i-1], f[i-1], False)
            result[i]["close_long"] = _cross(f[i], m[i], f[i-1], m[i-1], False)
            result[i]["close_short"] = _cross(f[i], m[i], f[i-1], m[i-1], True)
    elif strategy == "keltner_channel":
        basis, atr, trend = _ema(closes, ema_length), _atr(bars, atr_length), _ema(closes, trend_filter)
        upper = [b + atr_multiplier * a if a is not None else None for b, a in zip(basis, atr)]
        lower = [b - atr_multiplier * a if a is not None else None for b, a in zip(basis, atr)]
        for i in range(1, len(bars)):
            result[i]["long"] = gt(closes[i], trend[i]) and _cross(closes[i], lower[i], closes[i-1], lower[i-1], True) and le(bars[i]["low"], lower[i])
            result[i]["short"] = lt(closes[i], trend[i]) and _cross(closes[i], upper[i], closes[i-1], upper[i-1], False) and ge(bars[i]["high"], upper[i])
            result[i]["close_long"] = _cross(closes[i], basis[i], closes[i-1], basis[i-1], False)
            result[i]["close_short"] = _cross(closes[i], basis[i], closes[i-1], basis[i-1], True)
    else:
        regression, dev = _linear_regression(closes, length_lr)
        trend = _ema(closes, trend_ema)
        upper = [b + deviation_multiplier * d if b is not None else None for b, d in zip(regression, dev)]
        lower = [b - deviation_multiplier * d if b is not None else None for b, d in zip(regression, dev)]
        for i in range(1, len(bars)):
            uptrend = i >= 5 and gt(closes[i], trend[i]) and gt(regression[i], regression[i-5])
            downtrend = i >= 5 and lt(closes[i], trend[i]) and lt(regression[i], regression[i-5])
            result[i]["long"] = uptrend and _cross(closes[i], lower[i], closes[i-1], lower[i-1], True) and le(bars[i]["low"], lower[i])
            result[i]["short"] = downtrend and _cross(closes[i], upper[i], closes[i-1], upper[i-1], False) and ge(bars[i]["high"], upper[i])
            result[i]["close_long"] = _cross(closes[i], regression[i], closes[i-1], regression[i-1], False)
            result[i]["close_short"] = _cross(closes[i], regression[i], closes[i-1], regression[i-1], True)
    return result
