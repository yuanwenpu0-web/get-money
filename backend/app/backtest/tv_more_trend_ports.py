"""Closed-bar Eterna trend signal ports for the research market-order engine.

Signals preserve Pine entry and strategy.close conditions. Execution, equity
sizing and commission belong to the caller. Missing bars, intrabar execution,
market hours, lot sizes and margin calls are not simulated here.
"""
from __future__ import annotations

import math

from .tv_trend_ports import _atr, _cross, _ema, _int, _number
from .pine_compare import gt, lt

SOURCES = {
    "ichimoku_cloud": "data/pine-crypto-collection/eterna/strategies/trend-following/ichimoku-cloud/strategy.pine",
    "chandelier_exit": "data/pine-crypto-collection/eterna/strategies/trend-following/chandelier-exit/strategy.pine",
}
# Both source files use 10000 initial capital, 10% equity and 0.075% commission.
DEFAULTS = {}


def _donchian_midpoint(bars, length):
    out = [None] * len(bars)
    for i in range(length - 1, len(bars)):
        window = bars[i-length+1:i+1]
        out[i] = (min(bar["low"] for bar in window) + max(bar["high"] for bar in window)) / 2
    return out


def tradingview_signals(candles, strategy, params=None):
    """Return one causal set of original entry and close signals per candle."""
    allowed = {
        "ichimoku_cloud": {"conversion_period", "base_period", "lagging_span2_period", "displacement"},
        "chandelier_exit": {"atr_length", "atr_multiplier", "trend_ema", "use_trend_filter"},
    }
    if strategy not in allowed:
        raise ValueError(f"unsupported TradingView strategy: {strategy}")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    params = dict(params or {})
    unknown = params.keys() - allowed[strategy]
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    if strategy == "ichimoku_cloud":
        conversion = _int(params.get("conversion_period", 9), "conversion_period")
        base = _int(params.get("base_period", 26), "base_period")
        span2 = _int(params.get("lagging_span2_period", 52), "lagging_span2_period")
        displacement = _int(params.get("displacement", 26), "displacement")
    else:
        length = _int(params.get("atr_length", 22), "atr_length")
        multiplier = _number(params.get("atr_multiplier", 3.0), "atr_multiplier", 0.5)
        trend_length = _int(params.get("trend_ema", 50), "trend_ema")
        use_filter = params.get("use_trend_filter", True)
        if not isinstance(use_filter, bool):
            raise ValueError("use_trend_filter must be a boolean")
    bars = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(v) and v > 0 for v in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        bars.append(bar)
    closes = [bar["close"] for bar in bars]
    signals = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    if strategy == "ichimoku_cloud":
        conv = _donchian_midpoint(bars, conversion)
        basis = _donchian_midpoint(bars, base)
        span_b = _donchian_midpoint(bars, span2)
        span_a = [(c + b) / 2 if c is not None and b is not None else None for c, b in zip(conv, basis)]
        for i in range(1, len(bars)):
            bullish = _cross(conv[i], basis[i], conv[i-1], basis[i-1], True)
            bearish = _cross(conv[i], basis[i], conv[i-1], basis[i-1], False)
            # Pine's plotted offset is visual; its conditions compare the lines
            # at [displacement], i.e. values computed that many bars ago.
            j = i - displacement
            if j >= 0 and span_a[j] is not None and span_b[j] is not None:
                a, b = span_a[j], span_b[j]
                signals[i]["long"] = gt(closes[i], max(a, b)) and bullish and gt(a, b)
                signals[i]["short"] = lt(closes[i], min(a, b)) and bearish and lt(a, b)
            signals[i]["close_long"] = bearish
            signals[i]["close_short"] = bullish
    else:
        atr, trend = _atr(bars, length), _ema(closes, trend_length)
        high, low = [None] * len(bars), [None] * len(bars)
        for i in range(length - 1, len(bars)):
            window = bars[i-length+1:i+1]
            high[i] = max(bar["high"] for bar in window) - atr[i] * multiplier
            low[i] = min(bar["low"] for bar in window) + atr[i] * multiplier
        for i in range(1, len(bars)):
            long_break = _cross(closes[i], high[i], closes[i-1], high[i-1], True)
            short_break = _cross(closes[i], low[i], closes[i-1], low[i-1], False)
            signals[i]["long"] = long_break and (not use_filter or gt(closes[i], trend[i]))
            signals[i]["short"] = short_break and (not use_filter or lt(closes[i], trend[i]))
            signals[i]["close_long"] = _cross(closes[i], high[i], closes[i-1], high[i-1], False)
            signals[i]["close_short"] = _cross(closes[i], low[i], closes[i-1], low[i-1], True)
    return signals
