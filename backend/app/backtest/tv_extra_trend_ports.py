"""Parabolic SAR source signal port, for the research market-order engine.

Pine's built-in SAR state is implemented according to the public TradingView
reference pseudocode. History must begin at the same bar as the Pine script:
SAR's initial trend and extreme point are recursive and depend on the first
two bars. Orders, position sizing, fills and margin calls are not simulated.
"""
from __future__ import annotations

import math

from .pine_compare import gt, lt
from .tv_trend_ports import _cross, _ema, _int, _number

SOURCES = {
    "parabolic_sar": "data/pine-crypto-collection/eterna/strategies/trend-following/parabolic-sar/strategy.pine",
}
# Original Pine strategy uses $10000, 10% of equity and 0.075% commission.
DEFAULTS = {}


def _sar(bars, start, increment, maximum):
    out = [None] * len(bars)
    if len(bars) < 2:
        return out
    below = gt(bars[1]["close"], bars[0]["close"])
    extreme = bars[1]["high"] if below else bars[1]["low"]
    result = bars[0]["low"] if below else bars[0]["high"]
    acceleration = start
    for i in range(1, len(bars)):
        first_trend_bar = i == 1
        high, low = bars[i]["high"], bars[i]["low"]
        result += acceleration * (extreme - result)
        if below and gt(result, low):
            first_trend_bar = True
            below = False
            result = max(high, extreme)
            extreme = low
            acceleration = start
        elif not below and lt(result, high):
            first_trend_bar = True
            below = True
            result = min(low, extreme)
            extreme = high
            acceleration = start
        if not first_trend_bar:
            if below and gt(high, extreme):
                extreme = high
                acceleration = min(acceleration + increment, maximum)
            elif not below and lt(low, extreme):
                extreme = low
                acceleration = min(acceleration + increment, maximum)
        if below:
            result = min(result, bars[i-1]["low"])
            if i > 1:
                result = min(result, bars[i-2]["low"])
        else:
            result = max(result, bars[i-1]["high"])
            if i > 1:
                result = max(result, bars[i-2]["high"])
        out[i] = result
    return out


def tradingview_signals(candles, strategy, params=None):
    """Return original SAR entry booleans; the Pine source has no close orders."""
    if strategy not in SOURCES:
        raise ValueError(f"unsupported TradingView strategy: {strategy}")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    params = dict(params or {})
    unknown = params.keys() - {"sar_start", "sar_increment", "sar_maximum", "trend_filter", "use_trend_filter"}
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    start = _number(params.get("sar_start", .02), "sar_start", .001)
    increment = _number(params.get("sar_increment", .02), "sar_increment", .001)
    maximum = _number(params.get("sar_maximum", .2), "sar_maximum", .01)
    trend_length = _int(params.get("trend_filter", 50), "trend_filter")
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
    sar = _sar(bars, start, increment, maximum)
    trend = _ema(closes, trend_length)
    signals = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    for i in range(1, len(bars)):
        signals[i]["long"] = _cross(closes[i], sar[i], closes[i-1], sar[i-1], True) and (not use_filter or gt(closes[i], trend[i]))
        signals[i]["short"] = _cross(closes[i], sar[i], closes[i-1], sar[i-1], False) and (not use_filter or lt(closes[i], trend[i]))
    return signals
