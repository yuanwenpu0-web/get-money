"""Closed-bar order conditions for selected non-Eterna Pine strategies.

This module does not emulate TradingView's broker. The caller must supply
chronological OHLC data, including UTC-aware bar opening times, and avoid gaps.
The source's string-form timestamp() defaults use UTC midnight.
"""
from __future__ import annotations

import math
from datetime import datetime
from numbers import Real
from .pine_compare import gt, ge, lt, le

SOURCES = {
    "bb_ema_eth": "data/pine-crypto-collection/dheeraj_eth/BB_EMA_ETHUSDT.pine",
}
DEFAULTS = {
    "bb_ema_eth": {"initial_capital": 1000.0, "position_pct": 1.0, "commission_rate": 0.0},
}


def _int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _time(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a UTC timestamp")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp with time zone") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be an ISO timestamp with time zone")
    return int(value.timestamp() * 1000)


def tradingview_signals(candles, code, params=None):
    """Return order conditions for BB EMA ETH/USDT 4h.

    ``time`` is the bar's open timestamp, ISO-8601 with timezone or Unix ms.
    The default Training segment is deliberately retained, including strict
    start/end comparisons and an end date at the beginning of 2022-12-31 UTC.
    Open positions are not closed merely because the date window has ended.
    """
    if code not in SOURCES:
        raise ValueError(f"unsupported TradingView strategy: {code}")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    params = dict(params or {})
    valid = {"ema_length", "bb_length", "bb_multiplier", "segment", "train_start", "train_end", "test_start", "test_end"}
    if params.keys() - valid:
        raise ValueError(f"unsupported parameters: {sorted(params.keys() - valid)}")
    ema_length = _int(params.get("ema_length", 100), "ema_length")
    bb_length = _int(params.get("bb_length", 20), "bb_length")
    multiplier = params.get("bb_multiplier", 1.5)
    if isinstance(multiplier, bool) or not isinstance(multiplier, Real) or not math.isfinite(multiplier):
        raise ValueError("bb_multiplier must be finite")
    segment = params.get("segment", "Training")
    if segment not in ("Training", "Testing", "All"):
        raise ValueError("segment must be Training, Testing or All")
    dates = {name: _time(params.get(name, default), name) for name, default in {
        "train_start": "2018-01-01T00:00:00+00:00", "train_end": "2022-12-31T00:00:00+00:00",
        "test_start": "2023-01-01T00:00:00+00:00", "test_end": "2025-12-31T00:00:00+00:00",
    }.items()}
    bars = []
    for candle in candles:
        bar = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(x) and x > 0 for x in bar.values()):
            raise ValueError("OHLC prices must be finite and positive")
        if bar["high"] < max(bar["open"], bar["close"], bar["low"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("invalid OHLC range")
        if "time" not in candle:
            raise ValueError("bar opening time is required")
        bar["time"] = _time(candle["time"], "bar time")
        bars.append(bar)
    closes = [bar["close"] for bar in bars]
    ema, basis, upper = [], [], []
    alpha = 2 / (ema_length + 1)
    for i, close in enumerate(closes):
        ema.append(close if i == 0 else alpha * close + (1 - alpha) * ema[-1])
        if i + 1 < bb_length:
            basis.append(None)
            upper.append(None)
            continue
        window = closes[i + 1 - bb_length:i + 1]
        mean = sum(window) / bb_length
        std = math.sqrt(sum((value - mean) ** 2 for value in window) / bb_length)
        basis.append(mean)
        upper.append(mean + multiplier * std)
    result = [{"long": False, "short": False, "close_long": False, "close_short": False} for _ in bars]
    for i in range(1, len(bars)):
        time = bars[i]["time"]
        start, end = (dates["train_start"], dates["train_end"]) if segment == "Training" else (dates["test_start"], dates["test_end"])
        in_range = segment == "All" or start < time < end
        if not in_range:
            continue
        if upper[i] is not None and upper[i - 1] is not None:
            result[i]["long"] = gt(closes[i], upper[i]) and le(closes[i - 1], upper[i - 1]) and gt(closes[i], ema[i])
        basis_exit = lt(closes[i], basis[i]) and ge(closes[i - 1], basis[i - 1])
        ema_exit = lt(closes[i], ema[i]) and ge(closes[i - 1], ema[i - 1])
        result[i]["close_long"] = basis_exit or ema_exit
    return result
