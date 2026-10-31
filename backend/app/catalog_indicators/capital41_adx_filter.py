"""Capital41 ADX Filter v3 — directional index with a trend-strength gate.

Source: pine-crypto-collection/capital41_btc_general/Capital41_ADX_Filter/Capital41_ADX_Filter_v3.pine

MIT License — Copyright (c) 2026 Capital41

`ta.dmi(diLength, adxSmoothing)` smooths DI over one length and ADX over another, so
the shared `_pine.dmi` helper (single length) is not enough: this module re-implements
the two-stage Wilder smoothing. The filter exports the DI direction, the two ADX
thresholds and every crossing the alerts fire on.

Dropped: DI fill, the trending background, the status badge table and alert dialogs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import crossover, crossunder, flags, integer, ohlcv, rma, true_range


def capital41_adx_filter(frame: pd.DataFrame, *, di_length: int = 14, adx_smoothing: int = 14,
                         trend_threshold: float = 20.0, strong_threshold: float = 35.0) -> pd.DataFrame:
    integer("di_length", di_length, 2)
    integer("adx_smoothing", adx_smoothing, 1)
    if not 5 <= trend_threshold <= 50:
        raise ValueError("参数 trend_threshold 超出允许范围")
    if not 20 <= strong_threshold <= 60:
        raise ValueError("参数 strong_threshold 超出允许范围")

    high, low, _ = ohlcv(frame, "high", "low", "close")
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    trur = rma(true_range(frame), di_length).replace(0, np.nan)
    di_plus = 100 * rma(plus_dm, di_length) / trur
    di_minus = 100 * rma(minus_dm, di_length) / trur
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = rma(dx, adx_smoothing)

    bull = flags(di_plus > di_minus)
    bear = flags(di_minus > di_plus)
    trending = flags(adx >= trend_threshold)
    strong = flags(adx >= strong_threshold)

    return pd.DataFrame({
        "di_plus": di_plus,
        "di_minus": di_minus,
        "adx": adx,
        "bull_direction": bull,
        "bear_direction": bear,
        "trending": trending,
        "strong_trend": strong,
        "di_cross_up": crossover(di_plus, di_minus),
        "di_cross_down": crossunder(di_plus, di_minus),
        "adx_cross_up": crossover(adx, pd.Series(trend_threshold, index=frame.index)),
        "adx_cross_down": crossunder(adx, pd.Series(trend_threshold, index=frame.index)),
        "adx_cross_strong": crossover(adx, pd.Series(strong_threshold, index=frame.index)),
    }, index=frame.index)
