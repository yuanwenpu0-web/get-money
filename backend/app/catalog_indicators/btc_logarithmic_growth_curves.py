"""Bitcoin Logarithmic Growth Curves — deterministic log-growth channel bands.

Source: pine-crypto-collection/krzyczak_logcurves/btc-logarithmic-growth-curves.js
(Pine v4 study in a .js file; MPL-2.0, © aaahopper / © krzyczak, "no rights
reserved, copy, paste, do what you want")

The channel is a deterministic function of bar time, not of price: a time
index counts days since the 2010-10-23 epoch (1279670400000 ms, clamped to a
minimum of 3 days), a weight curve `log10(t + 10) * t² - t` over 30000 sets
the curvature, and the top/bottom curves are `log(weight) + intercept +
slope * t` in log space. Ten bands interpolate the log range at fixed
Fibonacci fractions (1.0 / 0.9098 / 0.8541 / 0.7639 / 0.618 / 0.5 / 0.382 /
0.2361 / 0.1459 / 0.0902 / 0.0) and are exponentiated back to linear prices.

Each band is therefore computable for any bar whose epoch distance exceeds 2
days — no rolling window, no look-ahead. The chart price (``close``) is
returned alongside so callers can locate themselves inside the channel.

Dropped: the "hours back" recency filter (a plot-visibility hack), zone
display toggles and all line styling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import integer, number, ohlcv

_EPOCH_MS = 1279670400000
_MS_PER_DAY = 86400000
_BANDS = (("band_100", 1.0), ("band_9098", 0.9098), ("band_8541", 0.8541),
          ("band_7639", 0.7639), ("band_618", 0.618), ("band_mid", 0.5),
          ("band_382", 0.382), ("band_2361", 0.2361), ("band_1459", 0.1459),
          ("band_0902", 0.0902), ("band_000", 0.0))


def btc_logarithmic_growth_curves(
        frame: pd.DataFrame, *, high_intercept: float = 1.06930947,
        high_slope: float = 0.00076, low_intercept: float = -3.0269716,
        low_slope: float = 0.001329) -> pd.DataFrame:
    number("high_intercept", high_intercept, 0.0, 10.0)
    number("high_slope", high_slope, 0.0, 1.0)
    number("low_intercept", low_intercept, -10.0, 10.0)
    number("low_slope", low_slope, 0.0, 1.0)
    ohlcv(frame, "close")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("索引必须是 K 线开盘时间")
    timestamps_us = frame.index.tz_convert("UTC").asi8  # µs since epoch (pandas 3)
    timestamps_ms = timestamps_us // 1_000
    time_index = np.where(timestamps_ms < _EPOCH_MS, 3.0,
                          (timestamps_ms - _EPOCH_MS) / _MS_PER_DAY)
    weight = (np.log10(time_index + 10.0) * time_index * time_index - time_index) / 30000.0
    usable = time_index > 2
    with np.errstate(divide="ignore", invalid="ignore"):
        high_log = np.log(weight) + high_intercept + high_slope * time_index
        low_log = np.log(weight) + low_intercept + low_slope * time_index
    high_log = np.where(usable, high_log, np.nan)
    low_log = np.where(usable, low_log, np.nan)
    log_range = high_log - low_log

    output = {"close": frame["close"].astype(float)}
    for name, fraction in _BANDS:
        output[name] = pd.Series(np.exp(low_log + log_range * fraction), index=frame.index)
    return pd.DataFrame(output, index=frame.index)
