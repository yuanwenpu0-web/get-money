"""Storm Gauge, adapted from milesdeutscher/garchmethod.

Source: https://github.com/milesdeutscher/garchmethod/blob/e1e93d9bbced22ca74fc61858a62e4ca9e4a0607/pine-script/storm-gauge.pine

MIT License

Copyright (c) 2026 Miles Deutscher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def storm_gauge(frame: pd.DataFrame, *, vol_len: int = 20, pctile_len: int = 252,
                periods_year: int = 365, target_vol: float = 15.0, base_size: float = 10000,
                max_lev: float = 2.0, min_size: float = 0.25, calm_th: float = 33.0,
                storm_th: float = 67.0, simple_mode: bool = True, show_bands: bool = True,
                show_tint: bool = True, tint_transp: int = 94, show_table: bool = True) -> pd.DataFrame:
    bounds = (("vol_len", vol_len, 5, 100, True), ("pctile_len", pctile_len, 60, 1000, True),
              ("periods_year", periods_year, 1, math.inf, True), ("target_vol", target_vol, 1, 200, False),
              ("base_size", base_size, 100, math.inf, False), ("max_lev", max_lev, .5, 10, False),
              ("min_size", min_size, .05, 1, False), ("calm_th", calm_th, 1, 49, False),
              ("storm_th", storm_th, 51, 99, False), ("tint_transp", tint_transp, 80, 99, True))
    for name, value, lower, upper, integer in bounds:
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not lower <= value <= upper or
                (integer and not isinstance(value, int))):
            raise ValueError(f"参数 {name} 超出允许范围")
    for name, value in (("simple_mode", simple_mode), ("show_bands", show_bands),
                        ("show_tint", show_tint), ("show_table", show_table)):
        if not isinstance(value, bool):
            raise ValueError(f"参数 {name} 必须为布尔值")
    if "close" not in frame:
        raise ValueError("缺少 close 列")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    ratio = close / close.shift(1).replace(0, np.nan)
    ret = np.log(ratio.where(ratio > 0))
    daily_vol = ret.rolling(vol_len).std(ddof=0)
    ann_vol = daily_vol * math.sqrt(periods_year) * 100
    # Missing historical values contribute zero, as in Pine's conditional loop.
    count = pd.Series(0, index=frame.index, dtype=int)
    for lag in range(1, pctile_len + 1):
        count += (ann_vol > ann_vol.shift(lag)).astype(int)
    percentile = count * (100.0 / pctile_len)
    storm, calm = percentile > storm_th, percentile < calm_th
    raw_size = (target_vol / ann_vol.where(ann_vol > 0)).fillna(min_size)
    size = np.minimum(np.maximum(raw_size, min_size), max_lev)
    basis = close.ewm(span=vol_len, adjust=False, ignore_na=True).mean()
    band = basis * daily_vol * math.sqrt(vol_len)
    prior_storm = storm.shift(1, fill_value=False)
    return pd.DataFrame({"daily_vol": daily_vol, "annual_vol": ann_vol,
                         "vol_percentile": percentile, "is_calm": calm, "is_storm": storm,
                         "regime": np.select([storm, calm], ["STORM", "CALM"], default="NORMAL"),
                         "size_multiplier": size, "dollar_size": base_size * size,
                         "upper_band": (basis + band) if show_bands else np.nan,
                         "lower_band": (basis - band) if show_bands else np.nan,
                         "storm_tint": storm & show_tint,
                         "storm_begins": storm & ~prior_storm,
                         "storm_ends": ~storm & prior_storm}, index=frame.index)
