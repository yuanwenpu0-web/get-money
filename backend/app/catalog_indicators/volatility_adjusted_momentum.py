"""Volatility-Adjusted Momentum Oscillator, adapted from TPTBusiness.

Source: https://github.com/TPTBusiness/VAMOStrategy/blob/96da7a5aa4c03c83d55b019760e66bc91d6120b0/Indicator.pine

MIT License

Copyright (c) 2025 TPTBusiness

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


def volatility_adjusted_momentum(frame: pd.DataFrame, *, mom_length: int = 14,
                                 vol_length: int = 14, signal_length: int = 9,
                                 ob_level: float = 70, os_level: float = 30) -> pd.DataFrame:
    for name, value in (("mom_length", mom_length), ("vol_length", vol_length),
                        ("signal_length", signal_length)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"参数 {name} 必须为正整数")
    for name, value in (("ob_level", ob_level), ("os_level", os_level)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"参数 {name} 必须为非负有限数值")
    missing = {"high", "low", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    high, low, close = (pd.to_numeric(frame[col], errors="raise").astype(float)
                        for col in ("high", "low", "close"))
    previous = close.shift(1)
    tr = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1, skipna=True)
    # Wilder smoothing seeded by the mean of the first vol_length valid values.
    atr_values = np.full(len(frame), np.nan)
    count, total, atr = 0, 0., np.nan
    for i, value in enumerate(tr.to_numpy()):
        if not np.isnan(value):
            if count < vol_length:
                count += 1
                total += value
                if count == vol_length:
                    atr = total / vol_length
            else:
                atr = (atr * (vol_length - 1) + value) / vol_length
        atr_values[i] = atr
    volatility = pd.Series(atr_values, index=frame.index)
    denominator = close.shift(mom_length).replace(0, np.nan)
    momentum = 100 * (close - denominator) / denominator
    vamo = momentum / volatility.replace(0, np.nan) * 100
    signal = vamo.ewm(span=signal_length, adjust=False, ignore_na=True).mean()
    return pd.DataFrame({"momentum": momentum, "volatility": volatility, "vamo": vamo,
                         "signal": signal, "overbought": vamo > ob_level,
                         "oversold": vamo < os_level}, index=frame.index)
