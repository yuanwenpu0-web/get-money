"""Prior H1 candles from Tim1l's H1 Candle as True Shadow on M15.

Source: https://github.com/Tim1l/PineCryptoStrategies/blob/424ddd3f2ebbdddefeabb24ae1d5f4dc6b6601a4/h1_candles_on_m15.pine

MIT License

Copyright (c) 2025 [Your Name or Username]

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

Input is time-indexed UTC 15-minute OHLC, representing bar opening times. A
previous hour missing any of its four bars is returned as NaN instead of
inventing an H1 candle from incomplete data.
"""
from __future__ import annotations

import pandas as pd


def h1_candles_on_m15(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if missing := required - set(frame.columns):
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.hasnans or not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("索引必须为有时区、递增且唯一的 K 线开盘时间")
    utc = idx.tz_convert("UTC")
    if ((utc.minute % 15 != 0) | (utc.second != 0) | (utc.microsecond != 0) | (utc.nanosecond != 0)).any():
        raise ValueError("K 线时间必须对齐 15 分钟边界")
    ohlc = frame.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise").astype(float)
    ohlc.index = utc
    group = ohlc.groupby(utc.floor("h"))
    hourly = group.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    complete = group.count().eq(4).all(axis=1)
    hourly = hourly.where(complete, float("nan"))
    previous = hourly.reindex(utc.floor("h") - pd.Timedelta(hours=1))
    previous.index = idx
    previous.columns = ["h1_open", "h1_high", "h1_low", "h1_close"]
    previous["bullish"] = (previous["h1_close"] >= previous["h1_open"]).where(previous["h1_close"].notna())
    return previous
