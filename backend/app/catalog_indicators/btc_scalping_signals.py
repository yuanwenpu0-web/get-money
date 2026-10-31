"""BTC 1–5 min Scalping Signals, adapted from Abhii58.

Source: https://github.com/Abhii58/Btc-trading-1---5-min-strategy-script/blob/826b272f2d15b6c1d7bfa578c63f9f9e8124b657/btc_scalping_signals.pinescript

MIT License

Copyright (c) 2025 Abhii58

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

import pandas as pd

from .rsi_trading_signals import rsi_trading_signals


def btc_scalping_signals(frame: pd.DataFrame, *, ema_fast_len: int = 5,
                         ema_slow_len: int = 13, rsi_len: int = 7,
                         rsi_buy: int = 50, rsi_sell: int = 50,
                         macd_fast: int = 8, macd_slow: int = 21,
                         macd_sig: int = 5) -> pd.DataFrame:
    for name, value in (("ema_fast_len", ema_fast_len), ("ema_slow_len", ema_slow_len),
                        ("rsi_len", rsi_len), ("macd_fast", macd_fast),
                        ("macd_slow", macd_slow), ("macd_sig", macd_sig)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"参数 {name} 必须为正整数")
    for name, value in (("rsi_buy", rsi_buy), ("rsi_sell", rsi_sell)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"参数 {name} 必须为整数")
    missing = {"open", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    opening = pd.to_numeric(frame["open"], errors="raise").astype(float)
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False, ignore_na=True).mean()
    fast, slow = ema(close, ema_fast_len), ema(close, ema_slow_len)
    rsi = rsi_trading_signals(frame, rsi_length=rsi_len)["rsi"]
    macd = ema(close, macd_fast) - ema(close, macd_slow)
    signal = ema(macd, macd_sig)
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return pd.DataFrame({"ema_fast": fast, "ema_slow": slow, "rsi": rsi,
                         "macd": macd, "macd_signal": signal,
                         "long_signal": cross_up & (rsi > rsi_buy) & (macd > signal) & (close > opening),
                         "short_signal": cross_down & (rsi < rsi_sell) & (macd < signal) & (close < opening)},
                        index=frame.index)
