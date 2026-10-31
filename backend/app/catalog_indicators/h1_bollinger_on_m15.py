"""Hourly Bollinger Bands (default period 24) projected onto 15-minute bars.

Source: https://github.com/Tim1l/PineCryptoStrategies/blob/424ddd3f2ebbdddefeabb24ae1d5f4dc6b6601a4/h1_bb_on_m15_shifted.pine

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

Same read as the daily variant: upstream freezes a lookahead hourly band for the
whole hour, this port publishes the previous completed hour, and hours missing
any of their four M15 bars stay empty.
"""
from __future__ import annotations

import pandas as pd

from ._pine import integer, number, ohlcv
from ._tf import aligned_index, previous_period_bollinger


def h1_bollinger_on_m15(frame: pd.DataFrame, *, bb_length: int = 24,
                        bb_std: float = 2.0) -> pd.DataFrame:
    integer("bb_length", bb_length, 1)
    number("bb_std", bb_std, 0.1)
    (close,) = ohlcv(frame, "close")
    utc = aligned_index(frame, 15)
    return previous_period_bollinger(close, utc, rule="h", step=pd.Timedelta(hours=1),
                                     expected_bars=4, length=bb_length, multiplier=bb_std)
