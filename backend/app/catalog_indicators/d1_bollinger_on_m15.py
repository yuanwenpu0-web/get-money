"""Daily Bollinger Bands (default period 90) projected onto 15-minute bars.

Source: https://github.com/Tim1l/PineCryptoStrategies/blob/424ddd3f2ebbdddefeabb24ae1d5f4dc6b6601a4/D1_bb90_on_m15.pine

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

Upstream requests the daily series with `barmerge.lookahead_on` and then freezes
the value for the whole day, so its band already contains that day's close. This
port keeps one band per day but publishes the previous completed day only: a
bar at 00:15 on day D sees day D-1's bands. A day whose 96 M15 bars are not all
present is dropped instead of being built from partial data.
"""
from __future__ import annotations

import pandas as pd

from ._pine import integer, number, ohlcv
from ._tf import aligned_index, previous_period_bollinger


def d1_bollinger_on_m15(frame: pd.DataFrame, *, bb_length: int = 90,
                        bb_std: float = 2.0) -> pd.DataFrame:
    integer("bb_length", bb_length, 1)
    number("bb_std", bb_std, 0.1)
    (close,) = ohlcv(frame, "close")
    utc = aligned_index(frame, 15)
    return previous_period_bollinger(close, utc, rule="D", step=pd.Timedelta(days=1),
                                     expected_bars=96, length=bb_length, multiplier=bb_std)
