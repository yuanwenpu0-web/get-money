"""Smart Money Concepts — order blocks, FVGs, BOS, equal highs/lows, premium/discount.

Source: pine-crypto-collection/tamheed_pro/indicators/smart-money-concepts.pine

No license header upstream (unattributed snippet).

Four per-bar pattern families: order blocks (the previous bar is opposite-coloured
and its range is smaller than 1.5x the current body), fair value gaps (a three-bar
imbalance: low above the high two bars back, mirrored down), breaks of structure
(close crossing the last confirmed 5/5 swing), equal highs/lows (liquidity: a
high equal to the high one or two bars back), and the 50-bar range premium /
discount split at the equilibrium midpoint. A "LONG" bias needs a discount close
plus a bullish BOS on the same bar; "SHORT" mirrors it.

Dropped: all boxes, labels and the dashboard table; the dead `obLookback` input
(declared but never used upstream); the box-pool trimming (the drawn boxes only
live for a fixed number of bars). Order-block activity is approximated as the
count of triggers in the last 20 bars, matching the upstream box lifetime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import flags, highest, lowest, nullable_int, ohlcv, pivothigh, pivotlow, sum_


def smart_money_concepts(frame: pd.DataFrame) -> pd.DataFrame:
    open_, high, low, close = ohlcv(frame, "open", "high", "low", "close")

    previous_range = (high.shift(1) - low.shift(1))
    is_bullish_ob = flags((close.shift(1) < open_.shift(1)) & (close > open_)
                          & ((close - open_) > previous_range * 1.5))
    is_bearish_ob = flags((close.shift(1) > open_.shift(1)) & (close < open_)
                          & ((open_ - close) > previous_range * 1.5))

    bullish_fvg = flags((low > high.shift(2)) & (close > open_))
    bearish_fvg = flags((high < low.shift(2)) & (close < open_))

    swing_high = pivothigh(high, 5, 5)
    swing_low = pivotlow(low, 5, 5)
    last_swing_high = swing_high.ffill()
    last_swing_low = swing_low.ffill()
    bullish_bos = flags((last_swing_high.notna() & (close > last_swing_high)
                         & (close.shift(1) <= last_swing_high)))
    bearish_bos = flags((last_swing_low.notna() & (close < last_swing_low)
                         & (close.shift(1) >= last_swing_low)))

    equal_highs = flags((high == high.shift(1)) | (high == high.shift(2)))
    equal_lows = flags((low == low.shift(1)) | (low == low.shift(2)))

    range_high = highest(high, 50)
    range_low = lowest(low, 50)
    equilibrium = (range_high + range_low) / 2.0
    premium_zone = flags(close > equilibrium)
    discount_zone = flags(close < equilibrium)
    bias = pd.Series("NEUTRAL", index=frame.index, dtype=object)
    bias = bias.mask(discount_zone & bullish_bos, "LONG").mask(premium_zone & bearish_bos, "SHORT")

    return pd.DataFrame({
        "is_bullish_ob": is_bullish_ob,
        "is_bearish_ob": is_bearish_ob,
        "bullish_ob_active": nullable_int(sum_(is_bullish_ob.astype(float), 20)),
        "bearish_ob_active": nullable_int(sum_(is_bearish_ob.astype(float), 20)),
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": equilibrium,
        "premium_zone": premium_zone,
        "discount_zone": discount_zone,
        "bias": bias,
    }, index=frame.index)
