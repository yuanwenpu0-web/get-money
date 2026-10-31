"""Bitcoin Market Regime Detector [RegimeRisk] — five-regime classifier.

Source: https://github.com/RegimeRisk/bitcoin-market-regime-detector (reference
release published on TradingView as "Bitcoin Market Regime Detector [RegimeRisk]")

Mozilla Public License 2.0 — https://mozilla.org/MPL/2.0/

The upstream script is an open, rule-based reference: five regimes decided in a
fixed priority order from an EMA trend, its percent slope, an ATR percentile and
ADX. The table, background colours and alerts are presentation only; this port
returns the underlying series and the regime label per bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, boolean, dmi, ema, integer, number, ohlcv, percentrank


def bitcoin_market_regime(frame: pd.DataFrame, *, trend_len: int = 100, slope_len: int = 20,
                          vol_len: int = 20, vol_rank_len: int = 200, vol_high: float = 80.0,
                          adx_len: int = 14, adx_range: float = 20.0, adx_trend: float = 25.0,
                          show_table: bool = True) -> pd.DataFrame:
    integer("trend_len", trend_len, 10)
    integer("slope_len", slope_len, 2)
    integer("vol_len", vol_len, 2)
    integer("vol_rank_len", vol_rank_len, 20)
    number("vol_high", vol_high, 50, 100)
    integer("adx_len", adx_len, 2)
    number("adx_range", adx_range, 5)
    number("adx_trend", adx_trend, 10)
    boolean("show_table", show_table)
    _, _, close = ohlcv(frame, "high", "low", "close")

    trend = ema(close, trend_len)
    slope_base = trend.shift(slope_len).replace(0, np.nan)
    slope = (trend - slope_base) / slope_base * 100
    above = close > trend
    atr_pct = atr(frame, vol_len) / close.replace(0, np.nan) * 100
    vol_rank = percentrank(atr_pct, vol_rank_len)
    di_plus, di_minus, adx = dmi(frame, adx_len)

    is_vol = (vol_rank >= vol_high).fillna(False).astype(bool)
    is_range = (~is_vol & (adx < adx_range)).fillna(False).astype(bool)
    trending = ~is_vol & ~is_range
    is_bull = (trending & above & (slope > 0) & (adx >= adx_trend)).fillna(False).astype(bool)
    is_bear = (trending & ~above & (slope < 0) & (adx >= adx_trend)).fillna(False).astype(bool)
    is_transition = ~(is_vol | is_range | is_bull | is_bear)

    regime = pd.Series(np.select([is_vol, is_range, is_bull, is_bear],
                                 ["Volatility", "Range", "Bull", "Bear"], default="Transition"),
                       index=frame.index, dtype=object)
    return pd.DataFrame({
        "ema": trend,
        "ema_slope": slope,
        "atr_percent": atr_pct,
        "vol_percentile": vol_rank,
        "di_plus": di_plus,
        "di_minus": di_minus,
        "adx": adx,
        "is_volatility": is_vol,
        "is_range": is_range,
        "is_bull": is_bull,
        "is_bear": is_bear,
        "is_transition": is_transition,
        "regime": regime,
        "regime_changed": regime.ne(regime.shift(1)) & regime.shift(1).notna(),
    }, index=frame.index)
