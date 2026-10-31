"""Volume Flow Anomaly · AnomIQ preview — volume z-scores and absorption.

Source: https://github.com/AnomIQ/volume-flow-anomaly (published on TradingView as
"Volume Flow Anomaly · AnomIQ Preview")

Upstream ships no licence file; the header states the script is a preview of the
commercial AnomIQ scanner. Only the formulas it publishes are reproduced here.

Buy/sell volume is approximated from bar geometry, exactly as upstream warns:
`(close - low) / (high - low)` of the bar's volume counts as buy volume, the rest
as sell volume. The window sums are then z-scored against `lookback` bars. Surge,
absorption and extreme-volume thresholds are inputs; the absorption gates
(`|move| < 0.3%`, `rpi < 0.8`, cleanliness `< 35`) are the upstream constants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, flags, integer, number, ohlcv, sma, stdev


def _zscore(series: pd.Series, length: int) -> pd.Series:
    mean, spread = sma(series, length), stdev(series, length)
    return ((series - mean) / spread.where(spread > 0, np.nan)).mask(spread <= 0, 0.0)


def volume_flow_anomaly(frame: pd.DataFrame, *, window_len: int = 5, lookback: int = 100,
                        vol_z_min: float = 2.0, vol_z_early: float = 1.3,
                        clean_min: float = 50.0, move_min: float = 0.3) -> pd.DataFrame:
    integer("window_len", window_len, 2, 20)
    integer("lookback", lookback, 30, 500)
    number("vol_z_min", vol_z_min, 0.5)
    number("vol_z_early", vol_z_early, 0.5)
    number("clean_min", clean_min, 0)
    number("move_min", move_min, 0)
    high, low, close, open_, volume = ohlcv(frame, "high", "low", "close", "open", "volume")

    price_range = high - low
    buy_fraction = ((close - low) / price_range.where(price_range > 0, np.nan)).mask(
        price_range <= 0, 0.5)
    buy_volume = volume * buy_fraction
    sell_volume = volume - buy_volume
    window_volume = volume.rolling(window_len).sum()
    window_buy = buy_volume.rolling(window_len).sum()
    window_sell = sell_volume.rolling(window_len).sum()

    vol_z = _zscore(window_volume, lookback)
    buy_z = _zscore(window_buy, lookback)
    sell_z = _zscore(window_sell, lookback)

    net_move = (close - open_).rolling(window_len).sum()
    abs_move = (close - open_).abs().rolling(window_len).sum()
    cleanliness = (net_move.abs() / abs_move.where(abs_move > 0, np.nan) * 100).mask(abs_move <= 0, 0.0)
    window_open = open_.shift(window_len - 1)
    window_move = ((close - window_open) / window_open.where(window_open > 0, np.nan) * 100).mask(
        window_open <= 0, 0.0)
    ratio = (window_move.abs() / (window_volume / 1_000_000).where(window_volume > 0, np.nan)).mask(
        window_volume <= 0, 0.0)

    vol_z_change = vol_z - vol_z.shift(2)
    early_warning = flags((vol_z >= vol_z_early) & (vol_z < vol_z_min) & (cleanliness < 35)
                          & (vol_z_change > 0))
    upside_surge = flags((vol_z >= vol_z_min) & (cleanliness >= clean_min) & (window_move >= move_min))
    downside_surge = flags((vol_z >= vol_z_min) & (cleanliness >= clean_min) & (window_move <= -move_min))
    absorption = flags((vol_z >= vol_z_min) & (window_move.abs() < 0.3) & (ratio < 0.8))
    bull_lean = flags(absorption & (window_buy > window_sell))
    bear_lean = flags(absorption & (window_sell >= window_buy))
    extreme = flags(vol_z >= 3.5)

    return pd.DataFrame({
        "buy_fraction": buy_fraction,
        "window_volume": window_volume,
        "window_buy_volume": window_buy,
        "window_sell_volume": window_sell,
        "vol_z": vol_z,
        "buy_z": buy_z,
        "sell_z": sell_z,
        "trend_cleanliness": cleanliness,
        "window_move": window_move,
        "window_volume_ratio": ratio,
        "vol_z_change": vol_z_change,
        "early_warning": early_warning,
        "upside_surge": upside_surge,
        "downside_surge": downside_surge,
        "absorption": absorption,
        "absorption_bull": bull_lean,
        "absorption_bear": bear_lean,
        "extreme_volume": extreme,
    }, index=frame.index)
