"""Capital41 ATR Volatility Bands — ATR envelope around an EMA/SMA basis.

Source: https://github.com/Capital41/TradingView-Indicators
        (Capital41_ATR_Volatility_Bands_v2.pine)

MIT License — Copyright (c) 2026 Capital41

Bands are the basis plus/minus `atr_mult` ATRs. The recent-range box is returned
as its two edges per bar, and a volatility expansion is flagged when ATR trades
`exp_factor` times above its own SMA. The box drawing and the background tint
are presentation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (atr, boolean, choice, crossover, crossunder, ema, flags, highest,
                    integer, lowest, number, ohlcv, sma)


def capital41_atr_volatility_bands(frame: pd.DataFrame, *, basis_type: str = "EMA",
                                   basis_len: int = 50, atr_len: int = 14, atr_mult: float = 1.5,
                                   box_lookback: int = 20, exp_factor: float = 1.2,
                                   show_basis: bool = False, show_bands: bool = True,
                                   show_vol_box: bool = True,
                                   show_expansion: bool = False) -> pd.DataFrame:
    choice("basis_type", basis_type, ("EMA", "SMA"))
    integer("basis_len", basis_len, 5)
    integer("atr_len", atr_len, 1)
    number("atr_mult", atr_mult, 0.1, 5)
    integer("box_lookback", box_lookback, 5, 200)
    number("exp_factor", exp_factor, 1, 3)
    for name, value in (("show_basis", show_basis), ("show_bands", show_bands),
                        ("show_vol_box", show_vol_box), ("show_expansion", show_expansion)):
        boolean(name, value)
    high, low, close = ohlcv(frame, "high", "low", "close")

    basis = ema(close, basis_len) if basis_type == "EMA" else sma(close, basis_len)
    atr_line = atr(frame, atr_len)
    upper, lower = basis + atr_line * atr_mult, basis - atr_line * atr_mult
    atr_average = sma(atr_line, atr_len)
    expansion = flags((atr_average > 0) & (atr_line > atr_average * exp_factor))

    return pd.DataFrame({
        "basis": basis,
        "atr": atr_line,
        "upper_band": upper,
        "lower_band": lower,
        "range_high": highest(high, box_lookback),
        "range_low": lowest(low, box_lookback),
        "atr_average": atr_average,
        "volatility_expansion": expansion,
        "expansion_start": expansion & ~expansion.shift(1, fill_value=False),
        "upper_band_cross": crossover(close, upper),
        "lower_band_cross": crossunder(close, lower),
    }, index=frame.index)
