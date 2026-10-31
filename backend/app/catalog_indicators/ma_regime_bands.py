"""MA Regime Bands — triple-MA alignment regime with ATR stop zones.

Source: pine-collection/indicators/trend_direction/ma_regime_bands/ma_regime_bands.pine
(WavesUnchained, MIT)

Three moving averages (SMA / EMA / DEMA / TEMA / JMA) over `len1..len3` define the
regime: a bar is bullish when the close is above all three and the stack is sorted
ma1 >= ma2 >= ma3 (bearish mirrored). The regime only counts as *strong* when the
MA spread, normalised by ATR, clears a minimum strength threshold that can be fixed
("Off"), scaled per MA type ("By MA Type") or auto-normalised against the average
absolute AMA step over `auto_lookback` bars ("Auto").

On a flip, the highest/lowest MA becomes the entry with an ATR stop
(`mult * atr` away); the level persists until the opposite flip or a neutral exit.

Port notes:
- `barstate.isconfirmed` is dropped: a historical replay only contains confirmed
  bars, so `bull_raw`/`bear_raw`/`neutral` are evaluated on every bar.
- JMA is the Pine f_jma recursion with the upstream call parameters (phase=0,
  power=2) and `var` state seeded from the first close.
- `strength_mode="Auto"` needs `auto_lookback` (>= 50) bars before its reference
  average exists; until then `auto_factor` falls back to 1.0 exactly like Pine.

Dropped: all plots/fills, regime labels, background colouring, the alert toggle
and the ATR stop-zone visuals (the entry/stop levels themselves are kept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, choice, ema, flags, integer, ma, number, ohlcv, rma, sma

_MA_TYPES = ("SMA", "EMA", "DEMA", "TEMA", "JMA")
_STRENGTH_FACTORS = {"SMA": 1.00, "EMA": 1.00, "DEMA": 1.05, "TEMA": 0.80, "JMA": 0.85}


def _jma(source: pd.Series, length: int) -> pd.Series:
    """Pine f_jma with phase=0, power=2 (the upstream default call)."""
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** 2
    phase_ratio = 1.5  # phase / 100 + 1.5 for phase = 0
    values = source.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    e0 = e1 = e2 = np.nan
    jma = np.nan
    for position, price in enumerate(values):
        if np.isnan(price):
            continue
        previous_e0 = price if np.isnan(e0) else e0
        previous_jma = price if np.isnan(jma) else jma
        e0 = (1 - alpha) * price + alpha * previous_e0
        e1 = (price - e0) * (1 - beta) + beta * (0.0 if np.isnan(e1) else e1)
        e2 = ((e0 + phase_ratio * e1 - previous_jma) * (1 - alpha) ** 2
              + alpha ** 2 * (0.0 if np.isnan(e2) else e2))
        jma = previous_jma + e2
        out[position] = jma
    return pd.Series(out, index=source.index)


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.ema over a series with leading NaNs (nested DEMA/TEMA layers).

    Pine seeds with the SMA once the last `length` values are all defined, so a
    leading NaN pushes the seed back; `_pine.ema` would seed early by skipping the
    gap, so nested layers are built on this local variant.
    """
    positions = series.notna().to_numpy().nonzero()[0]
    if len(positions) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    tail = series.iloc[positions[0]:]
    smoothed = ema(tail, length)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.iloc[positions[0]:] = smoothed.to_numpy()
    return out


def _moving_average(source: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type in ("SMA", "EMA"):
        return ma(source, length, ma_type)
    if ma_type == "DEMA":
        ema1 = ma(source, length, "EMA")
        return 2.0 * ema1 - _ema(ema1, length)
    if ma_type == "TEMA":
        ema1 = ma(source, length, "EMA")
        ema2 = _ema(ema1, length)
        return 3.0 * ema1 - 3.0 * ema2 + _ema(ema2, length)
    return _jma(source, length)


def ma_regime_bands(frame: pd.DataFrame, *, ma_type: str = "DEMA", len1: int = 21,
                    len2: int = 55, len3: int = 89, atr_len: int = 20,
                    mult: float = 2.0, strength_mode: str = "By MA Type",
                    base_min_strength: float = 0.6, auto_lookback: int = 200) -> pd.DataFrame:
    choice("ma_type", ma_type, _MA_TYPES)
    integer("len1", len1, 1)
    integer("len2", len2, 1)
    integer("len3", len3, 1)
    integer("atr_len", atr_len, 1)
    number("mult", mult, 0.1)
    choice("strength_mode", strength_mode, ("Off", "By MA Type", "Auto"))
    number("base_min_strength", base_min_strength, 0.0)
    integer("auto_lookback", auto_lookback, 50)
    (close,) = ohlcv(frame, "close")

    ma1 = _moving_average(close, len1, ma_type)
    ma2 = _moving_average(close, len2, ma_type)
    ma3 = _moving_average(close, len3, ma_type)
    ama = (ma1 + ma2 + ma3) / 3.0
    average_atr = atr(frame, atr_len)
    stop_distance = average_atr * mult

    highest_ma = pd.concat([ma1, ma2, ma3], axis=1).max(axis=1)
    lowest_ma = pd.concat([ma1, ma2, ma3], axis=1).min(axis=1)
    spread_atr = ((highest_ma - lowest_ma) / average_atr).where(average_atr > 0, 0.0)

    min_strength_by_type = base_min_strength * _STRENGTH_FACTORS[ma_type]
    ama_step = (rma((ama - ama.shift(1)).abs(), atr_len) / average_atr
                ).where(average_atr > 0, 0.0)
    ama_step_ref = sma(ama_step, auto_lookback)
    auto_factor = (ama_step / ama_step_ref).where(ama_step_ref > 0, 1.0)
    min_strength = (min_strength_by_type if strength_mode == "By MA Type"
                    else base_min_strength if strength_mode == "Off"
                    else base_min_strength * auto_factor)

    strong_trend = flags(spread_atr >= min_strength)
    bull_raw = (close >= ma1) & (close >= ma2) & (close >= ma3) \
        & (ma1 >= ma2) & (ma2 >= ma3)
    bear_raw = (close <= ma1) & (close <= ma2) & (close <= ma3) \
        & (ma1 <= ma2) & (ma2 <= ma3)
    bull = flags(bull_raw & strong_trend)
    bear = flags(bear_raw & strong_trend)
    neutral = flags(~bull & ~bear)

    bull_flip = bull & ~bull.shift(1, fill_value=False)
    bear_flip = bear & ~bear.shift(1, fill_value=False)
    exit_flip = flags((neutral & ~neutral.shift(1, fill_value=False))
                      | ((bull.shift(1, fill_value=False) & ~bull) & ~bear)
                      | ((bear.shift(1, fill_value=False) & ~bear) & ~bull))

    n = len(frame)
    long_entry = np.full(n, np.nan)
    long_stop = np.full(n, np.nan)
    short_entry = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    highs = highest_ma.to_numpy(dtype=float)
    lows = lowest_ma.to_numpy(dtype=float)
    current_long_entry = current_long_stop = np.nan
    current_short_entry = current_short_stop = np.nan
    for bar in range(n):
        if bull_flip.iat[bar]:
            current_long_entry = highs[bar]
            current_long_stop = highs[bar] - stop_distance.iat[bar]
            current_short_entry = current_short_stop = np.nan
        if bear_flip.iat[bar]:
            current_short_entry = lows[bar]
            current_short_stop = lows[bar] + stop_distance.iat[bar]
            current_long_entry = current_long_stop = np.nan
        if exit_flip.iat[bar]:
            current_long_entry = current_long_stop = np.nan
            current_short_entry = current_short_stop = np.nan
        long_entry[bar] = current_long_entry
        long_stop[bar] = current_long_stop
        short_entry[bar] = current_short_entry
        short_stop[bar] = current_short_stop

    return pd.DataFrame({
        "ma1": ma1, "ma2": ma2, "ma3": ma3, "ama": ama,
        "atr": average_atr,
        "highest_ma": highest_ma, "lowest_ma": lowest_ma,
        "spread_atr": spread_atr,
        "min_strength": min_strength,
        "strong_trend": strong_trend,
        "bull": bull, "bear": bear, "neutral": neutral,
        "bull_flip": bull_flip, "bear_flip": bear_flip, "exit_flip": exit_flip,
        "long_entry": pd.Series(long_entry, index=frame.index),
        "long_stop": pd.Series(long_stop, index=frame.index),
        "short_entry": pd.Series(short_entry, index=frame.index),
        "short_stop": pd.Series(short_stop, index=frame.index),
    }, index=frame.index)
