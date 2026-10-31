"""Capital41 Regime Matrix v3 — Hurst, ADX and volatility slope pick one of four regimes.

Source: pine-crypto-collection/capital41_btc_general/Capital41_RegimeMatrix/Capital41_Regime_Matrix_v3.pine

MIT License — Copyright (c) 2026 Capital41

Hurst is estimated with the script's own range-over-deviation ratio (`log(range/stdev) /
log(length)`, clamped to 0..1). Four regimes come out of it:

* TREND — Hurst at or above `trend_hurst`, ADX at or above `adx_trend_min`, ADX rising;
* EXPANSION — volatility slope above its threshold and ADX rising, unless trending;
* COMPRESSION — volatility slope below the negative threshold and ADX below its cap;
* MEAN REVERSION — everything else.

Each regime gets its own EMA length (base length times 1.35 / 0.95 / 0.65 / 0.75) and its own
ATR stop multiplier; the adaptive EMA and both stops are published per bar, plus the 0-100
"pressure" blend of normalised ADX, Hurst and price drift.

Dropped: the adaptive EMA plot, the stop lines, the regime background and labels, the badge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, dmi, ema, flags, highest, integer, lowest, number, ohlcv, stdev


def _hurst(close: pd.Series, length: int) -> pd.Series:
    spread = highest(close, length) - lowest(close, length)
    deviation = stdev(close, length)
    ratio = (spread / deviation.where(deviation > 0)).where(deviation > 0)
    raw = np.log(ratio) / np.log(length)
    return raw.clip(0.0, 1.0).fillna(0.5)


def capital41_regime_matrix(frame: pd.DataFrame, *, base_ema_length: int = 50, atr_length: int = 14,
                            hurst_length: int = 100, trend_hurst: float = 0.58,
                            mean_reversion_hurst: float = 0.46, adx_length: int = 14,
                            adx_slope_length: int = 5, adx_trend_min: float = 22.0,
                            adx_compression_max: float = 16.0, volatility_length: int = 20,
                            volatility_slope_length: int = 5,
                            volatility_slope_threshold: float = 0.20,
                            trend_stop_multiplier: float = 2.8, expansion_stop_multiplier: float = 2.2,
                            compression_stop_multiplier: float = 1.3,
                            mean_reversion_stop_multiplier: float = 1.8,
                            pressure_smoothing: int = 3) -> pd.DataFrame:
    integer("base_ema_length", base_ema_length, 5)
    integer("atr_length", atr_length, 1)
    integer("hurst_length", hurst_length, 20, 300)
    number("trend_hurst", trend_hurst, 0.50, 0.90)
    number("mean_reversion_hurst", mean_reversion_hurst, 0.10, 0.50)
    integer("adx_length", adx_length, 2)
    integer("adx_slope_length", adx_slope_length, 1, 50)
    number("adx_trend_min", adx_trend_min, 5.0, 60.0)
    number("adx_compression_max", adx_compression_max, 5.0, 40.0)
    integer("volatility_length", volatility_length, 5, 100)
    integer("volatility_slope_length", volatility_slope_length, 1, 50)
    number("volatility_slope_threshold", volatility_slope_threshold, 0.01, 3.0)
    for name, value in (("trend_stop_multiplier", trend_stop_multiplier),
                        ("expansion_stop_multiplier", expansion_stop_multiplier),
                        ("compression_stop_multiplier", compression_stop_multiplier),
                        ("mean_reversion_stop_multiplier", mean_reversion_stop_multiplier)):
        number(name, value, 0.5, 6.0)
    integer("pressure_smoothing", pressure_smoothing, 1, 20)

    _, _, close = ohlcv(frame, "high", "low", "close")
    hurst = _hurst(close, hurst_length)
    _, _, adx = dmi(frame, adx_length)
    adx_slope = adx - adx.shift(adx_slope_length)
    log_return = np.log((close / close.shift(1)).where(close.gt(0) & close.shift(1).gt(0)))
    realized_volatility = stdev(log_return, volatility_length) * 100.0
    volatility_slope = realized_volatility - realized_volatility.shift(volatility_slope_length)

    trending = flags((hurst >= trend_hurst) & (adx >= adx_trend_min) & (adx_slope > 0))
    expanding = flags((volatility_slope > volatility_slope_threshold) & (adx_slope > 0) & ~trending)
    compressing = flags((volatility_slope < -volatility_slope_threshold) & (adx <= adx_compression_max))
    reverting = flags(~trending & ~expanding & ~compressing)

    regime = np.select([trending, expanding, compressing], [1, 2, 3], default=4)
    regime_code = pd.Series(regime, index=frame.index, dtype=int)
    labels = np.select([trending, expanding, compressing], ["TREND", "EXP", "COMP"], default="MR")
    regime_text = pd.Series(labels, index=frame.index, dtype=object)

    lengths = {"trend": max(5, round(base_ema_length * 1.35)),
               "expansion": max(5, round(base_ema_length * 0.95)),
               "compression": max(5, round(base_ema_length * 0.65)),
               "reversion": max(5, round(base_ema_length * 0.75))}
    emas = {key: ema(close, value) for key, value in lengths.items()}
    adaptive = emas["reversion"].copy()
    adaptive = adaptive.mask(regime_code.eq(1), emas["trend"]).mask(regime_code.eq(2), emas["expansion"])
    adaptive = adaptive.mask(regime_code.eq(3), emas["compression"])

    average_range = atr(frame, atr_length)
    multipliers = pd.Series(np.select([regime_code.eq(1), regime_code.eq(2), regime_code.eq(3)],
                                      [trend_stop_multiplier, expansion_stop_multiplier,
                                       compression_stop_multiplier],
                                      default=mean_reversion_stop_multiplier), index=frame.index)

    adx_normalised = ((adx - 10.0) / 35.0).clip(0.0, 1.0)
    hurst_normalised = ((hurst - 0.5) / 0.25).clip(0.0, 1.0)
    drift_normalised = ((((close - adaptive) / (average_range * 2.5)) * 0.5 + 0.5)
                        .clip(0.0, 1.0).where(average_range > 0, 0.5))
    pressure_raw = 100.0 * (0.45 * adx_normalised + 0.25 * hurst_normalised + 0.30 * drift_normalised)
    pressure = ema(pressure_raw, pressure_smoothing)

    return pd.DataFrame({
        "hurst": hurst,
        "adx": adx,
        "adx_slope": adx_slope,
        "realized_volatility": realized_volatility,
        "volatility_slope": volatility_slope,
        "trend_state": trending,
        "expansion_state": expanding,
        "compression_state": compressing,
        "mean_reversion_state": reverting,
        "regime_code": regime_code,
        "regime": regime_text,
        "regime_changed": flags(regime_code.ne(regime_code.shift(1))),
        "adaptive_ema": adaptive,
        "atr": average_range,
        "stop_multiplier": multipliers,
        "long_stop": adaptive - average_range * multipliers,
        "short_stop": adaptive + average_range * multipliers,
        "trend_up": flags(close >= adaptive),
        "pressure": pressure,
        "pressure_bull": flags(pressure >= 60),
        "pressure_bear": flags(pressure <= 40),
    }, index=frame.index)
