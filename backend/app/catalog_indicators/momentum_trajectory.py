"""动量轨迹：把 WaveTrend、StochRSI 与 MFI 当作运动向量，拆出一阶速度与二阶加速度。

Source: pine-collection/indicators/momentum/momentum_trajectory/momentum_trajectory.pine
(WavesUnchained, MIT)

Each oscillator is treated as a movement vector: the first difference of the smoothed
oscillator is its velocity, the difference of the velocity is its acceleration. A
positive velocity with negative acceleration ("Peaking") is the script's early
exhaustion warning, and the WaveTrend acceleration crossing zero while velocity is
still positive (or negative) is flagged explicitly.

`input.*` display switches (`showWT`, `showSt`, `showMfi`, `showDash`, `showAccBg`) and
the `alertsOnBarClose` gate are presentation only and dropped, together with the plots,
the state dots, the dashboard table and the alert conditions. `volume` is a required
column because the MFI leg needs it; upstream guards the value with
`math.max(nz(volume), 1.0)`, reproduced here as NaN→0 with a floor of 1.

`math.max(_wtD, syminfo.mintick)` uses the symbol's tick size as a floor for the
normalising deviation; a DataFrame carries no tick size, so only the zero case is
special-cased (flat input → 0, the ratio's limit).

State labels read "Flat" while velocity and acceleration are still undefined, because
Pine's `na` comparisons in `f_state` are simply false.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (crossunder, crossover, ema, flags, highest, integer, lowest, ohlcv,
                    rsi, sma, sum_)


def _state(velocity: pd.Series, acceleration: pd.Series) -> pd.Series:
    """Pine's `f_state`: the sign pair picks the label, zero/na reads as "Flat"."""
    rising = flags(velocity > 0)
    falling = flags(velocity < 0)
    speeding_up = flags(acceleration > 0)
    slowing_down = flags(acceleration < 0)
    state = pd.Series("Flat", index=velocity.index, dtype=object)
    state = state.mask(rising & speeding_up, "Building")
    state = state.mask(rising & slowing_down, "Peaking")
    state = state.mask(falling & slowing_down, "Collapsing")
    state = state.mask(falling & speeding_up, "Recovering")
    return state


def momentum_trajectory(frame: pd.DataFrame, *, wt_channel_length: int = 10,
                        wt_average_length: int = 21, rsi_length: int = 14,
                        stoch_length: int = 14, k_smoothing: int = 3,
                        mfi_length: int = 14, velocity_smoothing: int = 3) -> pd.DataFrame:
    integer("wt_channel_length", wt_channel_length, 1)
    integer("wt_average_length", wt_average_length, 1)
    integer("rsi_length", rsi_length, 1)
    integer("stoch_length", stoch_length, 1)
    integer("k_smoothing", k_smoothing, 1)
    integer("mfi_length", mfi_length, 1)
    integer("velocity_smoothing", velocity_smoothing, 1, 10)

    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3.0

    # WaveTrend: channel deviation normalised by its own average absolute distance.
    channel = ema(typical, wt_channel_length)
    deviation = ema((typical - channel).abs(), wt_channel_length)
    denominator = 0.015 * deviation
    normalized = ((typical - channel) / denominator.where(denominator.ne(0))).where(
        denominator.ne(0), 0.0)
    wt_osc = ema(normalized, wt_average_length)

    # StochRSI: ta.stoch of the RSI, then a light SMA on %K.
    rsi_values = rsi(close, rsi_length)
    window_high, window_low = highest(rsi_values, stoch_length), lowest(rsi_values, stoch_length)
    stoch_raw = 100.0 * (rsi_values - window_low) / (window_high - window_low).replace(0, np.nan)
    stoch_k = sma(stoch_raw, k_smoothing)

    # MFI: money flow split by the direction of the typical price.
    flow = typical * volume.fillna(0.0).clip(lower=1.0)
    change = typical.diff()
    positive = sum_(flow.where(change > 0, 0.0), mfi_length)
    negative = sum_(flow.where(change < 0, 0.0), mfi_length)
    mfi = 100.0 - 100.0 / (1.0 + positive / negative.replace(0, np.nan))
    mfi = mfi.mask(negative.eq(0), 100.0)

    wt_velocity = sma(wt_osc.diff(), velocity_smoothing)
    stoch_velocity = sma(stoch_k.diff(), velocity_smoothing)
    mfi_velocity = sma(mfi.diff(), velocity_smoothing)
    wt_acceleration = wt_velocity.diff()
    stoch_acceleration = stoch_velocity.diff()
    mfi_acceleration = mfi_velocity.diff()

    zero = pd.Series(0.0, index=frame.index)
    return pd.DataFrame({
        "wt_osc": wt_osc,
        "rsi": rsi_values,
        "stoch_rsi": stoch_raw,
        "stoch_k": stoch_k,
        "mfi": mfi,
        "wt_velocity": wt_velocity,
        "wt_acceleration": wt_acceleration,
        "stoch_velocity": stoch_velocity,
        "stoch_acceleration": stoch_acceleration,
        "mfi_velocity": mfi_velocity,
        "mfi_acceleration": mfi_acceleration,
        "wt_state": _state(wt_velocity, wt_acceleration),
        "stoch_state": _state(stoch_velocity, stoch_acceleration),
        "mfi_state": _state(mfi_velocity, mfi_acceleration),
        "wt_acceleration_peak": crossunder(wt_acceleration, zero) & flags(wt_velocity > 0),
        "wt_acceleration_recover": crossover(wt_acceleration, zero) & flags(wt_velocity < 0),
    }, index=frame.index)
