"""Session VWAP with standard-deviation bands and touch signals.

Source: pine-crypto-collection/anchored_vwap/SessionVWAP_SDBands.pine
(Anchored Session VWAP — SD Bands + Volume Signals; no explicit license header)

The VWAP and its standard-deviation bands anchor to the first bar of each
session: a session window (start/end "HHMM" in the given IANA timezone) resets
the cumulative price×volume and variance sums. Bands sit at ±`sd_mult` × the
session running standard deviation of the source around the VWAP. Touch
signals fire when the high crosses above the chosen upper band (sell) or the
low crosses below the chosen lower band (buy), gated by a volume
confirmation (current volume > SMA(volume[1], lookback) × multiplier) and a
per-direction dedup window. The VWAP state is bullish/bearish when price sits
above/below VWAP with above-average turnover.

Sessions are evaluated in the requested timezone; out-of-session bars report
NaN bands and a neutral state. Off-session volume still feeds the volume SMA.

Dropped: band fills, line styling, dashboard table, colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, flags, integer, number, ohlcv, sma

_SESSION_RE = {"America/New_York": "America/New_York", "Europe/London": "Europe/London",
               "Asia/Tokyo": "Asia/Tokyo", "UTC": "UTC"}


def _session_mask(utc: pd.DatetimeIndex, session: str, timezone: str):
    local = utc.tz_convert(timezone)
    start_hhmm, end_hhmm = session.split("-")
    start_minutes = int(start_hhmm[:2]) * 60 + int(start_hhmm[2:])
    end_minutes = int(end_hhmm[:2]) * 60 + int(end_hhmm[2:])
    minutes_of_day = local.hour * 60 + local.minute
    if start_minutes == end_minutes:
        # "0000-0000" means the full day; sessions reset at local midnight.
        mask = np.ones(len(utc), dtype=bool)
        resets = np.concatenate([[True], np.diff(local.day) != 0])
        return mask, resets
    if start_minutes < end_minutes:
        mask = (minutes_of_day >= start_minutes) & (minutes_of_day < end_minutes)
    else:
        mask = (minutes_of_day >= start_minutes) | (minutes_of_day < end_minutes)
    resets = mask & ~np.concatenate([[False], mask[:-1]])
    return mask, resets


def sessionvwap_sdbands(
        frame: pd.DataFrame, *, source: str = "hlc3", session: str = "0930-1600",
        timezone: str = "UTC", sd_mult_1: float = 1.0, sd_mult_2: float = 2.0,
        sd_mult_3: float = 3.0, use_vol_filter: bool = True, vol_lookback: int = 20,
        vol_mult: float = 1.2, signal_band: str = "2 SD", dedup_bars: int = 10) -> pd.DataFrame:
    choice("source", source, ("close", "hlc3", "hl2", "ohlc4"))
    if len(session.split("-")) != 2:
        raise ValueError("参数 session 需要形如 0930-1600")
    if timezone not in _SESSION_RE:
        raise ValueError(f"参数 timezone 不支持: {timezone}")
    number("sd_mult_1", sd_mult_1, 0.1)
    number("sd_mult_2", sd_mult_2, 0.1)
    number("sd_mult_3", sd_mult_3, 0.1)
    integer("vol_lookback", vol_lookback, 5, 200)
    number("vol_mult", vol_mult, 1.0, 5.0)
    choice("signal_band", signal_band, ("1 SD", "2 SD", "3 SD"))
    integer("dedup_bars", dedup_bars, 1, 200)
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    if source == "close":
        src = close
    elif source == "hlc3":
        src = (high + low + close) / 3
    elif source == "hl2":
        src = (high + low) / 2
    else:
        src = (high + low + close + frame["open"].astype(float)) / 4

    utc = frame.index
    in_session_array, new_session_array = _session_mask(utc, session, timezone)

    rows = len(frame)
    vwap = np.full(rows, np.nan)
    sd1u = np.full(rows, np.nan)
    sd1l = np.full(rows, np.nan)
    sd2u = np.full(rows, np.nan)
    sd2l = np.full(rows, np.nan)
    sd3u = np.full(rows, np.nan)
    sd3l = np.full(rows, np.nan)
    src_values = src.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    cum_pv = cum_v = cum_p2v = 0.0
    for position in range(rows):
        if new_session_array[position]:
            cum_pv = cum_v = cum_p2v = 0.0
        if not in_session_array[position]:
            continue
        value = src_values[position]
        vol = volume_values[position]
        cum_pv += value * vol
        cum_p2v += value * value * vol
        cum_v += vol
        if cum_v <= 0:
            continue
        mean = cum_pv / cum_v
        variance = max(cum_p2v / cum_v - mean * mean, 0.0)
        sd = float(np.sqrt(variance))
        vwap[position] = mean
        sd1u[position], sd1l[position] = mean + sd_mult_1 * sd, mean - sd_mult_1 * sd
        sd2u[position], sd2l[position] = mean + sd_mult_2 * sd, mean - sd_mult_2 * sd
        sd3u[position], sd3l[position] = mean + sd_mult_3 * sd, mean - sd_mult_3 * sd

    vwap_series = pd.Series(vwap, index=utc)
    band_map = {"1 SD": (sd1u, sd1l), "2 SD": (sd2u, sd2l), "3 SD": (sd3u, sd3l)}
    upper_trigger = pd.Series(band_map[signal_band][0], index=utc)
    lower_trigger = pd.Series(band_map[signal_band][1], index=utc)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)

    average_volume = sma(volume.shift(1), vol_lookback)
    vol_ok = (flags(volume > average_volume * vol_mult) if use_vol_filter
              else flags(pd.Series(True, index=utc)))

    sell_touch = np.zeros(rows, dtype=bool)
    buy_touch = np.zeros(rows, dtype=bool)
    upper_values = upper_trigger.to_numpy(dtype=float)
    lower_values = lower_trigger.to_numpy(dtype=float)
    for position in range(1, rows):
        if not in_session_array[position] or new_session_array[position]:
            continue
        if (not np.isnan(upper_values[position]) and not np.isnan(upper_values[position - 1])
                and highs[position] >= upper_values[position]
                and highs[position - 1] < upper_values[position - 1]):
            sell_touch[position] = True
        if (not np.isnan(lower_values[position]) and not np.isnan(lower_values[position - 1])
                and lows[position] <= lower_values[position]
                and lows[position - 1] > lower_values[position - 1]):
            buy_touch[position] = True

    last_sell = last_buy = None
    sell_signal_values = np.zeros(rows, dtype=bool)
    buy_signal_values = np.zeros(rows, dtype=bool)
    vol_ok_values = vol_ok.to_numpy(dtype=bool)
    for position in range(rows):
        sell_dedup = last_sell is None or (position - last_sell) >= dedup_bars
        buy_dedup = last_buy is None or (position - last_buy) >= dedup_bars
        sell_signal_values[position] = sell_touch[position] and vol_ok_values[position] and sell_dedup
        buy_signal_values[position] = buy_touch[position] and vol_ok_values[position] and buy_dedup
        if sell_signal_values[position]:
            last_sell = position
        if buy_signal_values[position]:
            last_buy = position

    sd_unit = np.where((~np.isnan(vwap)) & (~np.isnan(sd1u)), sd1u - vwap, np.nan)
    distance_sd = np.where(sd_unit > 0, (close.to_numpy(dtype=float) - vwap) / sd_unit, np.nan)
    vol_above_avg = average_volume.notna().to_numpy() & (average_volume.to_numpy() > 0) \
        & (volume_values > average_volume.to_numpy())
    state_bull = in_session_array & (~np.isnan(vwap)) & (close.to_numpy(dtype=float) > vwap) & vol_above_avg
    state_bear = in_session_array & (~np.isnan(vwap)) & (close.to_numpy(dtype=float) < vwap) & vol_above_avg
    state_label = pd.Series(np.where(state_bull, "Bullish",
                                     np.where(state_bear, "Bearish",
                                              np.where(in_session_array & ~np.isnan(vwap),
                                                       "Neutral", "-"))),
                            index=utc, dtype=object)

    return pd.DataFrame({
        "vwap": vwap_series,
        "sd1_upper": pd.Series(sd1u, index=utc),
        "sd1_lower": pd.Series(sd1l, index=utc),
        "sd2_upper": pd.Series(sd2u, index=utc),
        "sd2_lower": pd.Series(sd2l, index=utc),
        "sd3_upper": pd.Series(sd3u, index=utc),
        "sd3_lower": pd.Series(sd3l, index=utc),
        "in_session": flags(pd.Series(in_session_array, index=utc)),
        "new_session": flags(pd.Series(new_session_array, index=utc)),
        "distance_sd": pd.Series(distance_sd, index=utc),
        "volume_ok": vol_ok,
        "sell_touch": flags(pd.Series(sell_touch, index=utc)),
        "buy_touch": flags(pd.Series(buy_touch, index=utc)),
        "sell_signal": flags(pd.Series(sell_signal_values, index=utc)),
        "buy_signal": flags(pd.Series(buy_signal_values, index=utc)),
        "state": state_label,
    }, index=utc)
