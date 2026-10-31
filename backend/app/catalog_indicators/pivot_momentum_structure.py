"""Pivot Momentum Structure [WavesUnchained]  v2.9.1  —  catalog port.

Source: data/pine-collection/indicators/momentum/pivot_momentum_structure/
        pivot_momentum_structure.pine

Structure-oriented oscillator:
  * Selectable oscillator — RSI, WaveTrend, Stochastic RSI, CCI, Fisher, TSI,
    Williams %R, CMO or MFI.
  * Price pivots are the structural anchors; HH/HL/LH/LL classification with
    an ATR tolerance.
  * Regular and hidden bullish/bearish divergence between consecutive
    same-type pivots, gated by pivot distance and a minimum ATR price
    difference.
  * Momentum alignment (structure/oscillator agreement, distinct from
    divergence).

Ported computational core; drawing, labels, dashboard, alerts and the
oscillator-local-extreme anchor search are dropped (Exact Pivot Bar anchor).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr, cci, choice, ema, integer, mfi, number, ohlcv, pivothigh_strict,
    pivotlow_strict, rsi, sma, stochrsi, tsi, williams_r,
)


def _fisher(src, length):
    n = len(src)
    out = np.full(n, np.nan)
    val = 0.0
    fish = 0.0
    for i in range(n):
        lo = src[max(0, i - length + 1):i + 1].min()
        hi = src[max(0, i - length + 1):i + 1].max()
        x = 0.5 if hi <= lo else (src[i] - lo) / (hi - lo)
        x = 0.66 * (x - 0.5) + 0.67 * val
        x = max(-0.999, min(0.999, x))
        val = x
        fish = 0.5 * math.log((1 + x) / (1 - x)) + 0.5 * fish
        out[i] = fish
    return out


def _cmo(src, length):
    d = np.diff(src, prepend=src[0])
    gain = pd.Series(np.maximum(d, 0)).rolling(length, min_periods=length).sum()
    loss = pd.Series(np.maximum(-d, 0)).rolling(length, min_periods=length).sum()
    denom = gain + loss
    return np.where(denom != 0, 100 * (gain - loss) / np.where(denom == 0, np.nan, denom), 0.0)


_PRESET = {
    "Scalping": (5, 5, 8, 60, 0.15),
    "Intraday": (7, 7, 10, 100, 0.15),
    "Swing": (8, 8, 12, 120, 0.10),
    "Position": (10, 10, 15, 160, 0.10),
    "Major Structure": (14, 14, 20, 250, 0.15),
}


def pivot_momentum_structure(
    frame: pd.DataFrame,
    *,
    oscillator_type: str = "RSI",
    rsi_length: int = 14,
    rsi_signal_length: int = 5,
    wt_channel_length: int = 10,
    wt_average_length: int = 21,
    wt_signal_length: int = 4,
    stoch_rsi_length: int = 14,
    stoch_length: int = 14,
    cci_length: int = 20,
    fisher_length: int = 10,
    tsi_long_length: int = 25,
    tsi_short_length: int = 13,
    williams_length: int = 14,
    cmo_length: int = 14,
    mfi_length: int = 14,
    structure_preset: str = "Swing",
    pivot_left: int = 8,
    pivot_right: int = 8,
    minimum_pivot_distance: int = 12,
    maximum_pivot_distance: int = 120,
    minimum_price_difference_atr: float = 0.10,
    atr_length: int = 14,
    detect_regular: bool = True,
    detect_hidden: bool = True,
) -> pd.DataFrame:
    oscillator_type = choice("oscillator_type", oscillator_type,
                             ("RSI", "WaveTrend", "Stochastic RSI", "CCI", "Fisher", "TSI",
                              "Williams %R", "CMO", "MFI"))
    structure_preset = choice("structure_preset", structure_preset,
                              ("Scalping", "Intraday", "Swing", "Position", "Major Structure", "Custom"))
    for name, val, lo in [("rsi_length", rsi_length, 2), ("rsi_signal_length", rsi_signal_length, 1),
                          ("wt_channel_length", wt_channel_length, 2), ("wt_average_length", wt_average_length, 2),
                          ("stoch_rsi_length", stoch_rsi_length, 2), ("stoch_length", stoch_length, 2),
                          ("cci_length", cci_length, 2), ("fisher_length", fisher_length, 2),
                          ("tsi_long_length", tsi_long_length, 1), ("tsi_short_length", tsi_short_length, 1),
                          ("williams_length", williams_length, 2), ("cmo_length", cmo_length, 2),
                          ("mfi_length", mfi_length, 2), ("atr_length", atr_length, 1)]:
        integer(name, val, lo)
    if structure_preset == "Custom":
        pivot_left = integer("pivot_left", pivot_left, 1, 50)
        pivot_right = integer("pivot_right", pivot_right, 1, 50)
        minimum_pivot_distance = integer("minimum_pivot_distance", minimum_pivot_distance, 1, 500)
        maximum_pivot_distance = integer("maximum_pivot_distance", maximum_pivot_distance, 5, 2000)
        minimum_price_difference_atr = number("minimum_price_difference_atr", minimum_price_difference_atr, 0.0)
    else:
        pivot_left, pivot_right, minimum_pivot_distance, maximum_pivot_distance, minimum_price_difference_atr = _PRESET[structure_preset]

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    h = h.to_numpy(dtype=float); l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)
    src = frame["close"]
    hlc3 = (h + l + c) / 3.0
    hl2 = (h + l) / 2.0

    # oscillator
    if oscillator_type == "RSI":
        osc = rsi(src, rsi_length).to_numpy()
        sig_len = rsi_signal_length
    elif oscillator_type == "WaveTrend":
        esa = ema(pd.Series(hlc3), wt_channel_length).to_numpy()
        de = np.maximum(ema(pd.Series(np.abs(hlc3 - esa)), wt_channel_length).to_numpy(), 0.0)
        ci = np.where(de > 0, (hlc3 - esa) / np.where(de == 0, np.nan, 0.015 * de), 0.0)
        osc = ema(pd.Series(ci), wt_average_length).to_numpy()
        sig_len = wt_signal_length
    elif oscillator_type == "Stochastic RSI":
        sr = stochrsi(src, stoch_rsi_length).to_numpy()
        # stochastic of stoch-rsi then smoothing
        lo = pd.Series(sr).rolling(stoch_length, min_periods=stoch_length).min()
        hi = pd.Series(sr).rolling(stoch_length, min_periods=stoch_length).max()
        osc = (100 * (pd.Series(sr) - lo) / (hi - lo).replace(0, np.nan)).to_numpy()
        sig_len = 3
    elif oscillator_type == "CCI":
        osc = cci(frame, cci_length, "hlc3").to_numpy()
        sig_len = 9
    elif oscillator_type == "Fisher":
        osc = _fisher(hl2, fisher_length)
        sig_len = 1
    elif oscillator_type == "TSI":
        osc = tsi(src, tsi_short_length, tsi_long_length).to_numpy()
        sig_len = 7
    elif oscillator_type == "Williams %R":
        osc = williams_r(frame, williams_length).to_numpy()
        sig_len = 9
    elif oscillator_type == "CMO":
        osc = _cmo(c, cmo_length)
        sig_len = 9
    else:  # MFI
        osc = mfi(frame, mfi_length).to_numpy()
        sig_len = 9
    osc = np.nan_to_num(osc, nan=0.0)
    osc_signal = sma(pd.Series(osc), sig_len).to_numpy()

    atr_a = np.maximum(atr(frame, atr_length).to_numpy(dtype=float), 1e-8)
    ph = pivothigh_strict(frame["high"], pivot_left, pivot_right).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], pivot_left, pivot_right).to_numpy(dtype=float)

    structure = np.zeros(n, dtype=np.int64)
    div = np.zeros(n, dtype=np.int64)
    reg_bull = np.zeros(n, dtype=np.int64); reg_bear = np.zeros(n, dtype=np.int64)
    hid_bull = np.zeros(n, dtype=np.int64); hid_bear = np.zeros(n, dtype=np.int64)
    align_bull = np.zeros(n, dtype=np.int64); align_bear = np.zeros(n, dtype=np.int64)
    piv_high_price = np.full(n, np.nan); piv_low_price = np.full(n, np.nan)
    piv_high_bar = np.full(n, np.nan); piv_low_bar = np.full(n, np.nan)

    last_high = {"price": None, "bar": None, "osc": None}
    last_low = {"price": None, "bar": None, "osc": None}
    prev_high = {"price": None, "bar": None}
    prev_low = {"price": None, "bar": None}
    structure_now = 0

    for i in range(n):
        if not math.isnan(ph[i]):
            b = i - pivot_right
            price = ph[i]
            osc_v = osc[b]
            tol = atr_a[b] * 0.1
            if last_high["price"] is not None:
                # divergence vs previous high
                prev = last_high
                if prev["bar"] is not None:
                    dist = b - prev["bar"]
                    price_diff = abs(price - prev["price"])
                    if (minimum_pivot_distance <= dist <= maximum_pivot_distance
                            and price_diff >= atr_a[b] * minimum_price_difference_atr):
                        if detect_regular and price > prev["price"] and osc_v < prev["osc"]:
                            reg_bear[i] = 1
                        if detect_hidden and price < prev["price"] and osc_v > prev["osc"]:
                            hid_bear[i] = 1
                if price > last_high["price"] + tol:
                    structure_now = 1  # HH
                elif price < last_high["price"] - tol:
                    structure_now = 3  # LH
            prev_high = {"price": last_high["price"], "bar": last_high["bar"]}
            last_high = {"price": price, "bar": b, "osc": osc_v}
            piv_high_price[i] = price; piv_high_bar[i] = b
        if not math.isnan(pl[i]):
            b = i - pivot_right
            price = pl[i]
            osc_v = osc[b]
            tol = atr_a[b] * 0.1
            if last_low["price"] is not None:
                prev = last_low
                if prev["bar"] is not None:
                    dist = b - prev["bar"]
                    price_diff = abs(price - prev["price"])
                    if (minimum_pivot_distance <= dist <= maximum_pivot_distance
                            and price_diff >= atr_a[b] * minimum_price_difference_atr):
                        if detect_regular and price < prev["price"] and osc_v > prev["osc"]:
                            reg_bull[i] = 1
                        if detect_hidden and price > prev["price"] and osc_v < prev["osc"]:
                            hid_bull[i] = 1
                if price > last_low["price"] + tol:
                    structure_now = 2  # HL
                elif price < last_low["price"] - tol:
                    structure_now = 4  # LL
            prev_low = {"price": last_low["price"], "bar": last_low["bar"]}
            last_low = {"price": price, "bar": b, "osc": osc_v}
            piv_low_price[i] = price; piv_low_bar[i] = b

        structure[i] = structure_now
        if reg_bull[i] or hid_bull[i]:
            div[i] = 1
        elif reg_bear[i] or hid_bear[i]:
            div[i] = -1

    osc_rising = np.concatenate([[False], osc[1:] > osc[:-1]])
    osc_falling = np.concatenate([[False], osc[1:] < osc[:-1]])
    align_bull = ((structure == 1) | (structure == 2)) & osc_rising
    align_bear = ((structure == 3) | (structure == 4)) & osc_falling

    df = pd.DataFrame(index=idx)
    df["pms_osc"] = osc
    df["pms_osc_signal"] = osc_signal
    df["pms_structure"] = pd.Series(structure, index=idx).astype("Int64")
    df["pms_pivot_high_price"] = piv_high_price
    df["pms_pivot_low_price"] = piv_low_price
    df["pms_regular_bull_div"] = pd.Series(reg_bull, index=idx).astype("Int64")
    df["pms_regular_bear_div"] = pd.Series(reg_bear, index=idx).astype("Int64")
    df["pms_hidden_bull_div"] = pd.Series(hid_bull, index=idx).astype("Int64")
    df["pms_hidden_bear_div"] = pd.Series(hid_bear, index=idx).astype("Int64")
    df["pms_divergence"] = pd.Series(div, index=idx).astype("Int64")
    df["pms_momentum_bull"] = pd.Series(align_bull.astype(int), index=idx).astype("Int64")
    df["pms_momentum_bear"] = pd.Series(align_bear.astype(int), index=idx).astype("Int64")
    return df
