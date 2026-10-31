"""MTF Divergence — regular/hidden divergence across timeframes.

Source: pine-crypto-collection/btankutt_divergence/indicators/03-mtf-divergence/mtf-divergence.pine
(btankutt; no explicit license header in the source file)

On each requested timeframe the oscillator (RSI or the MACD histogram) is
scanned for pivots. Regular divergence: price makes a lower low while the
oscillator makes a higher low (bull) or price a higher high while the
oscillator a lower high (bear). Hidden divergence is the mirror (price lower
low + oscillator lower low = hidden bull; price higher high + oscillator
higher high = hidden bear). A signal fires on the first bar a divergence kind
becomes true; each timeframe keeps a persistent Bull/Bear/— state that the
latest signal overwrote.

The upstream `request.security(..., lookahead_on)` reads the oscillator of the
*previous completed* HTF bar (its `[1]`-shifted divergence conf); the
repo-wide convention reproduces this by computing divergences on HTF bars
aggregated from the input and publishing with a one-period lag. Timeframes
finer than the base bar spacing report false, as upstream.

Dropped: status table, on-chart labels and colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import boolean, choice, ema, flags, integer, ohlcv, pivothigh, pivotlow, rma
from ._tf import aligned_index, base_step_minutes, htf_frame


def _oscillator(frame: pd.DataFrame, osc_type: str, rsi_len: int, macd_fast: int,
                macd_slow: int, macd_sig: int) -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    if osc_type == "RSI":
        delta = close.diff()
        gain = rma(delta.clip(lower=0), rsi_len)
        loss = rma(-delta.clip(upper=0), rsi_len)
        values = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return values.mask(loss.eq(0) & gain.gt(0), 100.0)
    macd_line = ema(close, macd_fast) - ema(close, macd_slow)
    signal_line = ema(macd_line, macd_sig)
    return macd_line - signal_line


def _divergences(frame: pd.DataFrame, oscillator: pd.Series, pivot_left: int,
                 pivot_right: int) -> dict[str, pd.Series]:
    high, low, _ = ohlcv(frame, "high", "low", "close")
    osc_pivot_low = pivotlow(oscillator, pivot_left, pivot_right)
    osc_pivot_high = pivothigh(oscillator, pivot_left, pivot_right)
    rows = len(frame)
    prev_pl_osc = prev_pl_prc = np.nan
    prev_ph_osc = prev_ph_prc = np.nan
    reg_bull = np.zeros(rows, dtype=bool)
    reg_bear = np.zeros(rows, dtype=bool)
    hid_bull = np.zeros(rows, dtype=bool)
    hid_bear = np.zeros(rows, dtype=bool)
    lows = low.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    pl_values = osc_pivot_low.to_numpy(dtype=float)
    ph_values = osc_pivot_high.to_numpy(dtype=float)
    for position in range(rows):
        if not np.isnan(pl_values[position]):
            price = lows[position - pivot_right]
            value = pl_values[position]
            if not np.isnan(prev_pl_osc):
                if price < prev_pl_prc and value > prev_pl_osc:
                    reg_bull[position] = True
                if price > prev_pl_prc and value < prev_pl_osc:
                    hid_bull[position] = True
            prev_pl_osc = value
            prev_pl_prc = price
        if not np.isnan(ph_values[position]):
            price = highs[position - pivot_right]
            value = ph_values[position]
            if not np.isnan(prev_ph_osc):
                if price > prev_ph_prc and value < prev_ph_osc:
                    reg_bear[position] = True
                if price < prev_ph_prc and value > prev_ph_osc:
                    hid_bear[position] = True
            prev_ph_osc = value
            prev_ph_prc = price
    index = frame.index
    return {"regular_bull": pd.Series(reg_bull, index=index),
            "regular_bear": pd.Series(reg_bear, index=index),
            "hidden_bull": pd.Series(hid_bull, index=index),
            "hidden_bear": pd.Series(hid_bear, index=index)}


def mtf_divergence(
        frame: pd.DataFrame, *, osc_type: str = "RSI", rsi_len: int = 14,
        macd_fast: int = 12, macd_slow: int = 26, macd_sig: int = 9,
        pivot_left: int = 5, pivot_right: int = 5, show_regular: bool = True,
        show_hidden: bool = False, timeframes: tuple[int, ...] = (0, 60, 240),
        timeframe_labels: tuple[str, ...] | None = None) -> pd.DataFrame:
    choice("osc_type", osc_type, ("RSI", "MACD"))
    integer("rsi_len", rsi_len, 2, 100)
    integer("macd_fast", macd_fast, 1)
    integer("macd_slow", macd_slow, 2)
    integer("macd_sig", macd_sig, 1)
    integer("pivot_left", pivot_left, 1, 50)
    integer("pivot_right", pivot_right, 1, 50)
    boolean("show_regular", show_regular)
    boolean("show_hidden", show_hidden)
    utc = aligned_index(frame, 1)
    base = base_step_minutes(frame)
    labels = timeframe_labels or tuple(
        "chart" if minutes == 0 else f"{minutes}m" for minutes in timeframes)
    if len(labels) != len(timeframes):
        raise ValueError("timeframe_labels 与 timeframes 长度不一致")

    output = {}
    state = {label: "-" for label in labels}
    for label, minutes in zip(labels, timeframes):
        if minutes == 0:
            osc = _oscillator(frame, osc_type, rsi_len, macd_fast, macd_slow, macd_sig)
            div = _divergences(frame, osc, pivot_left, pivot_right)
            index = utc
        else:
            if minutes < base or minutes % base:
                false = pd.Series(False, index=utc)
                output[f"{label}_regular_bull"] = false
                output[f"{label}_regular_bear"] = false.copy()
                output[f"{label}_hidden_bull"] = false.copy()
                output[f"{label}_hidden_bear"] = false.copy()
                output[f"{label}_bull_signal"] = false.copy()
                output[f"{label}_bear_signal"] = false.copy()
                output[f"{label}_state"] = pd.Series("-", index=utc, dtype=object)
                continue
            htf = htf_frame(frame, minutes)
            osc = _oscillator(htf, osc_type, rsi_len, macd_fast, macd_slow, macd_sig)
            div = _divergences(htf, osc, pivot_left, pivot_right)
            index = htf.index
        # A signal fires on the bar a divergence kind first turns true; publish
        # the previous completed HTF bar's state (lookahead-free).
        fired_bull = ((div["regular_bull"] if show_regular else pd.Series(False, index=index))
                      | (div["hidden_bull"] if show_hidden else pd.Series(False, index=index)))
        fired_bear = ((div["regular_bear"] if show_regular else pd.Series(False, index=index))
                      | (div["hidden_bear"] if show_hidden else pd.Series(False, index=index)))
        bull_new = flags(fired_bull)
        bear_new = flags(fired_bear)
        states = np.full(len(index), "-", dtype=object)
        current = "-"
        bull_values = bull_new.to_numpy(dtype=bool)
        bear_values = bear_new.to_numpy(dtype=bool)
        for position in range(len(index)):
            if bull_values[position]:
                current = "Bull"
            if bear_values[position]:
                current = "Bear"
            states[position] = current
        state_series = pd.Series(states, index=index, dtype=object)

        if minutes == 0:
            output[f"{label}_regular_bull"] = div["regular_bull"]
            output[f"{label}_regular_bear"] = div["regular_bear"]
            output[f"{label}_hidden_bull"] = div["hidden_bull"]
            output[f"{label}_hidden_bear"] = div["hidden_bear"]
            output[f"{label}_bull_signal"] = bull_new
            output[f"{label}_bear_signal"] = bear_new
            output[f"{label}_state"] = state_series
        else:
            period = utc.floor(f"{minutes}min")
            previous = period - pd.Timedelta(minutes=minutes)
            lookup = pd.Series(range(len(previous)), index=previous)
            position = lookup.reindex(previous).to_numpy(dtype=float)  # exact match
            position = pd.Series(position).ffill().bfill().fillna(0).to_numpy(dtype=int)
            for kind, series in (("regular_bull", div["regular_bull"]),
                                 ("regular_bear", div["regular_bear"]),
                                 ("hidden_bull", div["hidden_bull"]),
                                 ("hidden_bear", div["hidden_bear"]),
                                 ("bull_signal", bull_new),
                                 ("bear_signal", bear_new)):
                values = series.reindex(sorted(set(previous))).fillna(False).astype(bool)
                lifted = values.reindex(previous).fillna(False)
                lifted.index = utc
                output[f"{label}_{kind}"] = lifted
            state_lookup = state_series.reindex(sorted(set(previous))).fillna("-")
            state_lifted = state_lookup.reindex(previous).fillna("-")
            state_lifted.index = utc
            output[f"{label}_state"] = state_lifted
        state[label] = state_series.iloc[-1] if len(state_series) else "-"

    any_bull = np.zeros(len(utc), dtype=bool)
    any_bear = np.zeros(len(utc), dtype=bool)
    for label in labels:
        any_bull |= output[f"{label}_bull_signal"].to_numpy(dtype=bool)
        any_bear |= output[f"{label}_bear_signal"].to_numpy(dtype=bool)
    output["any_bull"] = flags(pd.Series(any_bull, index=utc))
    output["any_bear"] = flags(pd.Series(any_bear, index=utc))
    return pd.DataFrame(output, index=utc)
