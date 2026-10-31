"""MSO — Market Stress Oscillator [WavesUnchained] — Pine v6 port (catalog, no plots).

Bi-directional Williams-VIX-Fix stress oscillator with JMA trend context and ADX
quality grading. For every confirmed stress extreme it records *peak-hold* clusters
and then grades the forward follow-through after the peak (acceptance score).

Computational core ported (display/logging/HTF dropped):
  * JMA smoothing of close (calculateJMA, aligned with the source's approximation)
  * Bi-directional WVF: wvfLong from highest-close, wvfShort from lowest-close
  * Z-scores of each WVF series over a rolling window (sd==0 -> 0)
  * ADX strength context (quality grade only, never gates a trigger)
  * Peak-hold clustering of the raw extremes (strongest Z per cluster)
  * Follow-through acceptance: highest-high / lowest-low after the peak bar,
    ATR-normalised, compared to minFollowATR -> accepted/rejected + 0..1 score

Naming note: in the source a "long" stress event is a *downside* stress (a sharp
dip, expecting a reversal up, label DA) and a "short" stress event is an *upside*
stress (a blow-off, expecting a reversal down, label UA). The column names
``long_*`` / ``short_*`` follow that convention. Pivots, zones, labels, tables,
alerts and the higher-timeframe "wind" overlay are display/logging only and are
not emitted.

Causality: event columns are stamped at the *resolution* bar (peak bar + reactBars),
using only data available up to that bar; regime / bias columns are per-bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    dmi,
    ema,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    sma,
    stdev,
)

# Per-timeframe presets (mirrors the source's f_sel table). "Auto" resolves to 1h
# because the Python port has no chart timeframe context.
_PRESETS = {
    "15m": dict(wvf=34, z=144, zthr=2.0, jma=55, adx_min=18, min_ext=2, cool=12, follow=0.7),
    "1h": dict(wvf=34, z=144, zthr=2.0, jma=89, adx_min=18, min_ext=2, cool=8, follow=0.6),
    "4h": dict(wvf=55, z=200, zthr=2.1, jma=144, adx_min=18, min_ext=2, cool=5, follow=0.5),
    "D": dict(wvf=89, z=252, zthr=2.1, jma=200, adx_min=18, min_ext=3, cool=5, follow=0.45),
    "W": dict(wvf=89, z=252, zthr=2.2, jma=233, adx_min=18, min_ext=3, cool=4, follow=0.45),
    "M": dict(wvf=144, z=300, zthr=2.5, jma=377, adx_min=18, min_ext=3, cool=3, follow=0.45),
}


def _z(x: pd.Series, win: int) -> pd.Series:
    """Pine f_z: (x - sma) / stdev, returning 0 when the window stdev is 0."""
    mu = sma(x, win).to_numpy(dtype=float)
    sd = stdev(x, win).to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    out = np.where(sd == 0.0, 0.0, (xv - mu) / np.where(sd == 0.0, 1.0, sd))
    return pd.Series(out, index=x.index)


def calculate_jma(src: pd.Series, length: int, phase: float = 0.0, power: float = 2.0) -> pd.Series:
    """JMA approximation — matches the source's calculateJMA (C++/CPM aligned)."""
    s = src.to_numpy(dtype=float)
    n = len(s)
    out = np.full(n, np.nan)
    if n == 0 or length < 1:
        return pd.Series(out, index=src.index)
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    a1 = 1.0 - alpha
    a1_2 = a1 * a1
    alpha_2 = alpha * alpha
    e0 = np.nan
    e1 = np.nan
    e2 = np.nan
    jma = np.nan
    for i in range(n):
        si = s[i]
        e0_prev = si if np.isnan(e0) else e0
        e0 = a1 * si + alpha * e0_prev
        e1_prev = 0.0 if np.isnan(e1) else e1
        e1 = (si - e0) * (1.0 - beta) + beta * e1_prev
        jma_prev = si if np.isnan(jma) else jma
        e2_prev = 0.0 if np.isnan(e2) else e2
        e2 = (e0 + phase_ratio * e1 - jma_prev) * a1_2 + alpha_2 * e2_prev
        jma = e2 + jma_prev
        out[i] = jma
    return pd.Series(out, index=src.index)


def mso_market_stress_oscillator(
    frame: pd.DataFrame,
    *,
    tf: str = "1h",
    # preset-resolved knobs (override any of them explicitly)
    wvf_len: int | None = None,
    z_win: int | None = None,
    z_thr: float | None = None,
    jma_len: int | None = None,
    adx_min: int | None = None,
    min_extreme_bars: int | None = None,
    cooldown_bars: int | None = None,
    min_follow_atr: float | None = None,
    # core settings
    adx_len: int = 14,
    atr_len: int = 14,
    jma_phase: float = 0.0,
    jma_power: float = 2.0,
    react_bars: int = 2,
    use_peak_hold: bool = True,
    use_slope_filter: bool = True,
    require_trend_align: bool = False,
    jma_slope_len: int = 5,
    bias_smooth_len: int = 5,
    bias_band: float = 0.8,
) -> pd.DataFrame:
    """Market Stress Oscillator: bi-directional WVF stress with JMA/ADX context.

    Returns a DataFrame (index = frame index) with the continuous stress/bias/ADX
    series plus event columns stamped at each stress resolution bar.
    """
    tf = choice("tf", tf, tuple(_PRESETS.keys()))
    p = _PRESETS[tf]
    wvf_len = integer("wvf_len", wvf_len if wvf_len is not None else p["wvf"], 5, 2000)
    z_win = integer("z_win", z_win if z_win is not None else p["z"], 20, 3000)
    z_thr = number("z_thr", z_thr if z_thr is not None else p["zthr"], 0.1, 10.0)
    jma_len = integer("jma_len", jma_len if jma_len is not None else p["jma"], 5, 1000)
    adx_min = integer("adx_min", adx_min if adx_min is not None else p["adx_min"], 5, 100)
    min_extreme_bars = integer(
        "min_extreme_bars", min_extreme_bars if min_extreme_bars is not None else p["min_ext"], 1, 200
    )
    cooldown_bars = integer("cooldown_bars", cooldown_bars if cooldown_bars is not None else p["cool"], 0, 1000)
    min_follow_atr = number(
        "min_follow_atr", min_follow_atr if min_follow_atr is not None else p["follow"], 0.0, 50.0
    )
    adx_len = integer("adx_len", adx_len, 5, 200)
    atr_len = integer("atr_len", atr_len, 1, 200)
    jma_phase = number("jma_phase", jma_phase, -100.0, 100.0)
    jma_power = number("jma_power", jma_power, 1.0, 8.0)
    react_bars = integer("react_bars", react_bars, 1, 20)
    use_peak_hold = boolean("use_peak_hold", use_peak_hold)
    use_slope_filter = boolean("use_slope_filter", use_slope_filter)
    require_trend_align = boolean("require_trend_align", require_trend_align)
    jma_slope_len = integer("jma_slope_len", jma_slope_len, 1, 500)
    bias_smooth_len = integer("bias_smooth_len", bias_smooth_len, 1, 500)
    bias_band = number("bias_band", bias_band, 0.0, 5.0)

    high, low, close = ohlcv(frame, "high", "low", "close")
    index = frame.index
    n = len(frame)
    hi = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    cl = close.to_numpy(dtype=float)

    # ---- base series -------------------------------------------------------
    atr_s = atr(frame, atr_len).to_numpy(dtype=float)
    _di_plus, _di_minus, adx_s = dmi(frame, adx_len)
    adx_a = adx_s.to_numpy(dtype=float)

    jma_s = calculate_jma(close, jma_len, jma_phase, jma_power).to_numpy(dtype=float)
    jma_slope = jma_s - np.concatenate(([np.nan], jma_s[:-1]))
    jma_slope_sm = ema(pd.Series(jma_slope, index=index), jma_slope_len).to_numpy(dtype=float)

    hhc = highest(close, wvf_len).to_numpy(dtype=float)
    llc = lowest(close, wvf_len).to_numpy(dtype=float)
    wvf_long = np.where(hhc != 0, (hhc - lo) / hhc * 100.0, np.nan)
    wvf_short = np.where(llc != 0, (hi - llc) / llc * 100.0, np.nan)
    zL = _z(pd.Series(wvf_long, index=index), z_win).to_numpy(dtype=float)
    zS = _z(pd.Series(wvf_short, index=index), z_win).to_numpy(dtype=float)

    bias_raw = zL - zS
    bias_sm = ema(pd.Series(bias_raw, index=index), bias_smooth_len).to_numpy(dtype=float)

    trend_long_ok = (
        (jma_slope_sm > 0) if use_slope_filter else (cl >= jma_s)
    )
    trend_short_ok = (
        (jma_slope_sm < 0) if use_slope_filter else (cl <= jma_s)
    )
    trend_strength_ok = adx_a >= adx_min
    adx_grade = np.minimum(1.0, adx_a / max(adx_min, 1))

    ext_long_raw = zL >= z_thr
    ext_short_raw = zS >= z_thr

    # streak length of the raw extreme (1-based), like ta.barssince(not raw)
    def _streak(flag: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=int)
        run = 0
        for i in range(n):
            run = run + 1 if flag[i] else 0
            out[i] = run
        return out

    ext_long_dur = _streak(ext_long_raw)
    ext_short_dur = _streak(ext_short_raw)

    # ---- output arrays -----------------------------------------------------
    long_event = np.full(n, np.nan)
    long_score = np.full(n, np.nan)
    long_follow = np.full(n, np.nan)
    long_peak_z = np.full(n, np.nan)
    long_peak_price = np.full(n, np.nan)
    long_peak_atr = np.full(n, np.nan)
    long_peak_idx = np.full(n, np.nan)
    short_event = np.full(n, np.nan)
    short_score = np.full(n, np.nan)
    short_follow = np.full(n, np.nan)
    short_peak_z = np.full(n, np.nan)
    short_peak_price = np.full(n, np.nan)
    short_peak_atr = np.full(n, np.nan)
    short_peak_idx = np.full(n, np.nan)
    last_event_side = np.full(n, np.nan)
    last_event_accepted = np.full(n, np.nan)

    # ---- sequential state machine -----------------------------------------
    long_in_clus = False
    lp_z = np.nan
    lp_price = np.nan
    lp_close = np.nan
    lp_high = np.nan
    lp_bar = -1
    lp_maxhigh = np.nan
    lp_bars = 0

    short_in_clus = False
    sp_z = np.nan
    sp_price = np.nan
    sp_close = np.nan
    sp_low = np.nan
    sp_bar = -1
    sp_minlow = np.nan
    sp_bars = 0

    pending_long = False
    pl_bar = -1
    pl_price = np.nan
    pl_close = np.nan
    pl_high = np.nan
    pl_maxhigh = np.nan
    pl_minlow = np.nan
    pl_atr = np.nan
    pl_z = np.nan

    pending_short = False
    ps_bar = -1
    ps_price = np.nan
    ps_close = np.nan
    ps_low = np.nan
    ps_minlow = np.nan
    ps_maxhigh = np.nan
    ps_atr = np.nan
    ps_z = np.nan

    last_long_event = -10 ** 9
    last_short_event = -10 ** 9
    last_side_val = np.nan
    last_acc_val = np.nan

    def _nmax(a, b):
        if np.isnan(a):
            return b
        if np.isnan(b):
            return a
        return max(a, b)

    def _nmin(a, b):
        if np.isnan(a):
            return b
        if np.isnan(b):
            return a
        return min(a, b)

    for i in range(n):
        # ---- peak-hold clustering (long / downside stress) ----
        if ext_long_raw[i]:
            if not long_in_clus:
                long_in_clus = True
                lp_z = zL[i]
                lp_price = lo[i]
                lp_close = cl[i]
                lp_high = hi[i]
                lp_bar = i
                lp_maxhigh = np.nan
                lp_bars = 0
            elif zL[i] > (lp_z if not np.isnan(lp_z) else zL[i]):
                lp_z = zL[i]
                lp_price = lo[i]
                lp_close = cl[i]
                lp_high = hi[i]
                lp_bar = i
                lp_maxhigh = np.nan
                lp_bars = 0
            else:
                lp_maxhigh = _nmax(lp_maxhigh, hi[i])
                lp_bars += 1
        elif long_in_clus:
            long_in_clus = False

        # ---- peak-hold clustering (short / upside stress) ----
        if ext_short_raw[i]:
            if not short_in_clus:
                short_in_clus = True
                sp_z = zS[i]
                sp_price = hi[i]
                sp_close = cl[i]
                sp_low = lo[i]
                sp_bar = i
                sp_minlow = np.nan
                sp_bars = 0
            elif zS[i] > (sp_z if not np.isnan(sp_z) else zS[i]):
                sp_z = zS[i]
                sp_price = hi[i]
                sp_close = cl[i]
                sp_low = lo[i]
                sp_bar = i
                sp_minlow = np.nan
                sp_bars = 0
            else:
                sp_minlow = _nmin(sp_minlow, lo[i])
                sp_bars += 1
        elif short_in_clus:
            short_in_clus = False

        prev_long_raw = ext_long_raw[i - 1] if i > 0 else False
        prev_short_raw = ext_short_raw[i - 1] if i > 0 else False
        end_long_cluster = bool(prev_long_raw) and (not ext_long_raw[i])
        end_short_cluster = bool(prev_short_raw) and (not ext_short_raw[i])

        ext_long_i = bool(ext_long_raw[i]) and ext_long_dur[i] >= min_extreme_bars
        ext_short_i = bool(ext_short_raw[i]) and ext_short_dur[i] >= min_extreme_bars

        long_qual_i = ext_long_i and (not require_trend_align or bool(trend_long_ok[i]))
        short_qual_i = ext_short_i and (not require_trend_align or bool(trend_short_ok[i]))

        if use_peak_hold:
            prev_dur = ext_long_dur[i - 1] if i > 0 else 0
            if require_trend_align:
                peak_bar = lp_bar - (lp_bars + 1)
                ok = bool(trend_long_ok[peak_bar]) if 0 <= peak_bar < n else False
            else:
                ok = True
            long_qual_at_peak = (prev_dur >= min_extreme_bars) and (not require_trend_align or ok)

            prev_dur_s = ext_short_dur[i - 1] if i > 0 else 0
            if require_trend_align:
                peak_bar_s = sp_bar - (sp_bars + 1)
                ok_s = bool(trend_short_ok[peak_bar_s]) if 0 <= peak_bar_s < n else False
            else:
                ok_s = True
            short_qual_at_peak = (prev_dur_s >= min_extreme_bars) and (not require_trend_align or ok_s)
        else:
            long_qual_at_peak = long_qual_i
            short_qual_at_peak = short_qual_i

        allow_long = (i - last_long_event) >= cooldown_bars
        allow_short = (i - last_short_event) >= cooldown_bars

        if use_peak_hold:
            long_trigger = (not pending_long) and allow_long and end_long_cluster and long_qual_at_peak
            short_trigger = (not pending_short) and allow_short and end_short_cluster and short_qual_at_peak
        else:
            long_trigger = (not pending_long) and allow_long and long_qual_i
            short_trigger = (not pending_short) and allow_short and short_qual_i

        if long_trigger:
            pending_long = True
            pl_bar = lp_bar
            pl_price = lp_price
            pl_close = lp_close
            pl_high = lp_high
            pl_maxhigh = lp_maxhigh
            pl_minlow = lp_price
            pl_atr = atr_s[lp_bar] if 0 <= lp_bar < n else atr_s[i]
            pl_z = lp_z

        if short_trigger:
            pending_short = True
            ps_bar = sp_bar
            ps_price = sp_price
            ps_close = sp_close
            ps_low = sp_low
            ps_minlow = sp_minlow
            ps_maxhigh = sp_price
            ps_atr = atr_s[sp_bar] if 0 <= sp_bar < n else atr_s[i]
            ps_z = sp_z

        # ---- pending long resolution ----
        if pending_long:
            if i > pl_bar:
                pl_maxhigh = _nmax(pl_maxhigh, hi[i])
                pl_minlow = _nmin(pl_minlow, lo[i])
            if (i - pl_bar) >= react_bars:
                denom = _nmax(pl_atr if not np.isnan(pl_atr) else atr_s[i], 1e-4)
                follow = (pl_maxhigh - pl_close) / denom
                score = max(0.0, min(1.0, follow / min_follow_atr))
                accepted = follow >= min_follow_atr
                long_event[i] = 1.0 if accepted else 0.0
                long_score[i] = score
                long_follow[i] = follow
                long_peak_z[i] = pl_z
                long_peak_price[i] = pl_price
                long_peak_atr[i] = pl_atr
                long_peak_idx[i] = float(pl_bar)
                last_side_val = -1.0
                last_acc_val = 1.0 if accepted else 0.0
                last_long_event = i
                pending_long = False
                pl_atr = np.nan

        # ---- pending short resolution ----
        if pending_short:
            if i > ps_bar:
                ps_minlow = _nmin(ps_minlow, lo[i])
                ps_maxhigh = _nmax(ps_maxhigh, hi[i])
            if (i - ps_bar) >= react_bars:
                denom = _nmax(ps_atr if not np.isnan(ps_atr) else atr_s[i], 1e-4)
                follow = (ps_close - ps_minlow) / denom
                score = max(0.0, min(1.0, follow / min_follow_atr))
                accepted = follow >= min_follow_atr
                short_event[i] = 1.0 if accepted else 0.0
                short_score[i] = score
                short_follow[i] = follow
                short_peak_z[i] = ps_z
                short_peak_price[i] = ps_price
                short_peak_atr[i] = ps_atr
                short_peak_idx[i] = float(ps_bar)
                last_side_val = 1.0
                last_acc_val = 1.0 if accepted else 0.0
                last_short_event = i
                pending_short = False
                ps_atr = np.nan

        last_event_side[i] = last_side_val
        last_event_accepted[i] = last_acc_val

    in_down = bias_sm > bias_band
    in_up = bias_sm < -bias_band
    regime = np.where(in_down, 1, np.where(in_up, -1, 0)).astype("int64")

    data = {
        "z_long": pd.Series(zL, index=index),
        "z_short": pd.Series(zS, index=index),
        "wvf_long": pd.Series(wvf_long, index=index),
        "wvf_short": pd.Series(wvf_short, index=index),
        "jma": pd.Series(jma_s, index=index),
        "jma_slope_sm": pd.Series(jma_slope_sm, index=index),
        "bias": pd.Series(bias_sm, index=index),
        "adx": pd.Series(adx_a, index=index),
        "adx_grade": pd.Series(adx_grade, index=index),
        "ext_long_raw": pd.Series(ext_long_raw.astype("int64"), index=index),
        "ext_short_raw": pd.Series(ext_short_raw.astype("int64"), index=index),
        "ext_long": pd.Series((ext_long_raw & (ext_long_dur >= min_extreme_bars)).astype("int64"), index=index),
        "ext_short": pd.Series((ext_short_raw & (ext_short_dur >= min_extreme_bars)).astype("int64"), index=index),
        "trend_long_ok": pd.Series(trend_long_ok.astype("int64"), index=index),
        "trend_short_ok": pd.Series(trend_short_ok.astype("int64"), index=index),
        "regime": pd.Series(regime, index=index),
        "long_event": pd.Series(long_event, index=index, dtype="Int64"),
        "long_score": pd.Series(long_score, index=index),
        "long_follow": pd.Series(long_follow, index=index),
        "long_peak_z": pd.Series(long_peak_z, index=index),
        "long_peak_price": pd.Series(long_peak_price, index=index),
        "long_peak_atr": pd.Series(long_peak_atr, index=index),
        "long_peak_idx": pd.Series(long_peak_idx, index=index, dtype="Int64"),
        "short_event": pd.Series(short_event, index=index, dtype="Int64"),
        "short_score": pd.Series(short_score, index=index),
        "short_follow": pd.Series(short_follow, index=index),
        "short_peak_z": pd.Series(short_peak_z, index=index),
        "short_peak_price": pd.Series(short_peak_price, index=index),
        "short_peak_atr": pd.Series(short_peak_atr, index=index),
        "short_peak_idx": pd.Series(short_peak_idx, index=index, dtype="Int64"),
        "last_event_side": pd.Series(last_event_side, index=index, dtype="Int64"),
        "last_event_accepted": pd.Series(last_event_accepted, index=index, dtype="Int64"),
    }
    return pd.DataFrame(data, index=index)
