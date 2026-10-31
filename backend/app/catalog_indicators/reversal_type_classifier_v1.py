"""Reversal Type Classifier v1 [WavesUnchained] — Pine v6 port (catalog, no plots).

Ex-post diagnostic: classifies each *confirmed* pivot reversal by outcome type
over an evaluation window after the pivot:

    1 = A_SNAPBACK   quick bounce that hits the snap target R fast
    2 = B_GRIND      sustained follow-through (>= minFollowR) above the pivot
    3 = C_FAKE       a stop-run / failed move (hit the fail threshold first)
    4 = D_CHOP       unclear / no decisive follow-through

Per pivot it also reports the WaveTrend / StochRSI / BB-width / ATR-rank / ADX
context at the pivot bar and a 0–100 quality score. The original table / label /
bgcolor / log widgets are dropped (display only); the per-pivot classification is
emitted at the pivot bar, plus a forward-filled ``last_type`` / ``last_side``
regime column.

Causality note: like the source, classification for a pivot confirmed at bar ``c``
is only knowable ``evalBars`` bars later (bar ``c + evalBars``), because it needs
the post-pivot outcome window. Results are therefore stamped at the *pivot* bar
(the event), while ``last_type`` updates at the *evaluation* bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    bb,
    boolean,
    dmi,
    ema,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    rsi,
    sma,
    stdev,
    stoch,
)

_TYPE_NAME = {1: "A_SNAPBACK", 2: "B_GRIND", 3: "C_FAKE", 4: "D_CHOP"}


def _wt(hlc3: pd.Series, ch_len: int, avg_len: int, sig_len: int):
    """LazyBear WaveTrend (matches Pine f_wt)."""
    esa = ema(hlc3, ch_len)
    dev = ema((hlc3 - esa).abs(), ch_len)
    hlc3_a = hlc3.to_numpy(dtype=float)
    esa_a = esa.to_numpy(dtype=float)
    dev_a = dev.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = pd.Series(np.where(dev_a != 0.0,
                                (hlc3_a - esa_a) / (0.015 * dev_a),
                                0.0), index=hlc3.index)
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, sig_len)
    return wt1, wt2


def _first_hit(high_win: np.ndarray, low_win: np.ndarray, price: float, risk: float,
               target_r: float, fail_r: float):
    """Mirror Pine f_firstHitLong / f_firstHitShort.

    Returns (first_target, first_fail) as 1-based bar counts measured from the
    oldest bar of the window (i.e. bars after the pivot). ``None`` if never hit.
    ``high_win`` / ``low_win`` are ordered oldest -> newest.
    """
    w = len(high_win)
    target = price + target_r * risk
    fail = price - fail_r * risk
    first_target = None
    first_fail = None
    for k in range(w):
        if first_target is None and high_win[k] >= target:
            first_target = k + 1
        if first_fail is None and low_win[k] <= fail:
            first_fail = k + 1
    return first_target, first_fail


def _quality_score(div, extreme, trend_against, chop, mfe_r, mae_r, atr_rnk,
                   grind_target_r, min_follow_r, fake_fail_r):
    s = 50.0
    s += 18.0 if div else 0.0
    s += 12.0 if extreme else -8.0
    s -= 14.0 if trend_against else 0.0
    s -= 8.0 if chop else 0.0
    s += 12.0 if mfe_r >= grind_target_r else (5.0 if mfe_r >= min_follow_r else -12.0)
    s -= 10.0 if mae_r >= fake_fail_r else 0.0
    s += 4.0 if (20.0 <= atr_rnk <= 80.0) else -3.0
    return float(min(100.0, max(0.0, s)))


def _pivot_strict(series: pd.Series, left: int, right: int, highest_pivot: bool) -> pd.Series:
    """Strict local-extremum pivot (Pine ta.pivotlow / ta.pivothigh semantics).

    A pivot bar ``i`` requires the value to be strictly less (low) or greater
    (high) than *every* neighbour in ``[i-left, i-1]`` and ``[i+1, i+right]``.
    The value is reported at the confirmation bar ``i + right`` (matching Pine's
    lag), so a caller reading ``series[c]`` at a confirm bar gets the pivot value
    with ``c - right`` the actual pivot bar.

    NOTE: kept local (not in _pine) because the shared _pine._pivot currently
    uses a looser comparison that over-reports pivot lows; this port needs the
    strict definition. See project memory (pivot bug).
    """
    s = series.astype(float)
    v = s.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    if n < left + right + 1:
        return pd.Series(out, index=s.index)
    if highest_pivot:
        l_neigh = s.rolling(left).max().shift(1) if left else pd.Series(np.nan, index=s.index)
        r = s[::-1]
        r_neigh = r.rolling(right).max().shift(1)[::-1] if right else pd.Series(np.nan, index=s.index)
        neigh = np.full(n, -np.inf)
        if left:
            neigh = np.fmax(neigh, l_neigh.fillna(-np.inf).to_numpy())
        if right:
            neigh = np.fmax(neigh, r_neigh.fillna(-np.inf).to_numpy())
        cmp_ok = v > neigh
    else:
        l_neigh = s.rolling(left).min().shift(1) if left else pd.Series(np.nan, index=s.index)
        r = s[::-1]
        r_neigh = r.rolling(right).min().shift(1)[::-1] if right else pd.Series(np.nan, index=s.index)
        neigh = np.full(n, np.inf)
        if left:
            neigh = np.fmin(neigh, l_neigh.fillna(np.inf).to_numpy())
        if right:
            neigh = np.fmin(neigh, r_neigh.fillna(np.inf).to_numpy())
        cmp_ok = v < neigh
    mask = np.ones(n, dtype=bool)
    if left:
        mask &= ~np.isnan(l_neigh.to_numpy())
    if right:
        mask &= ~np.isnan(r_neigh.to_numpy())
    valid = mask & cmp_ok
    idx = np.where(valid)[0]
    out[idx + right] = v[idx]
    return pd.Series(out, index=s.index)


def reversal_type_classifier_v1(
    frame: pd.DataFrame,
    *,
    piv_len: int = 5,
    eval_bars: int = 24,
    min_move_atr: float = 0.30,
    ch_len: int = 10,
    avg_len: int = 21,
    sig_len: int = 4,
    os_level: float = -45.0,
    ob_level: float = 45.0,
    atr_len: int = 14,
    risk_atr_mult: float = 1.5,
    snap_target_r: float = 1.2,
    grind_target_r: float = 2.0,
    fake_fail_r: float = 1.0,
    min_follow_r: float = 0.45,
    snap_bars: int = 6,
    ema_len: int = 100,
    adx_len: int = 14,
    adx_trend_min: float = 22.0,
    atr_rank_len: int = 100,
    chop_adx_max: float = 16.0,
    srsi_len: int = 14,
    srsi_smooth: int = 3,
    bb_len: int = 20,
    show_snap: bool = True,
    show_grind: bool = True,
    show_fake: bool = True,
    show_chop: bool = True,
) -> pd.DataFrame:
    """Classify confirmed pivot reversals by outcome type (Snapback/Grind/Fake/Chop)."""
    piv_len = integer("piv_len", piv_len, 2, 30)
    eval_bars = integer("eval_bars", eval_bars, 5, 200)
    min_move_atr = number("min_move_atr", min_move_atr, 0.0, 50.0)
    ch_len = integer("ch_len", ch_len, 1, 200)
    avg_len = integer("avg_len", avg_len, 1, 200)
    sig_len = integer("sig_len", sig_len, 1, 200)
    os_level = number("os_level", os_level, -100.0, 0.0)
    ob_level = number("ob_level", ob_level, 0.0, 100.0)
    atr_len = integer("atr_len", atr_len, 1, 200)
    risk_atr_mult = number("risk_atr_mult", risk_atr_mult, 0.1, 50.0)
    snap_target_r = number("snap_target_r", snap_target_r, 0.2, 50.0)
    grind_target_r = number("grind_target_r", grind_target_r, 0.5, 50.0)
    fake_fail_r = number("fake_fail_r", fake_fail_r, 0.3, 50.0)
    min_follow_r = number("min_follow_r", min_follow_r, 0.0, 50.0)
    snap_bars = integer("snap_bars", snap_bars, 1, 50)
    ema_len = integer("ema_len", ema_len, 10, 500)
    adx_len = integer("adx_len", adx_len, 5, 200)
    adx_trend_min = number("adx_trend_min", adx_trend_min, 5.0, 100.0)
    atr_rank_len = integer("atr_rank_len", atr_rank_len, 20, 1000)
    chop_adx_max = number("chop_adx_max", chop_adx_max, 5.0, 100.0)
    srsi_len = integer("srsi_len", srsi_len, 3, 200)
    srsi_smooth = integer("srsi_smooth", srsi_smooth, 1, 100)
    bb_len = integer("bb_len", bb_len, 5, 500)
    show_snap = boolean("show_snap", show_snap)
    show_grind = boolean("show_grind", show_grind)
    show_fake = boolean("show_fake", show_fake)
    show_chop = boolean("show_chop", show_chop)

    high, low, close = ohlcv(frame, "high", "low", "close")
    index = frame.index
    n = len(frame)
    hi = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    cl = close.to_numpy(dtype=float)
    hlc3 = (hi + lo + cl) / 3.0

    # ---- base series --------------------------------------------------------
    wt1, _wt2 = _wt(pd.Series(hlc3, index=index), ch_len, avg_len, sig_len)
    wt1_a = wt1.to_numpy(dtype=float)
    atr_s = atr(frame, atr_len).to_numpy(dtype=float)
    ema_s = ema(close, ema_len).to_numpy(dtype=float)
    _plus_di, _minus_di, adx_s = dmi(frame, adx_len)
    adx_s = adx_s.to_numpy(dtype=float)

    atr_lo = lowest(pd.Series(atr_s, index=index), atr_rank_len).to_numpy(dtype=float)
    atr_hi = highest(pd.Series(atr_s, index=index), atr_rank_len).to_numpy(dtype=float)
    atr_rank = np.where(
        atr_hi != atr_lo,
        100.0 * (atr_s - atr_lo) / np.where(atr_hi != atr_lo, atr_hi - atr_lo, 1.0),
        50.0,
    )

    rsi14 = rsi(close, srsi_len)
    srsi_raw = stoch(rsi14, rsi14, rsi14, srsi_len)
    srsi_k = ema(srsi_raw, srsi_smooth).to_numpy(dtype=float)

    bb_basis, bb_up, bb_low = bb(close, bb_len, 2.0)
    bb_width = (bb_up - bb_low).to_numpy(dtype=float)
    bbw_lo = lowest(pd.Series(bb_width, index=index), atr_rank_len).to_numpy(dtype=float)
    bbw_hi = highest(pd.Series(bb_width, index=index), atr_rank_len).to_numpy(dtype=float)
    bbw_rnk = np.where(
        bbw_hi != bbw_lo,
        100.0 * (bb_width - bbw_lo) / np.where(bbw_hi != bbw_lo, bbw_hi - bbw_lo, 1.0),
        50.0,
    )

    pl = _pivot_strict(low, piv_len, piv_len, False)
    ph = _pivot_strict(high, piv_len, piv_len, True)
    pl_a = pl.to_numpy(dtype=float)
    ph_a = ph.to_numpy(dtype=float)

    bars_from_pivot = eval_bars + piv_len

    # ---- pivot record tables ------------------------------------------------
    # Long pivots: keyed by confirm bar c -> pivot_bar / context at pivot_bar.
    long_recs: dict[int, dict] = {}
    prev_price = np.nan
    prev_wt = np.nan
    for c in range(n):
        pv = pl_a[c]
        if np.isnan(pv):
            continue
        pb = c - piv_len
        wt_p = wt1_a[pb]
        rec = {
            "pivot_bar": pb, "price": pv, "wt": wt_p,
            "atr": atr_s[pb], "ema": ema_s[pb], "adx": adx_s[pb],
            "atr_rank": atr_rank[pb], "srsi": srsi_k[pb], "bbw": bbw_rnk[pb],
            "close": cl[pb],
            "prev_price": prev_price, "prev_wt": prev_wt,
        }
        long_recs[c] = rec
        prev_price, prev_wt = pv, wt_p

    short_recs: dict[int, dict] = {}
    prev_price = np.nan
    prev_wt = np.nan
    for c in range(n):
        pv = ph_a[c]
        if np.isnan(pv):
            continue
        pb = c - piv_len
        wt_p = wt1_a[pb]
        rec = {
            "pivot_bar": pb, "price": pv, "wt": wt_p,
            "atr": atr_s[pb], "ema": ema_s[pb], "adx": adx_s[pb],
            "atr_rank": atr_rank[pb], "srsi": srsi_k[pb], "bbw": bbw_rnk[pb],
            "close": cl[pb],
            "prev_price": prev_price, "prev_wt": prev_wt,
        }
        short_recs[c] = rec
        prev_price, prev_wt = pv, wt_p

    # ---- output arrays ------------------------------------------------------
    def blank():
        return np.full(n, np.nan)

    long_type = np.full(n, np.nan)
    long_score = blank()
    long_mfe = blank()
    long_mae = blank()
    long_wt = blank()
    long_div = np.full(n, np.nan)
    long_ext = np.full(n, np.nan)
    long_ta = np.full(n, np.nan)
    long_chp = np.full(n, np.nan)
    long_ar = blank()
    long_adx = blank()
    long_srsi = blank()
    long_bbw = blank()

    short_type = np.full(n, np.nan)
    short_score = blank()
    short_mfe = blank()
    short_mae = blank()
    short_wt = blank()
    short_div = np.full(n, np.nan)
    short_ext = np.full(n, np.nan)
    short_ta = np.full(n, np.nan)
    short_chp = np.full(n, np.nan)
    short_ar = blank()
    short_adx = blank()
    short_srsi = blank()
    short_bbw = blank()

    last_type = np.full(n, np.nan)
    last_side = np.array([None] * n, dtype=object)

    def classify_long(rec, i):
        pb = rec["pivot_bar"]
        price = rec["price"]
        risk = rec["atr"] * risk_atr_mult
        if risk <= 0 or np.isnan(risk):
            return None
        win_lo = max(pb + 1, 0)
        win_hi = i + 1
        if win_hi - win_lo < 1:
            return None
        h_win = hi[win_lo:win_hi]
        l_win = lo[win_lo:win_hi]
        post_hi = float(np.nanmax(h_win))
        post_lo = float(np.nanmin(l_win))
        impulse_ok = (post_hi - price) >= min_move_atr * rec["atr"]
        if not impulse_ok:
            return None
        first_snap, first_fail = _first_hit(h_win, l_win, price, risk, snap_target_r, fake_fail_r)
        first_grind, _ = _first_hit(h_win, l_win, price, risk, grind_target_r, fake_fail_r)
        mfe_r = (post_hi - price) / risk
        mae_r = max(0.0, price - post_lo) / risk

        pp = rec["prev_price"]
        pw = rec["prev_wt"]
        div = (not np.isnan(pp)) and (not np.isnan(pw)) and (price < pp) and (rec["wt"] > pw)
        extreme = rec["wt"] <= os_level
        trend_against = (rec["close"] < rec["ema"]) and (rec["adx"] >= adx_trend_min)
        chop = rec["adx"] <= chop_adx_max

        good_fail_snap = (first_snap is not None) and (first_fail is None or first_snap < first_fail)
        good_fail_grind = (first_grind is not None) and (first_fail is None or first_grind < first_fail)

        if good_fail_snap and first_snap <= snap_bars:
            t = 1
        elif good_fail_grind or (mfe_r >= min_follow_r and mae_r < fake_fail_r and cl[i] > price):
            t = 2
        elif (first_fail is not None and (first_snap is None or first_fail < first_snap)) or mfe_r < min_follow_r:
            t = 3
        else:
            t = 4

        score = _quality_score(div, extreme, trend_against, chop, mfe_r, mae_r,
                               rec["atr_rank"], grind_target_r, min_follow_r, fake_fail_r)
        return t, score, mfe_r, mae_r, div, extreme, trend_against, chop

    def classify_short(rec, i):
        pb = rec["pivot_bar"]
        price = rec["price"]
        risk = rec["atr"] * risk_atr_mult
        if risk <= 0 or np.isnan(risk):
            return None
        win_lo = max(pb + 1, 0)
        win_hi = i + 1
        if win_hi - win_lo < 1:
            return None
        h_win = hi[win_lo:win_hi]
        l_win = lo[win_lo:win_hi]
        post_hi = float(np.nanmax(h_win))
        post_lo = float(np.nanmin(l_win))
        impulse_ok = (price - post_lo) >= min_move_atr * rec["atr"]
        if not impulse_ok:
            return None
        # short target on low, fail on high
        first_snap, first_fail = _first_hit_short(l_win, h_win, price, risk, snap_target_r, fake_fail_r)
        first_grind, _ = _first_hit_short(l_win, h_win, price, risk, grind_target_r, fake_fail_r)
        mfe_r = (price - post_lo) / risk
        mae_r = max(0.0, post_hi - price) / risk

        pp = rec["prev_price"]
        pw = rec["prev_wt"]
        div = (not np.isnan(pp)) and (not np.isnan(pw)) and (price > pp) and (rec["wt"] < pw)
        extreme = rec["wt"] >= ob_level
        trend_against = (rec["close"] > rec["ema"]) and (rec["adx"] >= adx_trend_min)
        chop = rec["adx"] <= chop_adx_max

        good_fail_snap = (first_snap is not None) and (first_fail is None or first_snap < first_fail)
        good_fail_grind = (first_grind is not None) and (first_fail is None or first_grind < first_fail)

        if good_fail_snap and first_snap <= snap_bars:
            t = 1
        elif good_fail_grind or (mfe_r >= min_follow_r and mae_r < fake_fail_r and cl[i] < price):
            t = 2
        elif (first_fail is not None and (first_snap is None or first_fail < first_snap)) or mfe_r < min_follow_r:
            t = 3
        else:
            t = 4

        score = _quality_score(div, extreme, trend_against, chop, mfe_r, mae_r,
                               rec["atr_rank"], grind_target_r, min_follow_r, fake_fail_r)
        return t, score, mfe_r, mae_r, div, extreme, trend_against, chop

    # evaluate every confirm bar (in bar order so last_type is chronological)
    evals = []  # (eval_bar, side, type)
    for c in range(n):
        i = c + eval_bars
        if i >= n:
            continue
        if c in long_recs:
            r = classify_long(long_recs[c], i)
            if r is not None:
                t, score, mfe, mae, div, ext, ta, chp = r
                pb = long_recs[c]["pivot_bar"]
                long_type[pb] = t
                long_score[pb] = score
                long_mfe[pb] = mfe
                long_mae[pb] = mae
                long_wt[pb] = long_recs[c]["wt"]
                long_div[pb] = 1.0 if div else 0.0
                long_ext[pb] = 1.0 if ext else 0.0
                long_ta[pb] = 1.0 if ta else 0.0
                long_chp[pb] = 1.0 if chp else 0.0
                long_ar[pb] = long_recs[c]["atr_rank"]
                long_adx[pb] = long_recs[c]["adx"]
                long_srsi[pb] = long_recs[c]["srsi"]
                long_bbw[pb] = long_recs[c]["bbw"]
                evals.append((i, "long", t))
        if c in short_recs:
            r = classify_short(short_recs[c], i)
            if r is not None:
                t, score, mfe, mae, div, ext, ta, chp = r
                pb = short_recs[c]["pivot_bar"]
                short_type[pb] = t
                short_score[pb] = score
                short_mfe[pb] = mfe
                short_mae[pb] = mae
                short_wt[pb] = short_recs[c]["wt"]
                short_div[pb] = 1.0 if div else 0.0
                short_ext[pb] = 1.0 if ext else 0.0
                short_ta[pb] = 1.0 if ta else 0.0
                short_chp[pb] = 1.0 if chp else 0.0
                short_ar[pb] = short_recs[c]["atr_rank"]
                short_adx[pb] = short_recs[c]["adx"]
                short_srsi[pb] = short_recs[c]["srsi"]
                short_bbw[pb] = short_recs[c]["bbw"]
                evals.append((i, "short", t))

    # last_type / last_side: most recent evaluated pivot, forward filled
    for i, side, t in evals:
        last_type[i] = t
        last_side[i] = side
    last_type = pd.Series(last_type, index=index).ffill().to_numpy()
    last_side = pd.Series(last_side, index=index).ffill()

    data = {
        "long_type": pd.Series(long_type, index=index, dtype="Int64"),
        "long_score": pd.Series(long_score, index=index),
        "long_mfe_r": pd.Series(long_mfe, index=index),
        "long_mae_r": pd.Series(long_mae, index=index),
        "long_wt_at_pivot": pd.Series(long_wt, index=index),
        "long_div": pd.Series(long_div, index=index, dtype="Int64"),
        "long_extreme": pd.Series(long_ext, index=index, dtype="Int64"),
        "long_trend_against": pd.Series(long_ta, index=index, dtype="Int64"),
        "long_chop": pd.Series(long_chp, index=index, dtype="Int64"),
        "long_atr_rank": pd.Series(long_ar, index=index),
        "long_adx": pd.Series(long_adx, index=index),
        "long_srsi": pd.Series(long_srsi, index=index),
        "long_bbw_rank": pd.Series(long_bbw, index=index),
        "short_type": pd.Series(short_type, index=index, dtype="Int64"),
        "short_score": pd.Series(short_score, index=index),
        "short_mfe_r": pd.Series(short_mfe, index=index),
        "short_mae_r": pd.Series(short_mae, index=index),
        "short_wt_at_pivot": pd.Series(short_wt, index=index),
        "short_div": pd.Series(short_div, index=index, dtype="Int64"),
        "short_extreme": pd.Series(short_ext, index=index, dtype="Int64"),
        "short_trend_against": pd.Series(short_ta, index=index, dtype="Int64"),
        "short_chop": pd.Series(short_chp, index=index, dtype="Int64"),
        "short_atr_rank": pd.Series(short_ar, index=index),
        "short_adx": pd.Series(short_adx, index=index),
        "short_srsi": pd.Series(short_srsi, index=index),
        "short_bbw_rank": pd.Series(short_bbw, index=index),
        "last_type": pd.Series(last_type, index=index, dtype="Int64"),
        "last_side": pd.Series(last_side, index=index),
    }
    return pd.DataFrame(data, index=index)


def _first_hit_short(low_win: np.ndarray, high_win: np.ndarray, price: float, risk: float,
                     target_r: float, fail_r: float):
    """Mirror Pine f_firstHitShort: target on low, fail on high."""
    w = len(low_win)
    target = price - target_r * risk
    fail = price + fail_r * risk
    first_target = None
    first_fail = None
    for k in range(w):
        if first_target is None and low_win[k] <= target:
            first_target = k + 1
        if first_fail is None and high_win[k] >= fail:
            first_fail = k + 1
    return first_target, first_fail
