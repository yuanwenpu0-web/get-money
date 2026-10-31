"""Liquidity Hunter [WavesUnchained] v3.5.2 — Python port.

Computational core only. Dropped (display/logging only): all ``plot*`` /
``label`` / ``line`` / ``box`` / ``table`` / ``alert*`` calls. The per-bar event
series, active-level scoring, regime context, momentum (WaveTrend + MFI) and
Fair-Value-Gap detection are preserved as analytic columns.

The indicator is inherently stateful: equal-high/low levels are tracked in a
growing registry (matched within an ATR-scaled tolerance, kept fresh, scored,
and swept/reclaimed). We therefore run a single deterministic bar-by-bar pass
that mirrors the Pine ``var`` arrays ``bslLevels`` / ``sslLevels`` /
``recentEvents`` / ``gapLevels``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr as _atr,
    ema as _ema,
    highest as _highest,
    lowest as _lowest,
    sma as _sma,
    pivotlow_strict,
    pivothigh_strict,
)

_MINTICK = 1e-8


def _parse_session(active_session: str):
    """Return (start_min, end_min) as HHMM ints, or None if unparseable."""
    try:
        a, b = active_session.strip().split("-")
        return int(a), int(b)
    except Exception:
        return None


def _f_level_score(touches, age_bars, dist_atr, max_level_age, max_level_distance):
    touch_score = min(touches, 5) * 20.0
    age_penalty = min(age_bars / max_level_age, 1.0) * 18.0
    dist_penalty = min(dist_atr, max_level_distance) * 7.0
    return max(0.0, touch_score + 18.0 - age_penalty - dist_penalty)


def _f_context_bonus(is_bearish, in_premium, is_trending, trend_up, max_bonus):
    bonus = 0.0
    if is_bearish and in_premium:
        bonus += max_bonus * 0.5
    if (not is_bearish) and (not in_premium):
        bonus += max_bonus * 0.5
    if not is_trending:
        bonus += max_bonus * 0.5
    elif is_bearish and (not trend_up):
        bonus += max_bonus * 0.5
    elif (not is_bearish) and trend_up:
        bonus += max_bonus * 0.5
    return min(max_bonus, bonus)


def _f_event_score(
    touches, wick_atr, close_back_atr, vol_ratio, session_boost,
    wt_osc, mfi, is_bearish, use_momentum, struct_bonus, ctx_bonus,
    volume_is_real, wt_ext_thr,
):
    touch_score = min(touches, 5) * 12.0
    wick_score = min(wick_atr, 1.5) * 18.0
    close_score = min(close_back_atr, 1.5) * 24.0
    vol_score = max(0.0, min(vol_ratio, 3.0) - 1.0) * 12.0 if volume_is_real else 0.0
    core_scale = 1.0 if volume_is_real else 147.0 / 123.0

    wt_bonus = 0.0
    mfi_bonus = 0.0
    if use_momentum:
        if is_bearish and wt_osc > wt_ext_thr:
            wt_bonus = 10.0
        elif (not is_bearish) and wt_osc < -wt_ext_thr:
            wt_bonus = 10.0
        if volume_is_real and is_bearish and mfi > 80.0:
            mfi_bonus = 10.0
        elif volume_is_real and (not is_bearish) and mfi < 20.0:
            mfi_bonus = 10.0

    core_score = (touch_score + wick_score + close_score + vol_score) * core_scale
    return min(100.0, core_score + session_boost + wt_bonus + mfi_bonus + struct_bonus + ctx_bonus)


COLUMNS = [
    "atr14", "regime", "in_premium", "is_trending", "trend_up", "lh_wt_osc", "lh_mfi",
    "volume_is_real", "vol_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_position",
    "bsl_sweep", "ssl_sweep", "bsl_breakout", "ssl_breakout", "bsl_hunt", "ssl_hunt",
    "bearish_reclaim", "bullish_reclaim", "bearish_reclaim_score", "bullish_reclaim_score",
    "recent_bull_score", "recent_bear_score", "exhaustion_bear", "exhaustion_bull",
    "primary_bsl_price", "primary_bsl_score", "primary_bsl_touches",
    "secondary_bsl_price", "secondary_bsl_score", "secondary_bsl_touches",
    "primary_ssl_price", "primary_ssl_score", "primary_ssl_touches",
    "secondary_ssl_price", "secondary_ssl_score", "secondary_ssl_touches",
    "active_bsl_count", "active_ssl_count", "near_primary_bsl", "near_primary_ssl",
    "bias_value", "fvg_new", "fvg_new_size_pct", "fvg_unfilled",
]

INT64_COLS = [
    "in_premium", "is_trending", "trend_up", "volume_is_real",
    "bsl_sweep", "ssl_sweep", "bsl_breakout", "ssl_breakout", "bsl_hunt", "ssl_hunt",
    "bearish_reclaim", "bullish_reclaim", "exhaustion_bear", "exhaustion_bull",
    "primary_bsl_touches", "secondary_bsl_touches", "primary_ssl_touches", "secondary_ssl_touches",
    "active_bsl_count", "active_ssl_count", "near_primary_bsl", "near_primary_ssl",
    "bias_value", "fvg_new", "fvg_unfilled",
]


def liquidity_hunter(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 8,
    eq_tolerance_atr: float = 0.12,
    min_touches: int = 2,
    max_tracked_levels: int = 30,
    max_level_age: int = 400,
    max_level_distance: float = 8.0,
    min_sweep_atr: float = 0.05,
    reclaim_bars: int = 3,
    min_close_back_atr: float = 0.10,
    min_volume_ratio: float = 1.2,
    hunt_vol_multiplier: float = 1.5,
    hunt_wick_ratio_min: float = 0.45,
    hunt_close_pos_max: float = 0.45,
    enable_stop_hunts: bool = True,
    regime_lookback: int = 50,
    er_trend_threshold: float = 0.35,
    context_bonus_max: float = 12.0,
    enable_regime_context: bool = False,
    enable_level_flip: bool = False,
    use_session_filter: bool = False,
    active_session: str = "0700-1700",
    event_memory_bars: int = 30,
    lh_wt_ext_thr: float = 53.0,
    lh_mfi_len: int = 14,
    enable_exhaustion: bool = False,
    min_gap_size_pct: float = 0.05,
) -> pd.DataFrame:
    """Liquidity Hunter — equal-high/low liquidity map with sweep / stop-hunt /
    reclaim detection and Fair-Value-Gap magnets.

    Returns a per-bar ``DataFrame`` (index preserved) with the analytic columns
    listed in ``COLUMNS``.
    """
    if not {"open", "high", "low", "close"}.issubset(frame.columns):
        raise ValueError("frame must contain open/high/low/close columns")
    has_vol = "volume" in frame.columns

    o = frame["open"].astype(float).to_numpy()
    h = frame["high"].astype(float).to_numpy()
    l = frame["low"].astype(float).to_numpy()
    c = frame["close"].astype(float).to_numpy()
    n = len(c)
    idx = frame.index

    vol = frame["volume"].astype(float).to_numpy() if has_vol else np.full(n, np.nan)
    v = np.nan_to_num(vol, nan=0.0)

    atr14 = _atr(frame, 14).to_numpy()
    bar_range = np.maximum(h - l, _MINTICK)
    upper_wick = (h - np.maximum(o, c)) / bar_range
    lower_wick = (np.minimum(o, c) - l) / bar_range
    close_pos = (c - l) / bar_range

    # ---- Momentum Hybrid: WaveTrend + MFI ----
    hlc3 = (h + l + c) / 3.0
    esa = _ema(pd.Series(hlc3, index=idx), 10).to_numpy()
    d = _ema(pd.Series(np.abs(hlc3 - esa), index=idx), 10).to_numpy()
    ci = (hlc3 - esa) / (0.015 * np.maximum(d, _MINTICK))
    lh_wt_osc = _ema(pd.Series(ci, index=idx), 21).to_numpy()
    typ = (h + l + c) / 3.0
    typ_raw = typ * v
    typ_prev = np.concatenate([[np.nan], typ[:-1]])
    pos = np.where(typ > typ_prev, typ_raw, 0.0)
    neg = np.where(typ < typ_prev, typ_raw, 0.0)
    win = max(1, lh_mfi_len)
    pos_sum = np.convolve(pos, np.ones(win), mode="full")[:n]
    neg_sum = np.convolve(neg, np.ones(win), mode="full")[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi_ratio = np.where(neg_sum == 0, np.inf, pos_sum / np.where(neg_sum == 0, 1.0, neg_sum))
    lh_mfi = np.where(neg_sum == 0, 100.0, 100.0 - 100.0 / (1.0 + mfi_ratio))

    # ---- Regime context ----
    rh = _highest(pd.Series(h, index=idx), regime_lookback).to_numpy()
    rl = _lowest(pd.Series(l, index=idx), regime_lookback).to_numpy()
    rmid = (rh + rl) / 2.0
    in_premium = c > rmid
    if n > regime_lookback:
        close_lb = np.concatenate([[np.nan] * regime_lookback, c[: n - regime_lookback]])
    else:
        close_lb = np.full(n, np.nan)
    net_change = np.abs(c - close_lb)
    diff = np.abs(c - np.concatenate([[np.nan], c[:-1]]))
    sum_change = np.convolve(np.nan_to_num(diff, nan=0.0), np.ones(regime_lookback), mode="full")[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(sum_change > 0, net_change / np.where(sum_change > 0, sum_change, 1.0), 0.0)
    is_trending = efficiency >= er_trend_threshold
    trend_up = c > close_lb
    regime = np.where(
        is_trending,
        np.where(trend_up, "Trend Up", "Trend Down"),
        "Range",
    )

    # ---- Volume ----
    avg_vol = _sma(pd.Series(v, index=idx), 20).to_numpy()
    vol_ratio = np.where(avg_vol > 0, v / np.where(avg_vol > 0, avg_vol, 1.0), 1.0)
    cum_vol = np.cumsum(v)
    volume_is_real = (cum_vol > 0) & (~np.isnan(vol))

    # ---- Session ----
    sess = _parse_session(active_session)
    if use_session_filter and sess is not None:
        start_min, end_min = sess
        hm = idx.hour * 100 + idx.minute
        in_session = (hm >= start_min) & (hm <= end_min)
    else:
        in_session = np.ones(n, dtype=bool)

    # ---- Detectors ----
    ph = pivothigh_strict(pd.Series(h, index=idx), pivot_len, pivot_len).to_numpy()
    pl = pivotlow_strict(pd.Series(l, index=idx), pivot_len, pivot_len).to_numpy()

    min_gap_frac = min_gap_size_pct / 100.0

    # ---- Stateful registries ----
    bsl_levels: list[dict] = []
    ssl_levels: list[dict] = []
    recent_events: list[dict] = []
    gap_levels: list[dict] = []

    out = {k: [] for k in COLUMNS}

    def _new_level(price, pivot_bar, touches=1):
        return {
            "price": float(price), "pivot_bar": int(pivot_bar),
            "last_touch_bar": int(pivot_bar), "touches": int(touches),
            "swept": False, "sweep_bar": None, "reclaimed": False,
            "reclaim_dir": 0, "last_score": 0.0,
        }

    for b in range(n):
        atr_b = atr14[b]
        close_b = c[b]
        high_b = h[b]
        low_b = l[b]
        vol_real_b = bool(volume_is_real[b])
        vol_ratio_b = vol_ratio[b]
        tol_abs = max(atr_b * eq_tolerance_atr, _MINTICK * 2.0)
        sweep_abs = atr_b * min_sweep_atr
        close_back_abs = atr_b * min_close_back_atr

        # --- 1. Register pivot highs (BSL candidates) ---
        phv = ph[b]
        if not np.isnan(phv):
            pivot_bar = b - pivot_len
            found = False
            for lvl in bsl_levels:
                if abs(lvl["price"] - phv) <= tol_abs:
                    lvl["touches"] += 1
                    lvl["last_touch_bar"] = pivot_bar
                    lvl["price"] = (lvl["price"] * (lvl["touches"] - 1) + phv) / lvl["touches"]
                    found = True
                    break
            if not found:
                bsl_levels.append(_new_level(phv, pivot_bar))
            if len(bsl_levels) > max_tracked_levels:
                bsl_levels.pop(0)

        # --- 2. Register pivot lows (SSL candidates) ---
        plv = pl[b]
        if not np.isnan(plv):
            pivot_bar = b - pivot_len
            found = False
            for lvl in ssl_levels:
                if abs(lvl["price"] - plv) <= tol_abs:
                    lvl["touches"] += 1
                    lvl["last_touch_bar"] = pivot_bar
                    lvl["price"] = (lvl["price"] * (lvl["touches"] - 1) + plv) / lvl["touches"]
                    found = True
                    break
            if not found:
                ssl_levels.append(_new_level(plv, pivot_bar))
            if len(ssl_levels) > max_tracked_levels:
                ssl_levels.pop(0)

        # --- 3. Level expiry ---
        if atr_b > 0:
            bsl_levels = [
                x for x in bsl_levels
                if not (b - x["last_touch_bar"] > max_level_age
                        or abs(close_b - x["price"]) / atr_b > max_level_distance)
            ]
            ssl_levels = [
                x for x in ssl_levels
                if not (b - x["last_touch_bar"] > max_level_age
                        or abs(close_b - x["price"]) / atr_b > max_level_distance)
            ]

        # --- 4. Fair-Value-Gap detection + fill ---
        fvg_new = 0
        fvg_new_size = np.nan
        if b >= 2:
            hi2 = h[b - 2]
            lo2 = l[b - 2]
            if low_b > hi2:
                gap_sz = low_b - hi2
                if gap_sz / close_b >= min_gap_frac:
                    gap_levels.insert(0, {"gap_low": hi2, "gap_high": low_b,
                                          "gap_size": gap_sz, "direction": 1})
                    if len(gap_levels) > 15:
                        gap_levels.pop()
                    fvg_new = 1
                    fvg_new_size = gap_sz / close_b * 100.0
            elif high_b < lo2:
                gap_sz = lo2 - high_b
                if gap_sz / close_b >= min_gap_frac:
                    gap_levels.insert(0, {"gap_low": high_b, "gap_high": lo2,
                                          "gap_size": gap_sz, "direction": -1})
                    if len(gap_levels) > 15:
                        gap_levels.pop()
                    fvg_new = -1
                    fvg_new_size = gap_sz / close_b * 100.0
        for gap in gap_levels:
            if gap["gap_size"] > 0:
                if gap["direction"] == 1 and low_b <= gap["gap_low"]:
                    gap["gap_size"] = 0
                elif gap["direction"] == -1 and high_b >= gap["gap_high"]:
                    gap["gap_size"] = 0
        fvg_unfilled = sum(1 for gap in gap_levels if gap["gap_size"] > 0)

        # --- 5. Sweep / reclaim engine (BSL) ---
        bsl_sweep = bsl_breakout = bsl_hunt = False
        bearish_reclaim = False
        bearish_reclaim_score = 0.0
        for lvl in bsl_levels:
            if lvl["touches"] >= min_touches and lvl["pivot_bar"] < b:
                wick_atr = max(0.0, (high_b - lvl["price"]) / atr_b) if atr_b > 0 else 0.0
                close_back_atr = max(0.0, (lvl["price"] - close_b) / atr_b) if atr_b > 0 else 0.0
                touched = high_b >= lvl["price"] + sweep_abs
                if (not lvl["swept"]) and touched:
                    lvl["swept"] = True
                    lvl["sweep_bar"] = b
                    is_breakout = close_b >= lvl["price"] + sweep_abs
                    if is_breakout:
                        bsl_breakout = True
                        if enable_level_flip:
                            ssl_levels.append(_new_level(lvl["price"], b, min_touches))
                            if len(ssl_levels) > max_tracked_levels:
                                ssl_levels.pop(0)
                    else:
                        bsl_sweep = True
                        if (enable_stop_hunts and upper_wick[b] >= hunt_wick_ratio_min
                                and close_pos[b] <= hunt_close_pos_max
                                and (not vol_real_b or vol_ratio_b >= hunt_vol_multiplier)):
                            bsl_hunt = True
                can_reclaim = lvl["swept"] and (not lvl["reclaimed"]) and (b <= lvl["sweep_bar"] + reclaim_bars)
                if (can_reclaim and close_b <= lvl["price"] - close_back_abs
                        and (not vol_real_b or vol_ratio_b >= min_volume_ratio)
                        and high_b >= lvl["price"]):
                    lvl["reclaimed"] = True
                    lvl["reclaim_dir"] = -1
                    bearish_reclaim = True
                    session_boost = 10.0 if in_session[b] else 0.0
                    struct_bonus = 10.0 if close_b < l[b - 1] else 0.0
                    ctx_bonus = (
                        _f_context_bonus(True, in_premium[b], is_trending[b], trend_up[b], context_bonus_max)
                        if enable_regime_context else 0.0
                    )
                    evt_score = _f_event_score(
                        lvl["touches"], wick_atr, close_back_atr, vol_ratio_b, session_boost,
                        lh_wt_osc[b], lh_mfi[b], True, enable_exhaustion, struct_bonus, ctx_bonus,
                        vol_real_b, lh_wt_ext_thr,
                    )
                    bearish_reclaim_score = max(bearish_reclaim_score, evt_score)
                    recent_events.insert(0, {
                        "event_bar": b, "event_price": high_b, "direction": -1,
                        "score": evt_score, "level_price": lvl["price"], "volume_ratio": vol_ratio_b,
                    })
                    if len(recent_events) > 6:
                        recent_events.pop()

        # --- 6. Sweep / reclaim engine (SSL) ---
        ssl_sweep = ssl_breakout = ssl_hunt = False
        bullish_reclaim = False
        bullish_reclaim_score = 0.0
        for lvl in ssl_levels:
            if lvl["touches"] >= min_touches and lvl["pivot_bar"] < b:
                wick_atr = max(0.0, (lvl["price"] - low_b) / atr_b) if atr_b > 0 else 0.0
                close_back_atr = max(0.0, (close_b - lvl["price"]) / atr_b) if atr_b > 0 else 0.0
                touched = low_b <= lvl["price"] - sweep_abs
                if (not lvl["swept"]) and touched:
                    lvl["swept"] = True
                    lvl["sweep_bar"] = b
                    is_breakout = close_b <= lvl["price"] - sweep_abs
                    if is_breakout:
                        ssl_breakout = True
                        if enable_level_flip:
                            bsl_levels.append(_new_level(lvl["price"], b, min_touches))
                            if len(bsl_levels) > max_tracked_levels:
                                bsl_levels.pop(0)
                    else:
                        ssl_sweep = True
                        if (enable_stop_hunts and lower_wick[b] >= hunt_wick_ratio_min
                                and (1.0 - close_pos[b]) <= hunt_close_pos_max
                                and (not vol_real_b or vol_ratio_b >= hunt_vol_multiplier)):
                            ssl_hunt = True
                can_reclaim = lvl["swept"] and (not lvl["reclaimed"]) and (b <= lvl["sweep_bar"] + reclaim_bars)
                if (can_reclaim and close_b >= lvl["price"] + close_back_abs
                        and (not vol_real_b or vol_ratio_b >= min_volume_ratio)
                        and low_b <= lvl["price"]):
                    lvl["reclaimed"] = True
                    lvl["reclaim_dir"] = 1
                    bullish_reclaim = True
                    session_boost = 10.0 if in_session[b] else 0.0
                    struct_bonus = 10.0 if close_b > h[b - 1] else 0.0
                    ctx_bonus = (
                        _f_context_bonus(False, in_premium[b], is_trending[b], trend_up[b], context_bonus_max)
                        if enable_regime_context else 0.0
                    )
                    evt_score = _f_event_score(
                        lvl["touches"], wick_atr, close_back_atr, vol_ratio_b, session_boost,
                        lh_wt_osc[b], lh_mfi[b], False, enable_exhaustion, struct_bonus, ctx_bonus,
                        vol_real_b, lh_wt_ext_thr,
                    )
                    bullish_reclaim_score = max(bullish_reclaim_score, evt_score)
                    recent_events.insert(0, {
                        "event_bar": b, "event_price": low_b, "direction": 1,
                        "score": evt_score, "level_price": lvl["price"], "volume_ratio": vol_ratio_b,
                    })
                    if len(recent_events) > 6:
                        recent_events.pop()

        # --- 7. recentEvents expiry + rollups ---
        recent_events = [e for e in recent_events if b - e["event_bar"] <= event_memory_bars]
        recent_bull_score = max((e["score"] for e in recent_events if e["direction"] == 1), default=0.0)
        recent_bear_score = max((e["score"] for e in recent_events if e["direction"] == -1), default=0.0)

        # --- 8. Active level scoring (primary = best score, secondary = next) ---
        p_bsl_s = p_bsl_p = p_bsl_t = s_bsl_s = s_bsl_p = s_bsl_t = np.nan
        p_bsl_s = 0.0
        s_bsl_s = 0.0
        p_bsl_t = 0
        s_bsl_t = 0
        active_bsl = 0
        cand = []
        for lvl in bsl_levels:
            if (lvl["touches"] >= min_touches and (not lvl["swept"]) and (not lvl["reclaimed"])
                    and lvl["price"] > close_b and atr_b > 0):
                age = b - lvl["last_touch_bar"]
                dist = (lvl["price"] - close_b) / atr_b
                score = _f_level_score(lvl["touches"], age, dist, max_level_age, max_level_distance)
                lvl["last_score"] = score
                active_bsl += 1
                cand.append((score, lvl["price"], lvl["touches"]))
        if cand:
            cand.sort(key=lambda x: x[0], reverse=True)
            p_bsl_s, p_bsl_p, p_bsl_t = cand[0]
            if len(cand) > 1:
                s_bsl_s, s_bsl_p, s_bsl_t = cand[1]
            else:
                s_bsl_s, s_bsl_p, s_bsl_t = 0.0, np.nan, np.nan
        else:
            p_bsl_s, p_bsl_p, p_bsl_t = 0.0, np.nan, np.nan

        p_ssl_s = p_ssl_p = p_ssl_t = s_ssl_s = s_ssl_p = s_ssl_t = np.nan
        p_ssl_s = 0.0
        s_ssl_s = 0.0
        p_ssl_t = 0
        s_ssl_t = 0
        active_ssl = 0
        cand2 = []
        for lvl in ssl_levels:
            if (lvl["touches"] >= min_touches and (not lvl["swept"]) and (not lvl["reclaimed"])
                    and lvl["price"] < close_b and atr_b > 0):
                age = b - lvl["last_touch_bar"]
                dist = (close_b - lvl["price"]) / atr_b
                score = _f_level_score(lvl["touches"], age, dist, max_level_age, max_level_distance)
                lvl["last_score"] = score
                active_ssl += 1
                cand2.append((score, lvl["price"], lvl["touches"]))
        if cand2:
            cand2.sort(key=lambda x: x[0], reverse=True)
            p_ssl_s, p_ssl_p, p_ssl_t = cand2[0]
            if len(cand2) > 1:
                s_ssl_s, s_ssl_p, s_ssl_t = cand2[1]
            else:
                s_ssl_s, s_ssl_p, s_ssl_t = 0.0, np.nan, np.nan
        else:
            p_ssl_s, p_ssl_p, p_ssl_t = 0.0, np.nan, np.nan

        near_bsl = (not np.isnan(p_bsl_p)) and atr_b > 0 and (p_bsl_p - close_b) / atr_b <= 1.0
        near_ssl = (not np.isnan(p_ssl_p)) and atr_b > 0 and (close_b - p_ssl_p) / atr_b <= 1.0

        # --- 9. Exhaustion context ---
        exhaustion_bear = enable_exhaustion and (bsl_sweep or bsl_hunt) and lh_wt_osc[b] > lh_wt_ext_thr
        exhaustion_bull = enable_exhaustion and (ssl_sweep or ssl_hunt) and lh_wt_osc[b] < -lh_wt_ext_thr

        # --- 10. Directional bias (dashboard summary) ---
        bias = 0
        bias += 2 if bullish_reclaim else (1 if recent_bull_score >= 65 else 0)
        bias -= 2 if bearish_reclaim else (1 if recent_bear_score >= 65 else 0)
        bias += 1 if (p_ssl_s > p_bsl_s and near_ssl) else 0
        bias -= 1 if (p_bsl_s > p_ssl_s and near_bsl) else 0

        # --- commit row ---
        out["atr14"].append(atr_b)
        out["regime"].append(regime[b])
        out["in_premium"].append(int(in_premium[b]))
        out["is_trending"].append(int(is_trending[b]))
        out["trend_up"].append(int(trend_up[b]))
        out["lh_wt_osc"].append(lh_wt_osc[b])
        out["lh_mfi"].append(lh_mfi[b])
        out["volume_is_real"].append(int(vol_real_b))
        out["vol_ratio"].append(vol_ratio_b)
        out["upper_wick_ratio"].append(upper_wick[b])
        out["lower_wick_ratio"].append(lower_wick[b])
        out["close_position"].append(close_pos[b])
        out["bsl_sweep"].append(int(bsl_sweep))
        out["ssl_sweep"].append(int(ssl_sweep))
        out["bsl_breakout"].append(int(bsl_breakout))
        out["ssl_breakout"].append(int(ssl_breakout))
        out["bsl_hunt"].append(int(bsl_hunt))
        out["ssl_hunt"].append(int(ssl_hunt))
        out["bearish_reclaim"].append(int(bearish_reclaim))
        out["bullish_reclaim"].append(int(bullish_reclaim))
        out["bearish_reclaim_score"].append(bearish_reclaim_score)
        out["bullish_reclaim_score"].append(bullish_reclaim_score)
        out["recent_bull_score"].append(recent_bull_score)
        out["recent_bear_score"].append(recent_bear_score)
        out["exhaustion_bear"].append(int(exhaustion_bear))
        out["exhaustion_bull"].append(int(exhaustion_bull))
        out["primary_bsl_price"].append(p_bsl_p)
        out["primary_bsl_score"].append(p_bsl_s)
        out["primary_bsl_touches"].append(p_bsl_t if p_bsl_t else np.nan)
        out["secondary_bsl_price"].append(s_bsl_p)
        out["secondary_bsl_score"].append(s_bsl_s)
        out["secondary_bsl_touches"].append(s_bsl_t if s_bsl_t else np.nan)
        out["primary_ssl_price"].append(p_ssl_p)
        out["primary_ssl_score"].append(p_ssl_s)
        out["primary_ssl_touches"].append(p_ssl_t if p_ssl_t else np.nan)
        out["secondary_ssl_price"].append(s_ssl_p)
        out["secondary_ssl_score"].append(s_ssl_s)
        out["secondary_ssl_touches"].append(s_ssl_t if s_ssl_t else np.nan)
        out["active_bsl_count"].append(active_bsl)
        out["active_ssl_count"].append(active_ssl)
        out["near_primary_bsl"].append(int(near_bsl))
        out["near_primary_ssl"].append(int(near_ssl))
        out["bias_value"].append(int(bias))
        out["fvg_new"].append(fvg_new)
        out["fvg_new_size_pct"].append(fvg_new_size)
        out["fvg_unfilled"].append(fvg_unfilled)

    df = pd.DataFrame(out, index=idx)
    for col in INT64_COLS:
        df[col] = df[col].astype("Int64")
    return df
