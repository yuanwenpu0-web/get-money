"""Tweezer & Kangaroo Zones [WavesUnchained]  v3.2.1  —  catalog port.

Source: data/pine-collection/indicators/market_structure/tweezer_kangaroo_zones/
        tweezer_kangaroo_zones.pine

Pattern + zone engine:
  * Tweezer top/bottom detection (similar highs/lows, optional wick-similarity
    and volume checks) and Kangaroo-tail pin-bar detection, both with a 0-1
    strength score and timeframe-sensitivity-adjusted thresholds.
  * Dynamic supply/demand zones built from each pattern's candle geometry,
    merged on overlap, with 5-component scoring (rejection / extreme proximity /
    impulse-before / freshness / cleanliness), reversal-vs-continuation
    classification and A/B/C grades.
  * HH/HL/LH/LL structure context feeding the zone structure bonus.

Ported computational core; HTF stack panel, supertrend/MOST integration,
drawing/labels/tooltips and the full confluence signal score are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, ema, highest, integer, lowest, number, ohlcv, pivothigh_strict, pivotlow_strict, sma

_MINTICK = 1e-8


def tweezer_and_kangaroo_zones(
    frame: pd.DataFrame,
    *,
    input_enable_tweezers: bool = True,
    tweezer_lookback: int = 3,
    tweezer_tolerance: float = 0.01,
    tweezer_require_long_body: bool = False,
    tweezer_require_trend: bool = False,
    tweezer_check_wicks: bool = True,
    tweezer_wick_tolerance: float = 2.0,
    tweezer_min_volume: float = 0.8,
    w_tweezer: float = 0.8,
    w_kangaroo: float = 1.2,
    input_enable_kangaroo: bool = True,
    kt_wick_body_min: float = 1.8,
    kt_body_frac_max: float = 0.50,
    kt_body_position: float = 0.3,
    kt_require_trend: bool = False,
    kt_min_volume: float = 0.9,
    zone_ttl: int = 400,
    invalidate_on_close: bool = True,
    zone_overlap_pct: float = 0.5,
    zone_merge_enabled: bool = True,
    ema_len: int = 200,
    tf_sensitivity: float = 1.0,
    tf_auto_detect: bool = True,
    auto_tune: bool = True,
    atr_len: int = 14,
    chop_len: int = 14,
    vol_len: int = 20,
    use_struct_context: bool = True,
    struct_lookback: int = 20,
    struct_bonus: float = 10.0,
    use_zone_score: bool = True,
    zone_score_min: float = 50.0,
    zone_score_top_n: int = 5,
    zone_half_life: int = 80,
    zone_max_touches: int = 4,
    zone_lookback: int = 100,
    grade_a_min: float = 75.0,
    grade_b_min: float = 55.0,
    reversal_bonus: float = 15.0,
    continuation_penalty: float = 10.0,
) -> pd.DataFrame:
    tweezer_lookback = integer("tweezer_lookback", tweezer_lookback, 1, 5)
    tweezer_tolerance = number("tweezer_tolerance", tweezer_tolerance, 0.0)
    tweezer_wick_tolerance = number("tweezer_wick_tolerance", tweezer_wick_tolerance, 1.1)
    tweezer_min_volume = number("tweezer_min_volume", tweezer_min_volume, 0.0)
    w_tweezer = number("w_tweezer", w_tweezer, 0.5, 1.5)
    w_kangaroo = number("w_kangaroo", w_kangaroo, 0.5, 1.5)
    kt_wick_body_min = number("kt_wick_body_min", kt_wick_body_min, 0.0)
    kt_body_frac_max = number("kt_body_frac_max", kt_body_frac_max, 0.0, 1.0)
    kt_body_position = number("kt_body_position", kt_body_position, 0.0, 1.0)
    kt_min_volume = number("kt_min_volume", kt_min_volume, 0.0)
    zone_ttl = integer("zone_ttl", zone_ttl, 1)
    zone_overlap_pct = number("zone_overlap_pct", zone_overlap_pct, 0.0, 1.0)
    ema_len = integer("ema_len", ema_len, 1)
    tf_sensitivity = number("tf_sensitivity", tf_sensitivity, 0.5, 2.0)
    atr_len = integer("atr_len", atr_len, 1)
    chop_len = integer("chop_len", chop_len, 2)
    vol_len = integer("vol_len", vol_len, 1)
    struct_lookback = integer("struct_lookback", struct_lookback, 5, 50)
    struct_bonus = number("struct_bonus", struct_bonus, 0.0, 20.0)
    zone_score_min = number("zone_score_min", zone_score_min, 0.0, 100.0)
    zone_score_top_n = integer("zone_score_top_n", zone_score_top_n, 1, 20)
    zone_half_life = integer("zone_half_life", zone_half_life, 10, 500)
    zone_max_touches = integer("zone_max_touches", zone_max_touches, 1, 10)
    zone_lookback = integer("zone_lookback", zone_lookback, 20, 500)
    grade_a_min = number("grade_a_min", grade_a_min, 50.0, 100.0)
    grade_b_min = number("grade_b_min", grade_b_min, 30.0, 80.0)
    reversal_bonus = number("reversal_bonus", reversal_bonus, 0.0, 30.0)
    continuation_penalty = number("continuation_penalty", continuation_penalty, 0.0, 30.0)

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float); vol = vol.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    body_hi = np.maximum(o, c); body_lo = np.minimum(o, c); body = body_hi - body_lo
    rng = np.maximum(h - l, _MINTICK)
    u_wick = h - body_hi; l_wick = body_lo - l
    body_avg = ema(pd.Series(body), 14).to_numpy()
    white = o < c; black = o > c
    atr_a = np.maximum(atr(frame, atr_len).to_numpy(dtype=float), _MINTICK)
    rel_vol = vol / np.maximum(1.0, sma(pd.Series(vol), vol_len).to_numpy())
    real = bool(np.any(vol > 0))
    ema_a = ema(frame["close"], ema_len).to_numpy(dtype=float)

    # tf sensitivity
    if isinstance(idx, pd.DatetimeIndex) and n > 1:
        med_min = float(np.median(np.diff(idx.values).astype("timedelta64[s]").astype(float))) / 60.0
    else:
        med_min = 60.0
    if tf_auto_detect:
        tf_sens = 0.7 if med_min <= 15 else 1.0 if med_min <= 120 else 1.3 if med_min <= 240 else 1.5
    else:
        tf_sens = tf_sensitivity
    tweezer_tol_eff = tweezer_tolerance * tf_sens
    kt_wick_base = kt_wick_body_min / tf_sens
    kt_body_base = kt_body_frac_max * tf_sens

    tr_sum = pd.Series(np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))).rolling(chop_len).sum().to_numpy()
    hh_chop = highest(frame["high"], chop_len).to_numpy()
    ll_chop = lowest(frame["low"], chop_len).to_numpy()
    chop = np.array([100 * math.log10((tr_sum[i] / max(1e-6, hh_chop[i] - ll_chop[i]))) / math.log10(chop_len)
                     if i + 1 >= chop_len and hh_chop[i] - ll_chop[i] > 0 and tr_sum[i] > 0 else np.nan for i in range(n)])

    lookback_high = highest(frame["high"], zone_lookback).to_numpy()
    lookback_low = lowest(frame["low"], zone_lookback).to_numpy()

    ph = pivothigh_strict(frame["high"], struct_lookback, struct_lookback).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], struct_lookback, struct_lookback).to_numpy(dtype=float)

    zones: list = []
    last_swing_high = last_swing_low = float("nan")
    prev_swing_high = prev_swing_low = float("nan")

    out = {k: np.full(n, np.nan) for k in [
        "tweezer_top_strength", "tweezer_bot_strength", "kangaroo_bear_strength",
        "kangaroo_bull_strength", "chop", "rel_vol", "supply_top", "supply_bottom",
        "supply_score", "demand_top", "demand_bottom", "demand_score", "zone_count"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "tweezer_top", "tweezer_bot", "kangaroo_bear", "kangaroo_bull",
        "supply_grade_a", "demand_grade_a"]}

    def calc_rejection(is_supply, pattern_strength):
        r = rng[i]
        if r <= 0:
            return 0.5
        if is_supply:
            wick_pct = u_wick[i] / r; close_pos = 1.0 - (c[i] - l[i]) / r
        else:
            wick_pct = l_wick[i] / r; close_pos = (c[i] - l[i]) / r
        base = min(1.0, wick_pct * 0.6 + close_pos * 0.4)
        return max(base, pattern_strength) if (pattern_strength is not None and not math.isnan(pattern_strength) and pattern_strength > 0) else base

    def calc_extreme(is_supply, top, bot):
        lb_range = lookback_high[i] - lookback_low[i]
        if lb_range <= 0:
            return 0.5
        if is_supply:
            dist = lookback_high[i] - top
            g = 1.0 - min(1.0, dist / (lb_range * 0.5))
        else:
            dist = bot - lookback_low[i]
            g = 1.0 - min(1.0, dist / (lb_range * 0.5))
        local = 0.0
        if is_supply and not math.isnan(last_swing_high):
            local = 1.0 - min(1.0, abs(last_swing_high - top) / (atr_a[i] * 3))
        elif (not is_supply) and not math.isnan(last_swing_low):
            local = 1.0 - min(1.0, abs(bot - last_swing_low) / (atr_a[i] * 3))
        return max(g, local)

    def calc_impulse(is_supply):
        if is_supply:
            move = max(0.0, h[i - 1] - h[i - 6]) if i >= 6 else 0.0
        else:
            move = max(0.0, l[i - 6] - l[i - 1]) if i >= 6 else 0.0
        return min(1.0, move / max(1e-6, atr_a[i] * 2))

    def calc_score(rej, ext, imp, fresh, clean):
        return 30.0 * rej + 25.0 * ext + 20.0 * imp + 15.0 * fresh + 10.0 * clean

    def new_zone(is_supply, top, bot, ptype, strength):
        if top <= bot:
            return False
        mid_new = (top + bot) / 2
        for z in zones:
            if z["active"] and z["supply"] == is_supply and abs(mid_new - (z["top"] + z["bot"]) / 2) < atr_a[i] * 0.8:
                return False
        # overlap
        for z in zones:
            if z["active"] and z["supply"] == is_supply:
                if bot < z["top"] and top > z["bot"]:
                    o_top = min(top, z["top"]); o_bot = max(bot, z["bot"])
                    overlap_pct = (o_top - o_bot) / min(top - bot, z["top"] - z["bot"])
                    if overlap_pct >= zone_overlap_pct:
                        if zone_merge_enabled:
                            z["top"] = max(z["top"], top); z["bot"] = min(z["bot"], bot)
                            rej = calc_rejection(is_supply, strength); ext = calc_extreme(is_supply, z["top"], z["bot"]); imp = calc_impulse(is_supply)
                            z["rejection"] = max(z["rejection"], rej); z["extreme"] = max(z["extreme"], ext); z["impulse"] = max(z["impulse"], imp)
                            mult = w_kangaroo if ptype == "K" else w_tweezer
                            fresh = 1.0
                            clean = 1.0 - min(1.0, z["touches"] / zone_max_touches)
                            z["score"] = calc_score(z["rejection"], z["extreme"], z["impulse"], fresh, clean) * mult + z["struct_pts"]
                            z["pattern_bar"] = i
                            return True
                        return False
        rej = calc_rejection(is_supply, strength); ext = calc_extreme(is_supply, top, bot); imp = calc_impulse(is_supply)
        struct_pts = 0.0
        if use_struct_context:
            if is_supply and (bearish or near_swing_high):
                struct_pts = struct_bonus
            elif (not is_supply) and (bullish or near_swing_low):
                struct_pts = struct_bonus
        mult = w_kangaroo if ptype == "K" else w_tweezer
        zone_h = abs(top - bot)
        h_score = 1.0 - min(1.0, zone_h / (atr_a[i] * 1.2))
        base = calc_score(rej, ext, imp, 1.0, 1.0) * (0.7 + 0.3 * h_score)
        init = base * mult + struct_pts
        if use_zone_score and init < zone_score_min:
            return False
        ztype = "R" if (ext >= 0.6 and imp >= 0.5) else "C"
        adj = init + reversal_bonus if ztype == "R" else init - continuation_penalty
        adj = max(0.0, min(100.0, adj))
        grade = "A" if (adj >= grade_a_min and 0 <= 1) else "B" if adj >= grade_b_min else "C"
        zones.append({"top": top, "bot": bot, "supply": is_supply, "born": i, "active": True,
                      "retested": False, "score": adj, "rejection": rej, "extreme": ext, "impulse": imp,
                      "touches": 0, "last_touch": i, "pattern": ptype, "mult": mult,
                      "struct_pts": struct_pts, "type": ztype, "grade": grade})
        return True

    for i in range(n):
        if not math.isnan(ph[i]):
            prev_swing_high = last_swing_high; last_swing_high = ph[i]
        if not math.isnan(pl[i]):
            prev_swing_low = last_swing_low; last_swing_low = pl[i]
        is_hh = not (math.isnan(last_swing_high) or math.isnan(prev_swing_high)) and last_swing_high > prev_swing_high
        is_hl = not (math.isnan(last_swing_low) or math.isnan(prev_swing_low)) and last_swing_low > prev_swing_low
        is_lh = not (math.isnan(last_swing_high) or math.isnan(prev_swing_high)) and last_swing_high < prev_swing_high
        is_ll = not (math.isnan(last_swing_low) or math.isnan(prev_swing_low)) and last_swing_low < prev_swing_low
        bullish = is_hh or is_hl
        bearish = is_lh or is_ll
        near_swing_high = any(not math.isnan(ph[j]) for j in range(max(0, i - struct_lookback), i + 1))
        near_swing_low = any(not math.isnan(pl[j]) for j in range(max(0, i - struct_lookback), i + 1))

        up_trend = c[i] > ema_a[i]
        down_trend = c[i] < ema_a[i]
        chop_i = chop[i]
        if auto_tune and not math.isnan(chop_i):
            body_frac_eff = kt_body_base * (1 - 0.5 * max(0.0, (chop_i - 50) / 50))
            wick_body_eff = kt_wick_base * (1 + 0.25 * max(0.0, (chop_i - 50) / 50))
        else:
            body_frac_eff = kt_body_base
            wick_body_eff = kt_wick_base

        # tweezer
        t_top = t_bot = False
        t_top_strength = t_bot_strength = float("nan")
        t_top_back = t_bot_back = None
        if input_enable_tweezers:
            if (not tweezer_require_trend) or up_trend:
                for k in range(1, min(tweezer_lookback, i) + 1):
                    j = i - k
                    if white[j] and black[i]:
                        level_diff = abs(h[i] - h[j]) / h[i]
                        if level_diff <= tweezer_tol_eff:
                            body_ok = (body[j] > body_avg[j] and body[i] > body_avg[i] * 0.5) if tweezer_require_long_body else True
                            wick_ok = True; wick_sim = 1.0
                            if tweezer_check_wicks:
                                w1 = h[j] - body_hi[j]; w2 = h[i] - body_hi[i]
                                ratio = max(w1, w2) / max(1e-6, min(w1, w2))
                                wick_ok = ratio <= tweezer_wick_tolerance
                                wick_sim = 1.0 - min(1.0, (ratio - 1.0) / max(1e-6, tweezer_wick_tolerance - 1.0))
                            vol_ok = (not real) or rel_vol[i] >= tweezer_min_volume
                            if body_ok and wick_ok and vol_ok:
                                t_top = True; t_top_back = k
                                t_top_strength = (1.0 - min(1.0, level_diff / max(1e-6, tweezer_tol_eff))) * 0.6 + wick_sim * 0.4
                                break
            if (not tweezer_require_trend) or down_trend:
                for k in range(1, min(tweezer_lookback, i) + 1):
                    j = i - k
                    if black[j] and white[i]:
                        level_diff = abs(l[i] - l[j]) / l[i]
                        if level_diff <= tweezer_tol_eff:
                            body_ok = (body[j] > body_avg[j] and body[i] > body_avg[i] * 0.5) if tweezer_require_long_body else True
                            wick_ok = True; wick_sim = 1.0
                            if tweezer_check_wicks:
                                w1 = body_lo[j] - l[j]; w2 = body_lo[i] - l[i]
                                ratio = max(w1, w2) / max(1e-6, min(w1, w2))
                                wick_ok = ratio <= tweezer_wick_tolerance
                                wick_sim = 1.0 - min(1.0, (ratio - 1.0) / max(1e-6, tweezer_wick_tolerance - 1.0))
                            vol_ok = (not real) or rel_vol[i] >= tweezer_min_volume
                            if body_ok and wick_ok and vol_ok:
                                t_bot = True; t_bot_back = k
                                t_bot_strength = (1.0 - min(1.0, level_diff / max(1e-6, tweezer_tol_eff))) * 0.6 + wick_sim * 0.4
                                break

        # kangaroo
        kt_bear = kt_bull = False
        kt_bear_strength = kt_bull_strength = float("nan")
        if input_enable_kangaroo:
            wb_bear = u_wick[i] / max(1e-6, body[i]); wb_bull = l_wick[i] / max(1e-6, body[i])
            body_frac = body[i] / max(1e-6, rng[i])
            pos_bottom = (body_lo[i] - l[i]) / max(1e-6, rng[i])
            pos_top = (h[i] - body_hi[i]) / max(1e-6, rng[i])
            vol_ok = (not real) or rel_vol[i] >= kt_min_volume
            bear_basic = wb_bear >= wick_body_eff and body_frac <= body_frac_eff
            if bear_basic and pos_bottom <= kt_body_position and ((not kt_require_trend) or up_trend) and vol_ok:
                kt_bear = True
                kt_bear_strength = min(1.0, (u_wick[i] / max(1e-6, rng[i]) - 0.5) / 0.4)
            bull_basic = wb_bull >= wick_body_eff and body_frac <= body_frac_eff
            if bull_basic and pos_top <= kt_body_position and ((not kt_require_trend) or down_trend) and vol_ok:
                kt_bull = True
                kt_bull_strength = min(1.0, (l_wick[i] / max(1e-6, rng[i]) - 0.5) / 0.4)

        # zone creation
        if t_top and t_top_back is not None:
            new_zone(True, max(h[i], h[i - t_top_back]), max(body_hi[i], body_hi[i - t_top_back]), "T", t_top_strength)
        if t_bot and t_bot_back is not None:
            new_zone(False, min(body_lo[i], body_lo[i - t_bot_back]), min(l[i], l[i - t_bot_back]), "T", t_bot_strength)
        if kt_bear:
            new_zone(True, h[i], max(o[i], c[i]), "K", kt_bear_strength)
        if kt_bull:
            new_zone(False, min(o[i], c[i]), l[i], "K", kt_bull_strength)

        # zone update
        for z in zones:
            if not z["active"]:
                continue
            if i - z["born"] > zone_ttl:
                z["active"] = False
                continue
            if invalidate_on_close:
                invalidated = (c[i] > z["top"]) if z["supply"] else (c[i] < z["bot"])
            else:
                invalidated = (h[i] > z["top"]) if z["supply"] else (l[i] < z["bot"])
            if invalidated and i > z["born"]:
                z["active"] = False
                continue
            touched = (h[i] >= z["bot"] and l[i] <= z["top"]) and i > z["born"]
            if touched:
                z["touches"] += 1; z["last_touch"] = i; z["retested"] = True
                z["rejection"] = max(z["rejection"], calc_rejection(z["supply"], None))
                z["extreme"] = max(z["extreme"], calc_extreme(z["supply"], z["top"], z["bot"]))
                z["impulse"] = max(z["impulse"], calc_impulse(z["supply"]))
            age = i - z["born"]
            fresh = math.pow(0.5, age / zone_half_life)
            clean = 1.0 - min(1.0, z["touches"] / zone_max_touches)
            z["score"] = max(0.0, min(100.0, calc_score(z["rejection"], z["extreme"], z["impulse"], fresh, clean) * z["mult"] + z["struct_pts"]))
            z["grade"] = "A" if (z["score"] >= grade_a_min and z["touches"] <= 1) else "B" if z["score"] >= grade_b_min else "C"

        active = [z for z in zones if z["active"]]
        supplies = [z for z in active if z["supply"]]
        demands = [z for z in active if not z["supply"]]
        best_sup = max(supplies, key=lambda z: z["score"]) if supplies else None
        best_dem = max(demands, key=lambda z: z["score"]) if demands else None

        out["chop"][i] = chop_i
        out["rel_vol"][i] = rel_vol[i]
        out["zone_count"][i] = len(active)
        out["tweezer_top_strength"][i] = t_top_strength
        out["tweezer_bot_strength"][i] = t_bot_strength
        out["kangaroo_bear_strength"][i] = kt_bear_strength
        out["kangaroo_bull_strength"][i] = kt_bull_strength
        if best_sup is not None:
            out["supply_top"][i] = best_sup["top"]; out["supply_bottom"][i] = best_sup["bot"]
            out["supply_score"][i] = best_sup["score"]
            flags["supply_grade_a"][i] = 1 if best_sup["grade"] == "A" else 0
        if best_dem is not None:
            out["demand_top"][i] = best_dem["top"]; out["demand_bottom"][i] = best_dem["bot"]
            out["demand_score"][i] = best_dem["score"]
            flags["demand_grade_a"][i] = 1 if best_dem["grade"] == "A" else 0
        flags["tweezer_top"][i] = 1 if t_top else 0
        flags["tweezer_bot"][i] = 1 if t_bot else 0
        flags["kangaroo_bear"][i] = 1 if kt_bear else 0
        flags["kangaroo_bull"][i] = 1 if kt_bull else 0

    df = pd.DataFrame(index=idx)
    df["tkz_chop"] = out["chop"]
    df["tkz_rel_vol"] = out["rel_vol"]
    df["tkz_zone_count"] = pd.Series(out["zone_count"], index=idx).astype("Int64")
    df["tkz_tweezer_top_strength"] = out["tweezer_top_strength"]
    df["tkz_tweezer_bot_strength"] = out["tweezer_bot_strength"]
    df["tkz_kangaroo_bear_strength"] = out["kangaroo_bear_strength"]
    df["tkz_kangaroo_bull_strength"] = out["kangaroo_bull_strength"]
    df["tkz_supply_top"] = out["supply_top"]
    df["tkz_supply_bottom"] = out["supply_bottom"]
    df["tkz_supply_score"] = out["supply_score"]
    df["tkz_demand_top"] = out["demand_top"]
    df["tkz_demand_bottom"] = out["demand_bottom"]
    df["tkz_demand_score"] = out["demand_score"]
    for k, v in flags.items():
        df[f"tkz_{k}"] = pd.Series(v, index=idx).astype("Int64")
    return df
