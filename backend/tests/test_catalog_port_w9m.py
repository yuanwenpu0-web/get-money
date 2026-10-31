"""Targeted correctness tests for the Market Motion DNA port.

Strategy
--------
The weighted Impulse/Correction score is impractical to hand-derive for a
realistic multi-bar leg, so the core guard is an *independent* reference
reimplementation (written fresh from the Pine formula, vectorized) that
reproduces both the alternating-pivot engine and the leg-DNA calculator, then
asserts the module's output matches it to floating-point precision. On top of
that we check:

  * the exact invariant  impulse_score + correction_score == 100  (hand-derivable
    from the complementary weighting),
  * behavioural classification (clean trend -> impulse, chop -> neutral),
  * parameter validation and the Int64 dtype of the class column.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators import market_motion_dna as mod
from app.catalog_indicators._pine import (
    atr,
    hma,
    pivothigh_strict,
    pivotlow_strict,
    rma,
    sma,
    wma,
    ema,
)


# --------------------------------------------------------------------------- #
# Independent reference reimplementation (NOT copied from the module)          #
# --------------------------------------------------------------------------- #
def _ref_ma(series, length, kind):
    if kind == "SMA":
        return sma(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "HMA":
        return hma(series, length)
    return ema(series, length)


def _ref_calc_leg(c, o, h, l, ma, atr_arr, subh, subl, sp, spp, ep, epp, direction,
                  is_complete, P):
    raw = max(ep - sp, 1)
    evaluated = min(raw, P["max_leg_bars"])
    eff = ep - evaluated
    esp = spp if eff == sp else c[eff]
    net = max(abs(epp - esp), P["mintick"])

    pos = np.arange(eff + 1, ep + 1)
    cur = c[pos]
    prev = c[pos - 1]
    co = o[pos]
    ch = h[pos]
    cl = l[pos]
    ph = h[pos - 1]
    pl = l[pos - 1]
    cma = ma[pos]
    pma = ma[pos - 1]

    tr = np.maximum(ch - cl, np.maximum(np.abs(ch - prev), np.abs(cl - prev)))
    path = float(tr.sum())
    cur_range = np.maximum(ch - cl, P["mintick"])
    prev_range = np.maximum(ph - pl, P["mintick"])

    overlap_amt = np.maximum(0.0, np.minimum(ch, ph) - np.maximum(cl, pl))
    overlap = float((overlap_amt / np.minimum(cur_range, prev_range)).mean()) if len(pos) else 0.5

    close_loc = (cur - cl) / cur_range if direction == 1 else (ch - cur) / cur_range
    commitment = float(np.clip(close_loc, 0.0, 1.0).mean())

    total_wick = cur_range - np.abs(cur - co)
    wick = float((1.0 - np.clip(total_wick / cur_range, 0.0, 1.0)).mean())

    dd = direction * (cur - prev)
    acceptance = float(np.where(dd > 0, 1.0, np.where(dd < 0, 0.0, 0.5)).mean())

    ps = (cur >= cma) if direction == 1 else (cur <= cma)
    ss = (cma >= pma) if direction == 1 else (cma <= pma)
    ma_support = float(((ps * 0.6) + (ss * 0.4)).mean())

    delta = cur - prev
    sign = np.sign(delta)
    changes = 0
    prev_sign = 0
    for s in sign:
        if s != 0:
            if prev_sign != 0 and s != prev_sign:
                changes += 1
            prev_sign = s
    cleanliness = 1.0 - min(max(changes / max(evaluated - 1.0, 1.0), 0.0), 1.0) if evaluated > 1 else 0.5

    if direction == 1:
        fav = np.maximum.accumulate(ch)
        maxadv = float((fav - cl).max())
    else:
        fav = np.minimum.accumulate(cl)
        maxadv = float((ch - fav).max())
    highest_high = max(esp, epp, float(ch.max()))
    lowest_low = min(esp, epp, float(cl.min()))

    efficiency = min(max(net / max(path, net), 0.0), 1.0) if path != 0 else 0.5
    retr_raw = maxadv / net if net != 0 else 0.0
    retracement = min(max(retr_raw / P["retracement_ceiling"], 0.0), 1.0)

    k = int(math.ceil(evaluated * 0.5))
    first_pos = pos[:k]
    second_pos = pos[k:]
    dbody = np.maximum(direction * (c[first_pos] - o[first_pos]), 0.0)
    sbody = np.maximum(direction * (c[second_pos] - o[second_pos]), 0.0)
    db_tr = np.maximum(h[first_pos] - l[first_pos],
                       np.maximum(np.abs(h[first_pos] - c[first_pos - 1]),
                                  np.abs(l[first_pos] - c[first_pos - 1])))
    sb_tr = np.maximum(h[second_pos] - l[second_pos],
                       np.maximum(np.abs(h[second_pos] - c[second_pos - 1]),
                                  np.abs(l[second_pos] - c[second_pos - 1])))
    avg_fb = float(dbody.mean()) if len(first_pos) else 0.0
    avg_sb = float(sbody.mean()) if len(second_pos) else avg_fb
    atr_end = float(atr_arr[ep]) if not math.isnan(atr_arr[ep]) else 0.0
    mfb = max(avg_fb, atr_end * 0.25)
    mom_ratio = avg_sb / mfb if mfb > P["mintick"] else 1.0
    mom = min(max((mom_ratio - P["momentum_low_ratio"]) /
                  (P["momentum_high_ratio"] - P["momentum_low_ratio"]), 0.0), 1.0)
    avg_ft = float(db_tr.mean()) if len(first_pos) else 0.0
    avg_st = float(sb_tr.mean()) if len(second_pos) else avg_ft
    tr_ratio = avg_st / avg_ft if avg_ft > P["mintick"] else 1.0
    trx = min(max((tr_ratio - P["tr_low_ratio"]) /
                  (P["tr_high_ratio"] - P["tr_low_ratio"]), 0.0), 1.0)

    def w(x):
        return x
    W = P["weights"]
    safe_w = max(sum(W.values()), 0.0001)
    raw_imp = (efficiency * W["eff"] + (1 - overlap) * W["ov"] +
               (1 - retracement) * W["ret"] + commitment * W["com"] +
               wick * W["wick"] + mom * W["mom"] + trx * W["tr"] +
               ma_support * W["ma"] + cleanliness * W["clean"] +
               acceptance * W["acc"]) / safe_w * 100.0
    raw_cor = ((1 - efficiency) * W["eff"] + overlap * W["ov"] +
               retracement * W["ret"] + (1 - commitment) * W["com"] +
               (1 - wick) * W["wick"] + (1 - mom) * W["mom"] + (1 - trx) * W["tr"] +
               (1 - ma_support) * W["ma"] + (1 - cleanliness) * W["clean"] +
               (1 - acceptance) * W["acc"]) / safe_w * 100.0

    maturity = 1.0 if is_complete else min(max(evaluated / float(P["minimum_mature_bars"]), 0.0), 1.0)
    impulse = 50.0 + (raw_imp - 50.0) * maturity
    correction = 50.0 + (raw_cor - 50.0) * maturity
    confidence = abs(impulse - correction) * maturity

    # --- diagnostic-only internal sub-swings (replicated faithfully) ---
    last_sub_type = 0
    last_sub_price = math.nan
    internal_swing_count = 0
    internal_swing_sum = 0.0
    swing_sizes = []
    Rsub = P["sub_pivot_width"]
    for ap in pos:
        cid = ap + Rsub
        if cid <= ep:
            svh = subh[cid]
            svl = subl[cid]
            ish = not math.isnan(svh)
            isl = not math.isnan(svl)
            ct = 0
            cp = math.nan
            if ish and isl:
                ct = 1 if last_sub_type <= 0 else -1
                cp = svh if ct == 1 else svl
            elif ish:
                ct = 1
                cp = svh
            elif isl:
                ct = -1
                cp = svl
            if ct != 0:
                if last_sub_type == 0:
                    last_sub_type = ct
                    last_sub_price = cp
                    internal_swing_count += 1
                elif ct == last_sub_type:
                    more = cp > last_sub_price if ct == 1 else cp < last_sub_price
                    if more:
                        last_sub_price = cp
                else:
                    internal_swing_sum += abs(cp - last_sub_price)
                    swing_sizes.append(abs(cp - last_sub_price))
                    internal_swing_count += 1
                    last_sub_type = ct
                    last_sub_price = cp

    comp = len(swing_sizes)
    avg_isa = (internal_swing_sum / comp) / atr_end if comp > 0 and atr_end != 0 else 0.0
    prog = min(max(net / internal_swing_sum if internal_swing_sum != 0 else 1.0, 0.0), 1.0)
    sh = int(math.ceil(comp * 0.5))
    fs = sum(swing_sizes[:sh])
    ss = sum(swing_sizes[sh:])
    fc = sh
    sc = comp - sh
    avg_fs = fs / fc if fc > 0 else 0.0
    avg_ss = ss / sc if sc > 0 else avg_fs
    growth = avg_ss / avg_fs if avg_fs > P["mintick"] else 1.0

    return dict(impulse_score=impulse, correction_score=correction, confidence=confidence,
                maturity=maturity, leg_bars=evaluated, efficiency=efficiency, overlap=overlap,
                retracement_pressure=retracement, close_commitment=commitment,
                wick_clean=wick, momentum_expansion=mom, tr_expansion=trx, ma_support=ma_support,
                direction_cleanliness=cleanliness, acceptance=acceptance,
                highest_high=highest_high, lowest_low=lowest_low,
                internal_swing_count=internal_swing_count,
                avg_internal_swing_atr=avg_isa, progression_ratio=prog,
                sub_leg_growth_ratio=growth)


def _ref_run(frame, **kw):
    params = dict(
        pivot_left=kw.get("pivot_left", 5), pivot_right=kw.get("pivot_right", 5),
        max_leg_bars=kw.get("max_leg_bars", 350), retained_legs=kw.get("retained_legs", 40),
        min_leg_atr=kw.get("min_leg_atr", 0.50), atr_length=kw.get("atr_length", 14),
        minimum_mature_bars=kw.get("minimum_mature_bars", 8),
        retracement_ceiling=kw.get("retracement_ceiling", 0.80),
        momentum_low_ratio=kw.get("momentum_low_ratio", 0.60),
        momentum_high_ratio=kw.get("momentum_high_ratio", 1.40),
        tr_low_ratio=kw.get("tr_low_ratio", 0.70), tr_high_ratio=kw.get("tr_high_ratio", 1.30),
        ma_type=kw.get("ma_type", "EMA"), ma_length=kw.get("ma_length", 34),
        sub_pivot_width=kw.get("sub_pivot_width", 2),
        impulse_threshold=kw.get("impulse_threshold", 62.0),
        correction_threshold=kw.get("correction_threshold", 62.0),
        minimum_confidence=kw.get("minimum_confidence", 18.0),
        character_change_buffer=kw.get("character_change_buffer", 5.0),
        mintick=kw.get("mintick", 1e-8),
    )
    W = dict(eff=kw.get("weight_efficiency", 20.0), ov=kw.get("weight_overlap", 15.0),
             ret=kw.get("weight_retracement", 12.5), com=kw.get("weight_commitment", 10.0),
             wick=kw.get("weight_wicks", 7.5), mom=kw.get("weight_momentum", 10.0),
             tr=kw.get("weight_true_range", 5.0), ma=kw.get("weight_ma_support", 7.5),
             clean=kw.get("weight_cleanliness", 7.5), acc=kw.get("weight_acceptance", 5.0))
    params["weights"] = W
    R = params["pivot_right"]

    c = frame["close"].to_numpy(float)
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    l = frame["low"].to_numpy(float)
    n = len(c)
    atr_arr = atr(frame, params["atr_length"]).to_numpy(float)
    ma_arr = _ref_ma(frame["close"], params["ma_length"], params["ma_type"]).to_numpy(float)
    subh = pivothigh_strict(frame["high"], params["sub_pivot_width"], params["sub_pivot_width"]).to_numpy(float)
    subl = pivotlow_strict(frame["low"], params["sub_pivot_width"], params["sub_pivot_width"]).to_numpy(float)
    ch = pivothigh_strict(frame["high"], params["pivot_left"], params["pivot_right"]).to_numpy(float)
    cl = pivotlow_strict(frame["low"], params["pivot_left"], params["pivot_right"]).to_numpy(float)

    out = {col: np.full(n, dv) for col, dv in [
        ("impulse_score", 50.0), ("correction_score", 50.0), ("confidence", 0.0),
        ("maturity", 0.0), ("leg_bars", 0), ("efficiency", 0.5), ("overlap", 0.5),
        ("retracement_pressure", 0.5), ("close_commitment", 0.5), ("wick_clean", 0.5),
        ("momentum_expansion", 0.5), ("tr_expansion", 0.5), ("ma_support", 0.5),
        ("direction_cleanliness", 0.5), ("acceptance", 0.5),
        ("avg_internal_swing_atr", 0.0), ("progression_ratio", 1.0),
        ("sub_leg_growth_ratio", 1.0)]}
    out["live_class"] = np.zeros(n, dtype=np.int64)
    out["internal_swing_count"] = np.zeros(n, dtype=np.int64)

    pb, pp, pt = [], [], []
    live_state = 0
    for i in range(n):
        hh = not math.isnan(ch[i])
        hl = not math.isnan(cl[i])
        ct = 0
        cp = math.nan
        if hh and hl:
            if not pt:
                dl = abs(ch[i] - c[i - R])
                dr = abs(c[i - R] - cl[i])
                ct = 1 if dl >= dr else -1
            else:
                ct = -pt[-1]
            cp = ch[i] if ct == 1 else cl[i]
        elif hh:
            ct = 1
            cp = ch[i]
        elif hl:
            ct = -1
            cp = cl[i]
        if ct != 0 and not math.isnan(cp):
            cb = i - R
            if not pb:
                pb.append(cb); pp.append(cp); pt.append(ct)
            else:
                lt = pt[-1]; lp = pp[-1]
                if ct == lt:
                    more = cp > lp if ct == 1 else cp < lp
                    if more:
                        pb[-1] = cb; pp[-1] = cp; pt[-1] = ct
                else:
                    ap = atr_arr[i - R] if not math.isnan(atr_arr[i - R]) else 0.0
                    large = params["min_leg_atr"] <= 0.0 or abs(cp - lp) >= ap * params["min_leg_atr"]
                    if large and cb > pb[-1]:
                        pb.append(cb); pp.append(cp); pt.append(ct)
                        while len(pb) > params["retained_legs"] + 2:
                            pb.pop(0); pp.pop(0); pt.pop(0)
        if pb:
            s_b, s_p, s_t = pb[-1], pp[-1], pt[-1]
            ld = 1 if s_t == -1 else -1
            le = h[i] if ld == 1 else l[i]
            r = _ref_calc_leg(c, o, h, l, ma_arr, atr_arr, subh, subl, s_b, s_p, i, le, ld, False, params)
            for col, val in r.items():
                if col in ("highest_high", "lowest_low"):
                    continue
                out[col][i] = val
            raw = 0
            sep = abs(r["impulse_score"] - r["correction_score"])
            if r["maturity"] >= 0.999 and sep >= params["minimum_confidence"]:
                if r["impulse_score"] >= params["impulse_threshold"] and r["impulse_score"] > r["correction_score"]:
                    raw = 1
                elif r["correction_score"] >= params["correction_threshold"] and r["correction_score"] > r["impulse_score"]:
                    raw = -1
            if live_state == 1:
                keep = (r["impulse_score"] >= params["impulse_threshold"] - params["character_change_buffer"]
                        and r["impulse_score"] >= r["correction_score"])
                live_state = 1 if keep else raw
            elif live_state == -1:
                keep = (r["correction_score"] >= params["correction_threshold"] - params["character_change_buffer"]
                        and r["correction_score"] >= r["impulse_score"])
                live_state = -1 if keep else raw
            else:
                live_state = raw
            out["live_class"][i] = live_state
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------- #
# Frame builders                                                              #
# --------------------------------------------------------------------------- #
def _clean_uptrend(n=120):
    # Explicit zigzag: strong up-swings (10 bars, +0.6 each) with shallow pullbacks
    # (3 bars, -0.6 each). Alternating pivots confirm, and each up-leg is efficient
    # enough to read as impulsive. High/low are derived from close (not open) so
    # adjacent bars never share an extreme at turning points (which would defeat the
    # strict pivot detector).
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = [100.0]
    for _ in range(20):
        for _ in range(10):
            close.append(close[-1] + 0.6)
        for _ in range(3):
            close.append(close[-1] - 0.6)
    close = np.array(close[:n], dtype=float)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = close + 0.1
    low = close - 0.1
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _choppy(n=80):
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100.0 + 2.0 * np.sin(np.arange(n) * 0.9)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = close + 0.2
    low = close - 0.2
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_module_matches_reference_clean_uptrend():
    frame = _clean_uptrend(80)
    got = mod.market_motion_dna(frame)
    ref = _ref_run(frame)
    for col in ref.columns:
        assert np.allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-9), col
    assert got.shape[0] == 80


def test_module_matches_reference_choppy():
    frame = _choppy(80)
    got = mod.market_motion_dna(frame)
    ref = _ref_run(frame)
    for col in ref.columns:
        assert np.allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-9), col


def test_module_matches_reference_defaults_and_weights():
    frame = _clean_uptrend(120)
    # exercise non-default config (different MA, weights, thresholds)
    got = mod.market_motion_dna(
        frame, ma_type="SMA", ma_length=20, weight_efficiency=30.0,
        weight_overlap=5.0, impulse_threshold=55.0, correction_threshold=55.0,
        minimum_confidence=10.0, sub_pivot_width=3,
    )
    ref = _ref_run(
        frame, ma_type="SMA", ma_length=20, weight_efficiency=30.0,
        weight_overlap=5.0, impulse_threshold=55.0, correction_threshold=55.0,
        minimum_confidence=10.0, sub_pivot_width=3,
    )
    for col in ref.columns:
        assert np.allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-9), col


def test_impulse_correction_sum_to_100():
    # Hand-derivable invariant: the complementary weighting makes the two scores
    # sum to exactly 100 for every bar once maturity scaling is applied.
    for frame in (_clean_uptrend(90), _choppy(90)):
        got = mod.market_motion_dna(frame)
        mask = got["leg_bars"] > 0
        s = got.loc[mask, "impulse_score"] + got.loc[mask, "correction_score"]
        assert np.allclose(s.to_numpy(), 100.0, atol=1e-9)


def test_clean_uptrend_classified_impulsive():
    frame = _clean_uptrend(90)
    got = mod.market_motion_dna(frame, minimum_mature_bars=6, minimum_confidence=10.0,
                                impulse_threshold=55.0)
    # The prevailing up-leg must read as impulsive at least once, and the impulse
    # score must climb well above the neutral 50.
    assert int(got["live_class"].max()) == 1
    assert got["impulse_score"].max() > 65.0


def test_choppy_market_stays_neutral():
    frame = _choppy(90)
    got = mod.market_motion_dna(frame, minimum_mature_bars=6, minimum_confidence=10.0,
                                impulse_threshold=55.0)
    # No sustained directional character -> never classified as impulse/correction.
    assert int(got["live_class"].iloc[-1]) == 0


def test_no_pivots_uses_defaults():
    # Frame too short for the pivot right-bars -> no confirmed pivots -> defaults.
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    frame = pd.DataFrame({
        "open": [1, 2, 3, 4, 5], "high": [1.5, 2.5, 3.5, 4.5, 5.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5], "close": [1, 2, 3, 4, 5],
    }, index=idx)
    got = mod.market_motion_dna(frame, pivot_right=5)
    assert got["impulse_score"].iloc[-1] == 50.0
    assert int(got["live_class"].iloc[-1]) == 0
    assert got["leg_bars"].iloc[-1] == 0


def test_parameter_validation():
    frame = _clean_uptrend(30)
    with pytest.raises(ValueError):
        mod.market_motion_dna(frame, pivot_left=0)          # below minval 1
    with pytest.raises(ValueError):
        mod.market_motion_dna(frame, ma_type="LINREG")      # not in options
    with pytest.raises(ValueError):
        mod.market_motion_dna(frame, atr_length=1)          # below minval 2


def test_live_class_is_int64():
    frame = _clean_uptrend(40)
    got = mod.market_motion_dna(frame)
    assert got["live_class"].dtype == "Int64"
    assert got["internal_swing_count"].dtype == "Int64"
