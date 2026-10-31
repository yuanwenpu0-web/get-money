"""Wyckoff Schematics [WavesUnchained]  v4.10.1  —  catalog port.

Source: data/pine-collection/indicators/market_structure/wyckoff_schematics/
        wyckoff_schematics.pine

Volume/spread-based Wyckoff schematic detector with an A–E phase state machine,
event labelling (PS/PSY, SC, BC, AR, ST, Spring, Test, SOS, SOW, LPS, LPSY,
UTAD, SOSB/SOWB), range/setup scoring, pivot-confirmed LPS/LPSY, phase-anchored
VWAP, effort-vs-result and No Supply / No Demand / VDU.

Ported computational core: all detectors, the accumulation/distribution A–E
state machine with range freeze / breakout / invalidation, cause & effect,
and both scores. Drawing, dashboard, alerts and the summary label are dropped.
HTF bias and relative strength need external series (request.security); HTF
bias falls back to the chart-timeframe EMA slope and relative strength is
neutral (no bonus).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, ema, integer, number, ohlcv, pivothigh_strict, pivotlow_strict, sma

_PHASE_CODE = {"RANGING": 0, "ACCUMULATION": 1, "DISTRIBUTION": 2, "MARKUP": 3, "MARKDOWN": 4}
_AE_CODE = {"": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _score_label(s):
    return "High Confidence" if s >= 80 else "Strong" if s >= 60 else "Moderate" if s >= 40 else "Weak"


def wyckoff_schematics(
    frame: pd.DataFrame,
    *,
    volume_lookback: int = 20,
    climax_multiplier: float = 2.0,
    spring_threshold: float = 2.0,
    sos_threshold: float = 70.0,
    range_establish_bars: int = 20,
    invalidation_factor: float = 0.33,
    pivot_len: int = 3,
    event_cooldown_bars: int = 10,
    markup_min_score: int = 50,
    htf_ema_period: int = 20,
) -> pd.DataFrame:
    volume_lookback = integer("volume_lookback", volume_lookback, 10, 100)
    range_establish_bars = integer("range_establish_bars", range_establish_bars, 5, 100)
    pivot_len = integer("pivot_len", pivot_len, 2, 10)
    event_cooldown_bars = integer("event_cooldown_bars", event_cooldown_bars, 1, 50)
    markup_min_score = integer("markup_min_score", markup_min_score, 0, 100)
    htf_ema_period = integer("htf_ema_period", htf_ema_period, 5, 100)
    climax_multiplier = number("climax_multiplier", climax_multiplier, 1.5, 5.0)
    spring_threshold = number("spring_threshold", spring_threshold, 0.5, 10.0)
    sos_threshold = number("sos_threshold", sos_threshold, 50.0, 95.0)
    invalidation_factor = number("invalidation_factor", invalidation_factor, 0.2, 2.0)

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float)
    l = l.to_numpy(dtype=float); c = c.to_numpy(dtype=float); vol = vol.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    volume_avg = sma(pd.Series(vol), volume_lookback).to_numpy(dtype=float)
    spread_avg = sma(pd.Series(h - l), 20).to_numpy(dtype=float)
    atr_a = np.maximum(atr(frame, 14).to_numpy(dtype=float), 1e-8)
    piv_low = pivotlow_strict(frame["low"], pivot_len, pivot_len).to_numpy(dtype=float)
    piv_high = pivothigh_strict(frame["high"], pivot_len, pivot_len).to_numpy(dtype=float)
    trend_ema = ema(frame["close"], 50).to_numpy(dtype=float)
    vol10 = sma(pd.Series(vol), 10).to_numpy(dtype=float)
    vol30 = sma(pd.Series(vol), 30).to_numpy(dtype=float)
    piv_low3 = pivotlow_strict(frame["low"], 3, 3).to_numpy(dtype=float)
    piv_high3 = pivothigh_strict(frame["high"], 3, 3).to_numpy(dtype=float)
    htf_ema = ema(frame["close"], htf_ema_period).to_numpy(dtype=float)

    real = bool(np.any(vol > 0))

    def new_phase(name, sub="", ae="", prior="UNKNOWN", i=0):
        return {"name": name, "sub": sub, "ae": ae, "schematic": "TBD", "startBar": i,
                "startPrice": c[i], "rangeHigh": h[i], "rangeLow": l[i], "confidence": 50.0,
                "rangeEstablishedBar": None, "climaxVolRatio": 0.0, "climaxPrice": None,
                "arExtreme": None, "arBar": None, "springUtadBar": None, "springUtadPrice": None,
                "springUtadVolRatio": None, "awaitingTest": False, "priorTrend": prior,
                "rangePivotTests": 0, "climaxBarOpposite": None, "springUtadLeft": False}

    def push(events, etype, bar, price, volr, spread, sig, desc="", above=False):
        events.append({"type": etype, "bar": bar, "price": price, "volumeRatio": volr,
                       "spread": spread, "significance": sig, "desc": desc, "above": above})
        if len(events) > 50:
            events.pop(0)

    def has_event(events, etype, start):
        return any(e["type"] == etype and e["bar"] >= start for e in events)

    def event_bar(events, etype, start):
        for e in events:
            if e["type"] == etype and e["bar"] >= start:
                return e["bar"]
        return -1

    events: list = []
    cur = None
    prev_phase = None
    composite = "NEUTRAL"
    setup_score = 0
    range_score = 0
    avwap = float("nan")
    vwap_num = vwap_den = 0.0
    spring_conf = utad_conf = False
    cause_built = effect_expected = pf_target = float("nan")

    out = {k: np.full(n, np.nan) for k in [
        "phase", "subphase", "phase_ae", "schematic", "setup_score", "range_score", "avwap",
        "range_high", "range_low", "confidence", "effort", "result", "effort_strength",
        "effort_rel", "direction"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "spring_confirmed", "utad_confirmed", "no_supply", "no_demand", "vdu",
        "sc", "bc", "ar", "st", "spring", "utad", "test", "sos", "sow", "lps", "lpsy",
        "sosb", "sowb", "ps", "psy"]}

    for i in range(n):
        vavg = volume_avg[i]
        spread_avg_i = spread_avg[i]
        atr_i = atr_a[i]
        if (math.isnan(vavg) or math.isnan(spread_avg_i) or vavg <= 0 or spread_avg_i <= 0
                or h[i] <= l[i] or c[i] <= 0):
            # carry forward current state output only
            if cur is not None:
                out["range_high"][i] = cur["rangeHigh"]
                out["range_low"][i] = cur["rangeLow"]
            continue

        vr = vol[i] / vavg if vavg > 0 else 1.0
        spread = h[i] - l[i]
        bar_range = h[i] - l[i]
        close_pos = (c[i] - l[i]) / bar_range if bar_range > 0 else 0.5

        # effort vs result
        effort = vol[i] / vavg if vavg > 0 else 0.0
        result = spread / spread_avg_i if spread_avg_i > 0 else 0.0
        rel = "CONFIRMATION"; strength = 50.0
        if effort > 2.0 and result < 0.8:
            rel = "CLIMAX"; strength = 90.0
        elif effort > 1.5 and result > 1.5 and close_pos < 0.3:
            rel = "CLIMAX"; strength = 82.0
        elif effort > 1.5 and result > 1.5 and close_pos > 0.7:
            strength = 80.0
        elif effort > 1.5 and result > 1.5:
            strength = 65.0
        elif effort < 0.8 and result > 1.2:
            rel = "DIVERGENCE"; strength = 70.0

        # prior trend (chart EMA slope proxy)
        slope = trend_ema[i] - trend_ema[i - 25] if i >= 25 and not math.isnan(trend_ema[i]) else 0.0
        if slope > trend_ema[i] * 0.001:
            prior_trend = "UP"
        elif slope < -trend_ema[i] * 0.001:
            prior_trend = "DOWN"
        else:
            prior_trend = "SIDEWAYS"
        # htf bias proxy (chart TF)
        if i >= 1 and not math.isnan(htf_ema[i]) and not math.isnan(htf_ema[i - 1]):
            htf_slope = htf_ema[i] - htf_ema[i - 1]
            if c[i] > htf_ema[i] * 1.005 and htf_slope > 0:
                htf_bias = "BULLISH"
            elif c[i] < htf_ema[i] * 0.995 and htf_slope < 0:
                htf_bias = "BEARISH"
            else:
                htf_bias = "NEUTRAL"
        else:
            htf_bias = "NEUTRAL"

        recent_vol_decline = (not math.isnan(vol10[i]) and not math.isnan(vol30[i]) and vol10[i] < vol30[i])

        # climax detection
        exhaustion_close = c[i] < o[i] and close_pos < 0.35
        absorption_close = c[i] > o[i] and close_pos > 0.45
        is_selling_climax = ((not real or vr >= climax_multiplier) and bar_range > atr_i * 1.5
                            and (exhaustion_close or absorption_close))
        exhaustion_close_b = c[i] > o[i] and close_pos > 0.65
        absorption_close_b = c[i] < o[i] and close_pos < 0.55
        is_buying_climax = ((not real or vr >= climax_multiplier) and bar_range > atr_i * 1.5
                           and (exhaustion_close_b or absorption_close_b))

        if cur is None:
            cur = new_phase("RANGING", prior=prior_trend, i=i)

        # --- SC / BC ---
        if is_selling_climax and cur["name"] != "ACCUMULATION" and prior_trend != "UP":
            prev_phase = cur
            cur = new_phase("ACCUMULATION", "SC", "A", prior_trend, i)
            cur["climaxVolRatio"] = vr; cur["climaxPrice"] = l[i]; cur["climaxBarOpposite"] = h[i]
            composite = "ACCUMULATING"; spring_conf = utad_conf = False; vwap_num = vwap_den = 0.0
            push(events, "SC", i, l[i], vr, spread, 90.0, "Selling Climax")
            flags["sc"][i] = 1
            for j in range(1, min(21, i + 1)):
                jj = i - j
                rng = h[jj] - l[jj]
                cr = (c[jj] - l[jj]) / rng if rng > 0 else 0.5
                if (not real or vol[jj] > volume_avg[jj] * 1.1) and cr > 0.30:
                    push(events, "PS", jj, l[jj], (vol[jj] / volume_avg[jj]) if volume_avg[jj] > 0 else 1.0, rng, 60.0, "Preliminary Support")
                    flags["ps"][i] = 1
                    break
        elif is_buying_climax and cur["name"] != "DISTRIBUTION" and prior_trend != "DOWN":
            prev_phase = cur
            cur = new_phase("DISTRIBUTION", "BC", "A", prior_trend, i)
            cur["climaxVolRatio"] = vr; cur["climaxPrice"] = h[i]; cur["climaxBarOpposite"] = l[i]
            composite = "DISTRIBUTING"; spring_conf = utad_conf = False; vwap_num = vwap_den = 0.0
            push(events, "BC", i, h[i], vr, spread, 90.0, "Buying Climax", True)
            flags["bc"][i] = 1
            for j in range(1, min(21, i + 1)):
                jj = i - j
                rng = h[jj] - l[jj]
                cr = (c[jj] - l[jj]) / rng if rng > 0 else 0.5
                if (not real or vol[jj] > volume_avg[jj] * 1.1) and cr < 0.70:
                    push(events, "PSY", jj, h[jj], (vol[jj] / volume_avg[jj]) if volume_avg[jj] > 0 else 1.0, rng, 60.0, "Preliminary Supply", True)
                    flags["psy"][i] = 1
                    break

        phase_age = i - cur["startBar"]
        range_frozen = cur["rangeEstablishedBar"] is not None
        is_ad = cur["name"] in ("ACCUMULATION", "DISTRIBUTION")
        is_accum = cur["name"] == "ACCUMULATION"

        # range expansion
        if not range_frozen:
            if h[i] > cur["rangeHigh"]:
                cur["rangeHigh"] = h[i]
            if l[i] < cur["rangeLow"]:
                cur["rangeLow"] = l[i]
            if is_ad and phase_age >= range_establish_bars:
                cur["rangeEstablishedBar"] = i
                range_frozen = True

        # AVWAP
        if is_ad and real:
            vwap_num += vol[i] * c[i]; vwap_den += vol[i]
            if vwap_den > 0:
                avwap = vwap_num / vwap_den

        # --- phase A-E state machine ---
        if is_ad and phase_age > 0:
            ae = cur["ae"]
            climax_ref = cur["climaxPrice"] if cur["climaxPrice"] is not None else (cur["rangeLow"] if is_accum else cur["rangeHigh"])

            if ae == "A":
                ar_min = atr_i
                if cur["arBar"] is None and phase_age <= 15 and (not real or vr >= 0.8):
                    if is_accum and c[i] > o[i] and cur["climaxPrice"] is not None and h[i] - cur["climaxPrice"] >= ar_min and h[i] > cur["climaxBarOpposite"]:
                        cur["arExtreme"] = h[i]; cur["arBar"] = i
                        push(events, "AR", i, h[i], vr, spread, 70.0, "Automatic Rally", True)
                        flags["ar"][i] = 1
                    elif (not is_accum) and c[i] < o[i] and cur["climaxPrice"] is not None and cur["climaxPrice"] - l[i] >= ar_min and l[i] < cur["climaxBarOpposite"]:
                        cur["arExtreme"] = l[i]; cur["arBar"] = i
                        push(events, "AR", i, l[i], vr, spread, 70.0, "Automatic Reaction")
                        flags["ar"][i] = 1
                elif cur["arBar"] is not None:
                    ar_age = i - cur["arBar"]
                    narrow = spread < spread_avg_i * 1.2
                    vol_ok = (not real) or vr < cur["climaxVolRatio"]
                    if is_accum:
                        st_hit = l[i] >= climax_ref * 0.97 and l[i] <= climax_ref * 1.05 and vol_ok and narrow
                    else:
                        st_hit = h[i] <= climax_ref * 1.03 and h[i] >= climax_ref * 0.95 and vol_ok and narrow
                    if ar_age >= 3 and st_hit:
                        cur["ae"] = "B"; cur["sub"] = "ST"
                        push(events, "ST", i, l[i] if is_accum else h[i], vr, spread, 70.0, "Secondary Test", not is_accum)
                        flags["st"][i] = 1
                if cur["ae"] == "A" and phase_age > range_establish_bars:
                    cur["ae"] = "B"

            elif ae == "B":
                range_w = cur["rangeHigh"] - cur["rangeLow"]
                spring_depth = cur["rangeLow"] - l[i]
                lw = min(o[i], c[i]) - l[i]
                wick_ratio = lw / bar_range if bar_range > 0 else 0.0
                min_break = max(atr_i * 0.25, range_w * spring_threshold / 100.0)
                spring = (range_w > 0 and spring_depth >= min_break and c[i] > cur["rangeLow"]
                          and (c[i] > o[i] or close_pos > 0.55) and wick_ratio >= 0.25)
                utad_height = h[i] - cur["rangeHigh"]
                uw = h[i] - max(o[i], c[i])
                uw_ratio = uw / bar_range if bar_range > 0 else 0.0
                utad = (range_w > 0 and utad_height >= min_break and c[i] < cur["rangeHigh"]
                        and (c[i] < o[i] or close_pos < 0.45) and uw_ratio >= 0.25)
                if is_accum and range_frozen and spring:
                    cur["ae"] = "C"; cur["sub"] = "SPRING"; cur["schematic"] = "#1"
                    cur["springUtadBar"] = i; cur["springUtadPrice"] = l[i]; cur["springUtadVolRatio"] = vr
                    cur["awaitingTest"] = True; cur["springUtadLeft"] = False
                    push(events, "SPRING", i, l[i], vr, spread, 85.0, "Spring")
                    flags["spring"][i] = 1
                elif (not is_accum) and range_frozen and utad:
                    cur["ae"] = "C"; cur["sub"] = "UTAD"; cur["schematic"] = "#1"
                    cur["springUtadBar"] = i; cur["springUtadPrice"] = h[i]; cur["springUtadVolRatio"] = vr
                    cur["awaitingTest"] = True; cur["springUtadLeft"] = False
                    push(events, "UTAD", i, h[i], vr, spread, 85.0, "UTAD", True)
                    flags["utad"][i] = 1
                else:
                    narrow = spread < spread_avg_i * 1.2
                    vol_ok = (not real) or vr < cur["climaxVolRatio"]
                    if is_accum:
                        st_hit = l[i] >= climax_ref * 0.97 and l[i] <= climax_ref * 1.05 and vol_ok and narrow
                    else:
                        st_hit = h[i] <= climax_ref * 1.03 and h[i] >= climax_ref * 0.95 and vol_ok and narrow
                    if st_hit and not has_event(events, "ST", i - event_cooldown_bars):
                        cur["sub"] = "ST"
                        push(events, "ST", i, l[i] if is_accum else h[i], vr, spread, 65.0, "Secondary Test", not is_accum)
                        flags["st"][i] = 1
                    else:
                        range_prog = ((c[i] - cur["rangeLow"]) / range_w * 100.0) if range_w > 0 else 0.0
                        range_prog_dn = ((cur["rangeHigh"] - c[i]) / range_w * 100.0) if range_w > 0 else 0.0
                        if (is_accum and range_frozen and range_prog >= sos_threshold
                                and (not real or vr >= 1.5) and c[i] > o[i]):
                            cur["ae"] = "D"; cur["schematic"] = "#2"; cur["sub"] = "SOS"
                            push(events, "SOS", i, c[i], vr, spread, 80.0, "SOS #2")
                            flags["sos"][i] = 1
                        elif (not is_accum) and range_frozen and range_prog_dn >= sos_threshold and (not real or vr >= 1.5) and c[i] < o[i]:
                            cur["ae"] = "D"; cur["schematic"] = "#2"; cur["sub"] = "SOW"
                            push(events, "SOW", i, c[i], vr, spread, 80.0, "SOW #2", True)
                            flags["sow"][i] = 1

            elif ae == "C":
                if cur["awaitingTest"] and cur["springUtadBar"] is not None:
                    test_band = max(atr_i * 0.5, (cur["rangeHigh"] - cur["rangeLow"]) * 0.05)
                    if not cur["springUtadLeft"]:
                        leave_band = max(atr_i * 0.5, (cur["rangeHigh"] - cur["rangeLow"]) * 0.05)
                        cur["springUtadLeft"] = (c[i] > cur["springUtadPrice"] + leave_band) if is_accum else (c[i] < cur["springUtadPrice"] - leave_band)
                    else:
                        vol_ok = (not real) or vr < cur["springUtadVolRatio"]
                        if is_accum:
                            hit = l[i] <= cur["springUtadPrice"] + test_band and l[i] >= cur["springUtadPrice"] - test_band and vol_ok and c[i] > cur["springUtadPrice"]
                        else:
                            hit = h[i] >= cur["springUtadPrice"] - test_band and h[i] <= cur["springUtadPrice"] + test_band and vol_ok and c[i] < cur["springUtadPrice"]
                        if hit:
                            cur["awaitingTest"] = False; cur["ae"] = "D"; cur["sub"] = "TEST"
                            if is_accum:
                                spring_conf = True
                            else:
                                utad_conf = True
                            push(events, "TEST", i, l[i] if is_accum else h[i], vr, spread, 80.0, "Test", not is_accum)
                            flags["test"][i] = 1
                range_w = cur["rangeHigh"] - cur["rangeLow"]
                range_prog = ((c[i] - cur["rangeLow"]) / range_w * 100.0) if range_w > 0 else 0.0
                range_prog_dn = ((cur["rangeHigh"] - c[i]) / range_w * 100.0) if range_w > 0 else 0.0
                if cur["ae"] == "C" and is_accum and range_prog >= sos_threshold and (not real or vr >= 1.5) and c[i] > o[i]:
                    cur["awaitingTest"] = False; cur["ae"] = "D"; cur["sub"] = "SOS"
                    push(events, "SOS", i, c[i], vr, spread, 80.0, "SOS")
                    flags["sos"][i] = 1
                elif cur["ae"] == "C" and (not is_accum) and range_prog_dn >= sos_threshold and (not real or vr >= 1.5) and c[i] < o[i]:
                    cur["awaitingTest"] = False; cur["ae"] = "D"; cur["sub"] = "SOW"
                    push(events, "SOW", i, h[i], vr, spread, 80.0, "SOW", True)
                    flags["sow"][i] = 1

            elif ae == "D":
                range_w = cur["rangeHigh"] - cur["rangeLow"]
                if is_accum:
                    pbar = i - pivot_len
                    piv_vol_ratio = vol[pbar] / volume_avg[pbar] if (pbar >= 0 and volume_avg[pbar] > 0) else 1.0
                    in_support = range_w > 0 and pbar >= 0 and cur["rangeLow"] <= l[pbar] <= cur["rangeLow"] + range_w * 0.35
                    lps = ((not real or piv_vol_ratio < 0.85) and pbar >= 0 and c[pbar] > o[pbar]
                           and in_support and not math.isnan(piv_low[pbar]))
                    if lps and not has_event(events, "LPS", i - event_cooldown_bars):
                        cur["sub"] = "LPS"
                        push(events, "LPS", i, l[pbar] if pbar >= 0 else l[i], piv_vol_ratio, spread, 75.0, "LPS")
                        flags["lps"][i] = 1
                    else:
                        range_prog = ((c[i] - cur["rangeLow"]) / range_w * 100.0) if range_w > 0 else 0.0
                        if range_prog >= sos_threshold and (not real or vr >= 1.5) and c[i] > o[i] and not has_event(events, "SOS", i - event_cooldown_bars):
                            cur["sub"] = "SOS"
                            push(events, "SOS", i, c[i], vr, spread, 85.0, "SOS")
                            flags["sos"][i] = 1
                else:
                    pbar = i - pivot_len
                    piv_vol_ratio = vol[pbar] / volume_avg[pbar] if (pbar >= 0 and volume_avg[pbar] > 0) else 1.0
                    in_supply = range_w > 0 and pbar >= 0 and cur["rangeHigh"] - range_w * 0.35 <= h[pbar] <= cur["rangeHigh"]
                    lpsy = ((not real or piv_vol_ratio < 0.85) and pbar >= 0 and c[pbar] < o[pbar]
                            and in_supply and not math.isnan(piv_high[pbar]))
                    if lpsy and not has_event(events, "LPSY", i - event_cooldown_bars):
                        cur["sub"] = "LPSY"
                        push(events, "LPSY", i, h[pbar] if pbar >= 0 else h[i], piv_vol_ratio, spread, 75.0, "LPSY", True)
                        flags["lpsy"][i] = 1
                    else:
                        range_prog_dn = ((cur["rangeHigh"] - c[i]) / range_w * 100.0) if range_w > 0 else 0.0
                        if range_prog_dn >= sos_threshold and (not real or vr >= 1.5) and c[i] < o[i] and not has_event(events, "SOW", i - event_cooldown_bars):
                            cur["sub"] = "SOW"
                            push(events, "SOW", i, h[i], vr, spread, 85.0, "SOW", True)
                            flags["sow"][i] = 1

        # --- breakout / invalidation ---
        range_w = cur["rangeHigh"] - cur["rangeLow"]
        invalid_buffer = range_w * invalidation_factor
        can_markup = cur["ae"] == "D" and setup_score >= markup_min_score and (spring_conf or has_event(events, "SOS", cur["startBar"]) or has_event(events, "LPS", cur["startBar"]))
        can_markdown = cur["ae"] == "D" and setup_score >= markup_min_score and (utad_conf or has_event(events, "SOW", cur["startBar"]) or has_event(events, "LPSY", cur["startBar"]))
        if cur["name"] == "ACCUMULATION" and range_frozen:
            if can_markup and c[i] > cur["rangeHigh"] and (not real or vr >= 1.2):
                push(events, "SOSB", i, c[i], vr, spread, 88.0, "SOS breakout")
                flags["sosb"][i] = 1
                prev_phase = cur
                cur = new_phase("MARKUP", "SOSB", "E", prior_trend, i)
                cur["schematic"] = prev_phase["schematic"]; cur["rangeLow"] = prev_phase["rangeHigh"]
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0
            elif c[i] < cur["rangeLow"] - invalid_buffer:
                prev_phase = cur
                cur = new_phase("RANGING", "FAIL", "", prior_trend, i)
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0
        elif cur["name"] == "DISTRIBUTION" and range_frozen:
            if can_markdown and c[i] < cur["rangeLow"] and (not real or vr >= 1.2):
                push(events, "SOWB", i, c[i], vr, spread, 88.0, "SOW breakdown", True)
                flags["sowb"][i] = 1
                prev_phase = cur
                cur = new_phase("MARKDOWN", "SOWB", "E", prior_trend, i)
                cur["schematic"] = prev_phase["schematic"]; cur["rangeHigh"] = prev_phase["rangeLow"]
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0
            elif c[i] > cur["rangeHigh"] + invalid_buffer:
                prev_phase = cur
                cur = new_phase("RANGING", "FAIL", "", prior_trend, i)
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0
        elif cur["name"] == "MARKUP":
            if h[i] > cur["rangeHigh"]:
                cur["rangeHigh"] = h[i]
            if c[i] < cur["rangeLow"]:
                prev_phase = cur
                cur = new_phase("RANGING", "", "", prior_trend, i)
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0
        elif cur["name"] == "MARKDOWN":
            if l[i] < cur["rangeLow"]:
                cur["rangeLow"] = l[i]
            if c[i] > cur["rangeHigh"]:
                prev_phase = cur
                cur = new_phase("RANGING", "", "", prior_trend, i)
                composite = "NEUTRAL"; spring_conf = utad_conf = False; avwap = float("nan"); vwap_num = vwap_den = 0.0

        # --- cause & effect ---
        if cur["name"] in ("ACCUMULATION", "DISTRIBUTION"):
            fw = cur["rangeHigh"] - cur["rangeLow"]
            dur = i - cur["startBar"]
            cause_built = fw * dur
            box_size = max(atr_i * 0.25, fw * 0.05)
            cols = max(1.0, dur / 3.0)
            dist = min(cols * box_size, fw * 12.0)
            pf_target = (cur["rangeLow"] + dist) if cur["name"] == "ACCUMULATION" else (cur["rangeHigh"] - dist)
            effect_expected = abs(pf_target - (cur["rangeLow"] if cur["name"] == "ACCUMULATION" else cur["rangeHigh"]))

        # --- scoring (after phase update) ---
        nm = cur["name"]
        range_dur = i - cur["startBar"]
        if nm in ("ACCUMULATION", "DISTRIBUTION"):
            range_w2 = cur["rangeHigh"] - cur["rangeLow"]
            rs = 0
            if 10 <= range_dur < 20:
                rs += 10
            elif 20 <= range_dur <= 80:
                rs += 20
            elif range_dur > 80:
                rs += 12
            rw_atr = range_w2 / atr_i if atr_i > 0 else 0.0
            if 1.0 <= rw_atr <= 5.0:
                rs += 25
            elif rw_atr > 0.5:
                rs += 12
            if real and recent_vol_decline:
                rs += 20
            p3 = i - 3
            if p3 >= 0:
                pl_in = (not math.isnan(piv_low3[i])) and cur["rangeLow"] * 0.98 < l[p3] < cur["rangeLow"] + range_w2 * 0.4
                ph_in = (not math.isnan(piv_high3[i])) and cur["rangeHigh"] - range_w2 * 0.4 < h[p3] < cur["rangeHigh"] * 1.02
                if pl_in or ph_in:
                    cur["rangePivotTests"] += 1
            if cur["rangePivotTests"] >= 2:
                rs += 15
            if real and cur["climaxVolRatio"] >= 2.5:
                rs += 20
            elif real and cur["climaxVolRatio"] >= 1.5:
                rs += 10
            raw_max = 100.0 if real else 60.0
            range_score = int(round(min(rs / raw_max * 100.0, 100.0)))
        if nm in ("ACCUMULATION", "DISTRIBUTION", "MARKUP", "MARKDOWN"):
            is_trend = nm in ("MARKUP", "MARKDOWN")
            is_acc = nm in ("ACCUMULATION", "MARKUP")
            sb = (prev_phase["startBar"] if (is_trend and prev_phase is not None) else cur["startBar"])
            arB = event_bar(events, "AR", sb)
            stB = event_bar(events, "ST", sb)
            springB = event_bar(events, "SPRING" if is_acc else "UTAD", sb)
            testB = event_bar(events, "TEST", sb)
            sosB = event_bar(events, "SOS" if is_acc else "SOW", sb)
            sosbB = event_bar(events, "SOSB" if is_acc else "SOWB", sb)
            lpsB = event_bar(events, "LPS" if is_acc else "LPSY", sb)
            sc = 0
            if arB >= 0:
                sc += 8
            if stB >= 0:
                sc += 8
            if cur["ae"] in ("C", "D"):
                sc += 5
            if springB >= 0:
                sc += 18
            if testB >= 0 and (springB < 0 or springB < testB):
                sc += 18
            if sosB >= 0 and (testB < 0 or testB < sosB):
                sc += 14
            if sosbB >= 0:
                sc += 12
            if lpsB >= 0:
                sc += 10
            if cur["schematic"] == "#2":
                seq = sosB >= 0
            else:
                seq = springB >= 0 and testB >= 0 and sosB >= 0 and springB < testB and testB < sosB
            if seq:
                sc += 8
            if is_acc and htf_bias == "BULLISH":
                sc += 8
            elif (not is_acc) and htf_bias == "BEARISH":
                sc += 8
            if range_score >= 60:
                sc += 5
            setup_score = min(sc, 100)

        # NSD / VDU
        no_supply = (c[i] < o[i] and (h[i] - l[i]) < spread_avg_i * 0.7 and vr < 0.7)
        no_demand = (c[i] > o[i] and (h[i] - l[i]) < spread_avg_i * 0.7 and vr < 0.7)
        vdu = i >= 3 and not math.isnan(vol[i - 3]) and vol[i] < vol[i - 1] < vol[i - 2] < vol[i - 3]

        # --- record ---
        out["phase"][i] = _PHASE_CODE.get(cur["name"], 0)
        out["subphase"][i] = 0  # numeric placeholder; textual subphase not exported
        out["phase_ae"][i] = _AE_CODE.get(cur["ae"], 0)
        out["schematic"][i] = 1 if cur["schematic"] == "#1" else 2 if cur["schematic"] == "#2" else 0
        out["setup_score"][i] = setup_score
        out["range_score"][i] = range_score
        out["avwap"][i] = avwap
        out["range_high"][i] = cur["rangeHigh"]
        out["range_low"][i] = cur["rangeLow"]
        out["confidence"][i] = cur["confidence"]
        out["effort"][i] = effort
        out["result"][i] = result
        out["effort_strength"][i] = strength if real else np.nan
        out["effort_rel"][i] = {"CONFIRMATION": 1, "CLIMAX": 2, "DIVERGENCE": 3}.get(rel, 0) if real else np.nan
        out["direction"][i] = 1 if cur["name"] in ("ACCUMULATION", "MARKUP") else -1 if cur["name"] in ("DISTRIBUTION", "MARKDOWN") else 0
        flags["spring_confirmed"][i] = 1 if spring_conf else 0
        flags["utad_confirmed"][i] = 1 if utad_conf else 0
        flags["no_supply"][i] = 1 if (real and no_supply) else 0
        flags["no_demand"][i] = 1 if (real and no_demand) else 0
        flags["vdu"][i] = 1 if (real and vdu) else 0

    df = pd.DataFrame(index=idx)
    df["wyk_phase"] = pd.Series(out["phase"], index=idx).astype("Int64")
    df["wyk_phase_ae"] = pd.Series(out["phase_ae"], index=idx).astype("Int64")
    df["wyk_schematic"] = pd.Series(out["schematic"], index=idx).astype("Int64")
    df["wyk_direction"] = pd.Series(out["direction"], index=idx).astype("Int64")
    df["wyk_setup_score"] = pd.Series(out["setup_score"], index=idx).astype("Int64")
    df["wyk_range_score"] = pd.Series(out["range_score"], index=idx).astype("Int64")
    df["wyk_avwap"] = out["avwap"]
    df["wyk_range_high"] = out["range_high"]
    df["wyk_range_low"] = out["range_low"]
    df["wyk_effort"] = out["effort"]
    df["wyk_result"] = out["result"]
    df["wyk_effort_strength"] = out["effort_strength"]
    df["wyk_effort_rel"] = pd.Series(out["effort_rel"], index=idx).astype("Int64")
    for k, v in flags.items():
        df[f"wyk_{k}"] = pd.Series(v, index=idx).astype("Int64")
    return df
