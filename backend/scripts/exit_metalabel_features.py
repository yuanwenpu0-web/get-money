"""Pure-causal features and conditional labels for the exit meta-label probe.

Two decision layers, both strictly causal (only data available AT the decision
moment is used):

  T0 = order-creation moment: which TP arm (early vs late) is better?
  T1 = first in-hold favorable-excursion moment: exit now or keep holding?

No DB access, no production imports. OHLCV-derived features are NEGATIVE
CONTROLS (H2 stop-loss line); they are never candidates for implementation.
"""
from __future__ import annotations

import math

import numpy as np

from order_exit_common import net_return_pct, run_rule

DECISION_MFE_R = 0.7          # pre-committed, matches the "too late" class
MIN_UNIQUE = 2                # binary dummies are legitimate (H7)
MIN_CLASS_SHARE = 0.05        # but a 1/105 split is not usable


def _fav(b, side):
    return float(b["high"]) if side == "LONG" else float(b["low"])


def _adv(b, side):
    return float(b["low"]) if side == "LONG" else float(b["high"])


def _dir(side):
    return 1.0 if side == "LONG" else -1.0


def _safe(v, default=float("nan")):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _std(a):
    return float(np.std(a)) if len(a) > 1 else 0.0


def t0_features(hist, order, sentiment=None, funding=None):
    """Features knowable strictly BEFORE the order is created."""
    side, entry, sl, tp = order["side"], order["entry_price"], order["sl"], order["tp"]
    R = abs(entry - sl)
    if R <= 0 or len(hist) < 25:
        return {}
    d = _dir(side)
    cl = np.array([_safe(b["close"]) for b in hist], dtype=float)
    hi = np.array([_safe(b["high"]) for b in hist], dtype=float)
    lo = np.array([_safe(b["low"]) for b in hist], dtype=float)
    volu = np.array([_safe(b.get("volume"), 0.0) for b in hist], dtype=float)
    quote = np.array([_safe(b.get("quote_volume"), 0.0) for b in hist], dtype=float)
    ret = np.diff(cl) / np.maximum(cl[:-1], 1e-12) * d
    r20, r60 = ret[-20:], ret[-60:-20]
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    rng = float(np.max(hi[-128:]) - np.min(lo[-128:]))
    f = {
        # ---- OHLCV-derived NEGATIVE CONTROLS (H2) ----
        "atr14_over_R": float(np.mean(tr[-14:])) / R,
        "vol20": _std(r20),
        "vol_ratio_20_40": _std(r20) / max(_std(r60), 1e-12),
        "mom5": float(np.sum(ret[-5:])),
        "mom20": float(np.sum(r20)),
        "range_pos_128": (cl[-1] - np.min(lo[-128:])) / max(rng, 1e-12),
        "volume_ratio_20_40": (float(np.sum(volu[-20:]))
                               / max(float(np.sum(volu[-60:-20])), 1e-12)),
        "quote_ratio_20_40": (float(np.sum(quote[-20:]))
                              / max(float(np.sum(quote[-60:-20])), 1e-12)),
        "trend_slope20": float(np.polyfit(np.arange(20), cl[-20:], 1)[0]
                               / max(cl[-1], 1e-12) * d),
        "last_body_ratio": ((cl[-1] - _safe(hist[-1]["open"]))
                            / max(hi[-1] - lo[-1], 1e-12) * d),
        # ---- order geometry / metadata (not a price series) ----
        "tp_dist_R": abs(tp - entry) / R,
        "hour_of_day": float(order.get("hour_of_day", 0)),
        "score": _safe(order.get("score")),
        "prob_up": _safe(order.get("prob_up")),
        "is_short": 0.0 if side == "LONG" else 1.0,
        "is_15m": 1.0 if order.get("interval") == "15m" else 0.0,
    }
    # ---- non-OHLCV derivatives (P9 direction); sparse coverage, H8 ----
    # Freshness metadata is diagnostic only and must NOT become a feature:
    # it is nearly collinear with calendar time and would leak the sample date.
    if sentiment:
        f.update({k: _safe(v) for k, v in sentiment.items()
                  if not k.endswith("_age_hours")})
    if funding:
        f.update({k: _safe(v) for k, v in funding.items()
                  if not k.endswith("_age_hours")})
    return f


def find_decision_bar(ab_bars, side, entry, sl, tp):
    """First bar with favorable excursion >= 0.7R BEFORE any barrier fires.

    Barriers are checked in project order (gap SL, gap TP, SL, TP) so the
    decision point is genuinely actionable. Returns None if never reachable.
    """
    R = abs(entry - sl)
    if R <= 0 or not ab_bars:
        return None
    is_long = side == "LONG"
    fav_peak = float(entry)
    for k, b in enumerate(ab_bars):
        o, h, l = (float(b[k2]) for k2 in ("open", "high", "low"))
        fav, adv = (h, l) if is_long else (l, h)
        if (o <= sl if is_long else o >= sl):
            return None
        if (o >= tp if is_long else o <= tp):
            return None
        if (adv <= sl if is_long else adv >= sl):
            return None
        if (fav >= tp if is_long else fav <= tp):
            return None
        fav_peak = max(fav_peak, fav) if is_long else min(fav_peak, fav)
        if (fav_peak - entry if is_long else entry - fav_peak) >= DECISION_MFE_R * R:
            return k
    return None


def t1_features(ab_bars, k, side, entry, sl, tp):
    """Features knowable at decision bar k, using bars[0..k] ONLY."""
    R = abs(entry - sl)
    if R <= 0 or k is None or k < 0 or k >= len(ab_bars):
        return {}
    d = _dir(side)
    upto = ab_bars[:k + 1]
    cl = np.array([_safe(b["close"]) for b in upto], dtype=float)
    hi = np.array([_safe(b["high"]) for b in upto], dtype=float)
    lo = np.array([_safe(b["low"]) for b in upto], dtype=float)
    fav = hi if side == "LONG" else lo
    adv = lo if side == "LONG" else hi
    mfe = (float(np.max(fav)) - entry) if side == "LONG" else (entry - float(np.min(fav)))
    mae = (entry - float(np.min(adv))) if side == "LONG" else (float(np.max(adv)) - entry)
    cur = (cl[-1] - entry) * d
    ret = np.diff(cl) / np.maximum(cl[:-1], 1e-12) * d if len(cl) > 1 else np.array([])
    b = ab_bars[k]
    return {
        "mfe_R_sofar": mfe / R,
        "mae_R_sofar": mae / R,
        "giveback_R_sofar": (mfe - cur) / R,
        "cur_pnl_R": cur / R,
        "bars_held": float(k + 1),
        "bar_range_R": (float(b["high"]) - float(b["low"])) / R,
        "mom3_R": float(np.sum(ret[-3:])) * (entry / R) if len(ret) else 0.0,
        "dist_to_tp_R": ((tp - cl[-1]) if side == "LONG" else (cl[-1] - tp)) / R,
        "vol_sofar": _std(ret),
        "bar_body_R": ((cl[-1] - float(b["open"])) * d) / R,
    }


def t1_label(ab_bars, k, side, entry, sl, tp):
    """y1 = net(keep holding with original TP/SL) - net(exit at bar k close).

    Both legs use the same notional-basis cost model, so costs cancel in the
    difference; positive means 'push the take-profit later'.
    """
    if k is None or k + 1 >= len(ab_bars):
        return None
    rule = {"name": "baseline", "kind": "barrier", "tp": tp, "sl": sl,
            "R": abs(entry - sl), "activate_R": 0, "trail_R": 0, "max_bars": 0}
    px, _, _ = run_rule(ab_bars[k + 1:], side, entry, sl, rule)
    now = net_return_pct(entry, float(ab_bars[k]["close"]), side)
    later = net_return_pct(entry, px, side)
    if not (math.isfinite(now) and math.isfinite(later)):
        return None
    return {"y1": later - now, "hold_better": int(later > now),
            "exit_now_net_pct": now, "hold_net_pct": later}


def t0_label(rule_net):
    """y0 = net(late arm) - net(early arm); positive means 'push TP later'."""
    need = ("tp_x0.5", "tp_x0.75", "tp_x1.5", "tp_x2.0", "tp_x3.0", "no_tp_hold")
    if any(k not in rule_net for k in need):
        return None
    early = np.mean([rule_net["tp_x0.5"], rule_net["tp_x0.75"]])
    late = np.mean([rule_net["tp_x1.5"], rule_net["tp_x2.0"],
                    rule_net["tp_x3.0"], rule_net["no_tp_hold"]])
    y = float(late - early)
    return {"y0": y, "late_better": int(y > 0),
            "early_net_pct": float(early), "late_net_pct": float(late)}


def drop_degenerate(columns, records):
    """H7: remove constant / near-constant / mostly-missing columns."""
    keep = {}
    n = len(records)
    for key in columns:
        v = [r.get("X", {}).get(key, float("nan")) for r in records]
        finite = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
        if len(finite) < max(30, int(n * 0.5)):
            continue
        uniq = set(round(x, 10) for x in finite)
        if len(uniq) < MIN_UNIQUE:
            continue
        if len(uniq) == 2 and min(sum(x == u for x in finite) for u in uniq) < n * MIN_CLASS_SHARE:
            continue
        keep[key] = len(finite)
    return keep
