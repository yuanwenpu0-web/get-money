"""Pre-registered conditional rules + purged A/B for the meta-label probe.

Exactly TWO rules, both negative controls. No parameter grid (H2), no champion
re-selection on the test segment (H5).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from order_exit_common import bootstrap_ci, verdict
from exit_metalabel_stats import MIN_PAIRS, spearman, paired_perm_p, segment_diffs

PURGE_HOURS = 24.0
MIN_POOLED_TRADES = 30


def _finite(v):
    return [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]


def purge_split(recs, train_ratio=0.6):
    """Time split with a purged band so a forward label cannot cross the cut."""
    order = sorted(recs, key=lambda r: r["created_at"])
    cut_i = int(len(order) * train_ratio)
    cut_t = datetime.fromisoformat(order[cut_i]["created_at"]).timestamp()
    band = PURGE_HOURS * 3600
    train = [r for r in order[:cut_i]
             if datetime.fromisoformat(r["created_at"]).timestamp() < cut_t - band]
    test = [r for r in order[cut_i:]
            if datetime.fromisoformat(r["created_at"]).timestamp() > cut_t + band]
    return train, test, {"train": train_ratio, "test": round(1 - train_ratio, 2),
                         "purge_hours": PURGE_HOURS,
                         "train_trades": len(train), "test_trades": len(test)}


def metalabel_t0(recs, train, rng):
    """ONE pre-registered conditional rule: train picks a feature + median cut.

    The SIGN of the train IC decides which side gets the later TP; the test
    segment is READ ONLY (H5).
    """
    cols = sorted(set().union(*[set(r["X_t0"]) for r in train]))
    scored = []
    for key in cols:
        x = [r["X_t0"].get(key, float("nan")) for r in train]
        y = [r["y0"] for r in train]
        if len(_finite(x)) < MIN_PAIRS:
            continue
        ic = spearman(x, y)
        if np.isfinite(ic):
            scored.append((abs(ic), key, ic))
    if not scored:
        return None
    _, key, ic_train = max(scored)
    vals = np.array(_finite([r["X_t0"].get(key, float("nan")) for r in train]))
    thr = float(np.median(vals))
    sgn = 1.0 if ic_train >= 0 else -1.0
    out = []
    for r in recs:
        x = r["X_t0"].get(key, float("nan"))
        late = bool(np.isfinite(x) and (x - thr) * sgn > 0)
        arm = "tp_x2.0" if late else "tp_x0.5"
        out.append({"id": r["id"], "feature_value": x, "arm": arm,
                    "net_pct": r["rule_net_pct"][arm],
                    "paired_diff_pct": r["rule_net_pct"][arm] - r["baseline_net_pct"]})
    mapping = ("high -> tp_x2.0, low -> tp_x0.5" if sgn > 0
               else "high -> tp_x0.5, low -> tp_x2.0 (negative train IC)")
    return {"name": "metalabel_T0", "feature": key, "train_ic": ic_train,
            "threshold": thr, "mapping": mapping, "per_order": out}


def metalabel_t1(recs, train, rng):
    """Physically pre-registered dynamic rule: giveback gates an early exit."""
    vals = np.array(_finite([r["X_t1"].get("giveback_R_sofar", float("nan"))
                             for r in train]))
    if len(vals) < 10:
        return None
    thr = float(np.median(vals))
    out = []
    for r in recs:
        g = r["X_t1"].get("giveback_R_sofar", float("nan"))
        now, hold = r["exit_now_net_pct"], r["hold_net_pct"]
        pick = now if (np.isfinite(g) and g > thr) else hold
        out.append({"id": r["id"], "giveback_R": g,
                    "action": "exit_now" if pick == now else "hold",
                    "net_pct": pick, "paired_diff_pct": pick - hold})
    return {"name": "metalabel_T1", "feature": "giveback_R_sofar",
            "threshold": thr, "mapping": "high giveback -> exit now, else hold",
            "per_order": out}


def rule_block(recs, per_order, rng, baseline_key, kind):
    """Paired CI + permutation p + segments + 20-seed random baseline."""
    by_id = {p["id"]: p for p in per_order}
    rs = [r for r in recs if r["id"] in by_id]
    diffs = [by_id[r["id"]]["paired_diff_pct"] for r in rs]
    ci = bootstrap_ci(diffs)
    p = paired_perm_p(diffs, rng)
    seg = segment_diffs(diffs, [r["created_at"] for r in rs])
    is_t0 = kind == "T0"
    n_late = sum(1 for r in rs if by_id[r["id"]].get("arm") == "tp_x2.0")
    rand = []
    for s in range(20):
        rr = np.random.default_rng(20261110 + s * 7919)
        if is_t0:
            ids = [r["id"] for r in rs]
            late = set(rr.choice(ids, size=min(n_late, len(ids)),
                                 replace=False).tolist())
            pick = [r["rule_net_pct"]["tp_x2.0"] if r["id"] in late
                    else r["rule_net_pct"]["tp_x0.5"] for r in rs]
        else:
            pick = [r["exit_now_net_pct"] if rr.random() > 0.5
                    else r["hold_net_pct"] for r in rs]
        rand.append(float(np.mean([v - r[baseline_key]
                                  for v, r in zip(pick, rs)])))
    rand = np.asarray(rand)
    obs = float(np.mean(diffs))
    return {
        "trades": len(rs), "paired_diff_pct": obs,
        "paired_ci_low_pct": ci[0], "paired_ci_high_pct": ci[1],
        "p_perm": p, "segments": seg,
        "random_baseline": {"mean": float(rand.mean()), "std": float(rand.std()),
                            "p05": float(np.quantile(rand, 0.05)),
                            "p95": float(np.quantile(rand, 0.95)),
                            "percentile_of_observed": float(
                                (np.count_nonzero(rand < obs) + 0.5) / (len(rand) + 1))},
        "ci_verdict": verdict(ci, len(rs), MIN_POOLED_TRADES),
    }
