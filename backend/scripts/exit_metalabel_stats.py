"""Statistics for the Gate-1 conditional-exit meta-label probe.

All judgement vocabulary comes from strategy-backtest-playbook; this module
only supplies numbers. Reuses order_exit_common for CI / FDR / verdict.
"""
from __future__ import annotations

import math

import numpy as np

from order_exit_common import bootstrap_ci, bh_fdr, verdict

MIN_PAIRS = 30            # judgement.md §2
IC_GATE = 0.02            # H1
SIGN_SHARE_GATE = 0.60    # H1 cross-symbol consistency
Q_GATE = 0.10             # H7


def _pairs(x, y):
    """Pairwise-delete non-finite values, return aligned float arrays."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def _rank(v):
    """Average ranks for ties (scipy-free Spearman)."""
    order = np.argsort(v, kind="mergesort")
    out = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(x, y, min_pairs=MIN_PAIRS):
    a, b = _pairs(x, y)
    if len(a) < min_pairs:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def auc(x, y_binary):
    """Mann-Whitney AUC; NaN when a class is empty or pairs are too few."""
    a, b = _pairs(x, y_binary)
    pos, neg = a[b > 0.5], a[b <= 0.5]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    # _rank returns 0-based average ranks, so the minimum possible rank sum
    # for n_pos items is n_pos*(n_pos-1)/2 (NOT n_pos*(n_pos+1)/2).
    r = _rank(a)
    return float((r[b > 0.5].sum() - len(pos) * (len(pos) - 1) / 2)
                / (len(pos) * len(neg)))


def perm_p(x, y, rng, n=2000):
    """Two-sided permutation p for H0: IC == 0."""
    a, b = _pairs(x, y)
    if len(a) < MIN_PAIRS:
        return 1.0
    obs = abs(spearman(a, b))
    if not math.isfinite(obs):
        return 1.0
    null = np.empty(n)
    for i in range(n):
        null[i] = abs(spearman(a, rng.permutation(b)))
    return float((np.count_nonzero(null >= obs) + 1) / (n + 1))


def block_sign_share(x, y, symbols, obs_ic, rng, n=1000):
    """H1 'same-sign symbol share' as a symbol-level block bootstrap.

    Resamples WHOLE symbols (all orders of a symbol travel together), which
    preserves intra-symbol correlation and answers: how often would the IC
    keep its sign if the cohort of coins were drawn again?
    """
    ax, ay = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(ax) & np.isfinite(ay)
    a, b = ax[ok], ay[ok]
    if len(a) < MIN_PAIRS or not math.isfinite(obs_ic):
        return float("nan")
    syms = np.asarray(symbols)[ok]      # align with pairwise-deleted arrays
    idx_by_sym = {}
    for i, s in enumerate(syms):
        idx_by_sym.setdefault(s, []).append(i)
    keys = list(idx_by_sym)
    sign = 1.0 if obs_ic > 0 else -1.0
    same, valid = 0, 0
    for _ in range(n):
        picked = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([np.asarray(idx_by_sym[s], dtype=int) for s in picked])
        ic = spearman(a[idx], b[idx])
        if math.isfinite(ic):
            valid += 1
            same += int((ic > 0) == (sign > 0))
    return float(same / valid) if valid else float("nan")


def segment_ic(x, y, created_at, n_segments=4):
    """Per-window IC and sign stability (judgement.md §2)."""
    a, b = _pairs(x, y)
    t = np.asarray(created_at)[np.isfinite(np.asarray(x, dtype=float))
                               & np.isfinite(np.asarray(y, dtype=float))]
    order = np.argsort(t, kind="mergesort")
    # A 4-way split of a 105-order cohort has ~26 orders per segment, well
    # below the 30-pair IC gate. H7 asks for SIGN stability across windows,
    # not a standalone significant IC in each window, so use 10 pairs here.
    out = []
    for idx in np.array_split(order, n_segments):
        v = spearman(a[idx], b[idx], min_pairs=10)
        out.append(round(v, 6) if math.isfinite(v) else float("nan"))
    finite = [v for v in out if math.isfinite(v)]
    return {"ics": out,
            "n_positive": sum(1 for v in finite if v > 0),
            "n_negative": sum(1 for v in finite if v < 0)}


def paired_perm_p(diffs, rng, n=3000):
    """Sign-flip permutation p for a paired mean difference."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 1.0
    obs = abs(float(d.mean()))
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (n + 1))


def segment_diffs(diffs, created_at, n_segments=4):
    t = np.asarray(created_at)
    order = np.argsort(t, kind="mergesort")
    d = np.asarray(diffs, dtype=float)
    means = [float(np.mean(d[idx])) for idx in np.array_split(order, n_segments)]
    return {"means": means,
            "n_positive": sum(1 for v in means if v > 0),
            "n_negative": sum(1 for v in means if v < 0)}


def probe_columns(records, layer_key, target_key, binary_key, rng,
                  columns=None, label=""):
    """Full Gate-1 probe over one feature layer."""
    y = [r.get(target_key) for r in records]
    yb = [r.get(binary_key) for r in records]
    t = [r.get("created_at", "") for r in records]
    syms = [r.get("symbol", "") for r in records]
    if columns is None:
        columns = sorted(set().union(*[set(r.get(layer_key, {})) for r in records]))
    rows, pvals = [], []
    for key in columns:
        x = [r.get(layer_key, {}).get(key, float("nan")) for r in records]
        finite = sum(1 for v in x if isinstance(v, (int, float)) and math.isfinite(v))
        if finite < max(MIN_PAIRS, int(len(records) * 0.5)):
            continue
        uniq = len(set(round(v, 10) for v in x
                       if isinstance(v, (int, float)) and math.isfinite(v)))
        if uniq < 2:
            continue
        ic = spearman(x, y)
        a = auc(x, yb)
        p = perm_p(x, y, rng)
        pvals.append(p)
        rows.append({
            "column": key, "layer": label, "n_pairs": finite,
            "ic": None if not math.isfinite(ic) else round(ic, 6),
            "auc": None if not math.isfinite(a) else round(a, 6),
            "p_perm": None if not math.isfinite(p) else round(p, 6),
            "ic_ci95": [round(v, 6) for v in ic_bootstrap_ci(x, y, rng)],
            "sign_share": (None if not math.isfinite(ic)
                           else round(block_sign_share(x, y, syms, ic, rng), 4)),
            "segments": segment_ic(x, y, t),
        })
    for row, q in zip(rows, bh_fdr(pvals)):
        row["fdr_q"] = round(float(q), 6)
        ic = row["ic"]
        seg = max(row["segments"]["n_positive"], row["segments"]["n_negative"])
        gates = {
            "abs_ic_ge_0.02": bool(ic is not None and abs(ic) >= IC_GATE),
            "sign_share_ge_0.60": bool(row["sign_share"] is not None
                                       and row["sign_share"] >= SIGN_SHARE_GATE),
            "fdr_q_below_0.10": bool(q < Q_GATE),
            "segments_same_sign_ge_3of4": bool(seg >= 3),
        }
        row["gates"] = gates
        row["probe_pass"] = all(gates.values())
    return rows


def ic_bootstrap_ci(x, y, rng, n_boot=1000):
    """Bootstrap CI of the IC by resampling ORDERS (not bars)."""
    a, b = _pairs(x, y)
    if len(a) < MIN_PAIRS:
        return [float("nan"), float("nan")]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        ic = spearman(a[idx], b[idx])
        if math.isfinite(ic):
            vals.append(ic)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def self_calibration(records, positive_key, target_key, rng, n_seeds=20):
    """judgement.md §3 step 5: positive / negative / half-decay controls."""
    pos = np.array([r.get(positive_key, float("nan")) for r in records], dtype=float)
    y = np.array([r.get(target_key, float("nan")) for r in records], dtype=float)
    ok = np.isfinite(pos) & np.isfinite(y)
    pos, y = pos[ok], y[ok]
    if len(pos) < MIN_PAIRS:
        return {"error": "insufficient pairs for calibration"}
    pos_ic = spearman(pos, y)
    neg_ics, half_ics = [], []
    for s in range(n_seeds):
        rr = np.random.default_rng(20261110 + s * 7919)
        noise = rr.normal(size=len(pos))
        neg_ics.append(spearman(noise, y))
        half = 0.5 * (pos - pos.mean()) / max(pos.std(), 1e-12) + 0.5 * noise
        half_ics.append(spearman(half, y))
    neg = np.array([v for v in neg_ics if math.isfinite(v)])
    half = np.array([v for v in half_ics if math.isfinite(v)])
    return {
        "positive_control_ic_LOOKAHEAD": round(float(pos_ic), 6),
        "negative_control_mean_ic": round(float(neg.mean()), 6),
        "negative_control_p95_abs_ic": round(float(np.quantile(np.abs(neg), 0.95)), 6),
        "half_decay_control_mean_ic": round(float(half.mean()), 6),
        "half_decay_ratio_vs_positive": (round(float(abs(half.mean()) / abs(pos_ic)), 4)
                                         if abs(pos_ic) > 1e-9 else None),
        "n_seeds": n_seeds, "n_pairs": len(pos),
    }


def ci_verdict(values, n=None):
    ci = bootstrap_ci(values)
    return verdict(ci, n if n is not None else len(values)), ci
