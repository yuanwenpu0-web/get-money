"""Statistical helpers for the order-exit diagnosis.

Kept in a separate module so each delivered script stays under the 300-line
cap in the project playbook (H12). No production strategy imports here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.db import make_engine  # noqa: E402
from order_exit_common import bootstrap_ci, bh_fdr, verdict  # noqa: E402


def paired_perm_p(diffs, rng, n=4000):
    """Sign-flip permutation test for H0: mean paired difference == 0."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (n + 1))


def segment_signs(recs, rule, n_segments=4):
    order = sorted(recs, key=lambda r: r["created_at"])
    cuts = np.array_split(np.arange(len(order)), n_segments)
    signs, means = [], []
    for idx in cuts:
        d = [order[i]["rule_net_pct"][rule] - order[i]["baseline_net_pct"]
             for i in idx]
        means.append(float(np.mean(d)))
        signs.append(int(np.mean(d) > 0))
    return means, signs, sum(1 for m in means if m > 0), sum(1 for m in means if m < 0)


def rule_stats(recs, rules, rng):
    """Per-rule paired CI, permutation p, FDR q and time-segment signs (H4/H7)."""
    out, pvals = [], []
    for name in rules:
        vals = np.array([r["rule_net_pct"][name] for r in recs], dtype=float)
        diffs = vals - np.array([r["baseline_net_pct"] for r in recs], dtype=float)
        ci = bootstrap_ci(vals)
        dci = bootstrap_ci(diffs)
        p = paired_perm_p(diffs, rng)
        pvals.append(p)
        means, signs, npos, nneg = segment_signs(recs, name)
        out.append({
            "name": name, "trades": len(recs),
            "mean_return_pct": float(np.mean(vals)),
            "ci_low_pct": ci[0], "ci_high_pct": ci[1],
            "paired_diff_pct": float(np.mean(diffs)),
            "paired_ci_low_pct": dci[0], "paired_ci_high_pct": dci[1],
            "p_perm": p, "segment_means": means,
            "segments_positive": npos, "segments_negative": nneg,
            "ci_verdict": verdict(dci, len(recs)),
        })
    for row, q in zip(out, bh_fdr(pvals)):
        row["fdr_q"] = float(q)
        # H7: `pass` is reserved for ALL THREE gates together, not CI alone.
        row["three_gates_pass"] = bool(
            row["paired_ci_low_pct"] > 0 and q < 0.10
            and max(row["segments_positive"], row["segments_negative"]) >= 3)
    return out


def btc_change(window_start, window_end):
    """Sample-period benchmark, read live from local PG 1m klines."""
    try:
        e = make_engine(read_only=True)
        with e.connect() as c:
            rows = c.execute(text("""
                SELECT open_time, close FROM klines
                WHERE symbol='BTCUSDT' AND interval='1m'
                  AND open_time >= :a AND open_time <= :b
                ORDER BY open_time LIMIT 1
            """), {"a": window_start, "b": window_end}).first()
            last = c.execute(text("""
                SELECT open_time, close FROM klines
                WHERE symbol='BTCUSDT' AND interval='1m'
                  AND open_time >= :a AND open_time <= :b
                ORDER BY open_time DESC LIMIT 1
            """), {"a": window_start, "b": window_end}).first()
        if rows and last and rows[1]:
            return {"start": float(rows[1]), "end": float(last[1]),
                    "change_pct": float((last[1] - rows[1]) / rows[1] * 100)}
    except Exception as exc:  # benchmark is attribution, not a gate
        return {"error": repr(exc)[:200]}
    return {"error": "no BTCUSDT 1m bars in window"}
