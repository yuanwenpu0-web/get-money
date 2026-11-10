"""Statistical evaluation of kronos_order_probe.py output (Gate 1/4/5).

No trading-system writes. Produces docs/kronos-order-exit.json; the paired .md
report is written alongside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from app.db import make_engine  # noqa: E402
from kronos_order_common import (  # noqa: E402
    rw_features, auc_score, bootstrap_ci, bh_fdr,
)

MIN_POOLED_TRADES = 30
N_RANDOM_SEEDS = 20


def auc_bootstrap_ci(y, s, n_boot=2000, seed=20261110):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc_score(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def permutation_p(y, s, n_perm=2000, seed=20261110):
    rng = np.random.default_rng(seed)
    obs = auc_score(y, s)
    if np.isnan(obs):
        return float("nan"), float("nan")
    null = np.array([auc_score(y, rng.permutation(s)) for _ in range(n_perm)])
    return obs, float((np.count_nonzero(null >= obs) + 1) / (n_perm + 1))


def quick_summary(payload, rw_paths=120):
    """Compact evaluation of a secondary cohort (e.g. short-window live orders)."""
    recs = payload["records"]
    if not recs:
        return {"orders": 0}
    y = np.array([1 if r["label_path"] == 1 else 0 for r in recs], dtype=int)
    s = np.array([r["kronos_mfe"] for r in recs], dtype=float)
    obs, p = permutation_p(y, s)
    lo, hi = auc_bootstrap_ci(y, s)
    ret = np.array([r["net_return_pct"] for r in recs], dtype=float)
    v, rlo, rhi = ci_verdict(ret)
    return {
        "orders": len(recs), "symbols": len({r["symbol"] for r in recs}),
        "horizon_bars": payload["config"]["horizon_bars"],
        "positive_path_labels": int(y.sum()),
        "positive_reason_labels": int(sum(r["label_reason"] for r in recs)),
        "auc": None if np.isnan(obs) else round(obs, 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
        "p_perm": None if np.isnan(p) else round(p, 4),
        "trigger_rate_at_0_5": round(float(np.mean(s > 0.5)), 4),
        "mean_return_pct": round(float(ret.mean()), 4),
        "return_ci95": [round(rlo, 4), round(rhi, 4)],
        "verdict": v,
    }


def ci_verdict(returns, min_trades=MIN_POOLED_TRADES):
    lo, hi = bootstrap_ci(returns)
    if len(returns) < min_trades:
        return "insufficient", lo, hi
    if lo > 0:
        return "pass", lo, hi
    if hi < 0:
        return "negative", lo, hi
    return "inconclusive", lo, hi


def beta_pct(start_iso, end_iso):
    """Sample-period BTC move for beta attribution (read-only)."""
    try:
        engine = make_engine(read_only=True)
        a = int(datetime.fromisoformat(start_iso).timestamp() * 1000)
        b = int(datetime.fromisoformat(end_iso).timestamp() * 1000)
        with engine.connect() as c:
            first = c.execute(text(
                "SELECT close FROM klines WHERE symbol='BTCUSDT' AND interval='1m' "
                "AND open_time >= :a ORDER BY open_time LIMIT 1"), {"a": a}).scalar()
            last = c.execute(text(
                "SELECT close FROM klines WHERE symbol='BTCUSDT' AND interval='1m' "
                "AND open_time <= :b ORDER BY open_time DESC LIMIT 1"), {"b": b}).scalar()
        return round((last - first) / first * 100, 3) if first and last else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/kronos_order_probe.json")
    ap.add_argument("--out", default="docs/kronos-order-exit.json")
    ap.add_argument("--secondary-input", default="",
                     help="extra probe JSON (e.g. live short-window cohort) to embed")
    ap.add_argument("--rw-paths", type=int, default=120)
    args = ap.parse_args()

    payload = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    recs = payload["records"]
    if not recs:
        raise SystemExit("no usable records")

    y = np.array([1 if r["label_path"] == 1 else 0 for r in recs], dtype=int)
    y_reason = np.array([r["label_reason"] for r in recs], dtype=int)
    unresolved = int((np.array([r["label_path"] for r in recs]) == -1).sum())

    rng = np.random.default_rng(20261110)
    rw = [rw_features(r["hist_ratios"], r["side"], r["entry_price"],
                       r["tp"], r["sl"], r["horizon"], rng, n_paths=args.rw_paths)
          for r in recs]

    scores = {
        "kronos_mfe": np.array([r["kronos_mfe"] for r in recs], dtype=float),
        "kronos_hit": np.array([r["kronos_hit"] for r in recs], dtype=float),
        "kronos_end": np.array([r["kronos_end"] for r in recs], dtype=float),
        "rw_mfe": np.array([x["mfe_ratio"] for x in rw], dtype=float),
        "rw_p_tp": np.array([x["p_tp"] for x in rw], dtype=float),
        "positive_control_label_mfe": np.array([r["label_mfe"] for r in recs], dtype=float),
        "random_control": rng.random(len(recs)),
    }
    scores["half_decay_control"] = 0.5 * scores["positive_control_label_mfe"] + 0.5 * rng.random(len(recs))
    scores["weak_control_20pct"] = 0.2 * scores["positive_control_label_mfe"] + 0.8 * rng.random(len(recs))

    metrics, pvals, names = {}, [], []
    for name, s in scores.items():
        obs, p = permutation_p(y, s)
        lo, hi = auc_bootstrap_ci(y, s)
        trigger = float(np.mean(s > 0.5))
        metrics[name] = {
            "auc_vs_path_label": None if np.isnan(obs) else round(obs, 4),
            "auc_ci95": [round(lo, 4), round(hi, 4)],
            "auc_vs_close_reason": round(auc_score(y_reason, s), 4),
            "brier": round(float(np.mean((np.clip(s, 0, 1) - y) ** 2)), 4),
            "trigger_rate_at_0_5": round(trigger, 4),
            "degenerate": bool(trigger < 0.002 or trigger > 0.60),
            "p_perm": None if np.isnan(p) else round(p, 4),
        }
        names.append(name)
        pvals.append(1.0 if np.isnan(p) else p)
    qvals = bh_fdr(pvals)
    for name, q in zip(names, qvals):
        metrics[name]["fdr_q"] = round(q, 4)

    # Random-score distribution (H6): 20 seeds, then the observed score's percentile.
    rand_aucs = []
    for seed in range(N_RANDOM_SEEDS):
        r2 = np.random.default_rng(20261025 + seed)
        rand_aucs.append(auc_score(y, r2.random(len(recs))))
    rand_aucs = np.array(rand_aucs)
    obs_main = metrics["kronos_mfe"]["auc_vs_path_label"] or 0.5
    rand_pct = float((np.count_nonzero(rand_aucs < obs_main) + 1) / (N_RANDOM_SEEDS + 1))

    # Subgroup diagnostics: source / side / interval / strategy.
    def subgroup(key):
        out = {}
        for val in sorted({r[key] for r in recs}):
            idx = np.array([i for i, r in enumerate(recs) if r[key] == val])
            a = auc_score(y[idx], scores["kronos_mfe"][idx])
            out[val] = {"n": int(len(idx)), "positive": int(y[idx].sum()),
                        "auc": None if np.isnan(a) else round(a, 4)}
        return out

    subgroups = {k: subgroup(k) for k in ("source", "side", "interval", "strategy")}
    reasons: dict[str, int] = {}
    for r in recs:
        reasons[r["close_reason"]] = reasons.get(r["close_reason"], 0) + 1

    # Time segments: sign consistency must hold in >= 3/4 (H7).
    order = np.argsort([r["created_at"] for r in recs])
    segments = []
    for part in np.array_split(order, 4):
        a = auc_score(y[part], scores["kronos_mfe"][part])
        segments.append({"n": int(len(part)), "auc": None if np.isnan(a) else round(a, 4),
                         "sign_above_0_5": bool(a > 0.5)})

    # Trade-value groups (H4): filter orders by the pre-committed threshold 0.5.
    all_ret = np.array([r["net_return_pct"] for r in recs], dtype=float)
    sel_mask = scores["kronos_mfe"] > 0.5
    kronos_ret = all_ret[sel_mask]
    rw_ret = all_ret[np.array([x["mfe_ratio"] for x in rw]) > 0.5]
    groups = {}
    for gname, arr in (("all_orders", all_ret), ("kronos_filter", kronos_ret),
                       ("rw_filter", rw_ret)):
        v, lo, hi = ci_verdict(arr)
        groups[gname] = {"trades": int(len(arr)), "mean_return_pct": round(float(arr.mean()), 4) if len(arr) else None,
                         "ci_low_pct": round(lo, 4), "ci_high_pct": round(hi, 4), "verdict": v}

    # Random filter with the same number of orders, 20 seeds.
    k = int(sel_mask.sum())
    rand_filter_means = np.array([
        all_ret[np.random.default_rng(20261025 + s).choice(len(recs), size=k, replace=False)].mean()
        for s in range(N_RANDOM_SEEDS)
    ]) if k > 0 else np.array([])
    kronos_mean = float(kronos_ret.mean()) if k else float("nan")
    rf_pct = (float((np.count_nonzero(rand_filter_means < kronos_mean) + 1) / (N_RANDOM_SEEDS + 1))
              if k else None)

    main_v = groups["kronos_filter"]["verdict"]
    if main_v == "pass" and (rand_pct is None or rand_pct <= 0.95 or metrics["kronos_mfe"]["fdr_q"] >= 0.10):
        final_verdict = "likely_noise"
    else:
        final_verdict = main_v

    window = {"start": min(r["created_at"] for r in recs)[:10],
              "end": max(r["created_at"] for r in recs)[:10]}
    span_days = (datetime.fromisoformat(max(r["created_at"] for r in recs)).date()
                 - datetime.fromisoformat(min(r["created_at"] for r in recs)).date()).days + 1
    result = {
        "topic": "kronos-order-exit",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "strategy": "kronos_pre_entry_exit_probe",
        "interval": "+".join(sorted({r["interval"] for r in recs})),
        "days": span_days, "window": window,
        "universe": {"symbols": len({r["symbol"] for r in recs}),
                     "orders": len(recs), "skipped": payload.get("skipped", []),
                     "unresolved_labels": unresolved,
                     "positive_path_labels": int(y.sum()),
                     "positive_reason_labels": int(y_reason.sum()),
                     "close_reason_counts": reasons},
        "subgroup_auc": subgroups,
        "parameters": {"source": "zero-shot pretrained Kronos-small, no fitting",
                       "grid": {}, "threshold": 0.5,
                       "lookback": payload["config"]["lookback"],
                       "samples": payload["config"]["samples"],
                       "horizon_bars": payload["config"]["horizon_bars"],
                       "kronos_commit": payload["config"]["kronos_commit"]},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
                  "basis": "notional"},
        "split": {"train": 0.0, "valid": 0.0, "test": 1.0,
                  "note": "zero-shot; whole window is out-of-sample, threshold fixed a priori"},
        "beta_btc_pct": beta_pct(window["start"] + "T00:00:00+08:00", window["end"] + "T23:59:00+08:00"),
        "min_pooled_trades": MIN_POOLED_TRADES,
        "random_baseline_seeds": N_RANDOM_SEEDS,
        "scores": metrics,
        "self_calibration": {
            "positive_control_auc": metrics["positive_control_label_mfe"]["auc_vs_path_label"],
            "negative_control_auc": metrics["random_control"]["auc_vs_path_label"],
            "half_decay_control_auc": metrics["half_decay_control"]["auc_vs_path_label"],
            "weak_control_20pct_auc": metrics["weak_control_20pct"]["auc_vs_path_label"],
        },
        "random_baseline": {"auc_mean": round(float(rand_aucs.mean()), 4),
                            "auc_p95": round(float(np.quantile(rand_aucs, 0.95)), 4),
                            "kronos_percentile": round(rand_pct, 4)},
        "time_segments": segments,
        "segments_same_sign": sum(s["sign_above_0_5"] for s in segments),
        "groups": groups,
        "random_filter_baseline": {"trades_per_seed": k,
                                   "mean_p05": round(float(np.quantile(rand_filter_means, 0.05)), 4) if k else None,
                                   "mean_p95": round(float(np.quantile(rand_filter_means, 0.95)), 4) if k else None,
                                   "kronos_percentile": None if rf_pct is None else round(rf_pct, 4)},
        "baseline": {"random_percentile": round(rand_pct, 4)},
        "verdict": final_verdict,
        "verdict_by_ci": main_v,
        "default_enabled": False,
        "limitations": [
            "16-day window, 148 symbols, 36 take-profit outcomes -> low statistical power",
            "Generated paths are zero-shot samples; no fine-tuning on this order population",
            "Future window starts at the next complete bucket, dropping the remainder bar",
            "No funding, order-book depth, liquidation or partial-fill simulation",
            "Kronos pretraining cutoff cannot be independently verified",
        ],
        "source_runtime_seconds": payload.get("runtime_seconds"),
    }
    if args.secondary_input:
        extra = json.loads((ROOT / args.secondary_input).read_text(encoding="utf-8"))
        result["live_subset_short_window"] = quick_summary(extra, args.rw_paths)
        result["limitations"].append(
            "live cohort uses a shorter 24-bar (6h) horizon because its 24h future window is incomplete"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("universe", "scores", "self_calibration", "random_baseline",
                       "time_segments", "groups", "random_filter_baseline", "verdict")},
                     ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
