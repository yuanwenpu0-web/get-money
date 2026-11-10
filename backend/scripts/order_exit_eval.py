"""Statistical judgement for the trend-reversal order-exit diagnosis.

Paired per-order comparisons only: every rule is replayed on the SAME entry,
SAME bars, SAME cost. No champion is selected for production - the rule family
is a NEGATIVE CONTROL used to test whether moving the exit can rescue the
already-falsified OHLCV price signal (H2 / Gate 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from order_exit_common import bootstrap_ci, verdict  # noqa: E402
from order_exit_stats import rule_stats, btc_change  # noqa: E402

TWO_SIDED_RULES = ("tp_x0.5", "tp_x2.0", "no_tp_hold", "trail_0.5R")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/order_exit_paper.json")
    ap.add_argument("--secondary-input", default="output/order_exit_live.json")
    ap.add_argument("--out", default="docs/trend-reversal-order-exit.json")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    args = ap.parse_args()

    payload = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    live = json.loads((ROOT / args.secondary_input).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    # Main cohort: market-rule exits only. Administrative / archived closes
    # are not produced by an exit rule and would contaminate the A/B.
    cohort = [r for r in payload["records"] if r["market_exit"]]
    cohort.sort(key=lambda r: r["created_at"])
    core = [r for r in cohort if r["strategy"] == "supertrend_current"]
    rules = list(cohort[0]["rule_net_pct"].keys())

    real = np.array([r["real_net_pct"] for r in cohort], dtype=float)
    base = np.array([r["baseline_net_pct"] for r in cohort], dtype=float)
    calib = np.array([r["calib_net_pct"] for r in cohort], dtype=float)
    real_ci, base_ci = bootstrap_ci(real), bootstrap_ci(base)
    fidelity = sum(r["real_reason_class"] == r["replay_reason_class"]
                   for r in cohort) / len(cohort)

    stats = rule_stats(cohort, rules, rng)
    # Random-exit baseline: one mean per seed over the SAME orders.
    rand = np.array([[r["random_exit_net_pct"][s] for r in cohort]
                     for s in range(len(cohort[0]["random_exit_net_pct"]))])
    rand_means = rand.mean(axis=1)
    pct = lambda v: float(np.clip(np.count_nonzero(rand_means < v) / len(rand_means), 0, 1))

    # H5: train selects, test only reads. No tuning on the test segment.
    k = int(len(cohort) * args.train_ratio)
    train, test = cohort[:k], cohort[k:]
    tr_stats = rule_stats(train, [r for r in rules if r != "baseline"], rng)
    champion = max(tr_stats, key=lambda x: x["paired_diff_pct"])["name"]
    te_stats = rule_stats(test, [r for r in rules if r != "baseline"], rng)
    te_champ = next(x for x in te_stats if x["name"] == champion)
    full_champ = next(x for x in stats if x["name"] == champion)
    # H7 three-gate check on the READ-ONLY test segment, family-wide.
    tr_map = {x["name"]: x for x in tr_stats}
    full_map = {x["name"]: x for x in stats}
    oos_family = []
    for row in te_stats:
        gates = {
            "ci_low_positive": row["paired_ci_low_pct"] > 0,
            "fdr_q_below_0.10": row["fdr_q"] < 0.10,
            "segments_same_sign_ge_3of4": max(row["segments_positive"],
                                              row["segments_negative"]) >= 3,
        }
        oos_family.append({"name": row["name"],
                           "paired_diff_pct": row["paired_diff_pct"],
                           "ci_low_pct": row["paired_ci_low_pct"],
                           "ci_high_pct": row["paired_ci_high_pct"],
                           "fdr_q": row["fdr_q"], "gates": gates,
                           "train_paired_diff_pct": tr_map[row["name"]]["paired_diff_pct"],
                           "full_sample_fdr_q": full_map[row["name"]]["fdr_q"],
                           "three_gates_pass": row["three_gates_pass"]})
    n_oos_pass = sum(1 for x in oos_family if x["three_gates_pass"])

    # The central question: do "too early" and "too late" demand opposite exits?
    early = [r for r in cohort if r["class"] == "early"]
    late = [r for r in cohort if r["class"] == "late"]
    def paired(recs, rule):
        d = [r["rule_net_pct"][rule] - r["baseline_net_pct"] for r in recs]
        return round(float(np.mean(d)), 4) if d else None
    two_sided = {
        "early_n": len(early), "late_n": len(late),
        "neither_n": sum(1 for r in cohort if r["class"] == "neither"),
        "early_real_mean_pct": round(float(np.mean([r["real_net_pct"] for r in early])), 4) if early else None,
        "late_real_mean_pct": round(float(np.mean([r["real_net_pct"] for r in late])), 4) if late else None,
        "early_baseline_mean_pct": round(float(np.mean([r["baseline_net_pct"] for r in early])), 4) if early else None,
        "late_baseline_mean_pct": round(float(np.mean([r["baseline_net_pct"] for r in late])), 4) if late else None,
        "early_mean_post_run_R": float(np.mean([r["diag"]["post_exit_run_R"] for r in early])) if early else None,
        "late_mean_giveback_R": float(np.mean([r["diag"]["giveback_R"] for r in late])) if late else None,
        "by_class": {
            cls: {rule: paired(rs, rule) for rule in TWO_SIDED_RULES}
            for cls, rs in (("early", early), ("late", late))
        },
    }

    # Descriptive diagnostic quantities (oracle is LOOKAHEAD, never tradable).
    def dmean(key):
        v = [r["diag"].get(key) for r in cohort]
        v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
        return float(np.mean(v)) if v else None
    cr = [r["diag"]["capture_ratio"] for r in cohort
          if r["diag"].get("hold_mfe_R", 0) > 0.5
          and isinstance(r["diag"].get("capture_ratio"), float)
          and np.isfinite(r["diag"]["capture_ratio"])]
    diagnosis = {
        "mfe_R_full_window": dmean("mfe_R"),
        "hold_mfe_R": dmean("hold_mfe_R"),
        "post_exit_run_R_bounded_24h": dmean("post_exit_run_R"),
        "giveback_R": dmean("giveback_R"),
        "tp_dist_R": dmean("tp_dist_R"),
        "capture_ratio_median_favorable_only": float(np.median(cr)) if cr else None,
        "capture_ratio_favorable_n": len(cr),
        "oracle_net_pct_LOOKAHEAD": dmean("oracle_net_pct"),
        "actual_real_net_pct": float(np.mean(real)),
        "note": "oracle uses the best price inside (hold + 24h); look-ahead upper bound only",
    }

    # Live (DEMO) short-window cohort is reported separately, never merged.
    lrecs = [r for r in live["records"] if r["market_exit"]]
    lstats = rule_stats(lrecs, rules, rng)
    lreal = np.array([r["real_net_pct"] for r in lrecs], dtype=float)
    live_out = {
        "trades": len(lrecs), "ab_window_hours": live["config"]["ab_hours"],
        "real_mean_pct": float(np.mean(lreal)),
        "real_ci": bootstrap_ci(lreal),
        "verdict": verdict(bootstrap_ci(lreal), len(lrecs)),
        "champion_from_paper": champion,
        "champion_paired_diff_pct": next(x for x in lstats if x["name"] == champion)["paired_diff_pct"],
        "top3_by_paired_diff": sorted(
            [{"name": x["name"], "paired_diff_pct": x["paired_diff_pct"]}
             for x in lstats if x["name"] != "baseline"],
            key=lambda x: -x["paired_diff_pct"])[:3],
    }

    window = {"start": cohort[0]["created_at"][:10],
              "end": cohort[-1]["created_at"][:10],
              "days": (datetime.fromisoformat(cohort[-1]["created_at"]).date()
                       - datetime.fromisoformat(cohort[0]["created_at"]).date()).days + 1}
    start_ms = int(datetime.fromisoformat(cohort[0]["created_at"]).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(cohort[-1]["created_at"]).timestamp() * 1000)
    beta = btc_change(start_ms, end_ms)

    three_gates = {
        "test_ci_low_positive": te_champ["paired_ci_low_pct"] > 0,
        "fdr_q_below_0.10": te_champ["fdr_q"] < 0.10,
        "segments_same_sign_ge_3of4": max(te_champ["segments_positive"],
                                          te_champ["segments_negative"]) >= 3,
    }
    if len(test) < 30:
        main_verdict = "insufficient"
    elif all(three_gates.values()):
        main_verdict = "pass"
    elif te_champ["paired_ci_high_pct"] < 0 and te_champ["fdr_q"] < 0.10:
        main_verdict = "negative"
    else:
        main_verdict = "inconclusive"

    result = {
        "topic": "trend-reversal-order-exit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "supertrend_current",
        "interval": "15m+1h", "days": window["days"], "window": window,
        "trades": len(cohort), "ci": {"ci_low_pct": te_champ["paired_ci_low_pct"],
                                     "ci_high_pct": te_champ["paired_ci_high_pct"],
                                     "of": f"{champion} vs baseline, test segment"},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002,
                  "round_trip": 0.0012, "basis": "notional"},
        "universe": {"symbols": len({r["symbol"] for r in cohort}),
                     "orders_in": payload["orders_in"],
                     "skipped": payload["skipped"],
                     "admin_exits_excluded": sum(1 for r in payload["records"] if not r["market_exit"]),
                     "core_supertrend_current": len(core)},
        "parameters": {"source": "real orders + finite negative-control rule family",
                       "ab_window_hours": payload["config"]["ab_hours"],
                       "descriptive_window_hours": payload["config"]["horizon_hours"],
                       "grid": {"tp_scales": [0.5, 0.75, 1.5, 2.0, 3.0],
                                "sl_scales": [1.5, 2.0],
                                "atr_multiples": [1.5, 2.5, 4.0],
                                "trail_R": [0.5, 1.0, 2.0],
                                "hold_fractions": [0.25, 0.5, 1.0]},
                       "champion_selection": "train segment only; test segment read-only",
                       "note": "no parameter search for production (H2)"},
        "split": {"train": args.train_ratio, "test": round(1 - args.train_ratio, 2),
                  "train_trades": len(train), "test_trades": len(test)},
        "min_pooled_trades": 30, "random_baseline_seeds": len(rand_means),
        "groups": stats,
        "baseline": {
            "real_mean_pct": float(np.mean(real)), "real_ci": real_ci,
            "replay_mean_pct": float(np.mean(base)), "replay_ci": base_ci,
            "replay_verdict": verdict(base_ci, len(cohort)),
            "calib_mean_pct": float(np.mean(calib)),
            "calib_ci": bootstrap_ci(calib),
            "reason_class_match_rate": round(fidelity, 4),
            "random_percentile": pct(float(np.mean(base))),
            "champion_random_percentile": pct(full_champ["mean_return_pct"]),
        },
        "random_baseline": {"mean": float(rand_means.mean()),
                            "std": float(rand_means.std()),
                            "p05": float(np.quantile(rand_means, 0.05)),
                            "p95": float(np.quantile(rand_means, 0.95))},
        "champion": {"name": champion, "train_paired_diff_pct":
                     next(x for x in tr_stats if x["name"] == champion)["paired_diff_pct"],
                     "test": te_champ, "full_sample": full_champ},
        "out_of_sample_family": {
            "rules_tested": len(oos_family), "n_three_gates_pass": n_oos_pass,
            "note": "exploratory scan on read-only test segment; H5 forbids re-selecting a champion here",
            "rows": oos_family,
        },
        "three_gates": three_gates,
        "verdict": main_verdict,
        "baseline_verdict": verdict(real_ci, len(cohort)),
        "default_enabled": False,
        "diagnosis": diagnosis,
        "two_sided_conflict": two_sided,
        "beta_attribution": {"source": "local PG klines 1m BTCUSDT", "window": window,
                             **beta},
        "live_subset_short_window": live_out,
        "limitations": [
            "sl/tp are the values stored at close time; protective orders may have been updated during the hold, so a small amount of post-entry information cannot be excluded",
            "replay matches real exit-reason class in only part of the cohort (see baseline.reason_class_match_rate): exchange conditional orders trigger on mark price, bars use traded OHLC; fidelity is measured inside the real holding window (calib_*), while the A/B uses a fixed forward window",
            "the A/B forward window is FIXED from entry (24h paper / 6h live) and differs from each order's real holding time; all arms share it, so paired differences isolate the rule but the absolute levels are not a reproduction of production P&L",
            "paper 72h and live 6h forward windows are different cohorts and are never merged",
            "administrative closes (manual / env_switch_archived / research_hold) excluded from A/B",
            "10 legacy orders with sl on the favorable side listed in universe.skipped (H8)",
            "oracle_net_pct is a look-ahead upper bound and is not tradable",
            "no funding rate, order-book depth, liquidation or partial-fill modelling; notional basis only, do not subtract from margin-basis reports",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, default=str, indent=2),
                   encoding="utf-8")
    print(f"[eval] cohort={len(cohort)} champion={champion} "
          f"test_diff={te_champ['paired_diff_pct']:.4f}% "
          f"CI=[{te_champ['paired_ci_low_pct']:.4f},{te_champ['paired_ci_high_pct']:.4f}] "
          f"q={full_champ['fdr_q']:.3f} verdict={main_verdict} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
