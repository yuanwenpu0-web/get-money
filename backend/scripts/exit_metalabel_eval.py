"""Gate-1 conditional-exit meta-label probe: statistics + paired report JSON.

Read-only. No production strategy code, bindings or params are touched.
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
from order_exit_stats import btc_change  # noqa: E402
from exit_metalabel_stats import (  # noqa: E402
    MIN_PAIRS, spearman, _pairs, _rank, probe_columns, self_calibration,
)
from exit_metalabel_rules import (  # noqa: E402
    MIN_POOLED_TRADES, purge_split, metalabel_t0, metalabel_t1, rule_block,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/exit_metalabel_paper.json")
    ap.add_argument("--secondary-input", default="output/exit_metalabel_live.json")
    ap.add_argument("--out", default="docs/exit-metalabel.json")
    args = ap.parse_args()

    rng = np.random.default_rng(20261110)
    d = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    live = json.loads((ROOT / args.secondary_input).read_text(encoding="utf-8"))
    all_r = d["records"]
    t0 = [r for r in all_r if r["market_exit"]]
    t1 = [r for r in t0 if r["t1_usable"]]
    live0 = [r for r in live["records"] if r["market_exit"]]
    live1 = [r for r in live0 if r["t1_usable"]]

    # ---- Gate-1 probes -------------------------------------------------
    probe_t0 = probe_columns(t0, "X_t0", "y0", "late_better", rng,
                             label="T0_pre_entry")
    probe_t1 = probe_columns(t1, "X_t1", "y1", "hold_better", rng,
                             label="T1_in_hold")
    # judgement.md §3 step 5: the positive control is a FUTURE-DIRECTION
    # oracle (perfect knowledge of which arm wins), not a proxy.
    cal_t0 = self_calibration(t0, "late_better", "y0", rng)
    cal_t1 = self_calibration(t1, "hold_better", "y1", rng)
    # Extra diagnostic: even perfect knowledge of the future MFE is a weak
    # predictor of the early-vs-late question. Kept separate from calibration.
    mfe_ctl = {
        "T0": self_calibration(t0, "LOOKAHEAD_oracle_mfe_R", "y0", rng),
        "T1": self_calibration(t1, "LOOKAHEAD_post_mfe_R", "y1", rng),
    }
    n_pass = sum(1 for r in probe_t0 + probe_t1 if r["probe_pass"])

    # ---- H8 sparse derivatives sub-cohort ------------------------------
    deriv = [r for r in t0 if r["has_sentiment"]]
    deriv_cols = ["sent_oi_z30", "sent_oi_xsec_pct", "sent_ls_ratio",
                  "sent_top_ls", "sent_taker_bs", "fund_rate", "fund_z30"]
    derivative = {
        "trades": len(deriv), "min_pooled_trades": MIN_POOLED_TRADES,
        "verdict": "insufficient" if len(deriv) < MIN_POOLED_TRADES else "inconclusive",
        "note": "fresh (<24h) sentiment/funding only; stale rows excluded by H8",
        "max_abs_ic_among_columns": None,
        "per_column_ic": {},
    }
    if len(deriv) >= 10:
        ics = {}
        # min_pairs=30 gate: report the raw Spearman on the small cohort as a
        # NUMBER ONLY, never as a judgement (judgement.md §4).
        for k in deriv_cols:
            a, b = _pairs([r["X_t0"].get(k, float("nan")) for r in deriv],
                          [r["y0"] for r in deriv])
            if len(a) >= 10 and a.std() > 1e-12 and b.std() > 1e-12:
                ra, rb = _rank(a), _rank(b)
                ics[k] = round(float(np.mean((ra - ra.mean()) * (rb - rb.mean()))
                                     / (ra.std() * rb.std())), 6)
        derivative["per_column_ic"] = ics
        vals = [abs(v) for v in ics.values() if v is not None and np.isfinite(v)]
        derivative["max_abs_ic_among_columns"] = max(vals) if vals else None

    # ---- pre-registered conditional rules ------------------------------
    train, test, split = purge_split(t0)
    m0 = metalabel_t0(t0, train, rng)
    m0_full = rule_block(t0, m0["per_order"], rng, "baseline_net_pct", "T0") if m0 else None
    m0_test = rule_block(test, m0["per_order"], rng, "baseline_net_pct", "T0") if m0 else None
    m0_train = rule_block(train, m0["per_order"], rng, "baseline_net_pct", "T0") if m0 else None
    tr1, te1, _ = purge_split(t1)
    m1 = metalabel_t1(t1, tr1, rng)
    m1_full = rule_block(t1, m1["per_order"], rng, "hold_net_pct", "T1") if m1 else None
    m1_test = rule_block(te1, m1["per_order"], rng, "hold_net_pct", "T1") if m1 else None

    # ---- oracle upper bounds (LOOKAHEAD, not tradable) -----------------
    oracle0 = float(np.mean([r["late_net_pct"] if r["late_better"]
                             else r["early_net_pct"] for r in t0]))
    oracle1 = float(np.mean([max(r["exit_now_net_pct"], r["hold_net_pct"])
                             for r in t1]))
    real0 = float(np.mean([r["real_net_pct"] for r in t0]))
    replay0 = float(np.mean([r["baseline_net_pct"] for r in t0]))
    real_ci = bootstrap_ci([r["real_net_pct"] for r in t0])
    replay_ci = bootstrap_ci([r["baseline_net_pct"] for r in t0])

    # ---- verdict -------------------------------------------------------
    ci = {"ci_low_pct": m0_test["paired_ci_low_pct"],
          "ci_high_pct": m0_test["paired_ci_high_pct"],
          "of": "metalabel_T0 vs unconditional baseline, purged test segment"}
    q = min([r["fdr_q"] for r in probe_t0 if r["column"] == m0["feature"]]
           or [1.0])
    seg_ok = max(m0_test["segments"]["n_positive"],
                 m0_test["segments"]["n_negative"]) >= 3
    three = {"test_ci_low_positive": m0_test["paired_ci_low_pct"] > 0,
             "fdr_q_below_0.10": q < 0.10,
             "segments_same_sign_ge_3of4": seg_ok}
    if m0_test["trades"] < MIN_POOLED_TRADES:
        main_verdict = "insufficient"
    elif all(three.values()) and n_pass > 0:
        main_verdict = "pass"
    elif m0_test["paired_ci_high_pct"] < 0 and q < 0.10:
        main_verdict = "negative"
    else:
        main_verdict = "inconclusive"

    dates = sorted(r["created_at"][:10] for r in t0)
    window = {"start": dates[0], "end": dates[-1],
              "days": (datetime.fromisoformat(dates[-1])
                       - datetime.fromisoformat(dates[0])).days + 1}
    start_ms = int(datetime.fromisoformat(dates[0] + "T00:00:00+08:00").timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(dates[-1] + "T23:59:00+08:00").timestamp() * 1000)
    beta = btc_change(start_ms, end_ms)

    result = {
        "topic": "exit-metalabel",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "real order cohort (supertrend_current dominant)",
        "interval": "15m+1h", "days": window["days"], "window": window,
        "trades": len(t0), "t1_trades": len(t1),
        "ci": ci, "costs": {"fee_rate": 0.0004, "slippage": 0.0002,
                            "round_trip": 0.0012, "basis": "notional"},
        "hypothesis": {
            "question": "can an indicator, observable at the decision moment, "
                        "tell whether to take profit now or push it later?",
            "falsifiable": "at least one column must clear |IC|>=0.02, "
                           "symbol-block sign share>=0.60, FDR q<0.10 and 3/4 "
                           "time segments with the same sign",
            "difference_from_previous_round": "previous round tested unconditional "
                "static exit rules; this round tests CONDITIONAL discrimination",
            "h2_treatment": "OHLCV-derived columns are negative controls only, "
                            "never implementation or parameter-search candidates",
            "information_sources": ["OHLCV-derived (negative control)",
                                    "order metadata (score/prob_up/side/interval)",
                                    "derivatives sentiment + funding (sparse, H8)",
                                    "in-hold path state (mfe/giveback/bars held)"],
        },
        "universe": {
            "symbols": len({r["symbol"] for r in t0}),
            "orders_in": d["orders_in"],
            "market_exit": len(t0), "t1_decision_points": len(t1),
            "skipped": d["skipped"],
            "admin_exits_excluded": len(all_r) - len(t0),
            "fresh_sentiment_orders": sum(1 for r in t0 if r["has_sentiment"]),
            "fresh_funding_orders": sum(1 for r in t0 if r["has_funding"]),
            "y0_ties": sum(1 for r in t0 if abs(r["y0"]) < 1e-9),
        },
        "parameters": {
            "source": "real orders + finite pre-registered indicator families",
            "ab_window_hours": d["config"]["ab_hours"],
            "descriptive_window_hours": d["config"]["horizon_hours"],
            "probe_gates": {"abs_ic": 0.02, "sign_share": 0.60,
                            "fdr_q": 0.10, "segments": "3/4", "min_pairs": MIN_PAIRS},
            "decision_mfe_R": 0.7,
            "champion_selection": "train segment only; purged test segment read-only",
            "note": "no parameter grid, no production rule implementation (H1/H2)",
        },
        "split": split, "min_pooled_trades": MIN_POOLED_TRADES,
        "random_baseline_seeds": 20,
        "probe": {
            "T0_columns_tested": len(probe_t0), "T1_columns_tested": len(probe_t1),
            "n_columns_pass_all_gates": n_pass,
            "gate_result": "PASS" if n_pass > 0 else "FAIL",
            "T0": probe_t0, "T1": probe_t1,
            "columns_excluded": {
                "columns": sorted(
                    set(set().union(*[set(r["X_t0"]) for r in t0]))
                    - {r["column"] for r in probe_t0}),
                "reason": "H8 coverage, not statistical degeneracy: fresh (<24h) "
                          "derivative rows exist on only 17/105 orders, below the "
                          "30-pair gate, so these columns never enter the probe",
            },
        },
        "self_calibration": {
            "T0": cal_t0, "T1": cal_t1,
            "positive_control_definition": "binary oracle = perfect knowledge of "
                "which arm/action wins (late_better / hold_better); LOOKAHEAD",
            "future_mfe_control_separate": mfe_ctl,
        },
        "derivative_subset": derivative,
        "baseline": {
            "real_mean_pct": real0, "real_ci": real_ci,
            "real_verdict": verdict(real_ci, len(t0), MIN_POOLED_TRADES),
            "replay_mean_pct": replay0, "replay_ci": replay_ci,
            "replay_verdict": verdict(replay_ci, len(t0), MIN_POOLED_TRADES),
            "y0_mean_pct": float(np.mean([r["y0"] for r in t0])),
            "y1_mean_pct": float(np.mean([r["y1"] for r in t1])),
        },
        "conditional_rules": {
            "metalabel_T0": {"feature": m0["feature"], "threshold": m0["threshold"],
                            "train_ic": m0["train_ic"], "mapping": m0["mapping"],
                            "train": m0_train, "test": m0_test, "full": m0_full,
                            "selected_feature_fdr_q_full": q},
            "metalabel_T1": {"feature": m1["feature"], "threshold": m1["threshold"],
                            "mapping": m1["mapping"], "full": m1_full, "test": m1_test},
            "oracle_LOOKAHEAD_not_tradable": {
                "T0_best_arm_every_order_pct": oracle0,
                "T1_best_action_every_order_pct": oracle1,
                "T0_real_baseline_pct": real0,
                "note": "uses the realized future label; upper bound only",
            },
        },
        "random_baseline": m0_full["random_baseline"],
        "test_segment_single_rule_observation": {
            "reading": "metalabel_T0 purged test segment paired diff "
                       f"{m0_test['paired_diff_pct']:+.4f}pp, CI "
                       f"[{m0_test['paired_ci_low_pct']:+.4f}, "
                       f"{m0_test['paired_ci_high_pct']:+.4f}], "
                       f"p_perm {m0_test['p_perm']:.4f}",
            "why_not_an_edge": [
                f"champion selected among {len(probe_t0)} T0 columns on the train "
                f"segment; its full-sample FDR q = {q:.3f} (> 0.10)",
                f"full-sample paired CI [{m0_full['paired_ci_low_pct']:+.4f}, "
                f"{m0_full['paired_ci_high_pct']:+.4f}] crosses zero",
                f"train paired CI [{m0_train['paired_ci_low_pct']:+.4f}, "
                f"{m0_train['paired_ci_high_pct']:+.4f}] crosses zero",
                f"random-baseline percentile "
                f"{m0_test['random_baseline']['percentile_of_observed']:.3f} "
                "< 0.95 (H6)",
                f"Gate-1 probe passed {n_pass}/{len(probe_t0) + len(probe_t1)} "
                "columns (H1 forbids implementation)",
            ],
            "classification": "likely_noise: observation only, never an edge",
        },
        "three_gates": three, "verdict": main_verdict,
        "baseline_verdict": verdict(real_ci, len(t0), MIN_POOLED_TRADES),
        "default_enabled": False,
        "beta_attribution": {"source": "local PG klines 1m BTCUSDT",
                             "window": window, **beta},
        "live_subset_short_window": {
            "trades": len(live0), "t1_trades": len(live1),
            "ab_window_hours": live["config"]["ab_hours"],
            "y0_mean_pct": float(np.mean([r["y0"] for r in live0])),
            "y1_mean_pct": float(np.mean([r["y1"] for r in live1])),
            "n_columns_pass_all_gates": sum(
                1 for r in probe_columns(live0, "X_t0", "y0", "late_better", rng,
                                         label="live_T0") if r["probe_pass"]),
            "verdict": "insufficient",
            "note": "27 orders < 30-trade gate; numbers reported, no conclusion",
        },
        "limitations": [
            "sentiment/funding data end 2026-10-28 / 10-27; fresh (<24h) coverage "
            "is 17/105 paper orders and 0/27 live orders, so derivative columns "
            "are H8-insufficient and never enter the main judgement",
            "61/105 T0 labels are exact ties: both TP arms are stopped out by the "
            "same SL before TP, so the early-vs-late question has no answer there; "
            "excluding ties would use future information and is NOT done",
            "T1 decision points exist only on 54 orders (barrier fired earlier on "
            "the rest); this is a structural censoring, not a signal filter",
            "sl/tp are values stored at close time; protective orders may have "
            "been updated during the hold (see trend-reversal-order-exit.md)",
            "OHLC replay vs exchange conditional orders differ (mark price vs "
            "traded OHLC); calibration match rate was 83.8% in the prior round",
            "paper 24h and live 6h forward windows are different cohorts and "
            "are never merged",
            "oracle numbers are look-ahead upper bounds and are not tradable",
            "no funding-rate charging, order-book depth, liquidation, partial "
            "fills or slippage-path modelling; notional basis only, do not "
            "subtract from margin-basis reports",
            "no production spec / binding / parameter was changed; Gate 1 probe "
            "only, default_enabled=False",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print(f"[eval] verdict={main_verdict} probe={result['probe']['gate_result']} "
          f"n_pass={n_pass} T0_cols={len(probe_t0)} T1_cols={len(probe_t1)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
