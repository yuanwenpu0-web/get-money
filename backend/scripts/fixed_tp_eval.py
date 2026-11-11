"""Statistical judgement for ABSOLUTE fixed take-profit targets on online orders.

Question: does replacing the per-order ATR-scaled TP with one fixed profit
percentage for every order increase profit? Controlled A/B: TP is the ONLY
changed variable; entry, SL, bars, cost and forward window are identical.

Paper and live cohorts are never merged (different windows / venues).
No production code, spec or binding is touched.
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

MIN_TRADES = 30
FIX_PREFIX = "fix_"


def build_rows(recs, rules, rng, family=None):
    """Per-rule absolute level, paired difference, FDR and exit mechanics."""
    if not recs or not rules:
        return []
    stats = rule_stats(recs, rules, rng)
    wide = rule_stats(recs, family, rng) if family else None
    wmap = {x["name"]: x for x in wide} if wide else {}
    for row in stats:
        name = row["name"]
        ex = [r["rule_exit"][name] for r in recs]
        n = len(ex)
        row["tp_hit_rate"] = round(sum(e["reason"].startswith("tp") for e in ex) / n, 4)
        row["sl_hit_rate"] = round(sum(e["reason"].startswith("sl") for e in ex) / n, 4)
        row["horizon_exit_rate"] = round(
            sum(e["reason"] in ("horizon", "time") for e in ex) / n, 4)
        row["mean_hold_bars"] = round(float(np.mean([e["bar"] + 1 for e in ex])), 2)
        row["absolute_ci"] = [row["ci_low_pct"], row["ci_high_pct"]]
        row["absolute_verdict"] = verdict([row["ci_low_pct"], row["ci_high_pct"]], n)
        row["full_family_fdr_q"] = float(wmap.get(name, {}).get("fdr_q", row["fdr_q"]))
        row["three_gates_full_family_q"] = bool(
            row["paired_ci_low_pct"] > 0
            and row["full_family_fdr_q"] < 0.10
            and max(row["segments_positive"], row["segments_negative"]) >= 3)
    return stats


def gates(row):
    return {
        "absolute_ci_low_positive": row["ci_low_pct"] > 0,
        "paired_ci_low_positive": row["paired_ci_low_pct"] > 0,
        "fdr_q_below_0.10": row["fdr_q"] < 0.10,
        "full_family_q_below_0.10": row["full_family_fdr_q"] < 0.10,
        "segments_same_sign_ge_3of4": max(row["segments_positive"],
                                          row["segments_negative"]) >= 3,
    }


def cohort_report(recs, fixed, family, rng, train_ratio, label, ab_hours):
    recs = sorted(recs, key=lambda r: r["created_at"])
    n = len(recs)
    real = np.array([r["real_net_pct"] for r in recs], dtype=float)
    base = np.array([r["baseline_net_pct"] for r in recs], dtype=float)
    rand = np.array([[r["random_exit_net_pct"][s] for r in recs]
                     for s in range(len(recs[0]["random_exit_net_pct"]))])
    rand_means = rand.mean(axis=1)
    pct = lambda v: float(np.clip(np.count_nonzero(rand_means < v) / len(rand_means), 0, 1))

    rows = build_rows(recs, fixed, rng, family)
    n_abs_pass = sum(1 for x in rows if x["absolute_verdict"] == "pass")
    n_three_gates = sum(1 for x in rows if x["three_gates_pass"])
    n_full_family = sum(1 for x in rows if x["three_gates_full_family_q"])

    # Full-sample exploratory scan (H5: NOT used to select the champion).
    if n < MIN_TRADES:
        full_verdict = "insufficient"
    elif any(x["three_gates_pass"] and x["ci_low_pct"] > 0 for x in rows):
        full_verdict = "pass"
    elif all(x["ci_high_pct"] < 0 and x["paired_ci_high_pct"] < 0
             and x["fdr_q"] < 0.10 for x in rows):
        full_verdict = "negative"
    else:
        full_verdict = "inconclusive"

    # H5: train selects, test only reads.
    k = int(n * train_ratio)
    train, test = recs[:k], recs[k:]
    split_ok = len(train) >= MIN_TRADES and len(test) >= MIN_TRADES
    split = {"train_ratio": train_ratio, "train_trades": len(train),
             "test_trades": len(test), "min_pooled_trades": MIN_TRADES,
             "test_segment_usable": split_ok}
    oos = {"status": "not_evaluated"}
    main_verdict = "insufficient" if not split_ok else "inconclusive"
    if split_ok:
        tr = build_rows(train, fixed, rng, family)
        te = build_rows(test, fixed, rng, family)
        champ = max(tr, key=lambda x: x["paired_diff_pct"])["name"]
        tec = next(x for x in te if x["name"] == champ)
        g = gates(tec)
        oos = {
            "champion_from_train": champ,
            "train_paired_diff_pct": next(x for x in tr if x["name"] == champ)["paired_diff_pct"],
            "test": tec, "gates": g,
            "three_gates_pass": all(g.values()),
            "note": "test segment is read-only; no re-selection of champion",
        }
        if len(test) < MIN_TRADES:
            main_verdict = "insufficient"
        elif all(g.values()):
            main_verdict = "pass"
        elif (tec["ci_high_pct"] < 0 and tec["paired_ci_high_pct"] < 0
              and tec["fdr_q"] < 0.10):
            main_verdict = "negative"
        else:
            main_verdict = "inconclusive"

    best = max(rows, key=lambda x: x["paired_diff_pct"]) if rows else None
    return {
        "label": label, "trades": n, "ab_window_hours": ab_hours,
        "symbols": len({r["symbol"] for r in recs}),
        "strategies": sorted({r["strategy"] for r in recs}),
        "window": {"start": recs[0]["created_at"][:10], "end": recs[-1]["created_at"][:10]},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002,
                  "round_trip": 0.0012, "basis": "notional"},
        "baseline": {
            "real_mean_pct": float(np.mean(real)), "real_ci": bootstrap_ci(real),
            "real_verdict": verdict(bootstrap_ci(real), n),
            "replay_mean_pct": float(np.mean(base)), "replay_ci": bootstrap_ci(base),
            "replay_verdict": verdict(bootstrap_ci(base), n),
            "replay_random_percentile": pct(float(np.mean(base))),
            "tp_dist_pct_mean": round(float(np.mean(
                [abs(r["tp"] - r["entry_price"]) / r["entry_price"] * 100 for r in recs])), 4),
            "tp_dist_pct_median": round(float(np.median(
                [abs(r["tp"] - r["entry_price"]) / r["entry_price"] * 100 for r in recs])), 4),
        },
        "random_baseline": {
            "seeds": len(rand_means), "mean": float(rand_means.mean()),
            "std": float(rand_means.std()),
            "p05": float(np.quantile(rand_means, 0.05)),
            "p95": float(np.quantile(rand_means, 0.95)),
            "best_rule_mean_pct": best["mean_return_pct"] if best else None,
            "best_rule_percentile": pct(best["mean_return_pct"]) if best else None,
        },
        "groups": rows,
        "fixed_tp_grid": [x["name"] for x in rows],
        "n_absolute_profit_pass": n_abs_pass,
        "n_three_gates_pass_family_q": n_three_gates,
        "n_three_gates_pass_full_family_q": n_full_family,
        "full_sample_verdict": full_verdict,
        "split": split, "out_of_sample": oos,
        "verdict": main_verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="output/fixed_tp_paper.json")
    ap.add_argument("--live", default="output/fixed_tp_live.json")
    ap.add_argument("--calib-baseline", default="output/order_exit_paper.json")
    ap.add_argument("--out", default="docs/trend-reversal-fixed-tp.json")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    args = ap.parse_args()

    paper = json.loads((ROOT / args.paper).read_text(encoding="utf-8"))
    live = json.loads((ROOT / args.live).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    # Self-calibration: the default (no fixed grid) replay must be unchanged.
    calib_new = json.loads((ROOT / "output/fixed_tp_calib_paper.json").read_text(encoding="utf-8"))
    calib_old = json.loads((ROOT / args.calib_baseline).read_text(encoding="utf-8"))
    a = [r["baseline_net_pct"] for r in calib_new["records"]]
    b = [r["baseline_net_pct"] for r in calib_old["records"]]
    self_calib = {
        "old_records": len(b), "new_records": len(a),
        "old_rule_count": len(calib_old["records"][0]["rule_net_pct"]),
        "new_rule_count": len(calib_new["records"][0]["rule_net_pct"]),
        "max_abs_baseline_diff_pct": float(np.max(np.abs(np.array(a) - np.array(b)))),
        "passed": len(a) == len(b) and np.allclose(a, b)
                  and len(calib_new["records"][0]["rule_net_pct"])
                  == len(calib_old["records"][0]["rule_net_pct"]),
    }

    def split_payload(p):
        recs = [r for r in p["records"] if r["market_exit"]]
        all_rules = list(recs[0]["rule_net_pct"].keys())
        fixed = [x for x in all_rules if x.startswith(FIX_PREFIX)]
        legacy = [x for x in all_rules if x != "baseline" and not x.startswith(FIX_PREFIX)]
        return recs, fixed, fixed + legacy

    precs, pfixed, pfamily = split_payload(paper)
    lrecs, lfixed, lfamily = split_payload(live)
    # Main cohort: the trend-reversal strategy specifically.
    pcore = [r for r in precs if r["strategy"] == "supertrend_current"]

    cohorts = {
        "paper_trend_reversal": cohort_report(
            pcore, pfixed, pfamily, rng, args.train_ratio,
            "paper / supertrend_current / market-rule exits", paper["config"]["ab_hours"]),
        "paper_all_market_exits": cohort_report(
            precs, pfixed, pfamily, rng, args.train_ratio,
            "paper / all strategies / market-rule exits", paper["config"]["ab_hours"]),
        "live_binance_demo": cohort_report(
            lrecs, lfixed, lfamily, rng, args.train_ratio,
            "live (Binance DEMO) / supertrend_current", live["config"]["ab_hours"]),
    }
    primary = cohorts["paper_trend_reversal"]
    window = {"start": primary["window"]["start"], "end": primary["window"]["end"],
              "days": (datetime.fromisoformat(pcore[-1]["created_at"]).date()
                       - datetime.fromisoformat(pcore[0]["created_at"]).date()).days + 1}
    start_ms = int(datetime.fromisoformat(pcore[0]["created_at"]).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(pcore[-1]["created_at"]).timestamp() * 1000)

    result = {
        "topic": "trend-reversal-fixed-tp",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does a fixed absolute profit target increase profit on real online orders?",
        "strategy": "supertrend_current",
        "interval": "15m+1h", "days": window["days"], "window": window,
        "trades": primary["trades"],
        "ci": {"ci_low_pct": primary["out_of_sample"].get("test", {}).get("ci_low_pct"),
               "ci_high_pct": primary["out_of_sample"].get("test", {}).get("ci_high_pct"),
               "of": "train-selected champion vs baseline, read-only test segment"},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002,
                  "round_trip": 0.0012, "basis": "notional"},
        "universe": {
            "definition": "orders table rows actually produced by the running service "
                          "(paper simulator + Binance DEMO live); NO synthetic backtest orders",
            "paper_orders_in": paper["orders_in"], "paper_skipped": paper["skipped"],
            "live_orders_in": live["orders_in"], "live_skipped": live["skipped"],
            "paper_symbols": primary["symbols"],
            "paper_admin_exits_excluded": sum(1 for r in paper["records"]
                                              if not r["market_exit"]),
            "note": "manual / research_hold / env_switch_archived are administrative, not exit rules",
        },
        "parameters": {
            "fixed_tp_grid_pct": paper["config"]["fixed_tp_pcts"],
            "grid_rationale": "pre-registered to span an order of magnitude around the "
                              "observed per-order TP distance (~3.5% mean)",
            "controlled_variable": "TP only; SL, entry, bars, cost and forward window unchanged",
            "ab_window_hours": {"paper": paper["config"]["ab_hours"],
                                "live": live["config"]["ab_hours"]},
            "descriptive_window_hours": {"paper": paper["config"]["horizon_hours"],
                                         "live": live["config"]["horizon_hours"]},
            "candle_source": "derived from 1m via x - x%step, complete contiguous buckets only",
            "fdr_family": "8 fixed-TP rules; full-family sensitivity adds the 17 historical rules",
            "champion_selection": "train segment only (H5)",
            "no_production_parameter_search": True,
        },
        "split": primary["split"], "min_pooled_trades": MIN_TRADES,
        "random_baseline_seeds": primary["random_baseline"]["seeds"],
        "self_calibration": self_calib,
        "groups": primary["groups"],
        "baseline": primary["baseline"],
        "random_baseline": primary["random_baseline"],
        "cohorts": cohorts,
        "three_gates": primary["out_of_sample"].get("gates"),
        "verdict": primary["verdict"],
        "live_verdict": cohorts["live_binance_demo"]["verdict"],
        "live_full_sample_verdict": cohorts["live_binance_demo"]["full_sample_verdict"],
        "default_enabled": False,
        "beta_attribution": {"source": "local PG klines 1m BTCUSDT",
                             "window": window, **btc_change(start_ms, end_ms)},
        "limitations": [
            "real fills and replayed arms use different windows; only same-window PAIRED differences are comparable, absolute levels are not production P&L",
            "sl/tp are the values stored at close time and may have been updated by protective orders mid-hold; params_snapshot is all NULL",
            "exchange conditional orders trigger on mark price while bars use traded OHLC; replay reason-class fidelity was 83.8% in docs/trend-reversal-order-exit.md",
            "paper 24h and live 6h forward windows differ; cohorts are NEVER merged",
            "live cohort has 40 trades but its 40% test segment is below MIN_POOLED_TRADES=30, so no champion selection is done there",
            "no funding rate, order-book depth, liquidation or partial-fill modelling; notional basis, do not subtract from margin-basis reports",
            "13-16 day single-regime sample; no cross-regime robustness",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, default=str, indent=2),
                   encoding="utf-8")
    print(f"[eval] primary trades={primary['trades']} verdict={primary['verdict']} "
          f"full_sample={primary['full_sample_verdict']} "
          f"n_abs_pass={primary['n_absolute_profit_pass']} "
          f"n_three_gates={primary['n_three_gates_pass_family_q']} "
          f"self_calib={self_calib['passed']} -> {out}")
    for name, c in cohorts.items():
        print(f"  {name}: n={c['trades']} verdict={c['verdict']} "
              f"full={c['full_sample_verdict']} real={c['baseline']['real_mean_pct']:.4f}% "
              f"abs_pass={c['n_absolute_profit_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
