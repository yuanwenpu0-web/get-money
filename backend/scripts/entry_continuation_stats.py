"""第一阶段统计判定：入场后 1/5/10 分钟延续判断 vs 基准。

读 entry_continuation_study.py 的原始记录，输出 docs/entry-continuation-stage1.json。
判定链：绝对每笔 CI -> 配对差 CI -> 置换 p -> FDR -> 4 段时间同号 -> 随机早退分位。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from entry_continuation_common import (  # noqa: E402
    arm_stats, bh_fdr, bootstrap_ci, paired_perm_p, segment_means, verdict,
)

MIN_TRADES = 30


def cycle_block(records, k, rng, random_means):
    sub = [r for r in records if str(k) in r["arms"]]
    if not sub:
        return {"cycle_minutes": k, "trades": 0, "verdict": "insufficient"}
    arm = arm_stats([r["arms"][str(k)]["net"] for r in sub])
    base = arm_stats([r["base_net"] for r in sub])
    diff = np.array([r["arms"][str(k)]["net"] - r["base_net"] for r in sub], float)
    abs_ci = bootstrap_ci([r["arms"][str(k)]["net"] for r in sub])
    pair_ci = bootstrap_ci(diff)
    p = paired_perm_p(diff, rng)
    seg = segment_means(sub, str(k))
    reasons = [r["arms"][str(k)]["reason"] for r in sub]
    rm = np.asarray(random_means, float)
    return {
        "cycle_minutes": k, "trades": arm["trades"],
        "arm": arm, "baseline_same_subset": base,
        "delta": {
            "total_return_pct": arm["total_return_pct"] - base["total_return_pct"],
            "expectancy_pct": arm["expectancy_pct"] - base["expectancy_pct"],
            "win_rate": arm["win_rate"] - base["win_rate"],
            "payoff_ratio": arm["payoff_ratio"] - base["payoff_ratio"],
            "profit_factor": arm["profit_factor"] - base["profit_factor"],
            "max_drawdown_pct": arm["max_drawdown_pct"] - base["max_drawdown_pct"],
        },
        "absolute_ci": abs_ci,
        "paired_diff_pct": float(diff.mean()),
        "paired_ci": pair_ci, "p_perm": p,
        "segments": seg, "segments_positive": int(sum(m > 0 for m in seg)),
        "segments_negative": int(sum(m < 0 for m in seg)),
        "continuation_rate": round(reasons.count("continuation") / len(reasons), 4),
        "no_continuation_rate": round(reasons.count("no_continuation") / len(reasons), 4),
        "sl_in_window_rate": round(reasons.count("sl_in_window") / len(reasons), 4),
        "random_early": {
            "seeds": int(rm.size),
            "mean": float(rm.mean()) if rm.size else None,
            "p05": float(np.quantile(rm, 0.05)) if rm.size else None,
            "p95": float(np.quantile(rm, 0.95)) if rm.size else None,
            "arm_percentile": float((rm < arm["expectancy_pct"]).mean()) if rm.size else None,
        },
        "verdict": verdict(pair_ci, arm["trades"]),
        "absolute_verdict": verdict(abs_ci, arm["trades"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/entry_continuation_study.json")
    ap.add_argument("--out", default="docs/entry-continuation-stage1.json")
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    raw = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    recs = sorted(raw["records"], key=lambda r: r["time_ms"])
    rng = np.random.default_rng(args.seed)
    base = arm_stats([r["base_net"] for r in recs])
    blocks = []
    for k in raw["config"]["obs_minutes"]:
        blocks.append(cycle_block(recs, k, rng, raw["random_early"][str(k)]))
    q = bh_fdr([b["p_perm"] for b in blocks if b["trades"]])
    for b, qv in zip([b for b in blocks if b["trades"]], q):
        b["fdr_q"] = float(qv)
        b["three_gates_pass"] = bool(
            b["paired_ci"][0] > 0 and qv < 0.10
            and max(b["segments_positive"], b["segments_negative"]) >= 3)
        b["absolute_profit_pass"] = bool(b["absolute_ci"][0] > 0)

    live = [b for b in blocks if b["trades"] >= MIN_TRADES]
    best = max(live, key=lambda b: b["paired_diff_pct"]) if live else None
    if not live:
        main_verdict = "insufficient"
    elif any(b["three_gates_pass"] and b["absolute_profit_pass"] for b in live):
        main_verdict = "pass"
    elif all(b["paired_ci"][1] < 0 and b["fdr_q"] < 0.10 for b in live):
        main_verdict = "negative"
    else:
        main_verdict = "inconclusive"

    improved = [b["cycle_minutes"] for b in live
                if b["delta"]["expectancy_pct"] > 0 and b["delta"]["win_rate"] > 0
                and b["delta"]["total_return_pct"] > 0]
    reasons = []
    if not any(b["three_gates_pass"] for b in live):
        reasons.append("no cycle passes the three gates (paired CI low > 0, FDR q < 0.10, 3/4 segments same sign)")
    if not any(b["absolute_profit_pass"] for b in live):
        reasons.append("no cycle lifts the absolute per-trade CI lower bound above zero")
    if not improved:
        reasons.append("no cycle improves total return, expectancy and win rate together vs the same-subset baseline")
    worst = min(live, key=lambda b: b["random_early"]["arm_percentile"] or 0) if live else None
    if worst and (worst["random_early"]["arm_percentile"] or 0) < 0.95:
        reasons.append(f"{worst['cycle_minutes']}m arm sits at the {worst['random_early']['arm_percentile']:.2f} "
                       "percentile of matched-rate random early exits, i.e. no selection information")
    proceed = main_verdict == "pass"
    if not proceed:
        reasons.insert(0, "stage-2 expansion not triggered: the user's gate requires a clear improvement "
                          "on key metrics, which was not observed")

    result = {
        "topic": "entry-continuation-stage1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": 1,
        "question": "Does a 1/5/10-minute post-entry continuation check improve profit over the baseline?",
        "strategy": raw["strategy"], "interval": raw["interval"], "days": raw["days"],
        "window": raw["window"], "trades": base["trades"],
        "ci": {"ci_low_pct": best["paired_ci"][0] if best else None,
               "ci_high_pct": best["paired_ci"][1] if best else None,
               "of": f"{best['cycle_minutes']}m vs baseline paired difference" if best else None},
        "costs": raw["costs"], "universe": raw["universe"],
        "parameters": raw["parameters"], "config": raw["config"],
        "btc_change_pct": raw["btc_change_pct"],
        "baseline": base,
        "cycles": blocks,
        "min_pooled_trades": MIN_TRADES,
        "random_baseline": {"type": "matched-rate random early exit",
                            "seeds": raw["config"]["seeds"]},
        "verdict": main_verdict,
        "default_enabled": False,
        "stage2_decision": {"proceed": proceed, "reasons": reasons,
                            "trigger": "proceed only if a cycle passes all three gates AND absolute per-trade CI > 0"},
        "limitations": [
            "1m coverage limits the pool to 35 symbols: of 481 symbols with full 90-day 1h bars, only 35 have >=88% 1m minutes in the window; the other 500+ have a median of 16.8 days of 1m history",
            "the 90-day window starts 2026-08-12 and ends 2026-11-10; BTC rose over it, so short-horizon early exits carry a mechanical beta benefit",
            "an early exit is booked at the observation bar close with the standard 0.12% round-trip cost; the baseline books barrier touches at the barrier price",
            "the baseline uses a single full-position 3R target, whereas production supertrend_current scales out at 1R/2R/3R; absolute levels are therefore not production P&L",
            "no funding rate, order-book depth, liquidation or partial-fill modelling; notional basis, do not subtract from margin-basis reports",
            "observation windows of 1/5/10 minutes are far shorter than the 1.5xATR stop (mean 2.0%), so the stop is almost never touched inside the window (0.2-1.2%)",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, default=str, indent=2),
                   encoding="utf-8")
    print(f"[stats] trades={base['trades']} verdict={main_verdict} → {out}")
    for b in blocks:
        if not b.get("trades"):
            continue
        print(f"  {b['cycle_minutes']:>2}m n={b['trades']} exp={b['arm']['expectancy_pct']:+.4f}% "
              f"win={b['arm']['win_rate']:.3f} PF={b['arm']['profit_factor']:.3f} "
              f"MDD={b['arm']['max_drawdown_pct']:.1f} Δexp={b['delta']['expectancy_pct']:+.4f} "
              f"CI=[{b['paired_ci'][0]:+.4f},{b['paired_ci'][1]:+.4f}] q={b['fdr_q']:.3f} "
              f"rand_pct={b['random_early']['arm_percentile']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
