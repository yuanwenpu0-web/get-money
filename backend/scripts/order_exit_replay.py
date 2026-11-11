"""Offline per-order exit replay for the trend-reversal order cohort.

Read-only. For each real closed order we derive interval bars from 1m, cut a
strictly causal history (for ATR) and a fixed forward window, then replay a
finite NEGATIVE-CONTROL family of exit rules on the SAME bar sequence.
No production strategy code, bindings or params are touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.db import make_engine  # noqa: E402
from order_exit_common import (  # noqa: E402
    MARKET_EXIT_REASONS, TP_REASONS, SL_REASONS, net_return_pct, atr,
    run_rule, build_rules, diagnostics,
)

STEP_MS = {"15m": 900_000, "1h": 3_600_000}

# 1m -> derived interval bars; same bucket rule as data_service._candle_query.
CANDLE_SQL = """
WITH m AS (
  SELECT open_time, open, high, low, close, volume, quote_volume
  FROM klines
  WHERE symbol = :sym AND interval = '1m'
    AND open_time >= :start_ms AND open_time < :end_ms
), b AS (
  SELECT (open_time - open_time % :step) AS ot,
         count(*) AS n, min(open_time) AS first_t, max(open_time) AS last_t,
         max(high) AS high, min(low) AS low,
         sum(volume) AS volume, sum(quote_volume) AS quote_volume
  FROM m GROUP BY 1
)
SELECT b.ot, f.open, b.high, b.low, l.close, b.volume, b.quote_volume
FROM b
JOIN klines f ON f.symbol = :sym AND f.interval = '1m' AND f.open_time = b.first_t
JOIN klines l ON l.symbol = :sym AND l.interval = '1m' AND l.open_time = b.last_t
WHERE b.n = :minutes AND b.first_t = b.ot
  AND b.last_t = b.ot + (b.n - 1) * 60000
  AND b.ot >= :start_ms AND b.ot < :end_ms
ORDER BY b.ot
"""

ORDER_SQL = """
SELECT id, symbol, interval, side, source, strategy, pool, close_reason,
       entry_price, sl, tp, close_price, realized_pnl,
       created_at, closed_at
FROM orders
WHERE status = 'CLOSED' AND entry_price > 0 AND tp > 0 AND sl > 0
  AND close_price > 0 AND closed_at IS NOT NULL
  AND (:source = '' OR source = :source)
ORDER BY created_at, id
"""


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def reason_class(reason):
    if reason in TP_REASONS:
        return "TP"
    if reason in SL_REASONS:
        return "SL"
    if reason == "signal_reverse":
        return "REVERSE"
    return "ADMIN"


def classify(rec):
    """Pre-committed descriptive classes (thresholds fixed before the run)."""
    if rec["close_reason"] in TP_REASONS and rec["diag"].get("post_exit_run_R", 0) >= 1.0:
        return "early"
    if (rec["close_reason"] not in TP_REASONS
            and rec["diag"].get("hold_mfe_R", 0) >= 0.7
            and rec["real_net_pct"] < 0):
        return "late"
    return "neither"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="paper", help="paper | live | '' (all)")
    ap.add_argument("--horizon-hours", type=float, default=72.0,
                    help="data prefetch / descriptive forward window")
    ap.add_argument("--ab-hours", type=float, default=24.0,
                    help="FIXED forward window shared by every A/B rule")
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N orders only")
    ap.add_argument("--fixed-tp-grid", default="",
                    help="comma-separated ABSOLUTE TP percentages of entry, e.g. 0.5,1,2,4")
    ap.add_argument("--out", default="output/order_exit_replay_paper.json")
    args = ap.parse_args()
    fixed_grid = tuple(float(x) for x in args.fixed_tp_grid.split(",") if x.strip())
    if any(p <= 0 or p > 100 for p in fixed_grid):
        raise SystemExit("fixed TP grid must be in (0, 100] percent")

    t0 = time.perf_counter()
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        orders = [dict(r) for r in conn.execute(
            text(ORDER_SQL), {"source": args.source}).mappings().all()]
        orders = [o for o in orders if o["interval"] in STEP_MS]
        if args.limit:
            orders = orders[: args.limit]
        groups: dict[tuple[str, str], list] = {}
        for o in orders:
            groups.setdefault((o["symbol"], o["interval"]), []).append(o)
        cache = {}
        for (sym, interval), group in groups.items():
            step = STEP_MS[interval]
            hb = max(1, int(args.horizon_hours * 3_600_000 / step))
            buckets = [to_ms(o["created_at"]) // step * step for o in group]
            lo = min(buckets) - (args.lookback + 3) * step
            hi = max(buckets) + (hb + 3) * step
            rows = conn.execute(text(CANDLE_SQL), {
                "sym": sym, "step": step, "minutes": step // 60_000,
                "start_ms": lo, "end_ms": hi,
            }).mappings().all()
            cache[(sym, interval)] = [dict(r) for r in rows]

    skipped, records = [], []
    for o in orders:
        interval, step = o["interval"], STEP_MS[o["interval"]]
        hb = max(1, int(args.horizon_hours * 3_600_000 / step))
        ob = to_ms(o["created_at"]) // step * step
        candles = cache[(o["symbol"], interval)]
        hist = [c for c in candles if c["ot"] + step <= to_ms(o["created_at"])][-args.lookback:]
        future = [c for c in candles if ob + step <= c["ot"] <= ob + hb * step][:hb]
        if len(hist) < max(16, args.lookback // 4):
            skipped.append({"id": o["id"], "reason": "insufficient_history"}); continue
        if not future or future[0]["ot"] != ob + step:
            skipped.append({"id": o["id"], "reason": "missing_first_future_bar"}); continue
        if len(future) / hb < args.min_coverage:
            skipped.append({"id": o["id"], "reason": "low_future_coverage",
                            "bars": len(future), "expected": hb}); continue
        if any(any(float(c[k]) <= 0 for k in ("open", "high", "low", "close"))
               for c in hist + future):
            skipped.append({"id": o["id"], "reason": "nonpositive_price"}); continue
        entry, tp, sl = float(o["entry_price"]), float(o["tp"]), float(o["sl"])
        close = float(o["close_price"])
        side = o["side"]
        if abs(entry - sl) <= 0 or abs(tp - entry) <= 0:
            skipped.append({"id": o["id"], "reason": "degenerate_barriers"}); continue
        # Some legacy orders store the post-entry protective/trailing price in
        # `sl` (it sits on the FAVORABLE side of entry). Replaying those as an
        # entry-time stop is look-ahead, so they go to the H8 skip list.
        if (side == "LONG" and sl >= entry) or (side == "SHORT" and sl <= entry):
            skipped.append({"id": o["id"], "reason": "sl_not_adverse"}); continue
        if (side == "LONG" and tp <= entry) or (side == "SHORT" and tp >= entry):
            skipped.append({"id": o["id"], "reason": "tp_not_favorable"}); continue
        hold_bars = max(1, int((to_ms(o["closed_at"]) - ob) // step))
        # Counterfactual A/B: EVERY arm (including baseline) shares one FIXED
        # forward window measured from the entry, independent of when the real
        # order happened to close. Otherwise raising/removing TP could never
        # capture the post-TP move and the "too early" side would be censored.
        ab_bars = max(1, int(args.ab_hours * 3_600_000 / step))
        replay_bars = future[:max(1, min(len(future), ab_bars))]
        # Calibration only: replay the baseline inside the REAL holding window
        # to measure how well our OHLC replay reproduces production exits.
        calib_bars = future[:max(1, min(len(future), hold_bars))]
        rules = build_rules(entry, tp, sl, atr(hist), len(replay_bars), side,
                            fixed_tp_pcts=fixed_grid)
        rule_out, net = {}, {}
        for r in rules:
            px, idx, why = run_rule(replay_bars, side, entry, r["sl"], r)
            rule_out[r["name"]] = {"exit_price": px, "bar": idx, "reason": why}
            net[r["name"]] = net_return_pct(entry, px, side)
        cpx, _, creason = run_rule(calib_bars, side, entry, sl, rules[0])
        # Random-exit control: same SL protection, exit bar drawn at random.
        random_net = []
        for s in range(args.random_seeds):
            rr = np.random.default_rng(args.seed + s * 7919)
            k = int(rr.integers(1, len(replay_bars) + 1))
            px, _, _ = run_rule(replay_bars, side, entry, sl,
                                {"name": "rand", "kind": "time", "tp": None,
                                 "sl": sl, "R": abs(entry - sl),
                                 "activate_R": 0, "trail_R": 0, "max_bars": k})
            random_net.append(net_return_pct(entry, px, side))
        diag = diagnostics(future, side, entry, sl, tp,
                           rule_out["baseline"]["exit_price"],
                           rule_out["baseline"]["bar"], close, hold_bars,
                           post_cap=max(1, int(24 * 3_600_000 / step)))
        rec = {
            "id": o["id"], "symbol": o["symbol"], "interval": interval,
            "side": side, "source": o["source"], "strategy": o["strategy"],
            "pool": o["pool"], "close_reason": o["close_reason"],
            "created_at": o["created_at"].isoformat(),
            "closed_at": o["closed_at"].isoformat(),
            "entry_price": entry, "tp": tp, "sl": sl, "close_price": close,
            "realized_pnl": o["realized_pnl"], "hold_bars": hold_bars,
            "future_bars": len(future), "horizon_bars": hb,
            "ab_bars": len(replay_bars), "calib_bars": len(calib_bars),
            "post_bars": max(0, len(future) - len(calib_bars)),
            "market_exit": o["close_reason"] in MARKET_EXIT_REASONS,
            "real_net_pct": net_return_pct(entry, close, side),
            "baseline_net_pct": net["baseline"],
            "baseline_reason": rule_out["baseline"]["reason"],
            "calib_net_pct": net_return_pct(entry, cpx, side),
            "replay_reason_class": ("TP" if creason.startswith("tp")
                                    else "SL" if creason.startswith("sl")
                                    else "HOLD"),
            "calib_reason": creason,
            "real_reason_class": reason_class(o["close_reason"]),
            "rule_net_pct": net, "rule_exit": rule_out,
            "random_exit_net_pct": random_net, "diag": diag,
        }
        rec["class"] = classify(rec)
        records.append(rec)

    payload = {
        "config": {
            "source": args.source, "horizon_hours": args.horizon_hours,
            "ab_hours": args.ab_hours,
            "lookback": args.lookback, "min_coverage": args.min_coverage,
            "random_seeds": args.random_seeds, "seed": args.seed,
            "candle_source": "derived from 1m via x - x%step, complete contiguous buckets only",
            "causality": "history bars close strictly before created_at; future starts at next bucket",
            "fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
            "basis": "notional",
            "rule_family": "negative-control only; no parameter search (H2)",
            "fixed_tp_pcts": list(fixed_grid),
            "class_thresholds": {"early_post_run_R": 1.0, "late_hold_mfe_R": 0.7},
        },
        "orders_in": len(orders), "skipped": skipped, "records": records,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[replay] source={args.source} usable={len(records)} "
          f"skipped={len(skipped)} runtime={payload['runtime_seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
