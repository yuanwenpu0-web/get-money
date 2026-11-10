"""Build the offline, strictly-causal conditional-exit meta-label dataset.

Read-only. Reuses the exact bar/bucket/cost conventions of
order_exit_replay.py so T0 labels are comparable with the previous round.
No production strategy code, bindings or params are touched.
"""
from __future__ import annotations

import argparse
import json
import re
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
    MARKET_EXIT_REASONS, net_return_pct, atr, run_rule, build_rules,
)
from exit_metalabel_features import (  # noqa: E402
    t0_features, t1_features, t1_label, t0_label, find_decision_bar,
)

STEP_MS = {"15m": 900_000, "1h": 3_600_000}

CANDLE_SQL = """
WITH m AS (
  SELECT open_time, open, high, low, close, volume, quote_volume
  FROM klines WHERE symbol = :sym AND interval = '1m'
    AND open_time >= :start_ms AND open_time < :end_ms
), b AS (
  SELECT (open_time - open_time % :step) AS ot, count(*) AS n,
         min(open_time) AS first_t, max(open_time) AS last_t,
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
       entry_price, sl, tp, close_price, realized_pnl, note,
       created_at, closed_at
FROM orders
WHERE status = 'CLOSED' AND entry_price > 0 AND tp > 0 AND sl > 0
  AND close_price > 0 AND closed_at IS NOT NULL
  AND (:source = '' OR source = :source)
ORDER BY created_at, id
"""

SENT_SQL = """SELECT ts, oi_value, long_short_ratio, top_long_short,
                     taker_buy_sell FROM sentiment
              WHERE symbol = :sym ORDER BY ts"""
FUND_SQL = """SELECT funding_time, funding_rate FROM funding_rates
              WHERE symbol = :sym ORDER BY funding_time"""
SENT_ALL_SQL = "SELECT ts, oi_value FROM sentiment WHERE oi_value IS NOT NULL"

# H8: a "latest row" from weeks ago is stale, not coverage. sentiment is 1h
# and funding is 8h; anything older than 24h is treated as MISSING.
MAX_AGE_MS = 24 * 3_600_000

NOTE_SCORE = re.compile(r"score=([0-9.]+)")
NOTE_PROB = re.compile(r"prob_up=([0-9.]+)")


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def _latest_before(rows, cut_ms, key):
    """Last row with timestamp strictly < cut_ms (rows sorted ascending)."""
    arr = np.array([int(r[key]) for r in rows], dtype=np.int64)
    i = int(np.searchsorted(arr, cut_ms, side="left")) - 1
    return rows[i] if i >= 0 else None


def _sentiment_features(row, rows, cut_ms, xsec):
    """P9-direction features: 30d z-score + cross-sectional percentile."""
    if row is None or cut_ms - int(row["ts"]) > MAX_AGE_MS:
        return None
    ts = int(row["ts"])
    arr = np.array([int(r["ts"]) for r in rows], dtype=np.int64)
    lo = np.searchsorted(arr, ts - 30 * 86_400_000, side="left")
    hi = np.searchsorted(arr, ts, side="left")
    hist = np.array([float(r["oi_value"]) for r in rows[lo:hi]], dtype=float)
    mu, sd = (float(hist.mean()), float(hist.std())) if len(hist) > 2 else (0.0, 0.0)
    z = (float(row["oi_value"]) - mu) / sd if sd > 1e-12 else 0.0
    cross = xsec.get(ts)
    pct = (float(np.mean(cross <= float(row["oi_value"])))
           if cross is not None and len(cross) >= 20 else float("nan"))
    return {
        "sent_oi_z30": z, "sent_oi_xsec_pct": pct,
        "sent_ls_ratio": float(row["long_short_ratio"] or 0.0),
        "sent_top_ls": float(row["top_long_short"] or 0.0),
        "sent_taker_bs": float(row["taker_buy_sell"] or 0.0),
        "sent_age_hours": (cut_ms - ts) / 3_600_000,
    }


def _funding_features(row, rows, cut_ms):
    if row is None or cut_ms - int(row["funding_time"]) > MAX_AGE_MS:
        return None
    ts = int(row["funding_time"])
    arr = np.array([int(r["funding_time"]) for r in rows], dtype=np.int64)
    lo = np.searchsorted(arr, ts - 30 * 86_400_000, side="left")
    hi = np.searchsorted(arr, ts, side="left")
    hist = np.array([float(r["funding_rate"]) for r in rows[lo:hi]], dtype=float)
    mu, sd = (float(hist.mean()), float(hist.std())) if len(hist) > 2 else (0.0, 0.0)
    rate = float(row["funding_rate"] or 0.0)
    return {"fund_rate": rate,
            "fund_z30": (rate - mu) / sd if sd > 1e-12 else 0.0,
            "fund_age_hours": (cut_ms - ts) / 3_600_000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="paper", help="paper | live")
    ap.add_argument("--horizon-hours", type=float, default=72.0)
    ap.add_argument("--ab-hours", type=float, default=24.0)
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="output/exit_metalabel_paper.json")
    args = ap.parse_args()

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
        symbols = sorted({s for s, _ in groups})
        candles, sent, fund = {}, {}, {}
        xsec: dict[int, list[float]] = {}
        for r in conn.execute(text(SENT_ALL_SQL)).mappings().all():
            xsec.setdefault(int(r["ts"]), []).append(float(r["oi_value"]))
        xsec = {k: np.asarray(v, dtype=float) for k, v in xsec.items()}
        for sym in symbols:
            sent[sym] = [dict(r) for r in conn.execute(
                text(SENT_SQL), {"sym": sym}).mappings().all()]
            fund[sym] = [dict(r) for r in conn.execute(
                text(FUND_SQL), {"sym": sym}).mappings().all()]
        for (sym, interval), group in groups.items():
            step = STEP_MS[interval]
            hb = max(1, int(args.horizon_hours * 3_600_000 / step))
            buckets = [to_ms(o["created_at"]) // step * step for o in group]
            rows = conn.execute(text(CANDLE_SQL), {
                "sym": sym, "step": step, "minutes": step // 60_000,
                "start_ms": min(buckets) - (args.lookback + 3) * step,
                "end_ms": max(buckets) + (hb + 3) * step,
            }).mappings().all()
            candles[(sym, interval)] = [dict(r) for r in rows]

    skipped, records = [], []
    for o in orders:
        interval, step = o["interval"], STEP_MS[o["interval"]]
        hb = max(1, int(args.horizon_hours * 3_600_000 / step))
        ob = to_ms(o["created_at"]) // step * step
        cut = to_ms(o["created_at"])
        cs = candles[(o["symbol"], interval)]
        hist = [c for c in cs if c["ot"] + step <= cut][-args.lookback:]
        future = [c for c in cs if ob + step <= c["ot"] <= ob + hb * step][:hb]
        if len(hist) < max(16, args.lookback // 4):
            skipped.append({"id": o["id"], "reason": "insufficient_history"}); continue
        if not future or future[0]["ot"] != ob + step:
            skipped.append({"id": o["id"], "reason": "missing_first_future_bar"}); continue
        if len(future) / hb < args.min_coverage:
            skipped.append({"id": o["id"], "reason": "low_future_coverage"}); continue
        if any(any(float(c[k]) <= 0 for k in ("open", "high", "low", "close"))
               for c in hist + future):
            skipped.append({"id": o["id"], "reason": "nonpositive_price"}); continue
        entry, tp, sl = (float(o["entry_price"]), float(o["tp"]), float(o["sl"]))
        side = o["side"]
        if abs(entry - sl) <= 0 or abs(tp - entry) <= 0:
            skipped.append({"id": o["id"], "reason": "degenerate_barriers"}); continue
        if (side == "LONG" and sl >= entry) or (side == "SHORT" and sl <= entry):
            skipped.append({"id": o["id"], "reason": "sl_not_adverse"}); continue
        if (side == "LONG" and tp <= entry) or (side == "SHORT" and tp >= entry):
            skipped.append({"id": o["id"], "reason": "tp_not_favorable"}); continue
        ab_n = max(1, int(args.ab_hours * 3_600_000 / step))
        ab = future[:max(1, min(len(future), ab_n))]
        rules = build_rules(entry, tp, sl, atr(hist), len(ab), side)
        rule_net = {r["name"]: net_return_pct(entry, run_rule(
            ab, side, entry, r["sl"], r)[0], side) for r in rules}
        y0 = t0_label(rule_net)
        if y0 is None:
            skipped.append({"id": o["id"], "reason": "missing_t0_label"}); continue
        note = o.get("note") or ""
        sm, pm = NOTE_SCORE.search(note), NOTE_PROB.search(note)
        meta = {"side": side, "entry_price": entry, "sl": sl, "tp": tp,
                "interval": interval, "hour_of_day": o["created_at"].hour,
                "score": float(sm.group(1)) if sm else float("nan"),
                "prob_up": float(pm.group(1)) if pm else float("nan")}
        sr = _latest_before(sent[o["symbol"]], cut, "ts")
        fr = _latest_before(fund[o["symbol"]], cut, "funding_time")
        sf = _sentiment_features(sr, sent[o["symbol"]], cut, xsec)
        ff = _funding_features(fr, fund[o["symbol"]], cut)
        X0 = t0_features(hist, meta, sf, ff)
        k = find_decision_bar(ab, side, entry, sl, tp)
        X1 = t1_features(ab, k, side, entry, sl, tp) if k is not None else {}
        y1 = t1_label(ab, k, side, entry, sl, tp) if k is not None else None
        # LOOKAHEAD quantities used ONLY for the positive-control self-calibration
        # of the measuring instrument (judgement.md §3 step 5); never a feature.
        R = abs(entry - sl)
        fav = [float(b["high"] if side == "LONG" else b["low"]) for b in ab]
        oracle_mfe = ((max(fav) - entry) if side == "LONG"
                      else (entry - min(fav))) / R
        post_mfe = None
        if k is not None and k + 1 < len(ab):
            pf = fav[k + 1:]
            post_mfe = ((max(pf) - entry) if side == "LONG"
                        else (entry - min(pf))) / R
        records.append({
            "id": o["id"], "symbol": o["symbol"], "interval": interval,
            "side": side, "source": o["source"], "strategy": o["strategy"],
            "close_reason": o["close_reason"],
            "created_at": o["created_at"].isoformat(),
            "market_exit": o["close_reason"] in MARKET_EXIT_REASONS,
            "real_net_pct": net_return_pct(entry, float(o["close_price"]), side),
            "baseline_net_pct": rule_net["baseline"],
            "rule_net_pct": rule_net,
            "X_t0": X0, "X_t1": X1,
            "y0": y0["y0"], "late_better": y0["late_better"],
            "early_net_pct": y0["early_net_pct"], "late_net_pct": y0["late_net_pct"],
            "y1": (y1 or {}).get("y1"), "hold_better": (y1 or {}).get("hold_better"),
            "exit_now_net_pct": (y1 or {}).get("exit_now_net_pct"),
            "hold_net_pct": (y1 or {}).get("hold_net_pct"),
            "decision_bar": k, "t1_usable": y1 is not None,
            "LOOKAHEAD_oracle_mfe_R": oracle_mfe,
            "LOOKAHEAD_post_mfe_R": post_mfe,
            "has_sentiment": sf is not None, "has_funding": ff is not None,
            "sent_age_hours": (sf or {}).get("sent_age_hours"),
            "fund_age_hours": (ff or {}).get("fund_age_hours"),
        })

    payload = {
        "config": {
            "source": args.source, "horizon_hours": args.horizon_hours,
            "ab_hours": args.ab_hours, "lookback": args.lookback,
            "min_coverage": args.min_coverage,
            "candle_source": "derived from 1m via x - x%step, complete contiguous buckets only",
            "causality": "T0 features use bars closing strictly before created_at; "
                         "T1 features use bars[0..k] only; labels use future bars",
            "fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
            "basis": "notional",
            "derivative_max_age_hours": MAX_AGE_MS / 3_600_000,
            "note": "Gate-1 probe only; no rule implementation, no parameter grid (H1/H2)",
        },
        "orders_in": len(orders), "skipped": skipped, "records": records,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    t1n = sum(1 for r in records if r["t1_usable"])
    print(f"[metalabel] source={args.source} records={len(records)} "
          f"t1_usable={t1n} skipped={len(skipped)} "
          f"runtime={payload['runtime_seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
