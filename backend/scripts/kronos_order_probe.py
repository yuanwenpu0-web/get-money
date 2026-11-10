"""Offline Kronos order-exit probe: can a pre-entry forecast tell if TP is reached?

Read-only with respect to the trading system. For every closed order we cut a
strictly causal K-line window at the entry timestamp, sample N future paths
from Kronos, and score each path's favorable excursion toward TP. Labels come
from replaying the real subsequent bars with the identical triple-barrier rule.

Outputs a per-order JSON table consumed by kronos_order_eval.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
sys.path.insert(0, str(ROOT / ".workbuddy" / "tools" / "kronos"))

from app.db import make_engine  # noqa: E402
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
from kronos_order_common import (  # noqa: E402
    STEP_MS, HORIZON_BARS, TP_REASONS, replay_real, path_features, net_return_pct,
)

MODEL_REPO = "NeoQuasar/Kronos-small"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"

# 1m -> derived interval bars, exactly the same bucket rule as data_service._candle_query
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
SELECT id, symbol, interval, side, source, strategy, close_reason,
       entry_price, sl, tp, close_price, realized_pnl, margin, leverage,
       created_at, closed_at
FROM orders
WHERE status = 'CLOSED' AND entry_price > 0 AND tp > 0 AND sl > 0
  AND close_price > 0 AND closed_at IS NOT NULL
ORDER BY created_at, id
"""


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only first N orders")
    ap.add_argument("--source", default="", help="filter by order source, e.g. live")
    ap.add_argument("--horizon-bars", type=int, default=0,
                    help="override horizon bars for all intervals (short window for recent orders)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="output/kronos_order_probe.json")
    args = ap.parse_args()

    torch.set_num_threads(max(2, min(8, (os.cpu_count() or 8) // 3)))
    t_start = time.perf_counter()
    engine = make_engine(read_only=True)
    if args.horizon_bars:
        for k in HORIZON_BARS:
            HORIZON_BARS[k] = args.horizon_bars

    with engine.connect() as conn:
        orders = [dict(r) for r in conn.execute(text(ORDER_SQL)).mappings().all()]
        orders = [o for o in orders if o["interval"] in STEP_MS
                  and (not args.source or o["source"] == args.source)]
        if args.limit:
            orders = orders[: args.limit]

        skipped, samples = [], []
        groups: dict[tuple[str, str], list] = {}
        for o in orders:
            groups.setdefault((o["symbol"], o["interval"]), []).append(o)

        # Prefetch derived candles once per (symbol, interval) group.
        cache = {}
        for (sym, interval), group in groups.items():
            step = STEP_MS[interval]
            horizon = HORIZON_BARS[interval]
            buckets = [to_ms(o["created_at"]) // step * step for o in group]
            lo = min(buckets) - (args.lookback + 3) * step
            hi = max(buckets) + (horizon + 3) * step
            rows = conn.execute(text(CANDLE_SQL), {
                "sym": sym, "step": step, "minutes": step // 60_000,
                "start_ms": lo, "end_ms": hi,
            }).mappings().all()
            cache[(sym, interval)] = [dict(r) for r in rows]

    for o in orders:
        interval, step = o["interval"], STEP_MS[o["interval"]]
        horizon = HORIZON_BARS[interval]
        ob = to_ms(o["created_at"]) // step * step
        candles = cache[(o["symbol"], interval)]
        hist = [c for c in candles if c["ot"] < ob][-args.lookback:]
        expected_h = [ob - step * (args.lookback - i) for i in range(args.lookback)]
        future = [c for c in candles if ob + step <= c["ot"] <= ob + horizon * step][:horizon]
        expected_f = [ob + step * (i + 1) for i in range(horizon)]

        if len(hist) < args.lookback or [c["ot"] for c in hist] != expected_h:
            skipped.append({"id": o["id"], "reason": "incomplete_history"})
            continue
        if len(future) < horizon or [c["ot"] for c in future] != expected_f:
            skipped.append({"id": o["id"], "reason": "incomplete_future"})
            continue
        # Hard causality: last historical bar closed strictly before the order.
        if hist[-1]["ot"] + step - 1 >= to_ms(o["created_at"]):
            skipped.append({"id": o["id"], "reason": "lookahead_in_history"})
            continue

        hist_df = pd.DataFrame(hist)
        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")
        if hist_df[["open", "high", "low", "close"]].isnull().values.any():
            skipped.append({"id": o["id"], "reason": "nan_history"})
            continue
        x_df = hist_df[["open", "high", "low", "close"]].copy()
        x_df["volume"] = hist_df["volume"].fillna(0.0)
        x_df["amount"] = hist_df["quote_volume"].fillna(0.0)
        x_ts = pd.to_datetime(hist_df["ot"], unit="ms")
        y_ts = pd.Series(pd.to_datetime([c["ot"] for c in future], unit="ms"))

        real = replay_real(future, o["side"], o["entry_price"], o["tp"], o["sl"])
        # Past bar shapes ONLY (no future bar): the random-walk control must
        # not get any information that was unavailable at order time.
        ratios, prev = [], float(hist[0]["close"])
        for c in hist[1:]:
            if prev > 0:
                ratios.append([max(float(c["high"]), float(c["open"]), float(c["close"])) / prev,
                               min(float(c["low"]), float(c["open"]), float(c["close"])) / prev,
                               float(c["close"]) / prev])
            prev = float(c["close"])

        samples.append({
            "order": o, "x_df": x_df, "x_ts": x_ts, "y_ts": y_ts,
            "label": real, "label_reason": int(o["close_reason"] in TP_REASONS),
            "hist_ratios": ratios, "ref_close": float(hist[-1]["close"]),
            "horizon": horizon,
        })

    print(f"[probe] usable={len(samples)} skipped={len(skipped)} "
          f"orders={len(orders)} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

    print("[probe] loading Kronos weights ...", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO)
    model = Kronos.from_pretrained(MODEL_REPO)
    predictor = KronosPredictor(model, tokenizer, max_context=512, clip=5)
    predictor.model.eval()
    predictor.tokenizer.eval()

    for s in samples:
        s["hits"], s["mfe"], s["end"] = [], [], []
        s["ohlc_violations"], s["resolved_paths"] = 0, 0

    # predict_batch averages sample_count internally -> call with 1 and repeat
    # with fresh seeds to keep individual paths instead of a smoothed mean.
    by_group: dict[str, list] = {}
    for idx, s in enumerate(samples):
        by_group.setdefault(s["order"]["interval"], []).append(idx)

    with torch.no_grad():
        for seed_i in range(args.samples):
            torch.manual_seed(args.seed + seed_i * 7919)
            for interval, idxs in by_group.items():
                pred_len = HORIZON_BARS[interval]
                for start in range(0, len(idxs), args.batch_size):
                    chunk = idxs[start:start + args.batch_size]
                    preds = predictor.predict_batch(
                        df_list=[samples[i]["x_df"] for i in chunk],
                        x_timestamp_list=[samples[i]["x_ts"] for i in chunk],
                        y_timestamp_list=[samples[i]["y_ts"] for i in chunk],
                        pred_len=pred_len, T=args.temperature, top_p=args.top_p,
                        sample_count=1, verbose=False,
                    )
                    for i, pdf in zip(chunk, preds):
                        f = path_features(pdf, samples[i]["ref_close"],
                                          samples[i]["order"]["side"],
                                          samples[i]["order"]["entry_price"],
                                          samples[i]["order"]["tp"],
                                          samples[i]["order"]["sl"])
                        if f is None:
                            continue
                        samples[i]["hits"].append(f["hit"])
                        samples[i]["mfe"].append(f["mfe_ratio"])
                        samples[i]["end"].append(f["end_ratio"])
                        samples[i]["ohlc_violations"] += f["violations"]
                        samples[i]["resolved_paths"] += f["resolved"]
            print(f"[probe] seed {seed_i+1}/{args.samples} done "
                  f"elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

    for s in samples:
        s["score_hit"] = float(np.mean(s["hits"])) if s["hits"] else 0.0
        s["score_mfe"] = float(np.mean(s["mfe"])) if s["mfe"] else 0.0
        s["score_end"] = float(np.mean(s["end"])) if s["end"] else 0.0

    commit = ROOT / ".workbuddy" / "tools" / "kronos" / "COMMIT.txt"
    payload = {
        "config": {
            "model_repo": MODEL_REPO, "tokenizer_repo": TOKENIZER_REPO,
            "kronos_commit": commit.read_text().strip() if commit.exists() else "unknown",
            "lookback": args.lookback, "horizon_bars": HORIZON_BARS,
            "samples": args.samples, "seed": args.seed,
            "temperature": args.temperature, "top_p": args.top_p,
            "device": "cpu", "batch_size": args.batch_size,
            "causality": "last historical bar close_time < created_at; future starts at next bucket",
            "candle_source": "derived from 1m via x - x%step, complete contiguous buckets only",
            "fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
            "basis": "notional",
        },
        "skipped": skipped,
        "records": [{
            "id": s["order"]["id"], "symbol": s["order"]["symbol"],
            "interval": s["order"]["interval"], "side": s["order"]["side"],
            "source": s["order"]["source"], "strategy": s["order"]["strategy"],
            "close_reason": s["order"]["close_reason"],
            "created_at": s["order"]["created_at"].isoformat(),
            "closed_at": s["order"]["closed_at"].isoformat(),
            "entry_price": s["order"]["entry_price"], "tp": s["order"]["tp"],
            "sl": s["order"]["sl"], "close_price": s["order"]["close_price"],
            "realized_pnl": s["order"]["realized_pnl"],
            "label_path": s["label"]["label"], "label_mfe": s["label"]["mfe_ratio"],
            "label_end": s["label"]["end_ratio"], "label_reason": s["label_reason"],
            "kronos_hit": s["score_hit"], "kronos_mfe": s["score_mfe"],
            "kronos_end": s["score_end"], "path_hits": s["hits"],
            "ohlc_violations": s["ohlc_violations"],
            "resolved_paths": s["resolved_paths"], "horizon": s["horizon"],
            "net_return_pct": net_return_pct({
                "entry_price": s["order"]["entry_price"],
                "close_price": s["order"]["close_price"], "side": s["order"]["side"]}),
            "hist_ratios": s["hist_ratios"],
        } for s in samples],
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[probe] wrote {out} | runtime={payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
