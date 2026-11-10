"""趋势反转入场 × 预注册卖出信号：半年全市场受控 A/B 的原始结果采集。

固定同一批入场点，只换出场规则；所有臂共享同一段未来 K 线、同一名义本金
成本口径。含 Gate 1 条件化 IC 决策点、20 种子匹配触发率随机早退基线、
20 种子随机方向基线。只读数据库，不改任何生产代码／绑定。

用法：
  .venv\\Scripts\\python.exe backend/scripts/trend_exit_study.py --days 183 --out output/trend_exit_study.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.models import StrategyConfig  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402
from trend_exit_common import (  # noqa: E402
    DECISION_BAR, FEE_RATE, FORWARD_H, HORIZON, MIN_COVERAGE, MIN_GAP,
    SELL_SIGNALS, SLIPPAGE, TRIGGER_Q, attach_features, build_entries,
    decision_point, first_trigger, mirror_barriers, net_return_pct,
    replay, replay_from_barrier,
)
from trend_exit_data import (  # noqa: E402
    align_xsec, btc_change, connect, eligible_symbols, funding_xsec, iso,
    load_frame, sentiment_xsec,
)

CODE = "supertrend_current"
DAY = 86_400_000


def _px_row(bars: np.ndarray, barrier: tuple, H: int) -> np.ndarray:
    """第 j 列 = 若信号在 bar j 触发时的出场价；第 H 列 = 不触发（基线）。"""
    px, bj, why = barrier
    cutoff = bj if why != "time" else bj + 1
    row = np.full(H + 1, px, dtype=float)
    for j in range(H):
        if j >= cutoff:
            row[j] = px
        elif j + 1 < H:
            row[j] = bars[j + 1, 0]
        else:
            row[j] = bars[j, 3]
    return row


def _worker(task):
    symbols, cfg = task
    from app.strategy import specs  # noqa: F401  导入副作用完成 spec 注册
    H = HORIZON
    eng = connect()
    params, interval = cfg["params"], cfg["interval"]
    per_bar = 3_600_000 if interval == "1h" else 900_000
    expected = (cfg["end_ms"] - cfg["start_ms"]) // per_bar
    t_start = pd.Timestamp(cfg["start_ms"], unit="ms", tz="UTC")
    records, skipped, bars_list, barriers, eps, px_rows = [], [], [], [], [], []

    with eng.connect() as conn:
        for sym in symbols:
            df = load_frame(conn, sym, interval, cfg["warm_ms"], cfg["end_ms"])
            if len(df) < 200:
                skipped.append({"symbol": sym, "reason": "insufficient_bars"})
                continue
            cov = int((df.index >= t_start).sum()) / max(1, expected)
            if cov < cfg["min_coverage"]:
                skipped.append({"symbol": sym, "reason": "low_coverage", "coverage": round(cov, 4)})
                continue
            try:
                frame = prepare_frame(df, CODE, params, symbol=sym)
                sig = evaluate_signals(frame, CODE, params, symbol=sym)
            except Exception as exc:                       # noqa: BLE001 记录后继续
                skipped.append({"symbol": sym, "reason": "signal_error", "detail": str(exc)[:120]})
                continue
            fpct, frate = align_xsec(frame.index, cfg["funding"].get(sym, {}),
                                     28_800_000, with_rate=True)
            opct = align_xsec(frame.index, cfg["sentiment"].get(sym, {}), 3_600_000)
            score, trig = attach_features(frame, fpct, frate, opct)
            entries = build_entries(frame, sig)
            if not entries:
                skipped.append({"symbol": sym, "reason": "no_long_entries"})
                continue
            ohlc = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
            closes = frame["close"].to_numpy(dtype=float)
            tarr = {n: trig[n].to_numpy(dtype=bool) for n in SELL_SIGNALS}
            for e in entries:
                i = e["entry_i"]
                bars = ohlc[i:i + H]
                if bars.shape[0] < H or not np.isfinite(bars).all():
                    continue
                b_long = replay(bars, e["sl"], e["tp"], 1)
                m = mirror_barriers(e)
                b_short = replay(bars, m["sl"], m["tp"], -1)
                ep = e["entry_price"]
                rule_net, trig_j = {}, {}
                for name in SELL_SIGNALS:
                    j = first_trigger(tarr[name], i)
                    px = replay_from_barrier(bars, b_long, j)
                    rule_net[name] = round(net_return_pct(ep, px[0], 1), 6)
                    trig_j[name] = j
                records.append({
                    "symbol": sym, "time_ms": e["time_ms"],
                    "baseline_net": round(net_return_pct(ep, b_long[0], 1), 6),
                    "short_net": round(net_return_pct(ep, b_short[0], -1), 6),
                    "baseline_reason": b_long[2], "rule_net": rule_net, "trigger_j": trig_j,
                    "decision": decision_point(score, i, closes),
                })
                bars_list.append(bars)
                barriers.append(b_long)
                eps.append(ep)
                px_rows.append(_px_row(bars, b_long, H))

    n = len(eps)
    rng = np.random.default_rng(cfg["seed"])
    random_early = {name: [] for name in SELL_SIGNALS}
    random_dir = []
    if n:
        eps_a = np.asarray(eps, dtype=float)
        base_a = np.array([r["baseline_net"] for r in records], dtype=float)
        short_a = np.array([r["short_net"] for r in records], dtype=float)
        px_tab = np.asarray(px_rows, dtype=float)
        idx = np.arange(n)
        for name in SELL_SIGNALS:
            js = np.array([H if r["trigger_j"][name] is None else r["trigger_j"][name]
                           for r in records], dtype=int)
            for _ in range(cfg["seeds"]):
                px = px_tab[idx, rng.permutation(js)]
                net = (px - eps_a) / eps_a - (FEE_RATE * (1 + px / eps_a) + 2 * SLIPPAGE)
                random_early[name].append(float(np.mean(net * 100.0 - base_a)))
        for _ in range(cfg["seeds"]):
            flip = rng.random(n) < 0.5
            random_dir.append(float(np.where(flip, short_a, base_a).mean()))
    eng.dispose()
    return {"records": records, "skipped": skipped, "random_early": random_early,
            "random_dir": random_dir, "n": n, "symbols": len(symbols)}


def _weighted(results, key, seeds, name=None):
    total = sum(r["n"] for r in results) or 1
    out = []
    for s in range(seeds):
        vals = [(r[key][name][s] if name else r[key][s]) * r["n"] for r in results]
        out.append(round(sum(vals) / total, 6))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=183)
    ap.add_argument("--interval", default="1h", choices=("1h", "15m"))
    ap.add_argument("--limit", type=int, default=0, help="调试用：只取前 N 个币")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default="output/trend_exit_study.json")
    args = ap.parse_args()
    t0 = time.time()

    end_ms = (now_ms() // DAY) * DAY
    start_ms = end_ms - args.days * DAY
    warm_ms = start_ms - 60 * DAY
    eng = connect()
    db = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False, future=True)()
    with eng.connect() as conn:
        symbols, skipped = eligible_symbols(conn, args.interval, warm_ms, start_ms,
                                           end_ms, MIN_COVERAGE, args.limit)
        funding, fund_meta = funding_xsec(conn, warm_ms, end_ms)
        sentiment, sent_meta = sentiment_xsec(conn, warm_ms, end_ms)
        btc, btc_bars = btc_change(conn, start_ms, end_ms)
    cfg_row = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).first()
    overrides = json.loads(cfg_row.params or "{}") if cfg_row else {}
    overrides = {k: v for k, v in overrides.items() if k not in ("fee_rate", "slippage")}
    params = resolve_params(CODE, all_state(db), overrides, interval=args.interval)
    db.close()

    workers = args.workers or max(1, min((os.cpu_count() or 8) - 4, 16))
    chunks = [list(c) for c in np.array_split(np.asarray(symbols, dtype=object), workers) if len(c)]
    base = {"params": params, "interval": args.interval, "warm_ms": warm_ms,
            "start_ms": start_ms, "end_ms": end_ms, "min_coverage": MIN_COVERAGE,
            "seeds": args.seeds, "seed": args.seed}
    tasks = []
    for ch in chunks:
        cfg = dict(base, funding={s: funding.get(s, {}) for s in ch},
                   sentiment={s: sentiment.get(s, {}) for s in ch})
        tasks.append((ch, cfg))
    print(f"[study] {len(symbols)} symbols / {workers} workers / {args.days}d "
          f"{iso(start_ms)} → {iso(end_ms)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker, tasks))

    records = [r for res in results for r in res["records"]]
    skipped += [s for res in results for s in res["skipped"]]
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": CODE, "interval": args.interval, "days": args.days,
        "window": {"start_ms": start_ms, "end_ms": end_ms,
                   "start": iso(start_ms), "end": iso(end_ms)},
        "universe": {"pool": f"native-{args.interval}-full-window",
                     "symbols": len(symbols), "skipped": skipped,
                     "symbols_with_trades": len({r["symbol"] for r in records})},
        "parameters": {"source": "global_state + StrategyConfig overrides (cost keys removed)",
                       "fee_rate": params["fee_rate"], "slippage": params["slippage"],
                       "sl_atr_mult": params.get("sl_atr_mult"),
                       "target_r": params.get("target_r"),
                       "supertrend_factor": params.get("supertrend_factor"),
                       "regime_filter": params.get("regime_filter"),
                       "cost_filter": params.get("cost_filter")},
        "costs": {"fee_rate": FEE_RATE, "slippage": SLIPPAGE,
                  "round_trip": 0.0012, "basis": "notional"},
        "data": {"funding": dict(fund_meta, first=iso(fund_meta["first_ms"]),
                                 last=iso(fund_meta["last_ms"])),
                 "sentiment": dict(sent_meta, first=iso(sent_meta["first_ms"]),
                                   last=iso(sent_meta["last_ms"])),
                 "btc_change_pct": btc, "btc_bars_in_window": btc_bars,
                 "tail_fill": "native 1h + 1m aggregated tail buckets (native rows win)",
                 "native_window_rows_before_fill": None},
        "config": {"horizon": HORIZON, "forward_h": FORWARD_H, "decision_bar": DECISION_BAR,
                   "min_gap": MIN_GAP, "trigger_q": TRIGGER_Q,
                   "min_coverage": MIN_COVERAGE, "seeds": args.seeds,
                   "seed": args.seed, "workers": workers},
        "signals": list(SELL_SIGNALS), "n_records": len(records),
        "random_early": {n: _weighted(results, "random_early", args.seeds, n)
                         for n in SELL_SIGNALS},
        "random_dir": _weighted(results, "random_dir", args.seeds),
        "records": records, "runtime_seconds": round(time.time() - t0, 2),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[study] records={len(records)} symbols_with_trades="
          f"{out['universe']['symbols_with_trades']} skipped={len(skipped)} "
          f"runtime={out['runtime_seconds']}s → {path}", flush=True)


if __name__ == "__main__":
    main()
