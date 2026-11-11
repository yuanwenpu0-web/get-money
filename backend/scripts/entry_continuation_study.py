"""入场后 1/5/10 分钟「趋势是否延续」的受控 A/B（第一阶段：近 90 天线上币池）。

基准臂：1h supertrend_current 多头入场，1.5×ATR 止损 / 3R 止盈，全仓单目标。
处理臂：入场后第 1/5/10 分钟各合成一根观察 bar；
        该 bar 收盘沿方向优于入场价且未触 SL = 延续 -> 沿用基准出场；
        否则在该 bar 收盘提前平仓。
所有臂共享同一批入场点、同一段 1h 未来 K 线、同一名义本金成本 0.12%。
只读数据库；不改任何生产代码 / spec / 绑定。

用法：
  .venv\\Scripts\\python.exe backend/scripts/entry_continuation_study.py \\
      --days 90 --out output/entry_continuation_study.json
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
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.models import StrategyConfig  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402
from entry_continuation_common import (  # noqa: E402
    MINUTE, OBS_MINUTES, judge, obs_bar,
)
from trend_exit_common import (  # noqa: E402
    FEE_RATE, HORIZON, MIN_COVERAGE, MIN_GAP, SLIPPAGE, build_entries,
    net_return_pct, replay,
)
from trend_exit_data import connect, eligible_symbols, iso, load_frame  # noqa: E402

CODE = "supertrend_current"
DAY = 86_400_000
PAD_MINUTES = max(OBS_MINUTES) + 2      # 观察窗 + 余量，用于连续性校验

M1_SQL = """
SELECT open_time, open, high, low, close
FROM klines WHERE symbol=:sym AND interval='1m'
  AND open_time >= :a AND open_time < :b
ORDER BY open_time
"""


def _worker(task):
    symbols, cfg = task
    from app.strategy import specs  # noqa: F401  导入副作用完成 spec 注册
    eng = connect()
    params = cfg["params"]
    records, skipped = [], []
    with eng.connect() as conn:
        for sym in symbols:
            df = load_frame(conn, sym, cfg["interval"], cfg["warm_ms"], cfg["end_ms"])
            if len(df) < 200:
                skipped.append({"symbol": sym, "reason": "insufficient_1h_bars"})
                continue
            try:
                frame = prepare_frame(df, CODE, params, symbol=sym)
                sig = evaluate_signals(frame, CODE, params, symbol=sym)
            except Exception as exc:                      # noqa: BLE001
                skipped.append({"symbol": sym, "reason": "signal_error",
                                "detail": str(exc)[:120]})
                continue
            entries = build_entries(frame, sig)
            if not entries:
                skipped.append({"symbol": sym, "reason": "no_long_entries"})
                continue
            ohlc = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
            for e in entries:
                i = e["entry_i"]
                bars = ohlc[i:i + HORIZON]
                if bars.shape[0] < HORIZON or not np.isfinite(bars).all():
                    continue
                ep, sl, tp = e["entry_price"], e["sl"], e["tp"]
                base_px, base_j, base_why = replay(bars, sl, tp, 1)
                base_net = net_return_pct(ep, base_px)
                t0 = int(e["time_ms"])
                rows = [dict(r) for r in conn.execute(
                    text(M1_SQL), {"sym": sym, "a": t0,
                                   "b": t0 + PAD_MINUTES * MINUTE}).mappings().all()]
                arms = {}
                for k in OBS_MINUTES:
                    bar = obs_bar(rows, k, t0)
                    if bar is None:
                        continue
                    px, why = judge(bar, ep, sl, base_px)
                    arms[str(k)] = {"net": net_return_pct(ep, px), "reason": why,
                                    "obs_close": bar["close"]}
                if not arms:
                    skipped.append({"symbol": sym, "reason": "no_1m_window",
                                    "time_ms": t0})
                    continue
                records.append({
                    "symbol": sym, "time_ms": t0, "entry_price": ep,
                    "sl": sl, "tp": tp, "base_net": base_net,
                    "base_reason": base_why, "base_bar": int(base_j),
                    "arms": arms,
                    "sl_pct": (ep - sl) / ep * 100.0,
                    "tp_pct": (tp - ep) / ep * 100.0,
                })
    eng.dispose()
    return {"records": records, "skipped": skipped, "symbols": len(symbols)}


def _random_early(records, cycles, seeds, seed):
    """匹配实际早退率的随机早退基线：同周期、同观察窗收盘价，只随机选谁早退。"""
    rng = np.random.default_rng(seed)
    out = {}
    for k in cycles:
        sub = [r for r in records if str(k) in r["arms"]]
        if not sub:
            out[str(k)] = []
            continue
        base = np.array([r["base_net"] for r in sub], dtype=float)
        early = np.array([r["arms"][str(k)]["net"] for r in sub], dtype=float)
        p = float(np.mean([r["arms"][str(k)]["reason"] != "continuation" for r in sub]))
        means = []
        for _ in range(seeds):
            mask = rng.random(len(sub)) < p
            means.append(float(np.where(mask, early, base).mean()))
        out[str(k)] = means
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-1m-coverage", type=float, default=0.88,
                    help="1m 在窗口内的覆盖率下限（相对满分钟数），决定币池")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default="output/entry_continuation_study.json")
    args = ap.parse_args()
    t0 = time.time()

    end_ms = (now_ms() // DAY) * DAY
    start_ms = end_ms - args.days * DAY
    warm_ms = start_ms - 60 * DAY
    eng = connect()
    db = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False, future=True)()
    with eng.connect() as conn:
        base_syms, skipped = eligible_symbols(conn, args.interval, warm_ms, start_ms,
                                             end_ms, MIN_COVERAGE, 0)
        need = args.min_1m_coverage * args.days * 1440
        cov = {r["symbol"]: int(r["n"]) for r in conn.execute(text("""
            SELECT symbol, count(*) AS n FROM klines
            WHERE interval='1m' AND open_time >= :a AND open_time < :b
            GROUP BY symbol"""), {"a": start_ms, "b": end_ms}).mappings().all()}
    symbols = sorted([s for s in base_syms if cov.get(s, 0) >= need])
    for s in base_syms:
        if s not in symbols:
            skipped.append({"symbol": s, "reason": "insufficient_1m_coverage",
                            "days": round(cov.get(s, 0) / 1440, 1)})
    if args.limit:
        symbols = symbols[:args.limit]

    cfg_row = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).first()
    overrides = json.loads(cfg_row.params or "{}") if cfg_row else {}
    overrides = {k: v for k, v in overrides.items() if k not in ("fee_rate", "slippage")}
    params = resolve_params(CODE, all_state(db), overrides, interval=args.interval)
    db.close()

    workers = args.workers or max(1, min((os.cpu_count() or 8) - 4, 16))
    chunks = [list(c) for c in np.array_split(np.asarray(symbols, dtype=object), workers)
              if len(c)]
    base = {"params": params, "interval": args.interval, "warm_ms": warm_ms,
            "end_ms": end_ms}
    print(f"[cont] {len(symbols)} symbols / {workers} workers / {args.days}d "
          f"{iso(start_ms)} -> {iso(end_ms)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker, [(ch, base) for ch in chunks]))

    records = sorted([r for res in results for r in res["records"]],
                     key=lambda r: r["time_ms"])
    skipped += [s for res in results for s in res["skipped"]]
    btc = None
    with eng.connect() as conn:
        try:
            bdf = load_frame(conn, "BTCUSDT", "1h", start_ms, end_ms)
            if len(bdf) > 1:
                btc = round(float(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1) * 100, 4)
        except Exception:                                  # noqa: BLE001
            btc = None
    eng.dispose()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": CODE, "interval": args.interval, "days": args.days,
        "window": {"start_ms": start_ms, "end_ms": end_ms,
                   "start": iso(start_ms), "end": iso(end_ms)},
        "costs": {"fee_rate": FEE_RATE, "slippage": SLIPPAGE,
                  "round_trip": 0.0012, "basis": "notional"},
        "parameters": {"source": "global_state + StrategyConfig overrides (cost keys removed)",
                       "sl_atr_mult": params.get("sl_atr_mult"),
                       "target_r": params.get("target_r"),
                       "supertrend_factor": params.get("supertrend_factor"),
                       "regime_filter": params.get("regime_filter"),
                       "cost_filter": params.get("cost_filter")},
        "universe": {"symbols": len(symbols), "skipped": skipped,
                     "symbols_with_entries": len({r["symbol"] for r in records}),
                     "min_1m_coverage": args.min_1m_coverage,
                     "selection": "1h full-window coverage >= 90% AND 1m coverage >= "
                                  f"{args.min_1m_coverage:.0%} of window minutes "
                                  "(all qualifying symbols, no sampling)"},
        "config": {"horizon_1h": HORIZON, "min_gap_1h": MIN_GAP,
                   "obs_minutes": list(OBS_MINUTES), "seeds": args.seeds,
                   "seed": args.seed, "min_coverage": args.min_coverage},
        "btc_change_pct": btc, "n_records": len(records),
        "random_early": _random_early(records, OBS_MINUTES, args.seeds, args.seed),
        "records": records, "runtime_seconds": round(time.time() - t0, 2),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[cont] entries={len(records)} symbols_with_entries="
          f"{out['universe']['symbols_with_entries']} skipped={len(skipped)} "
          f"runtime={out['runtime_seconds']}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
