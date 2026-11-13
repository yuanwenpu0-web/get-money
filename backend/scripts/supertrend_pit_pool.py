r"""P17 / H-E · Supertrend「缩小币种池」的 PIT 可行性判定（双臂）。

预登记假设卡：`research/hypotheses/H-E-supertrend-pit-pool.md`（判据 C1–C5 事前写死）。

问题
----
全量池已证伪：`docs/supertrend-reversal-backtest.json` 500 币 ×1h×365 天、线上参数
factor2.5/r3.0 → 98738 笔 −0.228%/笔 CI[−0.2534,−0.202] negative，499 币中 107 币为正。
「缩小池」＝从这 107 个里挑，而这正是事后选币。本脚本回答：**存在不存在一个只用
T 时刻之前信息就能识别的币子集，让这套参数转正**。

做法
----
每币取数一次（全窗口 + 预热），按 fold 切片跑 `simulate`；成员构成是唯一变量：
- `all`        不选币（基线）
- `pit_liq_N`  事前 168h 成交额 TopN（可交易）
- `pit_vol_N`  事前 168h 成交额 Top100 内波动率 TopN（可交易）
- `posthoc_N`  **当期已实现收益** TopN（作弊臂，只用于量化偏差，不参与 gate）

所有臂共用同一份成交、同一套参数与成本，因此 `posthoc − pit` 的差就是泄漏的价值。

    .\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 1h  --days 365 --out docs/supertrend-pit-pool-1h
    .\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 15m --days 180 --out docs/supertrend-pit-pool-15m
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.backtest.execution import randomize_directions, simulate  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.research.pit_pool_arms import build_arms, summarise  # noqa: E402
from app.research.pit_pool_stats import arm_names, judge  # noqa: E402
from app.research.pit_universe import HOUR, rank_at  # noqa: E402
from app.strategy.coin_universe import BLACKLIST  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402
from supertrend_backtest import (  # noqa: E402
    CODE, DAY_MS, MIN_POOLED_TRADES, MIN_SEGMENT_BARS, load_segments,
)

WARMUP_BARS = 600
MIN_FOLD_BARS = 200
BASE_SEED = 20261025
COVERAGE = 0.95
_SESSION = None


def _db():
    global _SESSION
    if _SESSION is None:
        _SESSION = sessionmaker(bind=make_engine(read_only=True, null_pool=True),
                                autoflush=False, expire_on_commit=False, future=True)()
    return _SESSION


def evaluate_symbol(task: dict) -> dict:
    """一个币 = 一个任务：取数一次，逐 fold 切片撮合 + 多种子随机基线。"""
    symbol = task["symbol"]
    try:
        segments = load_segments(symbol, task["interval"], task["load_days"],
                                 task["end_ms"], MIN_SEGMENT_BARS)
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    if not segments:
        return {"symbol": symbol, "status": "skipped", "reason": "无满足长度的连续K线段"}

    params, cash = task["params"], task["cash"]
    fee, slip, interval = task["fee"], task["slippage"], task["interval"]
    warm = task["warmup_bars"] * interval_ms(interval)
    seeds = task["seeds"]
    folds = []
    for fold in task["folds"]:
        t0 = pd.Timestamp(fold["start"], unit="ms", tz="UTC")
        t1 = pd.Timestamp(fold["end"], unit="ms", tz="UTC")
        lo = pd.Timestamp(fold["start"] - warm, unit="ms", tz="UTC")
        trades: list[list] = []
        factor, bars, used = 1.0, 0, 0
        rnd = [1.0] * seeds
        for segment in segments:
            piece = segment.loc[(segment.index >= lo) & (segment.index < t1)]
            if len(piece) < MIN_FOLD_BARS:
                continue
            frame = prepare_frame(piece, CODE, params)
            signals = evaluate_signals(frame, CODE, params)
            equity, raw = simulate(frame, signals, params, cash, fee, slip, interval)
            eq = equity.loc[equity.index >= t0]
            if len(eq) < 2:
                continue
            used += 1
            bars += len(piece)
            # fold 账户收益取 T_f 起的权益比，排除预热段已实现盈亏
            factor *= float(eq.iloc[-1]) / float(eq.iloc[0])
            for trade in aggregate_positions(raw):
                if pd.Timestamp(trade["entry_time"]) >= t0:
                    trades.append([round(float(trade["return_pct"]), 4), trade["side"]])
            for k in range(seeds):
                rq, _ = simulate(frame, randomize_directions(signals, seed=BASE_SEED + k * 7919),
                                 params, cash, fee, slip, interval)
                rq = rq.loc[rq.index >= t0]
                if len(rq) >= 2:
                    rnd[k] *= float(rq.iloc[-1]) / float(rq.iloc[0])
        if not used:
            continue
        folds.append({"index": fold["index"], "trades": trades, "bars": bars,
                      "account_return_pct": round((factor - 1) * 100, 4),
                      "random_samples": [round((v - 1) * 100, 4) for v in rnd]})
    if not folds:
        return {"symbol": symbol, "status": "skipped", "reason": "各 fold 切片均不足最短根数"}
    return {"symbol": symbol, "status": "ok", "folds": folds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Supertrend PIT 选币可行性判定")
    ap.add_argument("--interval", default="1h", choices=("1h", "15m"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sizes", default="10,20,30")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--processes", type=int, default=0)
    ap.add_argument("--cash", type=float, default=10_000.0)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 币（冒烟）")
    ap.add_argument("--out", default="docs/supertrend-pit-pool")
    args = ap.parse_args(argv)

    sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
    processes = args.processes or max(1, min(24, (os.cpu_count() or 4) - 4))
    step = interval_ms(args.interval)
    db = sessionmaker(bind=make_engine(read_only=True), autoflush=False, future=True)()

    latest = db.execute(text("select max(open_time) from klines where interval=:i"),
                        {"i": args.interval}).scalar()
    end_ms = min(int(now_ms()), int(latest) + step) // DAY_MS * DAY_MS
    start_ms = end_ms - args.days * DAY_MS
    expected = args.days * DAY_MS // step
    rows = db.execute(text(
        "select symbol, count(*) c from klines where interval=:i "
        "and open_time>=:s and open_time<:e group by symbol"),
        {"i": args.interval, "s": start_ms, "e": end_ms}).all()
    qualified = sorted(r.symbol for r in rows if r.c >= COVERAGE * expected)
    symbols = qualified[: args.limit] if args.limit else qualified

    bounds = [start_ms + int(round(i * (end_ms - start_ms) / args.folds)) // step * step
              for i in range(args.folds + 1)]
    bounds[-1] = end_ms
    params = resolve_params(CODE, {"fee_rate": 0.0004, "slippage": 0.0002}, interval=args.interval)
    warm = WARMUP_BARS * step
    load_days = int(math.ceil((end_ms - start_ms + warm) / DAY_MS))

    fold_meta = []
    for i in range(args.folds):
        t0, t1 = bounds[i], bounds[i + 1]
        rank_rows = db.execute(text(
            "select symbol, open_time, close, quote_volume from klines "
            "where interval='1h' and open_time>=:s and open_time<:e"),
            {"s": t0 - 168 * HOUR, "e": t0}).all()
        ranked = rank_at(rank_rows, t0, excluded=BLACKLIST)
        fold_meta.append({"index": i, "start": t0, "end": t1,
                          "start_iso": datetime.fromtimestamp(t0 / 1000, timezone.utc).isoformat()[:10],
                          "end_iso": datetime.fromtimestamp(t1 / 1000, timezone.utc).isoformat()[:10],
                          "ranked": ranked, "eligible_count": len(ranked), "eligible": symbols})
        print(f"fold {i}: {fold_meta[-1]['start_iso']} → {fold_meta[-1]['end_iso']} · "
              f"PIT 合格 {len(ranked)} 币", flush=True)
    db.close()

    print(f"[PIT 选币] {args.interval} × {args.days} 天 | {len(symbols)} 币 × {args.folds} folds "
          f"| {processes} 进程 | {args.seeds} 种子 | 参数冻结 factor{params['supertrend_factor']}"
          f"/r{params['target_r']}", flush=True)
    tasks = [{"symbol": s, "interval": args.interval, "end_ms": end_ms, "load_days": load_days,
              "warmup_bars": WARMUP_BARS, "folds": [{"index": f["index"], "start": f["start"],
                                                    "end": f["end"]} for f in fold_meta],
              "params": params, "cash": args.cash, "fee": 0.0004, "slippage": 0.0002,
              "seeds": args.seeds} for s in symbols]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {pool.submit(evaluate_symbol, t): t["symbol"] for t in tasks}
        done = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"symbol": symbol, "status": "error",
                                "error": f"{type(exc).__name__}: {exc}"[:200]})
            done += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.monotonic() - started
                print(f"  {done}/{len(tasks)} 币 · {el:.0f}s", flush=True)
    elapsed = round(time.monotonic() - started, 1)
    results.sort(key=lambda r: r["symbol"])

    arms = build_arms(fold_meta, results, sizes, args.seeds)
    summary = summarise(arms, min_trades=MIN_POOLED_TRADES)
    verdicts = judge(summary, sizes)
    for meta in fold_meta:
        meta.pop("ranked", None)
        meta.pop("eligible", None)
    payload = {
        "name": "supertrend-pit-pool", "strategy": CODE,
        "hypothesis_card": "research/hypotheses/H-E-supertrend-pit-pool.md",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval, "days": args.days, "folds": args.folds,
        "window": {"start": datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat(),
                   "end": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat()},
        "universe": {"kind": "coverage_filtered", "requested": len(rows),
                     "qualified": len(qualified),
                     "evaluated": sum(1 for r in results if r["status"] == "ok"),
                     "coverage_threshold": COVERAGE, "expected_bars": expected,
                     "excluded": sorted(BLACKLIST),
                     "skipped": [f"{r['symbol']}: {r.get('reason') or r.get('error')}"
                                 for r in results if r["status"] != "ok"]},
        "parameters": params,
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "cash_per_symbol": args.cash,
                  "basis": "notional", "round_trip": 0.0012},
        "protocol": {"warmup_bars": WARMUP_BARS, "min_fold_bars": MIN_FOLD_BARS,
                     "min_segment_bars": MIN_SEGMENT_BARS, "sizes": list(sizes),
                     "seeds": args.seeds, "base_seed": BASE_SEED,
                     "min_pooled_trades": MIN_POOLED_TRADES,
                     "ranking_window_hours": 168, "ranking_interval": "1h",
                     "arms": arm_names(sizes)},
        "fold_meta": fold_meta, "processes": processes, "elapsed_seconds": elapsed,
        "summary": summary, "verdicts": verdicts,
    }
    from app.research.pit_pool_report import limitations, render
    payload["limitations"] = limitations(payload)
    payload["default_enabled"] = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
    out.with_suffix(".md").write_text(render(payload), encoding="utf-8")
    print(f"\noverall = {verdicts['overall']}")
    for name in arm_names(sizes):
        if name not in summary:
            continue
        ci = summary[name]["pooled"]
        print(f"  {name:14s} n={ci.get('trades', 0):6d} "
              f"mean={ci.get('mean_return_pct', 0.0):+8.4f}% "
              f"CI[{ci.get('ci_low_pct', 0.0):+8.4f},{ci.get('ci_high_pct', 0.0):+8.4f}] "
              f"{verdicts['arms'][name]['verdict']}")
    print(f"[输出] {out.with_suffix('.md')} / {out.with_suffix('.json')} · {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())