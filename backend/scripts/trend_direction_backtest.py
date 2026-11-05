"""趋势方向指标横评主脚本（1h K 线 / 近两日方向验证 / 分币种池最优）。

默认行为会先检查 1h K 线是否覆盖到「最后一个完整 UTC 小时」，
不足时自动调用 ``sync_recent_1h.py`` 去交易所取数（--no-sync 可关闭）。

用法：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/trend_direction_backtest.py
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/trend_direction_backtest.py \
        --days 180 --horizon 48 --workers 8 --out-prefix docs/trend-direction-backtest
    # 冒烟：只跑 12 个币
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/trend_direction_backtest.py --limit 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.models import Kline, SymbolPool  # noqa: E402
from app.strategy.pools import load_pool_spec, resolve_members  # noqa: E402
from app.research.trend_direction import (  # noqa: E402
    DEFAULT_COST, DEFAULT_HORIZON, EXCLUDED, SPECS, WARMUP_BARS,
    compute_directions, evaluate_symbol,
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
HOUR_NS = 3_600_000_000_000
CATALOG_TOTAL = len(SPECS) + len(EXCLUDED)

_WORKER: dict = {}


def _init_worker(start_ms: int, last_complete: int, horizon: int,
                 cost: float, days: int) -> None:
    """每个进程只建一次 engine，并预热 156 个指标的注册表。"""
    _WORKER["engine"] = make_engine(read_only=True)
    _WORKER["cfg"] = {"start_ms": start_ms, "last_complete": last_complete,
                      "horizon": horizon, "cost": cost, "days": days}
    import app.catalog_indicators  # noqa: F401  spawn 下预热 156 个指标模块的注册表


def _run_symbol(symbol: str) -> dict:
    import numpy as np
    import pandas as pd
    from app.models import Kline as _Kline

    cfg = _WORKER["cfg"]
    db = Session(_WORKER["engine"])
    try:
        rows = db.execute(
            select(_Kline.open_time, _Kline.open, _Kline.high, _Kline.low, _Kline.close,
                   _Kline.volume, _Kline.quote_volume, _Kline.trades)
            .where(_Kline.symbol == symbol, _Kline.interval == "1h",
                   _Kline.open_time >= cfg["start_ms"],
                   _Kline.open_time <= cfg["last_complete"])
            .order_by(_Kline.open_time.asc())
        ).mappings().all()
    finally:
        db.close()

    horizon = cfg["horizon"]
    if len(rows) < WARMUP_BARS + horizon + 2:
        return {"symbol": symbol, "status": "skip", "reason": "bars_short", "n_bars": len(rows)}
    frame = pd.DataFrame([dict(row) for row in rows])
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="time"
    )
    if not frame.index.is_unique:
        frame = frame[~frame.index.duplicated(keep="last")]
    if abs(int(frame.index[-1].timestamp() * 1000) - cfg["last_complete"]) > HOUR_MS:
        return {"symbol": symbol, "status": "skip", "reason": "stale", "n_bars": len(frame)}
    tail = frame.index[-(horizon + 2):]
    # pandas 3.x 的 DatetimeIndex 默认分辨率可能是 us，不能假设 asi8 是 ns；
    # 统一降到秒再比，1h = 3600。
    seconds = tail.to_numpy(dtype="datetime64[s]").astype("int64")
    if np.any(np.diff(seconds) != 3600):
        return {"symbol": symbol, "status": "skip", "reason": "gap_in_last_2d", "n_bars": len(frame)}
    for column in ("open", "high", "low", "close"):
        if not np.isfinite(frame[column].to_numpy(dtype="float64")).all():
            return {"symbol": symbol, "status": "skip", "reason": "bad_price", "n_bars": len(frame)}

    directions, errors = compute_directions(frame)
    try:
        payload = evaluate_symbol(frame, directions, errors, horizon=horizon,
                                  cost=cfg["cost"], eval_days=cfg["days"])
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "status": "error", "reason": f"evaluate: {exc}"[:200]}
    if payload is None:
        return {"symbol": symbol, "status": "skip", "reason": "no_decision", "n_bars": len(frame)}
    return {"symbol": symbol, "status": "ok", "payload": payload}


def _last_complete_and_universe(days: int, sync: bool, workers: int) -> tuple[int, int, list[str], dict]:
    engine = make_engine(read_only=True)
    expected = int(time.time() * 1000) // HOUR_MS * HOUR_MS - HOUR_MS
    with Session(engine) as db:
        db_max = int(db.execute(
            select(func.max(Kline.open_time)).where(Kline.interval == "1h")
        ).scalar() or 0)
    sync_info: dict = {"triggered": False, "expected_ms": expected, "db_max_before_ms": db_max}
    # 落后 ≤1 根属正常同步延迟，用库内最新完整小时即可；≥2 根才触发补取
    if sync and db_max < expected - HOUR_MS:
        print(f"[sync] 1h 数据落后 {int((expected - db_max) / HOUR_MS)} 小时，自动补齐…", flush=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "backend"))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "backend" / "scripts" / "sync_recent_1h.py"),
             "--workers", str(min(workers, 10))],
            cwd=ROOT, env=env,
        )
        sync_info.update({"triggered": True, "returncode": completed.returncode})
        with Session(make_engine(read_only=True)) as db:
            db_max = int(db.execute(
                select(func.max(Kline.open_time)).where(Kline.interval == "1h")
            ).scalar() or 0)
    last_complete = min(expected, db_max)
    sync_info["db_max_after_ms"] = db_max
    report_path = ROOT / "docs" / "sync-recent-1h-report.json"
    if report_path.exists():
        try:
            sync_info["last_sync_report"] = {
                k: json.loads(report_path.read_text(encoding="utf-8")).get(k)
                for k in ("status_counts", "rows_inserted", "last_complete_hour_ms")
            }
        except (ValueError, OSError):
            pass

    start_ms = last_complete - (days + 40) * DAY_MS  # 含预热余量
    with Session(make_engine(read_only=True)) as db:
        universe = [str(s) for s in db.execute(
            select(Kline.symbol).where(Kline.interval == "1h",
                                       Kline.open_time >= start_ms).distinct()
        ).scalars().all()]
    return last_complete, start_ms, universe, sync_info


def _pool_members(universe: list[str]) -> dict[str, list[str]]:
    """用原生 SQL 读池定义：ORM 模型比库里多一列 manual_override，直接查会报错。"""
    from types import SimpleNamespace
    from sqlalchemy import text

    with Session(make_engine(read_only=True)) as db:
        pools: dict[str, list[str]] = {}
        rows = db.execute(text(
            "select code, kind, symbols, rule, enabled from symbol_pools "
            "where enabled = 1 order by code"
        )).mappings().all()
        for row in rows:
            spec = load_pool_spec(SimpleNamespace(**dict(row)))
            pools[row["code"]] = resolve_members(
                spec["kind"], spec["symbols"], spec["rule"],
                universe=lambda: universe, volumes=lambda: {}, metrics=None,
            )
    return pools


def main() -> int:
    # 延迟导入：worker 走 spawn 会重新执行本模块，若顶层导入报告模块会连带加载
    # pandas 之外的统计栈，16 进程时把页面文件耗尽（实测 DLL load failed）。
    from app.research.trend_direction_report import aggregate_pool, write_reports

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="历史评估窗口（天）")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="持有根数，48=两天")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST, help="每笔往返双边成本")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个币（冒烟）")
    ap.add_argument("--symbols", default="", help="逗号分隔，覆盖池成员（调试）")
    ap.add_argument("--out-prefix", default="docs/trend-direction-backtest")
    ap.add_argument("--no-sync", action="store_true", help="不做自动补齐")
    args = ap.parse_args()

    last_complete, start_ms, universe, sync_info = _last_complete_and_universe(
        args.days, not args.no_sync, args.workers
    )
    print(f"[data] 最后完整小时 {datetime.fromtimestamp(last_complete / 1000, timezone.utc)}"
          f" | 1h 宇宙 {len(universe)} 币", flush=True)
    pools = _pool_members(universe)
    print("[pools] " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()), flush=True)
    explicit = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    targets = explicit or sorted({s for members in pools.values() for s in members} & set(universe))
    if args.limit:
        targets = targets[: args.limit]
    print(f"[run] {len(targets)} 币 × {len(SPECS)} 指标 × {args.workers} 进程", flush=True)

    rows: dict = {}
    skipped: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(targets))),
                             initializer=_init_worker,
                             initargs=(start_ms, last_complete, args.horizon, args.cost, args.days)) as ex:
        futures = {ex.submit(_run_symbol, s): s for s in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                result = {"symbol": futures[fut], "status": "error", "reason": str(exc)[:200]}
            if result["status"] == "ok":
                rows[result["symbol"]] = result["payload"]
            else:
                skipped.append(result)
            if i % 50 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] ok={len(rows)} skip={len(skipped)} "
                      f"{time.time() - started:.0f}s", flush=True)

    names = [spec.name for spec in SPECS]
    labels = {spec.name: spec.label for spec in SPECS}
    contracts = [{
        "name": spec.name, "label": spec.label, "source": spec.source,
        "columns": " / ".join(spec.columns), "mode": spec.mode,
        "params": spec.params, "note": spec.note, "alias_of": spec.alias_of,
    } for spec in SPECS]
    pool_results = [
        aggregate_pool(code, members, rows, names, args.seed)
        for code, members in pools.items()
    ]
    two_day_decision = next(
        (p["two_day_decision_time"] for p in rows.values()), None
    )
    indicator_errors: dict[str, list[str]] = {}
    for symbol, payload in rows.items():
        for name, rec in payload["indicators"].items():
            if rec.get("error"):
                indicator_errors.setdefault(name, []).append(f"{symbol}: {rec['error']}")

    skip_reasons: dict[str, int] = {}
    for item in skipped:
        skip_reasons[item["reason"]] = skip_reasons.get(item["reason"], 0) + 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "interval": "1h", "days": args.days, "horizon": args.horizon,
            "cost": args.cost, "workers": args.workers, "seed": args.seed,
            "evaluation_start_ms": start_ms,
            "command": (f"PYTHONPATH=backend .venv/Scripts/python.exe "
                        f"backend/scripts/trend_direction_backtest.py --days {args.days} "
                        f"--horizon {args.horizon} --cost {args.cost} --workers {args.workers} "
                        f"--out-prefix {args.out_prefix}"),
            "indicators": names,
        },
        "data": {
            "last_complete": datetime.fromtimestamp(last_complete / 1000, timezone.utc).isoformat(),
            "two_day_decision": two_day_decision,
            "catalog_total": CATALOG_TOTAL,
            "evaluable_indicators": len(names),
            "universe_size": len(universe),
            "target_symbols": len(targets),
            "evaluated_symbols": len(rows),
            "skip_reasons": skip_reasons,
            "sync": sync_info,
        },
        "labels": labels,
        "contracts": contracts,
        "excluded": EXCLUDED,
        "pools": pool_results,
        "indicator_errors": {k: {"count": len(v), "sample": v[:5]} for k, v in indicator_errors.items()},
        "skipped_symbols_sample": skipped[:60],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    md_path, json_path = write_reports(payload, ROOT / args.out_prefix)
    print(f"\n== 完成 == 用时 {payload['elapsed_seconds']}s | 参评 {len(rows)} 币 | 跳过 {len(skipped)}")
    for pool in pool_results:
        best = pool["best"]
        if best:
            print(f"  {pool['code']:44s} best={best['name']:42s} "
                  f"net={best['mean_net'] * 100:+.4f}%  q={best['q_value']:.3f}  {best['verdict']}")
    print(f"报告: {md_path} / {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
