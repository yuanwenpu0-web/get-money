#!/usr/bin/env python3
"""恢复 P9 衍生品情绪（/futures/data/*）增量采集链。

背景：sentiment 表停在 2026-10-28。原因不是接口失效，而是
`MarketDataService.sync_sentiment_many` 只有定义、无调用者，且每日 08:30
自动化所调的 `scripts/backfill_sentiment.py` 已从仓库删除。

本脚本只做一件事：把已有的 `sync_sentiment_many` 包成可调度 CLI。
接口每次返回最近约 500 根 1h（≈21 天），幂等 upsert，重跑安全；
DB 里更老的数据不会被覆盖或删除，新数据滚动前进。

用法（仓库根执行）：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_sentiment.py --dry-run
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_sentiment.py --limit 5
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_sentiment.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market.futures import set_public_request_interval  # noqa: E402
from app.models import JobLog  # noqa: E402


def iso(ms) -> str | None:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat() if ms else None


def snapshot(engine) -> dict:
    with engine.connect() as conn:
        lo, hi, rows, syms = conn.execute(text(
            "select min(ts), max(ts), count(*), count(distinct symbol) from sentiment")).one()
    return {"start_utc": iso(lo), "end_utc": iso(hi), "rows": int(rows),
            "symbols": int(syms),
            "span_days": round((int(hi) - int(lo)) / 86_400_000, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="P9 sentiment 增量采集（幂等 upsert）")
    ap.add_argument("--symbols", default="", help="逗号分隔白名单；缺省用 sentiment 表已有币")
    ap.add_argument("--workers", type=int, default=12, help="并发请求数（受公开接口门控排队）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个币，用于小样本验证")
    ap.add_argument("--request-interval", type=float, default=0.2,
                    help="公开 GET 最小放行间隔（秒），下限 0.02")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求、不写库")
    ap.add_argument("--out", default="output/sync_sentiment_report.json")
    args = ap.parse_args()

    engine = make_engine(read_only=False)
    with engine.connect() as conn:
        symbols = [str(r[0]) for r in conn.execute(text(
            "select distinct symbol from sentiment order by 1")).fetchall()]
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.limit > 0:
        symbols = symbols[:args.limit]
    if not symbols:
        print("币池为空，终止", file=sys.stderr)
        return 2

    before = snapshot(engine)
    print(json.dumps({"dry_run": args.dry_run, "symbols": len(symbols),
                      "sample": symbols[:5], "before": before,
                      "request_interval": args.request_interval},
                     ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    interval = set_public_request_interval(args.request_interval)
    db = Session(engine)
    started = time.perf_counter()
    try:
        service = MarketDataService(db)
        result = service.sync_sentiment_many(symbols, args.workers)
        duration = round(time.perf_counter() - started, 2)
        db.add(JobLog(job="sentiment_sync", level="info",
                      message=f"sentiment 增量采集 请求 {result['requested']} 币，"
                              f"写入 {result['inserted']} 行，失败 {len(result['errors'])} 个"))
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        print(f"采集失败，事务已回滚：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    after = snapshot(engine)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/sync_sentiment.py",
        "source": "Binance USD-M /futures/data/* (public, no API key)",
        "symbols": len(symbols), "workers": args.workers,
        "request_interval_seconds": interval,
        "requested": result["requested"], "inserted": result["inserted"],
        "error_count": len(result["errors"]), "errors": result["errors"][:30],
        "duration_seconds": duration,
        "coverage_before": before, "coverage_after": after,
        "note": "幂等 upsert；只滚动前进最近约 21 天窗口，更老历史不会被覆盖或删除",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "requested": report["requested"],
                      "inserted": report["inserted"], "errors": report["error_count"],
                      "duration_seconds": duration, "before": before, "after": after},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
