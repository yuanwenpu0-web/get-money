"""补齐 native 1h K 线的「尾部」缺口，一直到最后一个完整 UTC 小时。

与 ``backfill_native_1h.py`` 的区别：那个脚本只往历史起点之前回填，
不处理「昨天到今天」的缺口；本脚本专门补尾部，保证最近两天的 1h 资料可用。

用法：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_recent_1h.py --workers 8
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_recent_1h.py --dry-run
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/sync_recent_1h.py --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.db import engine as _app_engine  # noqa: E402
from app.state import all_state  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.models import Kline, SymbolPool  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from backfill_native_1h import fetch_1h_pages, upsert_1h  # noqa: E402

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
PAGE = 1000


def target_symbols(db: Session, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    from sqlalchemy import text

    found: set[str] = set()
    for interval in ("1h", "1m"):
        rows = db.execute(
            select(Kline.symbol).where(Kline.interval == interval).distinct()
        ).scalars().all()
        found.update(str(s) for s in rows)
    found.update(MAJOR_SYMBOLS)
    # ORM 模型比库里多 manual_override 列，池定义用原生 SQL 只取存在的列
    rows = db.execute(text(
        "select symbols from symbol_pools where kind = 'static'"
    )).scalars().all()
    for raw in rows:
        try:
            found.update(str(s).upper() for s in json.loads(raw or "[]"))
        except (ValueError, TypeError):
            continue
    return sorted(found)


def latest_1h(db: Session, symbol: str) -> int:
    value = db.execute(
        select(func.max(Kline.open_time)).where(
            Kline.symbol == symbol, Kline.interval == "1h"
        )
    ).scalar()
    return int(value or 0)


def sync_one(client: BinanceFuturesClient, symbol: str, last_complete: int,
             days: int, dry_run: bool) -> dict:
    db = Session(_app_engine)
    try:
        latest = latest_1h(db, symbol)
        if latest >= last_complete:
            return {"symbol": symbol, "status": "fresh", "latest": latest, "rows": 0}
        # 已有数据只补尾部（留 2 根重叠）；完全没有的币补一段历史
        start = (latest - 2 * HOUR_MS) if latest else (last_complete - days * DAY_MS)
        if dry_run:
            return {"symbol": symbol, "status": "stale", "latest": latest,
                    "target": last_complete, "rows": int((last_complete - start) // HOUR_MS) + 1}
        rows = fetch_1h_pages(client, symbol, start, last_complete)
        inserted = upsert_1h(db, symbol, rows)
        return {"symbol": symbol, "status": "ok", "latest": latest,
                "target": last_complete, "rows": inserted}
    except Exception as exc:  # noqa: BLE001 单币失败不阻塞
        return {"symbol": symbol, "status": "error", "reason": str(exc)[:200]}
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="逗号分隔白名单，缺省自动覆盖全宇宙+池成员")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", type=int, default=400, help="无历史数据币的回填宽度")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="docs/sync-recent-1h-report.json",
                    help="采集报告路径；默认保留历史路径，指定新路径可避免覆盖旧报告")
    args = ap.parse_args()

    explicit = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    db = Session(_app_engine)
    state = all_state(db)
    proxy = args.proxy or state.get("proxy_url") or None
    symbols = target_symbols(db, explicit)
    db.close()
    now_ms = int(time.time() * 1000)
    last_complete = now_ms // HOUR_MS * HOUR_MS - HOUR_MS
    print(f"目标 {len(symbols)} 币 | 最后完整小时 "
          f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(last_complete / 1000))} | proxy={proxy}",
          flush=True)
    if args.dry_run:
        print("sample:", symbols[:15], flush=True)

    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 32, len(symbols)))) as ex:
        futures = {
            ex.submit(sync_one, client, s, last_complete, args.days, args.dry_run): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"symbol": futures[fut], "status": "error", "reason": str(exc)[:200]}
            results.append(res)
            if i % 50 == 0 or i == len(symbols):
                print(f"  [{i}/{len(symbols)}] ...", flush=True)

    counts: dict[str, int] = {}
    for res in results:
        counts[res["status"]] = counts.get(res["status"], 0) + 1
    report = {
        "generated_at_ms": now_ms,
        "last_complete_hour_ms": last_complete,
        "universe_size": len(symbols),
        "dry_run": bool(args.dry_run),
        "status_counts": counts,
        "rows_inserted": sum(int(r.get("rows", 0)) for r in results if r["status"] == "ok"),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: {counts} | 写入 {report['rows_inserted']} 行 | 报告 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
