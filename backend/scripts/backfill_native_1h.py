"""回填 native 1h K 线历史到 klines 表（interval='1h'），扩展多币回测历史宽度。

为什么需要它：
- P0 释放的 native 1h 回落让 BTC/SOL 等大币拿到 ~400 天 1h，但全宇宙 521 币的
  native 1h 分布极不均（均值仅 ~1368 行，多数币 < 200 天）。多币 walk-forward 的
  窗口数被这一宽度强约束（内存档：多币约 90~400 天）。
- 现有 `MarketDataService.store` 只允许写 1m（其他周期由 1m 聚合），且
  `sync_history` 内部硬编码 1m，所以"回填 1h"必须走专用脚本、直接 upsert。
- 本脚本与现存 native 1h 行同源（fapi 合约公共K线），落库 interval='1h'，
  与 native 回落读取路径（data_service._merged_candle_rows 读 interval==target）一致。

用法：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_native_1h.py \
        --days 400 --workers 8
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_native_1h.py \
        --symbols BTCUSDT,ETHUSDT --days 400 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import engine as _app_engine  # noqa: E402
from app.state import all_state  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.market import universe  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.models import Kline  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402
from sqlalchemy.dialects.sqlite import insert as sqlite_insert  # noqa: E402

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
PAGE = 1000


def _insert_fn(db: Session):
    dialect = db.get_bind().dialect.name
    return pg_insert if dialect == "postgresql" else sqlite_insert


def upsert_1h(db: Session, symbol: str, rows: list[dict]) -> int:
    """直接 upsert interval='1h' 到 klines（绕过 store 的 1m 限制）。"""
    if not rows:
        return 0
    insert_fn = _insert_fn(db)
    for start in range(0, len(rows), 500):
        chunk = [
            {"symbol": symbol, "interval": "1h", **r}
            for r in rows[start : start + 500]
        ]
        stmt = insert_fn(Kline).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "open_time"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "quote_volume": stmt.excluded.quote_volume,
                "close_time": stmt.excluded.close_time,
                "trades": stmt.excluded.trades,
            },
        )
        db.execute(stmt)
    db.commit()
    return len(rows)


def fetch_1h_pages(client: BinanceFuturesClient, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """从 start_ms 向前翻页拉 1h K线，[start_ms, end_ms] 闭区间，升序。"""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        raw = client.klines(symbol, "1h", start_ms=cursor, limit=PAGE)
        parsed = [r for r in parse_klines(raw) if cursor <= r["open_time"] <= end_ms]
        if not parsed:
            break
        rows.extend(parsed)
        nxt = parsed[-1]["open_time"] + HOUR_MS
        if nxt <= cursor or len(raw) < PAGE:
            break
        cursor = nxt
    return rows


def backfill_one(db: Session, client: BinanceFuturesClient, symbol: str, days: int, now: int) -> dict:
    svc = MarketDataService(db)
    stats = svc.series_stats(symbol, "1h")
    earliest = int(stats.get("first_open_time") or 0)
    target_start = (now - days * DAY_MS) // HOUR_MS * HOUR_MS
    if earliest and earliest <= target_start + HOUR_MS:
        return {"symbol": symbol, "status": "skip", "reason": "already_covered",
                "earliest": earliest, "rows": 0}
    fetch_end = earliest if earliest else now
    if fetch_end <= target_start:
        return {"symbol": symbol, "status": "skip", "reason": "no_gap", "rows": 0}
    rows = fetch_1h_pages(client, symbol, target_start, fetch_end)
    inserted = upsert_1h(db, symbol, rows)
    db.close()
    return {"symbol": symbol, "status": "ok", "inserted": inserted,
            "fetched": len(rows), "earliest_before": earliest}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None, help="HTTP 代理，默认读 state.proxy_url")
    ap.add_argument("--days", type=int, default=400, help="目标历史宽度（天）")
    ap.add_argument("--symbols", default="", help="逗号分隔白名单，缺省用全宇宙")
    ap.add_argument("--workers", type=int, default=8, help="并发拉取线程数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（调试）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Session(_app_engine)
    state = all_state(db)
    proxy = args.proxy or state.get("proxy_url") or None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        blacklist = universe._blacklist(state)
        info = BinanceFuturesClient(testnet=False, proxy=proxy).exchange_info()
        symbols = [s for s in universe._filter(info.get("symbols", [])) if s not in set(blacklist)]
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"目标宇宙: {len(symbols)} 币 (days={args.days}, proxy={proxy})")
    if args.dry_run:
        print("  sample:", symbols[:15])
        return 0

    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    now = int(time.time() * 1000)
    results: list[dict] = []
    total_inserted = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 32, len(symbols)))) as ex:
        futs = {
            ex.submit(backfill_one, Session(_app_engine), client, s, args.days, now): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # 单币失败不阻塞
                res = {"symbol": sym, "status": "error", "reason": str(exc)[:200]}
            results.append(res)
            if res.get("status") == "ok":
                total_inserted += int(res.get("inserted", 0))
            print(f"[{i}/{len(symbols)}] {sym} {res.get('status')} "
                  f"+{res.get('inserted', res.get('reason', ''))}", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    report = {
        "days_target": args.days,
        "universe_size": len(symbols),
        "total_inserted": total_inserted,
        "backfilled": len(ok),
        "skipped": len([r for r in results if r.get("status") == "skip"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "results": results,
    }
    out = ROOT / "docs" / "backfill-native-1h-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 完成 ==  插入 {total_inserted} 行 1h，{len(ok)} 币扩展，"
          f"{report['skipped']} 跳过，{report['errors']} 错误")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
