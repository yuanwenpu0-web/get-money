"""回填 native 15m K 线历史到 klines 表（interval='15m'）。

为什么需要它：
- strong_pullback 策略以 15m 为基周期，但库里 15m 仅 BTC/ETH 有 ~200 天，
  其余币只有 ~10 天（521 币中 519 个约 1004 根），无法做严格 walk-forward。
  本脚本仿 backfill_native_1h.py 把 Top-N 币的 15m 补到 ≥300 天。
- 落库 interval='15m'，读取走 data_service 的 native 回落
  （NATIVE_FALLBACK_INTERVALS 已含 15m），与 1h 回填同路径。
- 注意：15m 数据量 = 4×1h（300 天 ≈ 28800 根/币），默认只建议 Top 60。

用法：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_native_15m.py \
        --top 60 --days 300 --workers 8
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_native_15m.py \
        --symbols BTCUSDT,ETHUSDT --days 300 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import engine as _app_engine  # noqa: E402
from app.market import universe  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.models import Kline  # noqa: E402
from app.state import all_state  # noqa: E402

INTERVAL = "15m"
STEP_MS = 900_000
DAY_MS = 86_400_000
PAGE = 1000


def _insert_fn(db: Session):
    dialect = db.get_bind().dialect.name
    return pg_insert if dialect == "postgresql" else sqlite_insert


def upsert_rows(db: Session, symbol: str, rows: list[dict]) -> int:
    """直接 upsert interval='15m' 到 klines（绕过 store 的 1m 限制）。"""
    if not rows:
        return 0
    insert_fn = _insert_fn(db)
    for start in range(0, len(rows), 500):
        chunk = [
            {"symbol": symbol, "interval": INTERVAL, **r}
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


def fetch_pages(client: BinanceFuturesClient, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """从 start_ms 向前翻页拉 15m K线，[start_ms, end_ms] 闭区间，升序。"""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        raw = client.klines(symbol, INTERVAL, start_ms=cursor, limit=PAGE)
        parsed = [r for r in parse_klines(raw) if cursor <= r["open_time"] <= end_ms]
        if not parsed:
            break
        rows.extend(parsed)
        nxt = parsed[-1]["open_time"] + STEP_MS
        if nxt <= cursor or len(raw) < PAGE:
            break
        cursor = nxt
    return rows


def backfill_one(db: Session, client: BinanceFuturesClient, symbol: str, days: int, now: int) -> dict:
    svc = MarketDataService(db)
    stats = svc.series_stats(symbol, INTERVAL)
    earliest = int(stats.get("first_open_time") or 0)
    target_start = (now - days * DAY_MS) // STEP_MS * STEP_MS
    if earliest and earliest <= target_start + STEP_MS:
        return {"symbol": symbol, "status": "skip", "reason": "already_covered",
                "earliest": earliest, "rows": 0}
    fetch_end = earliest if earliest else now
    if fetch_end <= target_start:
        return {"symbol": symbol, "status": "skip", "reason": "no_gap", "rows": 0}
    rows = fetch_pages(client, symbol, target_start, fetch_end)
    inserted = upsert_rows(db, symbol, rows)
    db.close()
    return {"symbol": symbol, "status": "ok", "inserted": inserted,
            "fetched": len(rows), "earliest_before": earliest}


def resolve_symbols(client: BinanceFuturesClient, state: dict, args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    blacklist = set(universe._blacklist(state))
    info = client.exchange_info()
    symbols = [s for s in universe._filter(info.get("symbols", [])) if s not in blacklist]
    if args.top:
        tickers = client.ticker_24hr()  # {symbol: quote_volume}
        symbols = sorted(symbols, key=lambda s: -tickers.get(s, 0.0))[: args.top]
    return symbols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None, help="HTTP 代理，默认读 state.proxy_url")
    ap.add_argument("--days", type=int, default=300, help="目标历史宽度（天）")
    ap.add_argument("--symbols", default="", help="逗号分隔白名单，缺省用全宇宙/--top")
    ap.add_argument("--top", type=int, default=0, help="按 24h 成交额取前 N 币（0=全宇宙）")
    ap.add_argument("--workers", type=int, default=8, help="并发拉取线程数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（调试）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Session(_app_engine)
    state = all_state(db)
    proxy = args.proxy or state.get("proxy_url") or None
    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    symbols = resolve_symbols(client, state, args)
    db.close()
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"目标宇宙: {len(symbols)} 币 (days={args.days}, proxy={proxy})")
    if args.dry_run:
        print("  sample:", symbols[:15])
        return 0

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
        "interval": INTERVAL,
        "days_target": args.days,
        "universe_size": len(symbols),
        "total_inserted": total_inserted,
        "backfilled": len(ok),
        "skipped": len([r for r in results if r.get("status") == "skip"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "results": results,
    }
    out = ROOT / "docs" / "backfill-native-15m-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 完成 ==  插入 {total_inserted} 行 15m，{len(ok)} 币扩展，"
          f"{report['skipped']} 跳过，{report['errors']} 错误")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
