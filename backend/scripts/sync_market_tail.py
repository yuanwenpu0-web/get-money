#!/usr/bin/env python3
"""补齐 native K 线（15m/4h/5m）与 funding_rates 的尾部缺口。

既有 backfill_native_* 只回填历史起点之前、sync_recent_1h 硬编码 1h、
sync_funding_many 无 CLI 调用者、4h 无 HTTP 采集脚本，故 15m/5m 停在
2026-10-25、4h 停 10-26、funding 停 10-27。本脚本只补尾部：目标币池严格
取各周期已有 distinct symbol（不扩池、不改口径），每币从 max(open_time) 留
2 根重叠补到最后一个完整 UTC bar，全部幂等 upsert，不碰策略/绑定/参数。

用法（仓库根）：
    .venv/Scripts/python.exe backend/scripts/sync_market_tail.py --dry-run
    .venv/Scripts/python.exe backend/scripts/sync_market_tail.py --limit 5
    .venv/Scripts/python.exe backend/scripts/sync_market_tail.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market.futures import BinanceFuturesClient, set_public_request_interval  # noqa: E402
from app.models import JobLog, Kline  # noqa: E402
from app.state import all_state  # noqa: E402

INTERVALS = {"5m": 300_000, "15m": 900_000, "4h": 14_400_000}
DEFAULT_INTERVALS = "15m,4h,5m"
PAGE = 1000
DAY_MS = 86_400_000
VALUE_COLS = ("open", "high", "low", "close", "volume", "quote_volume",
              "close_time", "trades")


def iso(ms) -> str | None:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat() if ms else None


def kline_coverage(engine, interval: str) -> dict:
    with engine.connect() as conn:
        lo, hi, rows, syms = conn.execute(text(
            "select min(open_time), max(open_time), count(*), count(distinct symbol) "
            "from klines where interval = :iv"), {"iv": interval}).one()
    return {"start_utc": iso(lo), "end_utc": iso(hi), "rows": int(rows),
            "symbols": int(syms)}


def funding_coverage(engine) -> dict:
    with engine.connect() as conn:
        lo, hi, rows, syms = conn.execute(text(
            "select min(funding_time), max(funding_time), count(*), "
            "count(distinct symbol) from funding_rates")).one()
    return {"start_utc": iso(lo), "end_utc": iso(hi), "rows": int(rows),
            "symbols": int(syms)}


def target_symbols(engine, interval: str, explicit: list[str], limit: int) -> list[str]:
    """只用该周期已有的币，避免把 5m（仅 2 币）等稀疏周期扩成全宇宙。"""
    with engine.connect() as conn:
        existing = {str(s) for s in conn.execute(text(
            "select distinct symbol from klines where interval = :iv"),
            {"iv": interval}).scalars().all()}
    if explicit:
        symbols = [s for s in explicit if s in existing]  # 不扩池：忽略非本周期已有币
    else:
        symbols = sorted(existing)
    return symbols[:limit] if limit else symbols


def fetch_pages(client: BinanceFuturesClient, symbol: str, interval: str,
                step_ms: int, start_ms: int, end_ms: int) -> list[dict]:
    """[start_ms, end_ms] 闭区间分页，升序；与 backfill_native_1h 同口径。"""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        raw = client.klines(symbol, interval, start_ms=cursor, limit=PAGE)
        parsed = [r for r in parse_klines(raw) if cursor <= r["open_time"] <= end_ms]
        if not parsed:
            break
        rows.extend(parsed)
        nxt = parsed[-1]["open_time"] + step_ms
        if nxt <= cursor or len(raw) < PAGE:
            break
        cursor = nxt
    return rows


def upsert_rows(db: Session, symbol: str, interval: str, rows: list[dict]) -> int:
    """直接 upsert 指定 interval，绕过 MarketDataService.store 的 1m 限制。"""
    if not rows:
        return 0
    insert_fn = pg_insert if db.get_bind().dialect.name == "postgresql" else sqlite_insert
    keys = ("open_time",) + VALUE_COLS
    for start in range(0, len(rows), 500):
        chunk = [{"symbol": symbol, "interval": interval,
                  **{k: r[k] for k in keys if k in r}}
                 for r in rows[start:start + 500]]
        stmt = insert_fn(Kline).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "open_time"],
            set_={c: getattr(stmt.excluded, c) for c in VALUE_COLS},
        )
        db.execute(stmt)
    db.commit()
    return len(rows)


def sync_one(engine, client, symbol: str, interval: str, step_ms: int,
             last_complete: int, dry_run: bool) -> dict:
    db = Session(engine)
    try:
        latest = int(db.execute(select(func.max(Kline.open_time)).where(
            Kline.symbol == symbol, Kline.interval == interval)).scalar() or 0)
        if latest >= last_complete:
            return {"symbol": symbol, "status": "fresh", "latest": latest, "rows": 0}
        start = (latest - 2 * step_ms) if latest else (last_complete - 30 * DAY_MS)
        if dry_run:
            return {"symbol": symbol, "status": "stale", "latest": latest,
                    "target": last_complete,
                    "rows": int((last_complete - start) // step_ms) + 1}
        rows = fetch_pages(client, symbol, interval, step_ms, start, last_complete)
        inserted = upsert_rows(db, symbol, interval, rows)
        return {"symbol": symbol, "status": "ok", "latest": latest,
                "target": last_complete, "rows": inserted}
    except Exception as exc:  # noqa: BLE001 单币失败不阻塞
        return {"symbol": symbol, "status": "error", "reason": str(exc)[:200]}
    finally:
        db.close()


def run_interval(engine, client, interval: str, explicit: list[str], limit: int,
                 workers: int, now_ms: int, dry_run: bool) -> dict:
    step_ms = INTERVALS[interval]
    last_complete = now_ms // step_ms * step_ms - step_ms
    symbols = target_symbols(engine, interval, explicit, limit)
    before = kline_coverage(engine, interval)
    print(f"[{interval}] {len(symbols)} 币 | 最后完整 bar {iso(last_complete)} "
          f"| before {before['end_utc']}", flush=True)
    if dry_run:
        return {"interval": interval, "universe_size": len(symbols),
                "last_complete_ms": last_complete, "before": before, "dry_run": True}
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 32, len(symbols)))) as ex:
        futures = {ex.submit(sync_one, engine, client, s, interval, step_ms,
                             last_complete, dry_run): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"symbol": futures[fut], "status": "error",
                       "reason": str(exc)[:200]}
            results.append(res)
            if i % 100 == 0 or i == len(symbols):
                print(f"  [{interval}] {i}/{len(symbols)}", flush=True)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "interval": interval, "universe_size": len(symbols),
        "last_complete_ms": last_complete, "before": before,
        "after": kline_coverage(engine, interval),
        "status_counts": counts,
        "rows_inserted": sum(int(r.get("rows", 0)) for r in results if r["status"] == "ok"),
        "errors": [f"{r['symbol']}: {r.get('reason')}" for r in results
                   if r["status"] == "error"][:30],
        "results": results,
    }


def run_funding(engine, days: int, workers: int, limit: int, now_ms: int,
                dry_run: bool) -> dict:
    with engine.connect() as conn:
        symbols = [str(s) for s in conn.execute(text(
            "select distinct symbol from funding_rates order by 1")).scalars().all()]
    if limit:
        symbols = symbols[:limit]
    before = funding_coverage(engine)
    # 按「最落后币」的水位定窗口：全局 max 会漏掉水位不齐的币，留下中间断层
    with engine.connect() as conn:
        watermark = int(conn.execute(text(
            "select min(mx) from (select max(funding_time) mx from funding_rates "
            "group by symbol) t")).scalar() or 0)
    window = days if days > 0 else min(30, max(3, (now_ms - watermark) // DAY_MS + 3))
    print(f"[funding] {len(symbols)} 币 | 窗口 {window} 天 | before {before['end_utc']}",
          flush=True)
    if dry_run:
        return {"universe_size": len(symbols), "days": window,
                "before": before, "dry_run": True}
    db = Session(engine)
    try:
        result = MarketDataService(db).sync_funding_many(symbols, window, workers)
        db.add(JobLog(job="funding_sync", level="info",
                      message=f"funding 尾部采集 请求 {result['requested']} 币，"
                              f"写入 {result['inserted']} 行，失败 {len(result['errors'])} 个"))
        db.commit()
    finally:
        db.close()
    return {"universe_size": len(symbols), "days": window,
            "before": before, "after": funding_coverage(engine),
            "requested": result["requested"], "inserted": result["inserted"],
            "error_count": len(result["errors"]), "errors": result["errors"][:30]}


def main() -> int:
    ap = argparse.ArgumentParser(description="native K 线与资金费尾部增量采集")
    ap.add_argument("--intervals", default=DEFAULT_INTERVALS,
                    help=f"逗号分隔，可选 {','.join(INTERVALS)}")
    ap.add_argument("--symbols", default="",
                    help="逗号分隔白名单；缺省用各周期已有 distinct symbol")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="每类只处理前 N 币（小样本验证）")
    ap.add_argument("--days", type=int, default=0,
                    help="funding 窗口天数；0=按水位自动（默认）")
    ap.add_argument("--request-interval", type=float, default=0.25,
                    help="公开 GET 最小放行间隔（秒）；klines 权重高，0.25 可避开 2400/min IP 限流")
    ap.add_argument("--no-funding", action="store_true", help="跳过资金费率采集")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求、不写库")
    ap.add_argument("--out", default="output/sync_market_tail_report.json")
    args = ap.parse_args()

    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    unknown = [s for s in intervals if s not in INTERVALS]
    if unknown:
        print(f"不支持的周期: {unknown}", file=sys.stderr)
        return 2
    explicit = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    engine = make_engine(read_only=False)
    state_db = Session(engine)
    proxy = all_state(state_db).get("proxy_url") or None
    state_db.close()
    now_ms = int(time.time() * 1000)
    print(json.dumps({"dry_run": args.dry_run, "intervals": intervals,
                      "funding": not args.no_funding, "symbols_override": explicit,
                      "limit": args.limit, "proxy_configured": bool(proxy)},
                     ensure_ascii=False), flush=True)

    gate = set_public_request_interval(args.request_interval) if not args.dry_run else None
    client = BinanceFuturesClient(testnet=False, proxy=proxy)
    started = time.perf_counter()
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/sync_market_tail.py",
        "source": "Binance USD-M /fapi/v1/klines + /fapi/v1/fundingRate (public, no API key)",
        "dry_run": bool(args.dry_run), "workers": args.workers,
        "request_interval_seconds": gate,
        "policy": "目标币池 = 各周期已有 distinct symbol；不扩池、不改口径；幂等 upsert",
        "kline_intervals": {},
    }
    try:
        for interval in intervals:
            report["kline_intervals"][interval] = run_interval(
                engine, client, interval, explicit, args.limit, args.workers,
                now_ms, args.dry_run)
            if not args.dry_run:
                time.sleep(3)  # 阶段间冷却，避开币安 IP 权重限流
        if not args.no_funding:
            report["funding"] = run_funding(engine, args.days, args.workers,
                                            args.limit, now_ms, args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"采集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report["duration_seconds"] = round(time.perf_counter() - started, 2)
    report["error_count"] = sum(
        v.get("status_counts", {}).get("error", 0)
        for v in report["kline_intervals"].values()
    ) + int(report.get("funding", {}).get("error_count", 0))
    if not args.dry_run:  # dry-run 不落盘，避免覆盖正式报告
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("error_count", "duration_seconds")}
    summary["kline_intervals"] = {
        k: {"after": v.get("after", {}).get("end_utc"), "rows": v.get("rows_inserted"),
            "counts": v.get("status_counts")}
        for k, v in report["kline_intervals"].items()}
    summary["funding"] = report.get("funding", {})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
