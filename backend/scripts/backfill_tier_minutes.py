r"""为「分档回测」选定的代表币种补齐 1m 历史数据。

背景
----
回测引擎坚持从 1m 原始K线聚合出所有周期（data_service._candle_query 固定
Kline.interval == "1m"，并要求桶内分钟齐全），而本地 1m 数据十分稀疏：
平均每个币种只有约 2500 根（≈42 小时），导致 1h 回测实际只能取到约 7 天数据。
本脚本按档位挑选代表币种后，把它们的 1m 历史补齐，使标准引擎能真正跑起来。

设计
----
* **只补需要的币种**：给全部 521 个币种补 90 天 1m 需要约 3400 万行、数小时；
  分档回测实际只用「每档代表币种」，量级降到百万行、几十分钟。
* **多线程下载 + 主线程串行写入**：网络 IO 是瓶颈用并发；SQLite 只有单写者，
  写入一律回到主线程，避免 "database is locked"。
* **复用项目原有链路**：仍走 BinanceFuturesClient + parse_klines + store(upsert)，
  数据格式与去重语义和既有同步逻辑完全一致；幂等，重复运行安全。

用法：
.\.venv\Scripts\python.exe backend/scripts/backfill_tier_minutes.py --per-tier 14 --days 90
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.ml.tiers import assign_tiers  # noqa: E402
from app.models import Kline  # noqa: E402

DAY_MS = 86_400_000
MINUTE_MS = 60_000
PAGE = 1000

_local = threading.local()


def _client() -> BinanceFuturesClient:
    """每个线程独立 client（httpx.Client 每次请求临时创建，无跨线程共享状态）。"""
    client = getattr(_local, "client", None)
    if client is None:
        client = BinanceFuturesClient(testnet=False)
        _local.client = client
    return client


def fetch_symbol(symbol: str, start_ms: int, end_ms: int, max_retries: int = 3) -> list[dict]:
    """分页抓取某币种 [start, end] 的全部 1m K线。"""
    collected: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        raw: list[list] | None = None
        for attempt in range(max_retries):
            try:
                raw = _client().klines(symbol, "1m", start_ms=cursor, limit=PAGE)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not raw:
            break
        parsed = [row for row in parse_klines(raw) if start_ms <= row["open_time"] <= end_ms]
        if not parsed:
            break
        collected.extend(parsed)
        nxt = parsed[-1]["open_time"] + MINUTE_MS
        if nxt <= cursor:
            break
        cursor = nxt
        if len(raw) < PAGE:
            break
    return collected


def existing_extent(db: Session, symbol: str) -> tuple[int, int]:
    """该币种现有 1m 数据的时间范围，用于跳过已覆盖区间、减少请求。"""
    row = db.execute(
        select(
            func.min(Kline.open_time),
            func.max(Kline.open_time),
        ).where(Kline.symbol == symbol, Kline.interval == "1m")
    ).first()
    if not row or row[0] is None:
        return 0, 0
    return int(row[0]), int(row[1])


def _jsonable(args) -> dict:
    """命令行参数转可序列化 dict：Path/pydantic 等类型 json.dumps 不认。"""
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", dest="as_of", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--per-tier", type=int, default=14, help="每档选取代表币种数")
    parser.add_argument("--days", type=int, default=90, help="补多少天历史")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--symbols", nargs="*", default=None, help="显式指定币种（跳过自动选币）")
    parser.add_argument("--dry-run", action="store_true", help="只选币种并预估，不下载")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "backfill-tier-minutes.json")
    args = parser.parse_args()

    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone")
    end_ms = int(as_of.timestamp() * 1000) // MINUTE_MS * MINUTE_MS
    target_start = end_ms - args.days * DAY_MS

    # 写库：不加 read_only，走连接池
    engine = make_engine()

    # ---- 选代表币种：用 tiers 模块（直查原生 1h 行，不受 1m 缺失影响）----
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols]
        tier_source = "explicit"
    else:
        with Session(engine) as db:
            _frame, groups = assign_tiers(db, None, "1h", end_ms, 30)
        symbols, tier_source = [], "auto"
        for key, members in groups.items():
            if members:
                symbols.extend(members[: args.per_tier])
        symbols = sorted(set(symbols))

    total_requests = args.days * 1440 // PAGE * len(symbols)
    estimate_sec = total_requests * 1.4 / max(args.workers, 1)
    print(json.dumps({
        "symbols": len(symbols), "days": args.days, "workers": args.workers,
        "tier_source": tier_source,
        "est_requests": total_requests,
        "est_minutes": round(estimate_sec / 60, 1),
        "est_rows_millions": round(args.days * 1440 * len(symbols) / 1e6, 2),
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        # 干跑也要落盘候选清单：tier_backtest 用它限定分档范围，避免全库 500+ 扫描
        args.output.write_text(
            json.dumps(
                {
                    "as_of": as_of.isoformat(),
                    "dry_run": True,
                    "params": _jsonable(args) | {"symbols_count": len(symbols)},
                    "tier_source": tier_source,
                    "symbols": symbols,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"候选清单已写入 {args.output}")
        return

    started = time.time()
    results: list[dict] = []
    with Session(engine) as db:
        service = MarketDataService(db)
        # 增量补：跳过已覆盖区间
        ranges: dict[str, tuple[int, int]] = {}
        for symbol in symbols:
            low, high = existing_extent(db, symbol)
            # 已有数据覆盖 [low, high]；需要 [target_start, low) 与 [high, end]
            ranges[symbol] = (target_start, low if low else target_start, high)
        pending = []
        for symbol, (start, low, high) in ranges.items():
            if low > 0 and start >= low - MINUTE_MS:
                # 新区段为空，只需补尾部
                pending.append((symbol, max(high + MINUTE_MS, target_start + MINUTE_MS), end_ms))
            else:
                pending.append((symbol, target_start, end_ms))
        pending = [(s, a, b) for s, a, b in pending if a <= b]
        print(json.dumps({"pending_jobs": len(pending)}, ensure_ascii=False), flush=True)

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_symbol, symbol, start, end): (symbol, start, end)
                for symbol, start, end in pending
            }
            for future in as_completed(futures):
                symbol, start, end = futures[future]
                done += 1
                try:
                    rows = future.result()
                    inserted = service.store(symbol, "1m", rows) if rows else 0
                    results.append({"symbol": symbol, "rows": len(rows), "inserted": inserted, "ok": True})
                except Exception as exc:
                    results.append({"symbol": symbol, "error": str(exc)[:200], "ok": False})
                if done % 20 == 0 or done == len(pending):
                    elapsed = time.time() - started
                    print(json.dumps({
                        "done": done, "total": len(pending),
                        "pct": round(done / max(len(pending), 1) * 100, 1),
                        "elapsed_min": round(elapsed / 60, 1),
                        "eta_min": round((elapsed / done * (len(pending) - done)) / 60, 1),
                        "rows_so_far": sum(r.get("inserted", 0) for r in results),
                    }, ensure_ascii=False), flush=True)
    engine.dispose()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    payload = {
        "as_of": as_of.isoformat(),
        "params": _jsonable(args) | {"symbols_count": len(symbols)},
        "elapsed_min": round((time.time() - started) / 60, 1),
        "total_rows": sum(r["inserted"] for r in ok),
        "succeeded": len(ok),
        "failed": failed,
        "symbols": symbols,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{len(ok)}/{len(pending)} 成功，{sum(r['inserted'] for r in ok):,} 行，"
          f"耗时 {payload['elapsed_min']} 分钟 -> {args.output}")


if __name__ == "__main__":
    main()
