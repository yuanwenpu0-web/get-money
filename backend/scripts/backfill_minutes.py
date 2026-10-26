"""Backfill the shared 1m source for the current universe and all open positions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import SessionLocal
from app.market.data_service import MarketDataService
from app.state import all_state
from app.trading.portfolio import open_orders
from app.trading.service import resolve_symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--symbols", nargs="+")
    args = parser.parse_args()
    if not 1 <= args.days <= 730 or not 1 <= args.workers <= 8:
        parser.error("days must be 1-730; workers must be 1-8")

    with SessionLocal() as db:
        symbols = set(args.symbols or resolve_symbols(db, all_state(db)))
        symbols.update(order.symbol for order in open_orders(db))

    def sync(symbol):
        with SessionLocal() as db:
            try:
                return MarketDataService(db).sync_history(symbol, "1m", args.days)
            except Exception as exc:
                return {"symbol": symbol, "error": str(exc)}

    results = []
    report = ROOT / "output" / "minute-source-backfill.json"
    report.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sync, symbol) for symbol in sorted(symbols)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(symbols)}] {json.dumps(result, ensure_ascii=False)}", flush=True)
            report.write_text(json.dumps({"days": args.days, "total": len(symbols), "results": results},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any("error" in result for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
