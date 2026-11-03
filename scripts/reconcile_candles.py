"""Repair conflicting historical candles only when archive and REST agree."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import sys

import httpx
from sqlalchemy import select, update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.models import Kline  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from backfill_all_history import RateLimitedClient  # noqa: E402
from backfill_taker_data import BASE, FIELDS, read_archive, same_candle, valid_candle  # noqa: E402

FIELDS_TO_WRITE = (*FIELDS, "close_time", "trades", "taker_buy_volume", "taker_buy_quote_volume")


def main():
    engine = make_engine(null_pool=True)
    logdir = ROOT / "output" / "candle_reconciliation"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with sqlite3.connect(ROOT / "output" / "all_history_backfill.sqlite") as state:
        targets = [(s, m) for s, m in state.execute("SELECT symbol,month FROM months WHERE status='partial' AND mismatched>0 ORDER BY symbol,month") if s in MAJOR_SYMBOLS]
    with httpx.Client(timeout=90, follow_redirects=True) as raw:
        client = RateLimitedClient(raw)
        for symbol, month in targets:
            url = f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
            archive = read_archive(client, url)
            if not archive:
                print(f"{symbol} {month}: 月文件未发布，暂不修改", flush=True)
                continue
            rows = {r["open_time"]: r for r in archive}
            with engine.connect() as conn:
                old = conn.execute(select(Kline).where(Kline.symbol == symbol, Kline.interval == "1m",
                    Kline.open_time >= min(rows), Kline.open_time <= max(rows))).all()
            mismatches = [r for r in old if r.open_time in rows and not same_candle(r, rows[r.open_time])]
            if not mismatches:
                print(f"{symbol}: 无不一致记录", flush=True)
                continue
            start, end = min(r.open_time for r in mismatches), max(r.open_time for r in mismatches)
            verified = {}
            while start <= end:
                response = client.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": symbol,
                    "interval": "1m", "startTime": start, "endTime": end, "limit": 1000})
                response.raise_for_status()
                page = parse_klines(response.json())
                if not page:
                    break
                verified.update({r["open_time"]: r for r in page})
                start = page[-1]["open_time"] + 60000
            candidates = []
            for existing in mismatches:
                source = rows[existing.open_time]
                api = verified.get(existing.open_time)
                if (api and valid_candle(source) and valid_candle(api)
                        and same_candle(SimpleNamespace(**api), source)
                        and api["taker_buy_volume"] == source["taker_buy_volume"]
                        and api["taker_buy_quote_volume"] == source["taker_buy_quote_volume"]):
                    candidates.append((existing, source))
            backup = [{"id": old_row.id, "symbol": symbol, "open_time": old_row.open_time,
                       **{k: getattr(old_row, k) for k in FIELDS_TO_WRITE}} for old_row, _ in candidates]
            path = logdir / f"{stamp}-{symbol}.json"
            path.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")
            repaired = 0
            with engine.begin() as conn:
                for old_row, source in candidates:
                    # Compare every original value under the UPDATE to avoid clobbering concurrent changes.
                    predicates = [Kline.id == old_row.id] + [getattr(Kline, k).is_not_distinct_from(getattr(old_row, k)) for k in FIELDS_TO_WRITE]
                    result = conn.execute(update(Kline).where(*predicates).values(**{k: source[k] for k in FIELDS_TO_WRITE}))
                    repaired += result.rowcount
            print(f"{symbol} {month}: 不一致 {len(mismatches)}，双来源确认 {len(candidates)}，修复 {repaired}，备份 {path}", flush=True)
    engine.dispose()


if __name__ == "__main__":
    main()
