"""Backfill BTC/ETH candle gaps and missing taker fields from Binance archives."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import httpx
from sqlalchemy import inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.models import Kline  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um"
FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


def archive_rows(client, url):
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    checksum = client.get(url + ".CHECKSUM")
    checksum.raise_for_status()
    expected = checksum.text.split()[0].lower()
    if hashlib.sha256(response.content).hexdigest() != expected:
        raise ValueError(f"文件校验失败: {url}")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"压缩文件内容异常: {url}")
        with archive.open(names[0]) as source:
            for row in csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig")):
                if row and row[0] == "open_time":
                    continue
                if len(row) < 11:
                    raise ValueError(f"K线字段不足: {url}")
                yield {"open_time": int(row[0]), "open": float(row[1]), "high": float(row[2]),
                       "low": float(row[3]), "close": float(row[4]), "volume": float(row[5]),
                       "close_time": int(row[6]), "quote_volume": float(row[7]), "trades": int(row[8]),
                       "taker_buy_volume": float(row[9]), "taker_buy_quote_volume": float(row[10])}


def read_archive(client, url):
    # Generator exceptions, including 404, occur on iteration.
    return list(archive_rows(client, url))


def api_day(client, symbol, day, end):
    """Use the official REST API where a daily archive is not yet published."""
    cursor = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    limit = min(cursor + 86_400_000 - 1, end)
    result = []
    from app.market.binance import parse_klines
    while cursor <= limit:
        response = client.get("https://fapi.binance.com/fapi/v1/klines", params={
            "symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": limit, "limit": 1000,
        })
        response.raise_for_status()
        page = parse_klines(response.json())
        if not page:
            break
        result.extend(page)
        cursor = page[-1]["open_time"] + 60000
    return result


def same_candle(existing, row):
    return (existing.close_time == row["close_time"] and existing.trades == row["trades"]
            and all(math.isclose(getattr(existing, k), row[k], rel_tol=1e-8, abs_tol=1e-8) for k in FIELDS))


def valid_candle(r):
    numbers = [r[k] for k in FIELDS] + [r["taker_buy_volume"], r["taker_buy_quote_volume"]]
    return (all(math.isfinite(x) for x in numbers) and r["open_time"] % 60000 == 0
            and r["close_time"] == r["open_time"] + 59999
            and min(r[k] for k in ("open", "high", "low", "close")) > 0
            and r["high"] >= max(r[k] for k in ("open", "low", "close"))
            and r["low"] <= min(r[k] for k in ("open", "high", "close"))
            and 0 <= r["taker_buy_volume"] <= r["volume"]
            and 0 <= r["taker_buy_quote_volume"] <= r["quote_volume"] and r["trades"] >= 0)


def apply_rows(engine, symbol, rows, start, end):
    rows = [r for r in rows if start <= r["open_time"] <= end]
    if not rows:
        return {"inserted": 0, "updated": 0, "mismatched": 0, "invalid_times": []}
    times = [r["open_time"] for r in rows]
    if len(times) != len(set(times)):
        raise ValueError("历史文件包含重复时间戳")
    invalid_times = [r["open_time"] for r in rows if not valid_candle(r)]
    rows = [r for r in rows if valid_candle(r)]
    result = {"inserted": 0, "updated": 0, "mismatched": 0, "invalid_times": invalid_times}
    if not rows:
        return result
    times = [r["open_time"] for r in rows]
    insert_fn = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    with engine.begin() as conn:
        existing = {r.open_time: r for r in conn.execute(select(Kline).where(
            Kline.symbol == symbol, Kline.interval == "1m", Kline.open_time >= min(times),
            Kline.open_time <= max(times))).all()}
        payload = []
        for row in rows:
            old = existing.get(row["open_time"])
            if old and not same_candle(old, row):
                result["mismatched"] += 1
                continue
            if old and old.taker_buy_volume is not None and old.taker_buy_quote_volume is not None:
                continue
            result["updated" if old else "inserted"] += 1
            payload.append({"symbol": symbol, "interval": "1m", **row})
        for offset in range(0, len(payload), 500):
            stmt = insert_fn(Kline).values(payload[offset:offset + 500])
            from sqlalchemy import func
            stmt = stmt.on_conflict_do_update(index_elements=["symbol", "interval", "open_time"], set_={
                "taker_buy_volume": func.coalesce(Kline.taker_buy_volume, stmt.excluded.taker_buy_volume),
                "taker_buy_quote_volume": func.coalesce(Kline.taker_buy_quote_volume, stmt.excluded.taker_buy_quote_volume),
            })
            conn.execute(stmt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], choices=["BTCUSDT", "ETHUSDT"])
    args = parser.parse_args()
    engine = make_engine(null_pool=True)
    # Add only the two nullable columns, without running unrelated application migrations.
    with engine.begin() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("klines")}
        for name in ("taker_buy_volume", "taker_buy_quote_volume"):
            if name not in cols:
                conn.execute(text(f"ALTER TABLE klines ADD COLUMN {name} FLOAT"))
    result = {}
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for symbol in args.symbols:
            with engine.connect() as conn:
                bounds = conn.execute(text("SELECT min(open_time), max(open_time) FROM klines WHERE symbol=:s AND interval='1m'"), {"s": symbol}).one()
            if bounds[0] is None:
                raise ValueError(f"{symbol} 没有已有历史范围")
            start, end = int(bounds[0]), min(int(bounds[1]), int(time.time() * 1000) // 60000 * 60000 - 60000)
            first = datetime.fromtimestamp(start / 1000, timezone.utc).date().replace(day=1)
            last = datetime.fromtimestamp(end / 1000, timezone.utc).date()
            totals = {"inserted": 0, "updated": 0, "mismatched": 0, "invalid_count": 0}
            month = first
            while month <= last:
                next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
                tag = month.strftime("%Y-%m")
                url = f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{tag}.zip"
                rows = read_archive(client, url)
                if not rows:
                    day = max(month, datetime.fromtimestamp(start / 1000, timezone.utc).date())
                    while day < next_month and day <= last:
                        tag_day = day.isoformat()
                        daily = f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{tag_day}.zip"
                        day_rows = read_archive(client, daily)
                        if not day_rows:
                            day_rows = api_day(client, symbol, day, end)
                        if not day_rows:
                            raise ValueError(f"历史文件及 API 无数据: {symbol} {tag_day}")
                        rows.extend(day_rows)
                        day += timedelta(days=1)
                part = apply_rows(engine, symbol, rows, start, end)
                for key in ("inserted", "updated", "mismatched"):
                    totals[key] += part[key]
                totals["invalid_count"] += len(part["invalid_times"])
                print(f"{symbol} {tag}: {part}", flush=True)
                month = next_month
            result[symbol] = {"start_ms": start, "end_ms": end, **totals}
    engine.dispose()
    path = ROOT / "output" / "taker_backfill_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return int(any(x["mismatched"] or x["invalid_count"] for x in result.values()))


if __name__ == "__main__":
    raise SystemExit(main())
