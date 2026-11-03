"""Resume a bounded, rate-limited USD-M USDT perpetual candle backfill."""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import threading
import time

import httpx
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from backfill_taker_data import BASE, api_day, apply_rows, read_archive, valid_candle  # noqa: E402

STATE = ROOT / "output" / "all_history_backfill.sqlite"
MIN_FREE = 40 * 1024**3
CORRECTIONS = ROOT / "output" / "candle_corrections.json"


class AccessBlocked(RuntimeError):
    """An IP ban or WAF block requires stopping all download workers."""


class RateLimitedClient:
    """One shared limiter for all workers; API and archives have separate pacing."""

    def __init__(self, client, archive_interval=0.4, api_interval=2.0):
        self.client = client
        self.intervals = {"api": api_interval, "archive": archive_interval}
        self.next_request = {"api": 0.0, "archive": 0.0}
        self.cooldown = 0.0
        self.blocked = False
        self.condition = threading.Condition()

    def _acquire(self, kind):
        with self.condition:
            while True:
                if self.blocked:
                    raise AccessBlocked("访问被封禁或防火墙拦截，停止请求")
                now = time.monotonic()
                delay = max(self.cooldown, self.next_request[kind]) - now
                if delay <= 0:
                    self.next_request[kind] = now + self.intervals[kind]
                    return
                self.condition.wait(timeout=delay)

    def _defer(self, seconds):
        with self.condition:
            self.cooldown = max(self.cooldown, time.monotonic() + seconds)
            self.condition.notify_all()

    @staticmethod
    def _retry_after(response):
        value = response.headers.get("Retry-After", "")
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return 60.0

    def get(self, url, **kwargs):
        kind = "api" if httpx.URL(url).host == "fapi.binance.com" else "archive"
        for attempt in range(4):
            self._acquire(kind)
            try:
                response = self.client.get(url, **kwargs)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                self._defer(min(60, 2 ** (attempt + 1)) + random.random())
                continue
            if response.status_code in (403, 418):
                with self.condition:
                    self.blocked = True
                    self.condition.notify_all()
                raise AccessBlocked(f"币安返回 HTTP {response.status_code}，停止全部请求")
            if response.status_code == 429:
                delay = max(self._retry_after(response), min(300, 30 * 2 ** attempt))
                # Respect the full server-specified delay; do not cap it.
                self._defer(delay + random.random())
                print(f"HTTP 429：所有下载线程退避至少 {delay:.0f} 秒", flush=True)
                if attempt == 3:
                    response.raise_for_status()
                continue
            if response.status_code in (408, 500, 502, 503, 504):
                if attempt == 3:
                    response.raise_for_status()
                self._defer(min(60, 2 ** (attempt + 1)) + random.random())
                continue
            return response
        raise RuntimeError("请求重试次数已用尽")


def free_space_check(engine):
    with engine.connect() as conn:
        if engine.dialect.name != "postgresql":
            raise RuntimeError("全量回补仅支持本地 PostgreSQL")
        path = conn.execute(text("show data_directory")).scalar_one()
    if not Path(path).is_dir():
        raise RuntimeError("无法检查 PostgreSQL 数据目录的可用空间")
    if shutil.disk_usage(path).free < MIN_FREE:
        raise RuntimeError("PostgreSQL 数据目录可用空间低于 40 GiB，已安全停止")


def tasks_for(contracts, first, end_ms, completed):
    for info in contracts:
        symbol = info["symbol"]
        start = max(int(first.timestamp() * 1000), int(info["onboardDate"]))
        start = (start + 59999) // 60000 * 60000
        end = min(end_ms, int(info.get("deliveryDate", end_ms + 60000)) - 60000)
        if start > end:
            continue
        start_day = datetime.fromtimestamp(start / 1000, timezone.utc).date()
        last_day = datetime.fromtimestamp(end / 1000, timezone.utc).date()
        month = start_day.replace(day=1)
        while month <= last_day:
            next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
            tag = month.strftime("%Y-%m")
            full = month > start_day.replace(day=1) and next_month <= last_day
            if not (full and (symbol, tag) in completed):
                yield symbol, month, next_month, start, end, full
            month = next_month


def apply_verified_corrections(symbol, by_time, corrections):
    for entry in corrections:
        if entry["symbol"] != symbol:
            continue
        original, fixed = entry["original"], entry["corrected"]
        key = original["open_time"]
        if key not in by_time:
            continue
        if by_time[key] != original:
            raise ValueError(f"{symbol} 源数据已变化，不应用先前的纠正记录")
        if ({k: v for k, v in original.items() if k != "volume"} !=
                {k: v for k, v in fixed.items() if k != "volume"} or not valid_candle(fixed)):
            raise ValueError(f"{symbol} 纠正记录无效")
        by_time[key] = fixed.copy()


def fetch_month(client, task):
    symbol, month, next_month, start, end, full = task
    start_day = datetime.fromtimestamp(start / 1000, timezone.utc).date()
    last_day = datetime.fromtimestamp(end / 1000, timezone.utc).date()
    tag = month.strftime("%Y-%m")
    url = f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{tag}.zip"
    rows = read_archive(client, url)
    by_time = {r["open_time"]: r for r in rows}
    if len(by_time) != len(rows):
        raise ValueError("月文件包含重复分钟")
    missing = []
    day = max(month, start_day)
    while day < next_month and day <= last_day:
        day_start = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        lo, hi = max(start, day_start), min(end, day_start + 86_400_000 - 60000)
        if any(t not in by_time for t in range(lo, hi + 1, 60000)):
            url = f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
            day_rows = read_archive(client, url)
            if not day_rows:
                day_rows = api_day(client, symbol, day, end)
            for row in day_rows:
                by_time.setdefault(row["open_time"], row)
            if any(t not in by_time for t in range(lo, hi + 1, 60000)):
                missing.append(day.isoformat())
        day += timedelta(days=1)
    if CORRECTIONS.exists():
        apply_verified_corrections(symbol, by_time, json.loads(CORRECTIONS.read_text(encoding="utf-8")))
    return list(by_time.values()), missing


def major_contracts(symbols):
    """Use the project's major-coin list in its existing priority order."""
    available = {x["symbol"]: x for x in symbols if x.get("quoteAsset") == "USDT"
                 and x.get("contractType") == "PERPETUAL"}
    missing = [s for s in MAJOR_SYMBOLS if s not in available]
    if missing:
        print(f"交易所未列出以下主流永续合约: {', '.join(missing)}", flush=True)
    return [available[s] for s in MAJOR_SYMBOLS if s in available]


def run(workers=3):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE)
    db.execute("CREATE TABLE IF NOT EXISTS months (symbol TEXT, month TEXT, status TEXT, inserted INTEGER, updated INTEGER, mismatched INTEGER, detail TEXT, PRIMARY KEY(symbol, month))")
    engine = make_engine(null_pool=True)
    try:
        free_space_check(engine)
        completed = set(db.execute("SELECT symbol,month FROM months WHERE status='done'"))
        with httpx.Client(timeout=90, follow_redirects=True) as raw:
            client = RateLimitedClient(raw)
            response = client.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
            response.raise_for_status()
            contracts = major_contracts(response.json()["symbols"])
            if not contracts:
                raise RuntimeError("交易所未返回主流币 U 本位永续合约，停止回补")
            now = datetime.now(timezone.utc)
            year = now.year - 3
            first = now.replace(year=year, day=min(now.day, calendar.monthrange(year, now.month)[1]), second=0, microsecond=0)
            end = int(now.timestamp() * 1000) // 60000 * 60000 - 60000
            print(f"主流 USDT 永续 {len(contracts)} 个，下载线程 {workers}，起点 {first.isoformat()}，币种: {', '.join(x['symbol'] for x in contracts)}", flush=True)
            tasks = iter(tasks_for(contracts, first, end, completed))
            incomplete = False
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="download") as pool:
                pending = {}
                def fill():
                    while len(pending) < workers:
                        task = next(tasks, None)
                        if task is None:
                            break
                        pending[pool.submit(fetch_month, client, task)] = task
                fill()
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = pending.pop(future)
                        symbol, month, next_month, start, end, full = task
                        tag = month.strftime("%Y-%m")
                        try:
                            rows, missing = future.result()
                        except AccessBlocked:
                            for f in pending:
                                f.cancel()
                            raise
                        except Exception as exc:
                            incomplete = True
                            db.execute("INSERT OR REPLACE INTO months VALUES (?,?,?,?,?,?,?)", (symbol, tag, "error", 0, 0, 0, f"{type(exc).__name__}: {exc}"))
                            db.commit()
                            print(f"{symbol} {tag} error={type(exc).__name__}", flush=True)
                            continue
                        free_space_check(engine)
                        month_start = int(datetime.combine(month, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
                        month_end = int(datetime.combine(next_month, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000) - 60000
                        try:
                            counts = apply_rows(engine, symbol, rows, max(start, month_start), min(end, month_end))
                        except ValueError as exc:
                            incomplete = True
                            db.execute("INSERT OR REPLACE INTO months VALUES (?,?,?,?,?,?,?)", (symbol, tag, "error", 0, 0, 0, str(exc)))
                            db.commit()
                            print(f"{symbol} {tag} 数据错误: {exc}", flush=True)
                            continue
                        status = "partial" if missing or counts["mismatched"] or counts["invalid_times"] else ("done" if full else "edge")
                        incomplete |= status == "partial"
                        db.execute("INSERT OR REPLACE INTO months VALUES (?,?,?,?,?,?,?)", (symbol, tag, status,
                                   counts["inserted"], counts["updated"], counts["mismatched"], json.dumps({"missing_days": missing, "invalid_times": counts["invalid_times"]})))
                        db.commit()
                        print(f"{symbol} {tag} {status} {counts} missing_days={len(missing)}", flush=True)
                    fill()
            if incomplete:
                raise RuntimeError("回补存在错误或未完成月份，进度已保存，可在检查后重新运行")
    finally:
        db.close()
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    run(parser.parse_args().workers)
