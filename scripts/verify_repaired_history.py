"""Verify repairs and refresh only the corresponding persisted progress records."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys

import httpx
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.models import Kline  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from backfill_all_history import CORRECTIONS, RateLimitedClient  # noqa: E402
from backfill_taker_data import BASE, read_archive, same_candle  # noqa: E402


def main():
    db = sqlite3.connect(ROOT / "output" / "all_history_backfill.sqlite")
    engine = make_engine(read_only=True, null_pool=True)
    entries = db.execute("SELECT symbol,month,detail FROM months WHERE status='partial' AND mismatched>0 AND month='2026-10'").fetchall()
    with httpx.Client(timeout=90, follow_redirects=True) as raw:
        client = RateLimitedClient(raw)
        for symbol, month, detail in entries:
            if symbol not in MAJOR_SYMBOLS:
                continue
            details = json.loads(detail)
            if details.get("missing_days") or details.get("invalid_times"):
                continue
            rows = read_archive(client, f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip")
            if not rows:
                continue
            sources = {row["open_time"]: row for row in rows}
            with engine.connect() as conn:
                stored = conn.execute(select(Kline).where(Kline.symbol == symbol, Kline.interval == "1m",
                    Kline.open_time >= min(sources), Kline.open_time <= max(sources))).all()
            problems = len(set(sources) - {row.open_time for row in stored})
            problems += sum(not same_candle(row, sources[row.open_time]) or row.taker_buy_volume is None
                            or row.taker_buy_quote_volume is None
                            or not math.isclose(row.taker_buy_volume or 0, sources[row.open_time]["taker_buy_volume"], rel_tol=1e-8, abs_tol=1e-8)
                            or not math.isclose(row.taker_buy_quote_volume or 0, sources[row.open_time]["taker_buy_quote_volume"], rel_tol=1e-8, abs_tol=1e-8)
                            for row in stored if row.open_time in sources)
            if not problems:
                db.execute("UPDATE months SET status='done',mismatched=0 WHERE symbol=? AND month=? AND status='partial'", (symbol, month))
                db.commit()
            print(f"{symbol} {month} 尚有 {problems} 条不一致", flush=True)
    corrections = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    for entry in corrections:
        symbol, fixed = entry["symbol"], entry["corrected"]
        with engine.connect() as conn:
            row = conn.execute(select(Kline).where(Kline.symbol == symbol, Kline.interval == "1m", Kline.open_time == fixed["open_time"])).one_or_none()
        if row and all(getattr(row, key) == value for key, value in fixed.items()):
            status = db.execute("SELECT detail,mismatched FROM months WHERE symbol=? AND month='2023-11' AND status='partial'", (symbol,)).fetchone()
            if status:
                detail, mismatch = json.loads(status[0]), status[1]
                detail["invalid_times"] = [t for t in detail.get("invalid_times", []) if t != fixed["open_time"]]
                if not (mismatch or detail.get("invalid_times") or detail.get("missing_days")):
                    db.execute("UPDATE months SET status='edge',detail=? WHERE symbol=? AND month='2023-11'", (json.dumps(detail), symbol))
                    db.commit()
                    print(f"{symbol} 2023-11 源数据修复已核验", flush=True)
    db.close()
    engine.dispose()


if __name__ == "__main__":
    main()
