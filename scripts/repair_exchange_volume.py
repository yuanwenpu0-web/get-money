"""Rebuild verified 1m volume from Binance aggTrades for known bad source bars."""
from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import httpx
from sqlalchemy.dialects.postgresql import insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.market.binance import parse_klines  # noqa: E402
from app.models import Kline  # noqa: E402
from backfill_all_history import RateLimitedClient  # noqa: E402
from backfill_taker_data import BASE, read_archive, valid_candle  # noqa: E402

TIMESTAMP = 1701347700000
CORRECTIONS = ROOT / "output" / "candle_corrections.json"


def rebuild(source, trades):
    """Require full trade-id continuity and agreement on every other candle field."""
    if not trades:
        raise ValueError("分钟内没有成交")
    for i, row in enumerate(trades):
        if len(row) < 7 or row[6].lower() not in ("true", "false"):
            raise ValueError("成交记录格式错误")
        if not source["open_time"] <= int(row[5]) < source["open_time"] + 60000:
            raise ValueError("成交时间超出分钟")
        if int(row[4]) < int(row[3]) or (i and (int(row[3]) != int(trades[i - 1][4]) + 1 or int(row[5]) < int(trades[i - 1][5]))):
            raise ValueError("成交 ID 或时间不连续")
    prices = [Decimal(row[1]) for row in trades]
    volumes = [Decimal(row[2]) for row in trades]
    if any(p <= 0 for p in prices) or any(q <= 0 for q in volumes):
        raise ValueError("成交价格或数量异常")
    buy = [i for i, row in enumerate(trades) if row[6].lower() == "false"]
    derived = {"open": prices[0], "high": max(prices), "low": min(prices), "close": prices[-1],
               "volume": sum(volumes), "quote_volume": sum(p*q for p, q in zip(prices, volumes)),
               "trades": sum(int(row[4]) - int(row[3]) + 1 for row in trades),
               "taker_buy_volume": sum(volumes[i] for i in buy),
               "taker_buy_quote_volume": sum(prices[i]*volumes[i] for i in buy)}
    for key, value in derived.items():
        if key != "volume" and Decimal(str(source[key])) != value:
            raise ValueError(f"逐笔成交与 K 线 {key} 不一致")
    fixed = {**source, "volume": float(derived["volume"])}
    if not valid_candle(fixed):
        raise ValueError("修复后 K 线仍然无效")
    return fixed


def main():
    results = []
    with httpx.Client(timeout=90, follow_redirects=True) as raw:
        client = RateLimitedClient(raw)
        for symbol in ("XRPUSDT", "DOTUSDT", "SUIUSDT"):
            archive = read_archive(client, f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-2023-11.zip")
            source = next(r for r in archive if r["open_time"] == TIMESTAMP)
            if valid_candle(source):
                raise ValueError(f"{symbol} 源 K 线已有效，请重新核查而非覆盖")
            response = client.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": symbol,
                "interval": "1m", "startTime": TIMESTAMP, "endTime": TIMESTAMP + 59999, "limit": 1})
            response.raise_for_status()
            if parse_klines(response.json()) != [source]:
                raise ValueError(f"{symbol} 历史 K 线与 REST 不一致")
            url = f"{BASE}/daily/aggTrades/{symbol}/{symbol}-aggTrades-2023-11-30.zip"
            response = client.get(url); response.raise_for_status()
            checksum = client.get(url + ".CHECKSUM"); checksum.raise_for_status()
            digest = hashlib.sha256(response.content).hexdigest()
            if digest != checksum.text.split()[0].lower():
                raise ValueError(f"{symbol} 成交文件校验失败")
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive_zip:
                names = [n for n in archive_zip.namelist() if n.endswith(".csv")]
                if len(names) != 1:
                    raise ValueError("成交压缩文件异常")
                with archive_zip.open(names[0]) as source_file:
                    reader = csv.reader(io.TextIOWrapper(source_file, encoding="utf-8-sig"))
                    trades = [r for r in reader if len(r) >= 7 and r[5].isdigit() and TIMESTAMP <= int(r[5]) < TIMESTAMP + 60000]
            corrected = rebuild(source, trades)
            results.append({"symbol": symbol, "original": source, "corrected": corrected,
                            "agg_trades_url": url, "agg_trades_sha256": digest, "aggregate_count": len(trades)})
            print(f"{symbol} 成交记录 {len(trades)}，原成交量 {source['volume']}，重建 {corrected['volume']}", flush=True)
    CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    temp = CORRECTIONS.with_suffix(".tmp")
    temp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(CORRECTIONS)
    engine = make_engine(null_pool=True)
    with engine.begin() as conn:
        for entry in results:
            stmt = insert(Kline).values(symbol=entry["symbol"], interval="1m", **entry["corrected"])
            inserted = conn.execute(stmt.on_conflict_do_nothing(index_elements=["symbol", "interval", "open_time"])).rowcount
            print(f"{entry['symbol']} 修复记录插入 {inserted}", flush=True)
    engine.dispose()


if __name__ == "__main__":
    main()
