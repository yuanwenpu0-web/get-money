r"""迭代5数据准备：写入黑名单 → 验证宇宙 → Top10（24h 成交额）→ 同步 1h 400 天 → 跨度核查。

经运行中的服务（127.0.0.1:8010）执行；K线按 (symbol, interval, open_time) upsert，可重跑。
用法：先启动服务（uvicorn app.main:app --host 127.0.0.1 --port 8010），再运行本脚本。
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8010"
PROXY = "http://127.0.0.1:7890"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import SessionLocal
from app.market.data_service import MarketDataService
BLACKLIST = ["ZECUSDT", "1000PEPEUSDT", "HYPEUSDT", "BNBUSDT", "BIRBUSDT", "BMTUSDT"]
TOP_N = 10
INTERVAL = "1h"
DAYS = 400

client = httpx.Client(base_url=BASE, trust_env=False, timeout=600.0)
proxy_client = httpx.Client(proxy=PROXY, timeout=60.0)

print("== 1. 写入黑名单（6 个负期望候选） ==")
before = client.get("/api/market/symbols").json()
print("写入前宇宙:", len(before["symbols"]), "| cached:", before["cached"])
updated = client.put("/api/settings", json={"symbol_blacklist": BLACKLIST}).json()
print("symbol_blacklist =", updated["symbol_blacklist"])

print("== 2. 验证黑名单生效（缓存键含过滤配置，立即生效） ==")
after = client.get("/api/market/symbols").json()
print("写入后宇宙:", len(after["symbols"]), "| cached:", after["cached"])
remaining = set(after["symbols"])
leaked = [symbol for symbol in BLACKLIST if symbol in remaining]
print("仍在宇宙中的黑名单币种:", leaked or "无（全部排除）")
assert not leaked, f"黑名单未生效: {leaked}"
print(f"宇宙 {len(before['symbols'])} → {len(after['symbols'])}")

print("== 3. 按 24h 成交额取 Top", TOP_N, "==")
tickers = proxy_client.get("https://fapi.binance.com/fapi/v1/ticker/24hr").json()
volumes = {row["symbol"]: float(row.get("quoteVolume") or 0) for row in tickers}
ranked = sorted(((symbol, volumes.get(symbol, 0.0)) for symbol in after["symbols"]),
                key=lambda item: item[1], reverse=True)
top = [symbol for symbol, _ in ranked[:TOP_N]]
for symbol, volume in ranked[:TOP_N]:
    print(f"  {symbol}: {volume / 1e6:.1f}M")

print("== 4. 同步", INTERVAL, DAYS, "天历史 ==")
response = client.post("/api/market/sync", json={
    "symbols": top, "source": "binance", "interval": INTERVAL, "days": DAYS,
})
response.raise_for_status()
result = response.json()
print(json.dumps({"requested": result.get("requested"), "inserted": result.get("inserted"),
                  "errors": result.get("errors")}, ensure_ascii=False))

print("== 5. 本地数据跨度核查（根数 + 起止 + span_days） ==")
db = SessionLocal()
step_ms = 3_600_000
spans = {}
for symbol in top:
    stats = MarketDataService(db).series_stats(symbol, INTERVAL)
    count, first, last = stats["count"], stats["first_open_time"], stats["last_open_time"]
    span_days = round((last - first) / 86_400_000, 1) if count else 0.0
    spans[symbol] = {"bars": count, "first": first, "last": last, "span_days": span_days}
    print(f"  {symbol}: {count} 根, span={span_days} 天 "
          f"({datetime.fromtimestamp(first / 1000, timezone.utc):%Y-%m-%d} → "
          f"{datetime.fromtimestamp(last / 1000, timezone.utc):%Y-%m-%d})")
db.close()
print("SYNC_DONE")
