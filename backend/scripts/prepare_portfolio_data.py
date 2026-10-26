"""迭代4数据准备：从运行中的服务取流动性过滤后的宇宙，按 24h 成交额取 Top N，
经服务同步 1h / N 天历史。幂等可重跑（K线按 (symbol, interval, open_time) upsert）。

用法：先启动服务（uvicorn app.main:app --host 127.0.0.1 --port 8010），再运行本脚本。
"""

import json
import sys

import httpx

BASE = "http://127.0.0.1:8010"
PROXY = "http://127.0.0.1:7890"
TOP_N = 20
INTERVAL = "1h"
DAYS = 200

client = httpx.Client(base_url=BASE, trust_env=False, timeout=600.0)
proxy_client = httpx.Client(proxy=PROXY, timeout=60.0)

print("== 1. 获取流动性过滤后的宇宙 ==")
universe = client.get("/api/market/symbols").json()
symbols = universe["symbols"]
print("宇宙数量:", len(symbols), "| cached:", universe["cached"], "| source:", universe["source"])

print("== 2. 拉取 24h 成交额并排序取 Top", TOP_N, "==")
tickers = proxy_client.get("https://fapi.binance.com/fapi/v1/ticker/24hr").json()
volumes = {row["symbol"]: float(row.get("quoteVolume") or 0) for row in tickers}
ranked = sorted(
    ((symbol, volumes.get(symbol, 0.0)) for symbol in symbols),
    key=lambda item: item[1], reverse=True,
)
top = [symbol for symbol, _ in ranked[:TOP_N]]
print("Top", TOP_N, "币种（24h 成交额，百万美元）:")
for symbol, volume in ranked[:TOP_N]:
    print(f"  {symbol}: {volume / 1e6:.1f}M")
missing = [symbol for symbol, volume in ranked[:TOP_N] if volume <= 0]
if missing:
    print("警告：以下币种 ticker 无成交额数据:", missing)

print("== 3. 经服务同步", INTERVAL, DAYS, "天历史 ==")
response = client.post("/api/market/sync", json={
    "symbols": top, "source": "binance", "interval": INTERVAL, "days": DAYS,
})
response.raise_for_status()
result = response.json()
print(json.dumps({
    "requested": result.get("requested"),
    "inserted": result.get("inserted"),
    "errors": result.get("errors"),
}, ensure_ascii=False))

print("== 4. 本地数据量核查 ==")
stats = client.get("/api/market/stats").json()["stats"]
by_symbol = {row["symbol"]: row for row in stats if row.get("interval") == INTERVAL}
for symbol in top:
    row = by_symbol.get(symbol)
    print(f"  {symbol}: {row['count'] if row else 0} 根")
short = [symbol for symbol in top if not by_symbol.get(symbol) or by_symbol[symbol]["count"] < 4000]
print("不足 4000 根的币种:", short or "无")
print("SYNC_DONE")
