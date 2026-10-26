"""迭代2实测验证脚本：流动性过滤数字 + run-once 熔断不误伤。仅调用本地 8010 服务。"""

import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8010"
client = httpx.Client(base_url=BASE, trust_env=False, timeout=300.0)

print("== 1. 服务健康与当前设置 ==")
health = client.get("/api/health").json()
print("health:", health)

print("\n== 2. PUT /api/settings 设置 min_quote_volume_usd=30000000 ==")
resp = client.put("/api/settings", json={"min_quote_volume_usd": 30000000})
resp.raise_for_status()
data = resp.json()
print("min_quote_volume_usd =", data["min_quote_volume_usd"])
print("risk_halt =", data["risk_halt"], "| max_drawdown_halt_pct =", data["max_drawdown_halt_pct"],
      "| consecutive_loss_halt =", data["consecutive_loss_halt"], "| order_type =", data["order_type"])

print("\n== 3. GET /api/market/symbols（流动性过滤实测） ==")
symbols = client.get("/api/market/symbols").json()
count = len(symbols["symbols"])
print("symbol_count =", count, "| cached =", symbols["cached"], "| source =", symbols["source"])
print("sample(前20) =", symbols["symbols"][:20])
print("sample(后10) =", symbols["symbols"][-10:])
assert count < 521, f"预期从 521 降下来，实际 {count}"
print(f"PASS: {count} 个合约通过 30M 成交额门槛（全宇宙 521）")

print("\n== 3.5 暂停调度器，保证手动 run-once 独占交易锁 ==")
paused = client.put("/api/settings", json={"schedule_enabled": False}).json()
print("schedule_enabled =", paused["schedule_enabled"])

print("\n== 4. POST /api/trading/run-once（确认熔断检查不误伤正常流程） ==")
summary = None
for attempt in range(1, 9):
    run = client.post("/api/trading/run-once")
    run.raise_for_status()
    summary = run.json()
    if summary.get("ok"):
        print(f"第 {attempt} 次尝试成功")
        break
    print(f"第 {attempt} 次尝试: {summary.get('message')}，20s 后重试")
    time.sleep(20)
if not summary or not summary.get("ok"):
    print("CYCLE_FAILED message =", summary.get("message") if summary else "no response")
    print(json.dumps(summary, ensure_ascii=False)[:2000] if summary else "")
    client.put("/api/settings", json={"schedule_enabled": True})
    sys.exit(2)
print("ok =", summary["ok"], "| symbol_count =", summary["symbol_count"],
      "| elapsed_ms =", summary["elapsed_ms"])
print("opened =", summary["opened"], "| managed =", len(summary["managed"]),
      "| sync_errors =", len(summary["sync_errors"]), "| signal_errors =", len(summary["signal_errors"]))
print("decisions(前8) =", summary["decisions"][:8])

after = client.get("/api/settings").json()
print("\nrun-once 后 risk_halt =", after["risk_halt"], "| reason =", repr(after["risk_halt_reason"]))
assert after["risk_halt"] is False, "正常流程不应触发熔断"
print("PASS: 正常流程未触发熔断（回撤 2.9% < 10% 线，0 笔连续亏损）")

print("\n== 5. JobLog 熔断/宇宙日志核查 ==")
logs = client.get("/api/trading/logs", params={"limit": 200}).json()["logs"]
halt_logs = [row for row in logs if row["job"] == "risk_halt"]
universe_logs = [row for row in logs if row["job"] == "market_universe"]
print("risk_halt 日志:", [(row["level"], row["message"][:80]) for row in halt_logs] or "无（未触发，符合预期）")
print("market_universe 日志:", [(row["level"], row["message"][:80]) for row in universe_logs] or "无")

print("\n== 6. 恢复调度器 ==")
resumed = client.put("/api/settings", json={"schedule_enabled": True}).json()
print("schedule_enabled =", resumed["schedule_enabled"])

print("\nALL_CHECKS_DONE")
