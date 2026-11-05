"""出场结构诊断：把「信号是否有效」和「出场规则是否在毁灭价值」分开。

事件研究口径：在每个信号上以次开价入场、固定持有 N 根、无止损、扣双边成本，
看单笔期望。若这个纯持有口径仍为负，说明信号本身没有 edge；
若为正而回测为负，说明止损/止盈/持仓上限在毁灭价值。

用法：
    .venv/Scripts/python.exe scripts/diagnose_flow_exit.py --days 360
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine
from app.market.data_service import MarketDataService
from app.strategy.registry import PrepareContext, get_spec
from app.strategy.rules import evaluate_signals, resolve_params
from app.config import DEFAULT_STATE

STRATEGY = "flow_imbalance_long"
MINUTE = 60_000


def event_study(frame: pd.DataFrame, params: dict, holds: tuple[int, ...], cost: float) -> dict:
    """纯持有口径：信号 bar 收盘后下一根开盘入场，持有 N 根后开盘离场。

    不使用止损、止盈、追踪止损，也不使用仓位与杠杆——只衡量方向本身的收益。
    """
    prepared = get_spec(STRATEGY).prepare(frame.copy(), params, PrepareContext())
    sig = evaluate_signals(prepared, STRATEGY, params)
    entries = np.flatnonzero(sig.direction.to_numpy() != 0)
    opens = prepared["open"].to_numpy(dtype=float)
    n = len(prepared)
    out = {}
    for hold in holds:
        returns = []
        for i in entries:
            exit_index = i + 1 + hold
            if exit_index >= n:
                continue
            entry = opens[i + 1]
            exit_price = opens[exit_index]
            if not (np.isfinite(entry) and np.isfinite(exit_price)) or entry <= 0:
                continue
            returns.append((exit_price / entry - 1) - cost)
        arr = np.array(returns)
        if len(arr) == 0:
            out[f"hold_{hold}"] = {"trades": 0}
            continue
        out[f"hold_{hold}"] = {
            "trades": int(len(arr)),
            "mean_pct": float(arr.mean()) * 100,
            "median_pct": float(np.median(arr)) * 100,
            "win_rate": float((arr > 0).mean()) * 100,
            "t_stat": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if len(arr) > 2 else None,
        }
    return out


def stop_study(frame: pd.DataFrame, params: dict, sl_mults: tuple[float, ...], target_r: float,
               max_hold: int, cost: float) -> dict:
    """带 ATR 止损/目标的单笔模拟：同一批信号，只改止损宽度，看期望如何变化。"""
    prepared = get_spec(STRATEGY).prepare(frame.copy(), params, PrepareContext())
    sig = evaluate_signals(prepared, STRATEGY, params)
    entries = np.flatnonzero(sig.direction.to_numpy() != 0)
    high = prepared["high"].to_numpy(dtype=float)
    low = prepared["low"].to_numpy(dtype=float)
    opens = prepared["open"].to_numpy(dtype=float)
    close = prepared["close"].to_numpy(dtype=float)
    atr = prepared["atr_abs"].to_numpy(dtype=float)
    n = len(prepared)
    out = {}
    for mult in sl_mults:
        returns = []
        for i in entries:
            if i + 1 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            entry = opens[i + 1]
            stop = close[i] - mult * atr[i]
            target = close[i] + target_r * mult * atr[i]
            if entry <= stop or target <= entry:
                continue
            result = None
            for j in range(i + 1, min(i + 1 + max_hold, n)):
                # 同一根内先判止损（保守）
                if low[j] <= stop:
                    result = stop / entry - 1
                    break
                if high[j] >= target:
                    result = target / entry - 1
                    break
            if result is None:
                exit_index = min(i + max_hold, n - 1)
                result = opens[exit_index] / entry - 1
            returns.append(result - cost)
        arr = np.array(returns)
        if len(arr) == 0:
            out[f"sl_{mult}"] = {"trades": 0}
            continue
        out[f"sl_{mult}"] = {
            "trades": int(len(arr)),
            "mean_pct": float(arr.mean()) * 100,
            "win_rate": float((arr > 0).mean()) * 100,
            "t_stat": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if len(arr) > 2 else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=360)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                                                         "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT"])
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--flow-window", type=int, default=48)
    parser.add_argument("--entry-quantile", type=float, default=0.95)
    args = parser.parse_args()

    cost = 2 * (args.fee + args.slippage)
    params = resolve_params(STRATEGY, DEFAULT_STATE, {
        "flow_window": args.flow_window,
        "entry_quantile": args.entry_quantile,
    })
    engine = make_engine(read_only=True, null_pool=True)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - args.days * 86_400_000
    from sqlalchemy.orm import Session
    session = Session(engine)
    service = MarketDataService(session)
    results = {}
    try:
        for symbol in args.symbols:
            frame = service.load_frame(symbol, "15m", start_ms=start_ms, end_ms=end_ms)
            if frame.empty:
                continue
            frame = frame.loc[(frame.index.as_unit("ms").asi8 + 15 * MINUTE) <= end_ms].copy()
            results[symbol] = {
                "event": event_study(frame, params, (4, 8, 16, 32, 64), cost),
                "stop": stop_study(frame, params, (1.0, 1.5, 2.0, 3.0, 5.0), 2.0, 96, cost),
            }
            print(f"{symbol} 完成", flush=True)
    finally:
        session.close()
        engine.dispose()

    lines = ["# 订单流策略出场结构诊断", "",
             f"{len(results)} 币，{args.days} 天，flow_window={args.flow_window}，"
             f"entry_quantile={args.entry_quantile}，双边成本 {cost * 100:.3f}%。", "",
             "## 纯持有（无止损止盈）", "",
             "| 币种 | 持有4根 | 持有8根 | 持有16根 | 持有32根 | 持有64根 |",
             "|---|---:|---:|---:|---:|---:|"]
    holds = ("hold_4", "hold_8", "hold_16", "hold_32", "hold_64")
    for symbol, r in results.items():
        cells = []
        for key in holds:
            v = r["event"].get(key, {})
            cells.append(f"{v.get('mean_pct', 0):+.4f} ({v.get('trades', 0)})")
        lines.append(f"| {symbol} | " + " | ".join(cells) + " |")
    lines += ["", "跨币均值：", ""]
    for key in holds:
        values = [r["event"][key]["mean_pct"] for r in results.values()
                  if r["event"].get(key, {}).get("trades", 0) > 0]
        if values:
            lines.append(f"- {key}: {np.mean(values):+.4f}%（{len(values)} 币）")
    lines += ["", "## 带 ATR 止损（target = 2R，最长 96 根）", "",
              "| 币种 | SL=1.0×ATR | SL=1.5×ATR | SL=2.0×ATR | SL=3.0×ATR | SL=5.0×ATR |",
              "|---|---:|---:|---:|---:|---:|"]
    for symbol, r in results.items():
        cells = []
        for mult in (1.0, 1.5, 2.0, 3.0, 5.0):
            v = r["stop"].get(f"sl_{mult}", {})
            cells.append(f"{v.get('mean_pct', 0):+.4f} ({v.get('trades', 0)})")
        lines.append(f"| {symbol} | " + " | ".join(cells) + " |")
    lines += ["", "跨币均值：", ""]
    for mult in (1.0, 1.5, 2.0, 3.0, 5.0):
        values = [r["stop"][f"sl_{mult}"]["mean_pct"] for r in results.values()
                  if r["stop"].get(f"sl_{mult}", {}).get("trades", 0) > 0]
        if values:
            lines.append(f"- SL={mult}×ATR: {np.mean(values):+.4f}%（{len(values)} 币）")
    lines += ["", "## 判读", "",
              "- 纯持有为正、带止损为负 → 止损在毁灭价值，应放宽或关闭止损。",
              "- 两者都为负 → 信号本身没有 edge，调参无用。", ""]
    md = "\n".join(lines)
    (ROOT / "docs" / "flow-imbalance-diagnosis.md").write_text(md, encoding="utf-8")
    (ROOT / "docs" / "flow-imbalance-diagnosis.json").write_text(
        json.dumps({"args": vars(args), "results": results}, ensure_ascii=False, indent=2,
                   default=lambda o: None), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
