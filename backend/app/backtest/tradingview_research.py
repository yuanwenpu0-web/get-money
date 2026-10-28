"""Research backtests of locally ported Pine signals on stored OHLC data."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from ..market.data_service import MarketDataService
from ..utils import interval_ms, now_ms
from .metrics import downsample, metrics_from_equity
from .tradingview_native import run_tradingview_backtest

MAX_BARS = 100_000


def run_research(db: Session, code: str, symbol: str, interval: str, days: int) -> dict:
    step = interval_ms(interval)
    count = days * 86_400_000 // step
    if count < 2 or count > MAX_BARS:
        raise ValueError(f"回测窗口须包含 2 至 {MAX_BARS} 根K线，请调整周期或天数")
    frame = MarketDataService(db).load_frame(symbol.upper(), interval, limit=count + 1)
    if not frame.empty:
        frame = frame.loc[frame.index.as_unit("ms").asi8 + step <= now_ms()].tail(count)
    if len(frame) != count:
        raise ValueError(f"本地已收盘K线不足：需要 {count} 根，实际 {len(frame)} 根")
    if frame.index.to_series().diff().iloc[1:].ne(pd.Timedelta(milliseconds=step)).any():
        raise ValueError("回测窗口有行情缺口，请先补齐数据")
    fields = ["open", "high", "low", "close"] + (["volume"] if "volume" in frame.columns else [])
    candles = [{"time": str(ts), **{key: float(row[key]) for key in fields}}
               for ts, row in frame.iterrows()]
    result = run_tradingview_backtest(candles, code)
    equity = pd.Series([row["equity"] for row in result["equity"]], index=frame.index)
    metrics = metrics_from_equity(equity, result["trades"], result["initial_capital"], interval)
    metrics.update({"symbol": symbol.upper(), "interval": interval, "strategy": code,
                    "position_pct": result["position_pct"], "commission_rate": result["commission_rate"]})
    assumptions = ["按源码默认参数，收盘产生市场单、下一根开盘成交；区间内预热。",
                   "末根未成交订单不执行，未平仓按末根收盘价估值。",
                   "不模拟滑点、保证金拒单与强平、交易所数量精度、盘中成交或实时逐 tick；结果不保证与 TradingView 完全一致。",
                   "交易明细最多返回最近 500 笔；指标包含全部已平仓交易。"]
    if code == "bb_ema_eth":
        assumptions.append("原脚本默认训练区间为 2018-01-01 至 2022-12-31 UTC（不含边界），区间外不产生订单。")
    return {
        "metrics": metrics, "equity_curve": downsample(equity),
        "trades": result["trades"][-500:], "open_position": result["open_position"],
        "assumptions": assumptions,
        "unfilled_final_orders": result["unfilled_final_orders"],
    }
