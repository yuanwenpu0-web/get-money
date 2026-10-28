"""Run the locally ported TradingView strategies on stored, contiguous candles.

Example: PYTHONPATH=backend python backend/scripts/backtest_tradingview_native.py
The window ends at the latest completed stored candle, not the current time.
"""

import argparse
import json

import pandas as pd
from sqlalchemy.orm import Session

from app.backtest.tradingview_native import SOURCES, run_tradingview_backtest
from app.db import make_engine
from app.market.data_service import MarketDataService
from app.utils import interval_ms, now_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    step = interval_ms(args.interval)
    if args.days < 1:
        parser.error("days must be positive")
    count = args.days * 86_400_000 // step
    if count < 201:
        parser.error("window must contain at least 201 candles")
    engine = make_engine(args.database_url, read_only=True)
    try:
        with Session(engine) as db:
            frame = MarketDataService(db).load_frame(args.symbol.upper(), args.interval, limit=count + 1)
        frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= now_ms()].tail(count)
        if len(frame) != count or frame.index.to_series().diff().iloc[1:].ne(pd.Timedelta(milliseconds=step)).any():
            raise ValueError("requested window is unavailable or contains candle gaps")
        fields = ["open", "high", "low", "close"] + (["volume"] if "volume" in frame.columns else [])
        candles = [{"time": str(index), **{key: float(row[key]) for key in fields}}
                   for index, row in frame.iterrows()]
        output = {"symbol": args.symbol.upper(), "interval": args.interval, "bars": len(candles),
                  "start_utc": str(frame.index[0]), "end_open_utc": str(frame.index[-1]), "strategies": {}}
        for code in SOURCES:
            result = run_tradingview_backtest(candles, code)
            curve = pd.Series([v["equity"] for v in result["equity"]])
            drawdown = ((curve / curve.cummax().clip(lower=result["initial_capital"])) - 1).min() * 100
            trades = result["trades"]
            output["strategies"][code] = {
                "source": result["source"], "initial_capital": result["initial_capital"],
                "position_pct": result["position_pct"], "commission_rate": result["commission_rate"],
                "final_equity": round(result["final_equity"], 2),
                "return_pct": round((result["final_equity"] / result["initial_capital"] - 1) * 100, 4),
                "max_drawdown_pct": round(float(drawdown), 4), "closed_trades": len(trades),
                "win_rate_pct": round(100 * sum(t["pnl"] > 0 for t in trades) / len(trades), 2) if trades else None,
                "open_position": result["open_position"], "unfilled_final_orders": result["unfilled_final_orders"],
            }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
