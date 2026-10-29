r"""Simple ZigZag Double —— 按 TradingView Pine v5 原脚本口径的本地回测入口。

这个策略**不走**项目通用「信号 → 市价单」管线：Pine 的
``strategy.entry(..., stop=LEVEL)`` 是持久止损挂单，由专用撮合器
``app/backtest/zigzag_execution.py`` 逐 bar 回放（OHLC 路径、bar 间 gap 按
次根开盘价、已穿越 stop 转 market 单、反手、time_end 平仓）。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --symbol ETHUSDT --interval 15m --days 180
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --params "{\"use_short\": false}"
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.backtest import runner  # noqa: E402
from app.db import make_engine  # noqa: E402

CODE = "simple_zigzag_double"


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple ZigZag Double 本地回测（Pine stop 单口径）")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--initial", type=float, default=None,
                    help="默认 1_000_000（对齐 Pine 默认权益口径）")
    ap.add_argument("--fee", type=float, default=None,
                    help="默认 0.001（Pine commission_value=0.1）")
    ap.add_argument("--slippage", type=float, default=None, help="默认 0")
    ap.add_argument("--params", default="{}",
                    help='JSON 参数覆盖，如 \'{"use_short": false}\'')
    ap.add_argument("--output", default=None, help="结果 JSON 路径")
    ap.add_argument("--end-ms", type=int, default=None)
    args = ap.parse_args()

    overrides = json.loads(args.params)
    db = sessionmaker(bind=make_engine(read_only=True))()
    result = runner.run_backtest(
        db, args.symbol, args.interval, CODE,
        days=args.days, initial_cash=args.initial, fee=args.fee,
        slippage=args.slippage, param_overrides=overrides, end_ms=args.end_ms,
    )
    m = result["metrics"]
    print(f"策略       : {CODE} / {args.symbol} {args.interval} / {args.days} 天")
    print(f"执行引擎   : {m['execution']}")
    print(f"K 线       : {m['data_quality']['raw_bars']} 根，缺口 {m['data_quality']['gap_count']} 处")
    print(f"区间       : {m['split']['test_start']} ~ {m['split']['test_end']}")
    print(f"挂单信号   : {m['signal_bars']} 根")
    print(f"成交       : {m['total_trades']} 笔")
    print(f"净收益     : {m['total_return_pct']} %   最大回撤 {m['max_drawdown_pct']} %")
    print(f"买入持有   : {m['buy_hold_return_pct']} %")
    print(f"胜率/PF    : {m['win_rate']} % / {m['profit_factor']}")
    print(f"手续费     : {m['total_fees']}   持仓占比 {m['exposure_pct']} %")
    print(f"权益点     : {len(result['equity_curve'])}   成交明细 {len(result['trades'])} 条")
    for warning in m["warnings"]:
        print(f"提示       : {warning}")

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "symbol": args.symbol, "interval": args.interval, "days": args.days,
                "initial_cash": m["initial_cash"], "fee_rate": m["fee_rate"],
                "slippage": m["slippage"], "param_overrides": overrides,
            },
            "metrics": m,
            "trades": result["trades"],
            "equity_curve": result["equity_curve"],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                       encoding="utf-8")
        print(f"结果已写入 : {out}")


if __name__ == "__main__":
    main()
