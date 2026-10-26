"""迭代8：回填 Top10 指标历史 + 恐惧贪婪指数历史（回测前置条件）。

币安 /futures/data/* 与 openInterestHist 支持 period=1d&limit=500 → 一次调用覆盖约 500 天；
FNG 支持 limit=500 日报历史。CoinGecko 市值排名无免费历史，回填时按当前常量注入
（报告需标注"市值排名存在轻微前视，结论需打折"）。

用法：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_metrics.py --proxy http://127.0.0.1:7890
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.market.metrics import fetch_market_ranks  # noqa: E402
from app.market.metrics_history import (  # noqa: E402
    FNG_SENTINEL,
    backfill_fear_greed,
    backfill_symbol_metrics,
)

TOP10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT",
         "DOGEUSDT", "JCTUSDT", "ORDERUSDT", "IRYSUSDT", "APRUSDT"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=TOP10)
    parser.add_argument("--proxy", default=None, help="如 http://127.0.0.1:7890")
    parser.add_argument("--period", default="1d", help="日报级：1d（一次覆盖 ~500 天）")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--db",
        default=None,
        help="可选：显式 SQLite 文件路径；留空则用 .env 的 DATABASE_URL",
    )
    args = parser.parse_args()

    symbols = [str(s).strip().upper() for s in args.symbols if str(s).strip()]
    proxy = args.proxy or None
    engine = make_engine(url=f"sqlite:///{args.db}" if args.db else None)
    client = BinanceFuturesClient(testnet=False, proxy=proxy)

    print(f"回填目标: {len(symbols)} 个币种 + FNG | period={args.period} limit={args.limit} "
          f"| 代理: {proxy or '直连'}")

    # 市值排名：无免费历史，取当前值按常量注入（前视风险见报告）
    ranks = fetch_market_ranks(proxy=proxy) or {}
    print(f"市值排行: {'已获取' if ranks else '获取失败(限流?)，排名按中性处理'}")

    with Session(engine) as db:
        fng = backfill_fear_greed(db, proxy=proxy, limit=args.limit)
        print(f"[{FNG_SENTINEL}] rows={fng['rows']} span_days={fng['span_days']}"
              + (f" ERROR={fng.get('error')}" if fng.get("error") else ""))
        total_rows, failed = 0, []
        for symbol in symbols:
            try:
                result = backfill_symbol_metrics(
                    client, db, symbol, period=args.period, limit=args.limit,
                    market_rank=ranks.get(symbol[:-4] if symbol.endswith("USDT") else symbol),
                )
            except Exception as exc:  # noqa: BLE001 - 单币种失败不阻断其余
                failed.append(symbol)
                print(f"[{symbol}] 失败: {str(exc)[:120]}")
                continue
            total_rows += result["rows"]
            if result["rows"] == 0:
                failed.append(symbol)
            print(f"[{symbol}] rows={result['rows']} span_days={result['span_days']}")
    engine.dispose()

    print(f"\n合计落库 {total_rows} 行；失败/空: {', '.join(failed) or '无'}")
    print("提示：市值排名为当前快照常量注入，回测中属轻微前视，结论需打折。")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
