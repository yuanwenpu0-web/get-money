"""P9：回填正交衍生品情绪数据（OI/多空比/taker 比）到 sentiment 表。

数据源：币安 /futures/data/*（免 Key 公开），每 symbol 4 个请求，
各返回 1h×500 条（≈21 天，API 硬上限 30 天）→ 不做长历史，定位横截面 IC 探针。

用法（从项目根运行）：
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_sentiment.py \
        --workers 8
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_sentiment.py \
        --symbols BTCUSDT,ETHUSDT --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import engine as _app_engine  # noqa: E402
from app.models import Base  # noqa: E402
from app.state import all_state  # noqa: E402

# 新表 Sentiment 由 create_all 幂等创建（应用启动期才会自动建，脚本期需显式调用）
Base.metadata.create_all(bind=_app_engine, tables=[Base.metadata.tables["sentiment"]])
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.market import universe  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None, help="HTTP 代理，默认读 state.proxy_url")
    ap.add_argument("--symbols", default="", help="逗号分隔白名单，缺省用全宇宙")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（调试）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Session(_app_engine)
    state = all_state(db)
    proxy = args.proxy or state.get("proxy_url") or None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        blacklist = universe._blacklist(state)
        info = BinanceFuturesClient(testnet=False, proxy=proxy).exchange_info()
        symbols = [s for s in universe._filter(info.get("symbols", [])) if s not in set(blacklist)]
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"目标宇宙: {len(symbols)} 币 (workers={args.workers}, proxy={proxy})")
    if args.dry_run:
        print("  sample:", symbols[:15])
        return 0

    svc = MarketDataService(db, BinanceFuturesClient(testnet=False, proxy=proxy))
    res = svc.sync_sentiment_many(symbols, workers=args.workers)
    report = {
        "requested": res["requested"],
        "inserted": res["inserted"],
        "error_count": len(res["errors"]),
        "errors": res["errors"][:20],
    }
    out = ROOT / "docs" / "backfill-sentiment.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"== 完成 ==  插入 {res['inserted']} 行，{res['requested']} 币，{len(res['errors'])} 错误")
    for err in res["errors"][:10]:
        print("  ERR", err)
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
