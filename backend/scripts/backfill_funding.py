r"""全宇宙资金费率回填：把 FundingRate 采集从 BTC/ETH 2 币扩展到库内全部 clean 币种。

P6 需要「独立信息源」覆盖全宇宙，才能把 funding_cum（累积费率/拥挤度代理）作为通用特征喂给
分层模型——此前库里只有 BTC/ETH 各 ~200 天，绝大多数币种没有资金费，funding 特征退化为全 0。

本脚本：
- 枚举库内全部 clean 币种（klines 表 distinct symbol，过滤非标准合约名）；
- 调用 MarketDataService.sync_funding_many 分页拉取资金费率历史（每 8 小时 1 条，
  1000 条约 333 天，一般 1~2 页），幂等 upsert，可重复运行；
- 输出覆盖统计（覆盖币种数、资金费总行数、时间跨度）。

用法：
.\.venv\Scripts\python.exe backend/scripts/backfill_funding.py --days 400 --workers 8
.\.venv\Scripts\python.exe backend/scripts/backfill_funding.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

REPO = Path(__file__).resolve().parents[2]  # 仓库根（docs 所在）
sys.path.insert(0, str(REPO / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.ml.tiers import is_clean_symbol  # noqa: E402
from app.models import FundingRate, Kline  # noqa: E402


def universe_symbols(db: Session, limit: int | None = None) -> list[str]:
    """库内全部 clean 币种（任意周期有 K线即视为可交易永续）。"""
    rows = db.execute(select(func.distinct(Kline.symbol))).scalars().all()
    cleaned = [s for s in rows if is_clean_symbol(s)]
    cleaned.sort()
    if limit:
        cleaned = cleaned[:limit]
    return cleaned


def coverage(db: Session) -> dict:
    """资金费当前覆盖：币种数、总行数、最早/最晚 funding_time。"""
    sym_count = db.execute(
        select(func.count(func.distinct(FundingRate.symbol)))
    ).scalar_one()
    row_count = db.execute(select(func.count(FundingRate.id))).scalar_one()
    bounds = db.execute(
        select(func.min(FundingRate.funding_time), func.max(FundingRate.funding_time))
    ).one()
    return {
        "symbols": int(sym_count),
        "rows": int(row_count),
        "first_ms": int(bounds[0] or 0),
        "last_ms": int(bounds[1] or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="", help="显式币种列表（逗号分隔）；缺省用全库 clean 宇宙")
    ap.add_argument("--days", type=int, default=400, help="每币回看天数（匹配回测深度）")
    ap.add_argument("--workers", type=int, default=8, help="并发拉取线程数")
    ap.add_argument("--limit", type=int, default=None, help="仅取宇宙前 N 个币种（抽样/调试）")
    ap.add_argument("--output", type=Path, default=REPO / "docs" / "backfill-funding.json")
    args = ap.parse_args()

    engine = make_engine()
    with Session(engine) as db:
        before = coverage(db)
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if is_clean_symbol(s.strip().upper())]
        else:
            symbols = universe_symbols(db, args.limit)
        print(f"待回填币种：{len(symbols)} 个；days={args.days}；workers={args.workers}")
        print(f"回填前覆盖：{before['symbols']} 币 / {before['rows']} 行")

        started = time.time()
        svc = MarketDataService(db)
        result = svc.sync_funding_many(symbols, days=args.days, workers=args.workers)
        after = coverage(db)

    first = datetime.fromtimestamp(after["first_ms"] / 1000, timezone.utc).date() if after["first_ms"] else None
    last = datetime.fromtimestamp(after["last_ms"] / 1000, timezone.utc).date() if after["last_ms"] else None
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "requested": len(symbols),
        "inserted": result["inserted"],
        "error_count": len(result["errors"]),
        "errors": result["errors"][:20],
        "coverage_before": before,
        "coverage_after": after,
        "elapsed_sec": round(time.time() - started, 1),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n本次插入：{result['inserted']} 行；错误 {len(result['errors'])} 条")
    print(f"回填后覆盖：{after['symbols']} 币 / {after['rows']} 行")
    if first:
        print(f"时间跨度：{first} ~ {last}")
    print(f"耗时 {report['elapsed_sec']}s；报告 -> {args.output}")


if __name__ == "__main__":
    main()
