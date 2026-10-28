"""每日资金费率收割建议（paper trading 清单，不下单）。

流程：增量同步 funding（近 3 天，幂等）→ 现货对过滤 → 选 K 个收割目标
→ 与上次持仓对比输出换手/成本 → 写 docs/harvest-latest.json 并打印清单。

用法：
    .venv/Scripts/python.exe scripts/harvest_daily.py [--k 10] [--no-sync] [--no-spot-filter]

P11 gate 结论依赖的口径：8h 桶、trailing 7 天、3 天重平衡——在
app/strategy/funding_harvest.py 中定义，勿在此处另设。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.models import FundingRate  # noqa: E402
from app.state import all_state  # noqa: E402
from app.strategy.funding_harvest import (  # noqa: E402
    match_spot_base,
    select_targets,
    rebalance_cost,
    spot_base_symbols,
    turnover,
)

LATEST = ROOT / "docs" / "harvest-latest.json"


def funding_symbols(db: Session) -> list[str]:
    rows = db.execute(select(FundingRate.symbol).distinct()).scalars().all()
    return sorted(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-rate", type=float, default=0.0,
                    help="trailing 8h 均费率下限（默认 0，负费率环境自动空仓）")
    ap.add_argument("--no-sync", action="store_true", help="跳过增量同步（调试用）")
    ap.add_argument("--no-spot-filter", action="store_true",
                    help="跳过现货对过滤（网络不可用时降级）")
    ap.add_argument("--sync-days", type=int, default=3)
    args = ap.parse_args()

    engine = make_engine()
    with Session(engine) as db:
        # 1) 增量同步（幂等 upsert，只补缺失事件）
        synced = 0
        if not args.no_sync:
            started = time.time()
            symbols = funding_symbols(db)
            svc = MarketDataService(db)
            result = svc.sync_funding_many(symbols, days=args.sync_days, workers=8)
            synced = result.get("inserted", 0)
            print(f"同步：{len(symbols)} 币，插入 {synced} 行，"
                  f"错误 {len(result.get('errors', []))} 条，"
                  f"耗时 {time.time() - started:.0f}s", flush=True)

        # 2) 现货对过滤（delta 中性的现货腿必须可买）
        eligible = None
        spot_note = "未过滤"
        if not args.no_spot_filter:
            proxy = all_state(db).get("proxy_url") or None
            try:
                bases = spot_base_symbols(proxy)
                symbols = funding_symbols(db)
                eligible = {s for s in symbols if match_spot_base(s, bases)}
                spot_note = f"现货可买 {len(eligible)}/{len(symbols)}"
            except Exception as exc:  # 网络失败时降级为不过滤，并在报告里标注
                spot_note = f"exchangeInfo 获取失败（{exc}），本次未过滤"
        print(f"现货过滤：{spot_note}")

        targets = select_targets(db, k=args.k, min_rate=args.min_rate, eligible=eligible)
        cur = [t["symbol"] for t in targets]

        # 3) 换手与成本（对比上次建议持仓）
        prev: list[str] = []
        if LATEST.exists():
            try:
                prev = json.loads(LATEST.read_text(encoding="utf-8")).get("held", [])
            except Exception:
                prev = []
        to = turnover(prev, cur)
        cost = rebalance_cost(prev, cur)

        report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "strategy": "funding_harvest delta-neutral (P11)",
            "k": args.k,
            "min_rate": args.min_rate,
            "spot_filter": spot_note,
            "synced_rows": synced,
            "held": cur,
            "targets": targets,
            "turnover_vs_prev": round(to, 4),
            "rebalance_cost_pct": round(cost * 100, 4),
            "avg_trail_annualized_pct": (
                round(sum(t["annualized_pct"] for t in targets) / len(targets), 2)
                if targets else 0.0
            ),
        }
        LATEST.parent.mkdir(parents=True, exist_ok=True)
        LATEST.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) 输出建议清单（paper trading：按此手动执行双腿）
    print(f"\n=== 收割建议 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ===")
    print(f"平均 trailing 年化：{report['avg_trail_annualized_pct']}%  "
          f"换手：{to:.0%}  本次重平衡成本：{cost * 100:.3f}%")
    if not targets:
        print("（无正费率候选——建议空仓）")
    for t in targets:
        action = "持有" if prev and t["symbol"] in prev else "开仓"
        print(f"  {action} {t['symbol']:<20} trailing {t['trail_rate_8h'] * 100:.4f}%/8h"
              f"  ≈年化 {t['annualized_pct']}%")
    print(f"\nJSON: {LATEST}")
    print("提醒：双腿 = 买入现货 + 等量做空永续；重平衡时平掉被移出名单的仓位。")


if __name__ == "__main__":
    main()
