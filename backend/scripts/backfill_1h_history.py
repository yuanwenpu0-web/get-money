"""回填 1h K 线历史，解除迭代 9 暴露的样本瓶颈。

迭代 9 发现：流动性宇宙多数币仅 ~42 天 1h 历史，导致 ML 训练 test_rows≈181 < 200
下限，AUC 结构性贴 0.5（样本越大越随机）。本脚本把 $30M 流动性宇宙逐币回填到
--days（默认 120）天，使 auto_train_days=60 能取到完整窗口。

用法（从项目根运行）:
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_1h_history.py \
        --proxy http://127.0.0.1:7890 --days 120
    PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/backfill_1h_history.py \
        --proxy http://127.0.0.1:7890 --dry-run   # 只打印目标宇宙
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import engine as _app_engine  # noqa: E402  (确认 db 模块可导入)
from app.state import all_state  # noqa: E402
from app.market.futures import BinanceFuturesClient  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.market import universe  # noqa: E402


def target_universe(db: Session, client: BinanceFuturesClient, min_volume: float,
                    blacklist: list[str]) -> list[str]:
    """复刻 universe._fetch：exchange_info 过滤永续 + 流动性 + 黑名单。"""
    info = client.exchange_info()
    symbols = universe._filter(info.get("symbols", []))
    if blacklist:
        blocked = set(blacklist)
        symbols = [s for s in symbols if s not in blocked]
    if min_volume <= 0:
        return symbols
    volumes = client.ticker_24hr()
    if not volumes:
        return symbols
    return [s for s in symbols if volumes.get(s, 0.0) >= min_volume]


def span_days(db: Session, symbol: str) -> tuple[int, int]:
    stats = MarketDataService(db).series_stats(symbol, "1h")
    if not stats["count"]:
        return 0, 0
    n, t0, t1 = stats["count"], stats["first_open_time"], stats["last_open_time"]
    span = int((t1 - t0) / 86_400_000) + 1
    return n, span


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--min-volume", type=float, default=None,
                    help="流动性门槛 USD，默认读 state.min_quote_volume_usd")
    ap.add_argument("--limit", type=int, default=0, help="只回填前 N 个（调试用）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Session(_app_engine)
    state = all_state(db)
    min_volume = args.min_volume if args.min_volume is not None else float(state.get("min_quote_volume_usd") or 0)
    blacklist = universe._blacklist(state)

    client = BinanceFuturesClient(testnet=False, proxy=args.proxy)
    symbols = target_universe(db, client, min_volume, blacklist)
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"目标宇宙: {len(symbols)} 币 (min_volume={min_volume:,.0f}, blacklist={len(blacklist)})")
    if args.dry_run:
        print("  sample:", symbols[:15])
        return 0

    svc = MarketDataService(db, client)
    DAY = 86_400_000
    results = []
    total_inserted = 0
    for i, sym in enumerate(symbols, 1):
        before_n, before_span = span_days(db, sym)
        try:
            res = svc.sync_history(sym, "1h", days=args.days)
            inserted = int(res.get("inserted", 0))
        except Exception as exc:  # 单币失败不阻塞
            results.append({"symbol": sym, "error": str(exc)[:200], "inserted": 0})
            print(f"[{i}/{len(symbols)}] {sym} FAIL {exc}")
            continue
        total_inserted += inserted
        after_n, after_span = span_days(db, sym)
        results.append({
            "symbol": sym, "inserted": inserted,
            "span_before": before_span, "span_after": after_span,
            "rows_after": after_n,
        })
        flag = "OK " if inserted else "skip"
        print(f"[{i}/{len(symbols)}] {sym} {flag} +{inserted} rows  span {before_span}d->{after_span}d")

    reached = [r for r in results if r.get("span_after", 0) >= args.days]
    report = {
        "days_target": args.days,
        "min_volume": min_volume,
        "universe_size": len(symbols),
        "total_inserted": total_inserted,
        "reached_target_days": len(reached),
        "results": results,
    }
    out = ROOT / "docs" / "backfill-1h-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 完成 ==  插入 {total_inserted} 行，{len(reached)}/{len(symbols)} 币达到 {args.days} 天")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
