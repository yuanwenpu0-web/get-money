r"""全币种参数寻优 — 命令行入口。

和前端「参数寻优」页走同一套引擎（`app.backtest.optimizer`），区别只在于：
网页版把结果落在 `optimization_runs` / `optimization_results` 表里供后续一键应用，
命令行适合在服务器上跑大批量、看完整表格。

用法
----
```
# 单币试跑
.\.venv\Scripts\python.exe backend\scripts\optimize_all.py --symbols BTCUSDT,ETHUSDT --combos 8

# 全币种（该周期上有足够历史的所有币种），64 组参数，默认并发 CPU-4
.\.venv\Scripts\python.exe backend\scripts\optimize_all.py --interval 15m --days 90

# 只跑某个池，并把结果写成 markdown
.\.venv\Scripts\python.exe backend\scripts\optimize_all.py --pool major --out data\opt-major
```

注意
----
- 结果默认**不写库、不改线上参数**。要生效请用网页版的一键应用（会经过
  `symbol_params` 校验并留痕），或在页面上挑选后再应用。
- `--workers` 默认留 4 核给巡检；跑满会让行情同步和下单挤不到 CPU。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402

from app.backtest.optimizer import (  # noqa: E402
    OptimizationRequest,
    default_workers,
    prepare_request,
    run_optimization,
    symbol_universe,
)
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import SymbolPool  # noqa: E402
from app.strategy.pools import pool_symbols  # noqa: E402


def _progress(payload: dict) -> None:
    done, total = payload["done"], payload["total"]
    bar_len = 24
    filled = int(bar_len * done / max(1, total))
    eta = payload.get("eta")
    eta_text = f"{eta:.0f}s" if isinstance(eta, (int, float)) else "--"
    print(
        f"\r[{'#' * filled}{'.' * (bar_len - filled)}] {done}/{total} "
        f"elapsed={payload.get('elapsed', 0):.0f}s eta={eta_text} {payload.get('current', ''):14s}",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="全币种参数寻优")
    parser.add_argument("--strategy", default="absorption_reversal")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--pool", default="", help="只跑某个币种池的成员")
    parser.add_argument("--symbols", default="", help="逗号分隔的币种列表，覆盖 --pool")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--combos", type=int, default=64)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0, help="0 = CPU-4")
    parser.add_argument("--objective", default="return_dd", choices=["return_dd", "sharpe"])
    parser.add_argument(
        "--delta-source", default="auto", choices=["auto", "lower", "est"],
        help="auto：有 1m 数据用 lower-tf delta（与实盘一致），否则退回 est",
    )
    parser.add_argument("--grid", default="", help="自定义搜索空间 JSON，如 '{\"target_r\":[1.5,2,3]}'")
    parser.add_argument("--out", default="", help="输出前缀（生成 .md 与 .json）")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif args.pool:
            pool = db.query(SymbolPool).filter(SymbolPool.code == args.pool).first()
            if pool is None:
                print(f"未知币种池: {args.pool}")
                return 2
            symbols = pool_symbols(db, pool)
        else:
            symbols = symbol_universe(db, args.interval, args.days)
        if not symbols:
            print("没有可用的币种（该周期历史不足）")
            return 2

        custom = json.loads(args.grid) if args.grid else None
        request, meta, grid = prepare_request(
            db,
            args.strategy,
            args.interval,
            days=args.days,
            train_ratio=args.train_ratio,
            combos=args.combos,
            workers=args.workers,
            min_trades=args.min_trades,
            objective=args.objective,
            delta_source=args.delta_source,
            custom_grid=custom,
        )
    finally:
        db.close()

    workers = request.workers or default_workers()
    print(
        f"{args.strategy} × {args.interval}：{len(symbols)} 币 × {meta['combos']} 组参数"
        f"（搜索空间 {meta['grid_size']}）· {workers} 进程 · {args.days} 天"
        f"（训练 {request.train_ratio:.0%} / 验证 {1 - request.train_ratio:.0%}）"
    )

    rows: list[dict] = []
    result = run_optimization(
        request=request,
        meta=meta,
        grid=grid,
        symbols=symbols,
        on_progress=_progress,
        on_result=rows.append,
        workers=workers,
    )
    print()

    ok = [row for row in rows if row.get("status") == "ok"]
    traded = [row for row in ok if row.get("eligible")]
    pooled = bootstrap_expectancy_ci(
        [{"return_pct": value} for row in ok for value in (row.get("valid_returns") or [])]
    )
    print(
        f"完成 {result['done']} / 跳过 {result['skipped']} / 失败 {result['failed']}"
        f" · {result['elapsed']:.0f}s"
    )
    print(f"有候选组合的币种：{len(traded)}/{len(ok)}")
    if pooled.get("trades"):
        print(
            f"验证段池化：{pooled['trades']} 笔，每笔期望 {pooled['mean_return_pct']:+.4f}% "
            f"95% CI [{pooled['ci_low_pct']:+.4f}%, {pooled['ci_high_pct']:+.4f}%] → {pooled['verdict']}"
        )
    verdicts = {
        key: sum(1 for row in ok if row.get("verdict") == key)
        for key in ("pass", "inconclusive", "negative", "insufficient", "no_trades")
    }
    print("判定分布：" + "  ".join(f"{k}={v}" for k, v in verdicts.items() if v))

    if rows:
        frame = pd.DataFrame(
            [
                {
                    "symbol": row["symbol"],
                    "verdict": row.get("verdict", ""),
                    "score": row.get("score", 0),
                    "train_%": (row.get("train") or {}).get("total_return_pct"),
                    "train_n": (row.get("train") or {}).get("total_trades"),
                    "valid_%": (row.get("valid") or {}).get("total_return_pct"),
                    "valid_n": (row.get("valid") or {}).get("total_trades"),
                    "ci_low": (row.get("ci") or {}).get("ci_low_pct"),
                    "params": json.dumps(row.get("best_combo") or {}, ensure_ascii=False),
                    "delta": row.get("delta_source", ""),
                }
                for row in ok
            ]
        )
        frame = frame.sort_values(["verdict", "score"], ascending=[True, False])
        print()
        with pd.option_context("display.width", 200, "display.max_colwidth", 70):
            print(frame.head(30).to_string(index=False))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
            payload = {
                "strategy": args.strategy,
                "interval": args.interval,
                "days": args.days,
                "combos": meta["combos"],
                "grid_size": meta["grid_size"],
                "workers": workers,
                "delta_source": args.delta_source,
                "summary": result,
                "pooled_ci": pooled,
                "results": rows,
            }
            out.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str, indent=1),
                encoding="utf-8",
            )
            print(f"\n已写出 {out.with_suffix('.csv')} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
