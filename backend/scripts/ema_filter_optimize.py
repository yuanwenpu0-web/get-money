r"""对所有策略统一跑 EMA 快慢线方向过滤的参数寻优。

用法
----
```
.\.venv\Scripts\python.exe backend\scripts\ema_filter_optimize.py \
    --pool major --interval 1h --days 180 --combos 12 --workers 8
```

只搜 `ema_trend_fast` / `ema_trend_slow` 两个参数，固定策略自身参数不变。
输出汇总表 `output/ema-filter-all.csv` 与同名 `.json`。
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
    build_tasks,
    default_workers,
    pick_best_per_symbol,
    prepare_request,
    run_optimization,
)
from app.db import SessionLocal  # noqa: E402
from app.models import SymbolPool  # noqa: E402
from app.strategy import specs  # noqa: F401  触发注册
from app.strategy.pools import pool_symbols  # noqa: E402
from app.strategy.registry import all_specs  # noqa: E402

# 走专用 broker 的策略：signals() 里 direction 恒为 0，通用 EMA 过滤不起作用，跳过。
DEDICATED_BROKER_CODES = {
    "opening_candle_breakout",
    "rgd_donchian_v2",
    "simple_zigzag_double",
}

# 只列「开启」组合：两键都取 >0，避免笛卡尔积里出现 (0,x)/(x,0) 这类与
# 「关闭」等价的重复档位（会污染最佳组合频率统计，且让 ema_on 计数失真）。
# 关闭基线 (0,0) 由 build_grid 自动从 base_params 补入；
# fast >= slow 的非法组合由 resolve_params 在 build_grid 的 legal 过滤里剔除。
# 全网格 5×4=20 档，需 --combos >= 20 才能覆盖全部合法组合。
EMA_GRID = {
    "ema_trend_fast": [9, 12, 20, 30, 50],
    "ema_trend_slow": [26, 50, 100, 200],
}


def _progress(payload: dict) -> None:
    done, total = payload["done"], payload["total"]
    bar_len = 24
    filled = int(bar_len * done / max(1, total))
    eta = payload.get("eta")
    eta_text = f"{eta:.0f}s" if isinstance(eta, (int, float)) else "--"
    print(
        f"\r[{'#' * filled}{'.' * (bar_len - filled)}] {done}/{total} "
        f"elapsed={payload.get('elapsed', 0):.0f}s eta={eta_text} {payload.get('current', ''):18s}",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="全策略 EMA 方向过滤寻优")
    parser.add_argument("--pool", default="major")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--combos", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--objective", default="return_dd", choices=["return_dd", "sharpe"])
    parser.add_argument("--delta-source", default="auto", choices=["auto", "lower", "est"])
    parser.add_argument("--out", default="output/ema-filter-all")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        pool = db.query(SymbolPool).filter(SymbolPool.code == args.pool).first()
        if pool is None:
            print(f"未知币种池: {args.pool}")
            return 2
        members = pool_symbols(db, pool)
        if not members:
            print(f"池 {args.pool} 没有可用币种")
            return 2
        universe = {args.interval: members}
        workers = args.workers or default_workers()

        codes = [
            s.code for s in all_specs()
            if s.code not in DEDICATED_BROKER_CODES
            and not s.code.startswith("research_")
        ]
        print(f"待跑策略 {len(codes)} 个: {codes}（{args.pool} 池 {len(members)} 币，{args.interval}，{workers} 进程）")
        all_rows = []
        for code in codes:
            print(f"\n=== {code} ===")
            try:
                request, meta, grids = prepare_request(
                    db, code, args.interval,
                    days=args.days,
                    train_ratio=args.train_ratio,
                    combos=args.combos,
                    workers=workers,
                    min_trades=args.min_trades,
                    objective=args.objective,
                    delta_source=args.delta_source,
                    custom_grid=EMA_GRID,
                )
            except Exception as exc:
                print(f"  跳过（prepare_request 失败）: {exc}")
                continue
            rows: list[dict] = []
            tasks = build_tasks(request, meta, grids, universe)
            run_optimization(
                request=request, meta=meta, grids=grids, symbols=universe,
                on_progress=_progress, on_result=rows.append, workers=workers,
            )
            print()
            best_map = pick_best_per_symbol(rows)
            best = list(best_map.values())
            if not best:
                print("  无有效结果")
                continue
            df = pd.DataFrame(best)
            df.insert(0, "strategy", code)
            all_rows.append(df)
            verdicts = df["verdict"].value_counts().to_dict()
            print(f"  判定分布: {verdicts}")
            # best_combo 是相对基准的差值 dict；EMA 参数以 ema_trend_fast 为准
            ema_on = df["best_combo"].apply(lambda p: bool((p or {}).get("ema_trend_fast")))
            print(f"  启用 EMA 过滤的币: {int(ema_on.sum())}/{len(df)}")
        if not all_rows:
            print("\n没有任何策略产出结果")
            return 1
        combined = pd.concat(all_rows, ignore_index=True)
        out_csv = Path(args.out).with_suffix(".csv")
        out_json = Path(args.out).with_suffix(".json")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
        out_json.write_text(
            json.dumps(combined.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n已写出 {out_csv} / {out_json}")
        # 按策略汇总
        summary = combined.groupby("strategy").agg(
            n=("symbol", "count"),
            verdict_dist=("verdict", lambda s: dict(s.value_counts())),
            ema_on=("best_combo", lambda s: int(sum(bool((p or {}).get("ema_trend_fast")) for p in s))),
        ).reset_index()
        print("\n=== 按策略汇总 ===")
        print(summary.to_string(index=False))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
