r"""RGD v2 — 15m 主周期 × 4 组（方向 × 成本）矩阵回测（编排入口）。

4 组矩阵（契约 §7.1）
--------------------
G1 Long & Short / fee 0.0004 / slip 0.0002   <- 主报告（项目全局成本）
G2 Long Only    / fee 0.0004 / slip 0.0002     现货仅多对照
G3 Long & Short / fee 0.0005 / slip 2ticks     脚本原样成本（TV 对齐，近似）
G4 Long Only    / fee 0.0005 / slip 2ticks

本模块只做装配：参数解析、进程池、按 symbol 排序汇总、报告落盘。
单币取数与 4 组回放在 `rgd_backtest_worker`；随机基线与池化在 `rgd_baselines`；
markdown 在 `rgd_backtest_report`。HTF attach 铁律见 worker 模块 docstring（§7.3.1）。

并行铁律（照抄 supertrend_backtest 的踩坑记录）
----------------------------------------------
- `simulate_rgd` 是纯 Python 逐 bar，线程被 GIL 锁死，必须进程池。
- 子进程必须在 numpy 初始化前把 BLAS 线程数钉成 1，否则 20 进程各开 N 线程会 OOM。
- worker 用只读 + NullPool 引擎，避免打满 PostgreSQL 的 max_connections。
- `as_completed` 结果按 symbol 排序后再池化，否则固定种子 bootstrap 每跑漂移。

用法
----
```
.\.venv\Scripts\python.exe scripts/rgd_backtest.py --interval 15m --days 180 --universe major --limit 8 --seeds 5
.\.venv\Scripts\python.exe scripts/rgd_backtest.py --interval 15m --days 180 --universe all --seeds 10
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# 必须在 numpy/BLAS 于子进程里初始化之前设置：spawn 出来的 worker 会继承这些变量。
for _threads_var in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

from sqlalchemy.orm import sessionmaker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.backtest.optimizer import symbol_universe  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.features.rgd import htf_interval_for  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.registry import get_spec  # noqa: E402
from app.strategy.rules import resolve_params  # noqa: E402
from app.utils import now_ms  # noqa: E402
from rgd_backtest_report import render_markdown  # noqa: E402
from rgd_backtest_worker import (  # noqa: E402,F401
    CODE, GROUPS, MIN_SEGMENT_BARS, evaluate_symbol, load_segments,
)
# `randomize_rgd` / `load_segments` 在这里一并导出：编排入口是契约 §7 对外的
# 脚本面，测试可以直接 `from rgd_backtest import ...`，不必知道内部分包。
from rgd_baselines import (  # noqa: E402,F401
    MIN_POOLED_TRADES, randomize_rgd, summarise_group,
)

DAY_MS = 86_400_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RGD v2 四组矩阵回测")
    parser.add_argument("--symbols", default="", help="逗号分隔；优先于 --universe")
    parser.add_argument("--universe", default="all", choices=("all", "major"))
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 个币种（冒烟用）")
    parser.add_argument("--cash", type=float, default=10_000.0, help="每币独立账户初始资金")
    parser.add_argument("--processes", type=int, default=0, help="进程数，0 = CPU 核数-2")
    parser.add_argument("--seeds", type=int, default=10, help="随机基线种子数")
    parser.add_argument("--min-bars", type=int, default=0, help="0 = max(800, lookback+200)")
    parser.add_argument("--out", default=str(ROOT / "docs" / "rgd-backtest"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = get_spec(CODE)
    processes = args.processes or max(1, min(24, (os.cpu_count() or 4) - 2))
    base_params = resolve_params(
        CODE, {"fee_rate": GROUPS[0]["fee_rate"], "slippage": GROUPS[0]["slippage"]},
        interval=args.interval,
    )
    min_bars = args.min_bars or max(MIN_SEGMENT_BARS, spec.lookback(base_params) + 200)
    htf_interval = htf_interval_for(
        args.interval, base_params["auto_htf"], int(base_params["htf_in"])
    )
    db = sessionmaker(
        bind=make_engine(read_only=True), autoflush=False, expire_on_commit=False, future=True
    )()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    insufficient: list[str] = []
    if not symbols:
        if args.universe == "major":
            available = set(symbol_universe(db, args.interval, args.days))
            symbols = [item for item in MAJOR_SYMBOLS if item in available]
            insufficient = [item for item in MAJOR_SYMBOLS if item not in available]
            if insufficient:
                print(f"  [主流池] 无足够历史覆盖、已跳过: {', '.join(insufficient)}")
        else:
            symbols = symbol_universe(db, args.interval, args.days)
    if args.limit:
        symbols = symbols[: args.limit]
    as_of = int(now_ms())
    print(
        f"[RGD v2] {args.interval} × {args.days} 天 | HTF {htf_interval} | "
        f"{len(symbols)} 币 × {len(GROUPS)} 组 | {processes} 进程 | "
        f"随机基线 {args.seeds} 种子 | 最短段 {min_bars} 根"
    )
    tasks = [
        {
            "symbol": symbol, "interval": args.interval, "days": args.days,
            "end_ms": as_of, "as_of": as_of, "min_bars": min_bars, "cash": args.cash,
            "base": base_params, "prepare_keys": list(spec.prepare_keys),
            "seeds": args.seeds, "base_seed": 20261025,
        }
        for symbol in symbols
    ]
    started = time.monotonic()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {pool.submit(evaluate_symbol, task): task["symbol"] for task in tasks}
        done = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # worker 崩溃不影响整轮
                results.append({"symbol": symbol, "status": "error",
                                "error": f"{type(exc).__name__}: {exc}"[:200]})
            done += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                print(f"  {done}/{len(tasks)} 币 · {elapsed:.0f}s · "
                      f"{done / elapsed * 60:.0f} 币/分", flush=True)
    elapsed = round(time.monotonic() - started, 1)

    # 固定种子 bootstrap 要求成交数组顺序稳定，否则每跑漂移 ±0.02pp。
    results.sort(key=lambda row: row["symbol"])
    ok = [row for row in results if row["status"] == "ok"]
    skipped = [
        f"{row['symbol']}: {row.get('reason') or row.get('error')}"
        for row in results if row["status"] != "ok"
    ]
    groups = [
        summarise_group([row["groups"][index] for row in ok], group)
        for index, group in enumerate(GROUPS)
    ]
    payload = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval,
        "htf_interval": htf_interval,
        "days": args.days,
        "window": {
            "start": datetime.fromtimestamp(
                (as_of - args.days * DAY_MS) / 1000, timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(as_of / 1000, timezone.utc).isoformat(),
        },
        "universe": {
            "kind": args.universe if not args.symbols else "explicit",
            "requested": len(symbols), "evaluated": len(ok), "skipped": skipped,
            # 主流池里因 15m 历史不足被剔除的币：显式列出，避免把「没数据」读成「没信号」
            "insufficient_history": insufficient,
        },
        "parameters": base_params,
        "costs": {
            "cash_per_symbol": args.cash,
            "groups": [
                {key: group[key] for key in ("id", "dir_mode", "fee_rate", "slippage", "slippage_note")}
                for group in GROUPS
            ],
        },
        "random_baseline_seeds": args.seeds,
        "min_pooled_trades": MIN_POOLED_TRADES,
        "min_bars": min_bars,
        "processes": processes,
        "elapsed_seconds": elapsed,
        "groups": groups,
        "baseline": groups[0],
    }

    out_md, out_json = Path(args.out + ".md"), Path(args.out + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    for group in groups:
        detail = {
            "strategy": CODE, "group": group, "interval": args.interval,
            "htf_interval": htf_interval, "days": args.days,
            "window": payload["window"], "universe": payload["universe"],
            "parameters": base_params,
            "costs": {"cash_per_symbol": args.cash, "fee_rate": group["fee_rate"],
                      "slippage": group["slippage_used"]},
            "min_bars": min_bars, "random_baseline_seeds": args.seeds,
            "min_pooled_trades": MIN_POOLED_TRADES,
        }
        Path(f"{args.out}-{group['id']}.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    baseline = payload["baseline"]
    ci = baseline["expectancy_ci"]
    print(
        f"\nG1 评估 {len(ok)} 币 | 成交 {baseline['trades']} 笔 | "
        f"池化每笔期望 {ci.get('mean_return_pct', 0.0):+.4f}% "
        f"CI [{ci.get('ci_low_pct', 0.0):+.4f}%, {ci.get('ci_high_pct', 0.0):+.4f}%] "
        f"-> {ci.get('verdict')}"
    )
    print(f"[输出] {out_md} / {out_json} / -G1..G4.json　耗时 {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
