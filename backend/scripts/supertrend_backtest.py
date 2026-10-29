r"""Supertrend 趋势反转 — 全币种 / 主流币 × 一年 K 线回测。

目的
----
用 `supertrend_current` 策略在**全部有数据覆盖的永续合约**（`--universe all`，默认）
或**主流池**（`--universe major`，`coin_universe.MAJOR_SYMBOLS` 的 16 个高流动性合约）
上做一次诚实的历史检验：
真实回放完工 K 线（信号收盘 → 次根开盘成交、费率/滑点、ATR 止损、R 倍数止盈、
反向翻转出场），不允许任何形式的未来函数。

与 `absorption_backtest.py` 的三点不同
--------------------------------------
1. **分片单位是币种，不是窗口**：一个 worker 拿到一个币，先把 K 线切成连续段，
   再把它要跑的**全部参数组合**跑完。取数与指标预热只做一次，网格里出场参数
   （`target_r`）变化不影响 `prepare` 的输出，按 `prepare_keys` 缓存复用。
2. **跨缺口不复权、不串味**：K 线上的数据缺口会把序列切成多段。若把各段拼起来
   再算 Supertrend，前一段的状态机会「流进」后一段，等于用缺口前的价格推断
   缺口后的方向。这里**逐段独立撮合**，账户权益按段复利结转
   （缺口处仓位已平，现金继续带着走），与真实账户的行为一致。
3. **出场语义**：`simulate` 在同一根上「先出场、不再进场」（`exited` 门控），
   所以翻转根只平仓，反向仓位要等下一次翻转才建立 —— 比「永远在市」的
   Supertrend 保守一根 K 线。这是全项目共用的撮合行为，不是本脚本的取舍。

对照组（决定「有没有 edge」，而不是「收益多少」）
------------------------------------------------
- **随机方向基线**：相同时点、相同仓位、相同风控、相同成本，只把方向 50/50 翻转
  （多种子）。策略账户收益若不能稳定超过它的 95 分位，说明方向判断没有信息量。
- **池化每笔期望的 bootstrap CI**：把所有币/所有段的成交池化后重采样。
  单币只有个位数成交，逐币 CI 必然跨 0；池化后 N 足够才能区分 edge 与噪声。
  `MIN_POOLED_TRADES` 是池化层的最小样本门槛 —— 实测 4 笔成交的 bootstrap 会给出
  「CI 下界 +0.86%」的假精确读数，必须降级成 insufficient。
- **多空分解**：翻转策略做多和做空的期望通常不对称（加密资产的长期漂移、
  资金费率方向），分开看能避免「多空互相对冲出一个看起来不显著的数」。

并行铁律（踩过的坑，不要改回去）
--------------------------------
- `simulate` 是纯 Python 逐 bar → 线程被 GIL 锁死零收益，必须进程池。
- 取数也在 worker 内做：父进程查好再 pickle 进子进程，通信开销比查询本身还贵。
- worker 用**只读 + NullPool** 引擎：默认连接池（5+10）在 20 个进程下会打满
  PostgreSQL 的 max_connections，而 worker 只做顺序查询，池化没有收益。
- 子进程必须在 numpy 初始化**之前**把 BLAS 线程数钉成 1，否则 20 个进程各开
  cpu_count 个 OpenBLAS 线程会直接 OOM 杀掉整个池。

用法
----
```
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py                      # 1h × 365 天 × 全币种
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py --universe major     # 只跑主流池
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py --limit 8            # 冒烟
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py --interval 4h
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py --symbols BTCUSDT,ETHUSDT --seeds 20
```
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# 必须在 numpy/BLAS 于子进程里初始化之前设置：进程池里 20+ 个进程各自默认开
# cpu_count 个 OpenBLAS 线程，每个线程预留栈缓冲，实测直接
# "Memory allocation still failed after 10 retries" 并杀掉整个池。
# 父进程的 numpy 早已 import，改不到它；关键是让 spawn 出来的子进程继承这些变量。
for _threads_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import pandas as pd
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import exposure_metrics, randomize_directions, simulate  # noqa: E402
from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.backtest.optimizer import symbol_universe  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.strategy.registry import get_spec  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import interval_ms, now_ms  # noqa: E402

CODE = "supertrend_current"
DAY_MS = 86_400_000
# 池化判定的最小成交笔数：与 `optimizer.MIN_POOLED_TRADES` 同一条判据。
# 4 笔成交的 bootstrap 会给出「CI 下界 +0.86%」的假精确读数，必须降级。
MIN_POOLED_TRADES = 30
# 单段最小 K 线数：低于此值指标预热都不够（lookback 默认 600 根）。
MIN_SEGMENT_BARS = 800

_SESSION = None


def _db():
    """worker 进程自己的只读会话（NullPool：一条连接、用完即关）。"""
    global _SESSION
    if _SESSION is None:
        engine = make_engine(read_only=True, null_pool=True)
        _SESSION = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False, future=True
        )()
    return _SESSION


# ---------------------------------------------------------------- 取数


def load_segments(symbol: str, interval: str, days: int, end_ms: int,
                  min_bars: int) -> list[pd.DataFrame]:
    """取一个币在该窗口上的**连续**K线片段（按缺口切段）。

    只取 OHLCV，不建 ML 特征：Supertrend 只需要 high/low/close，
    全量特征（ADX/MACD/funding/sentiment）在这一步纯属浪费。
    """
    service = MarketDataService(_db())
    as_of = min(end_ms, now_ms())
    start_ms = as_of - days * DAY_MS
    frame = service.load_frame(symbol, interval, start_ms=start_ms, end_ms=as_of)
    if frame.empty:
        return []
    step = interval_ms(interval)
    # 排除未收盘的桶：它的 OHLCV 是半截值，用它当信号等于用到未来数据。
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= as_of].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    valid = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    mask = valid.notna().all(axis=1)
    mask &= (valid[["open", "high", "low", "close"]] > 0).all(axis=1)
    mask &= valid["volume"].ge(0)
    mask &= valid["high"].ge(valid[["open", "close", "low"]].max(axis=1))
    mask &= valid["low"].le(valid[["open", "close", "high"]].min(axis=1))
    frame = frame.loc[mask]
    if frame.empty:
        return []
    diffs = frame.index.to_series().diff().dt.total_seconds() * 1000
    segments = []
    for _, segment in frame.groupby(diffs.ne(step).cumsum()):
        if len(segment) >= min_bars:
            segments.append(segment)
    return segments


# ---------------------------------------------------------------- worker


def evaluate_symbol(task: dict) -> dict:
    """一个币种 = 一个任务：取数一次，跑完全部参数组合。"""
    symbol = task["symbol"]
    try:
        segments = load_segments(
            symbol, task["interval"], task["days"], task["end_ms"], task["min_bars"]
        )
    except Exception as exc:  # 无数据/查询异常 — 跳过而不是中断整轮
        return {"symbol": symbol, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    if not segments:
        return {"symbol": symbol, "status": "skipped", "reason": "无满足长度的连续K线段"}

    cash, fee, slippage = task["cash"], task["fee"], task["slippage"]
    interval = task["interval"]
    prepared: dict[tuple, list] = {}
    combos = []
    for combo in task["combos"]:
        params = {**task["base"], **combo}
        # prepare 只依赖 prepare_keys：网格里不影响指标的参数可以复用同一份信号帧
        cache_key = tuple(params[key] for key in task["prepare_keys"])
        if cache_key not in prepared:
            # `simulate` 逐 bar 取 `row.atr_pct`：必须把**已富化**的帧喂给它，
            # 不能喂原始 OHLCV —— atr_pct 是本策略 prepare() 的产物，不是内置列。
            # 富化帧缓存后，`evaluate_signals` 再调用 prepare 会因为列已齐而直接返回。
            frames = [prepare_frame(segment, CODE, params) for segment in segments]
            prepared[cache_key] = (frames, [evaluate_signals(item, CODE, params) for item in frames])
        segment_list, signals_list = prepared[cache_key]

        equity_parts: list[pd.Series] = []
        all_trades: list[dict] = []
        factor = 1.0
        signal_bars = 0
        blocked: Counter = Counter()
        for segment, signals in zip(segment_list, signals_list):
            equity, trades = simulate(segment, signals, params, cash, fee, slippage, interval)
            # 段与段之间按复利结转：缺口处仓位已平，现金继续带着走到下一段
            equity_parts.append(equity * factor)
            factor *= float(equity.iloc[-1]) / cash
            all_trades.extend(aggregate_positions(trades))
            signal_bars += int(signals.direction.ne(0).sum())
            blocked.update(
                {
                    str(key): int(value)
                    for key, value in signals.block_reason[
                        signals.block_reason.ne("")
                    ].value_counts().items()
                }
            )
        combined = pd.concat(equity_parts)
        metrics = metrics_from_equity(combined, all_trades, cash, interval)

        # 随机方向基线：逐种子把每一段都翻转后复利，得到账户收益的一个样本
        samples = []
        for offset in range(task["seeds"]):
            seed = task["base_seed"] + offset * 7919
            random_factor = 1.0
            for segment, signals in zip(segment_list, signals_list):
                randomized = randomize_directions(signals, seed=seed)
                equity, _ = simulate(segment, randomized, params, cash, fee, slippage, interval)
                random_factor *= float(equity.iloc[-1]) / cash
            samples.append((random_factor - 1) * 100)

        combos.append(
            {
                "label": combo["label"],
                "trades": len(all_trades),
                "signal_bars": signal_bars,
                "return_pct": metrics.get("total_return_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe": metrics.get("sharpe"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "expectancy_cash": metrics.get("expectancy"),
                "total_fees": metrics.get("total_fees"),
                **exposure_metrics(combined, all_trades, interval),
                # 逐笔收益回传：主进程做池化 CI（单币样本太小，逐币 CI 必然跨 0）
                "trade_returns": [
                    [round(float(item["return_pct"]), 4), item["side"]] for item in all_trades
                ],
                "status_counts": dict(Counter(item["status"] for item in all_trades)),
                "blocked": dict(blocked),
                "random": {
                    "mean_pct": round(statistics.fmean(samples), 4) if samples else None,
                    "p95_pct": round(sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))], 4)
                    if samples
                    else None,
                    "p05_pct": round(sorted(samples)[int(len(samples) * 0.05)], 4) if samples else None,
                },
            }
        )
    return {
        "symbol": symbol,
        "status": "ok",
        "bars": int(sum(len(segment) for segment in segments)),
        "segments": len(segments),
        "combos": combos,
    }


# ---------------------------------------------------------------- 汇总


def _merge_counts(sequence) -> dict:
    totals: Counter = Counter()
    for mapping in sequence:
        for key, value in (mapping or {}).items():
            totals[str(key)] += int(value)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def _pooled(rows: list[list]) -> dict:
    """把 [(return_pct, side)] 池化后算每笔期望的 CI，并做多空分解。"""
    pair_list = [item for row in rows for item in row]
    trades = [{"return_pct": value} for value, _ in pair_list]
    ci = bootstrap_expectancy_ci(trades, min_trades=MIN_POOLED_TRADES)
    longs = [value for value, side in pair_list if side == "long"]
    shorts = [value for value, side in pair_list if side == "short"]
    return {
        "trades": len(trades),
        "expectancy_ci": ci,
        "side_split": {
            "long": bootstrap_expectancy_ci(
                [{"return_pct": v} for v in longs], min_trades=MIN_POOLED_TRADES
            ),
            "short": bootstrap_expectancy_ci(
                [{"return_pct": v} for v in shorts], min_trades=MIN_POOLED_TRADES
            ),
        },
    }


def summarise(results: list[dict], combos: list[dict]) -> list[dict]:
    """按参数组合聚合全市场：池化判据 + 账户口径分布。"""
    ok = [row for row in results if row["status"] == "ok"]
    summaries = []
    for index, combo in enumerate(combos):
        rows = [row["combos"][index] for row in ok]
        with_trades = [row for row in rows if row["trades"] > 0]
        returns = [row["return_pct"] for row in with_trades if row["return_pct"] is not None]
        pooled = _pooled([row["trade_returns"] for row in rows])
        random_means = [
            row["random"]["mean_pct"] for row in with_trades if row["random"]["mean_pct"] is not None
        ]
        summaries.append(
            {
                "label": combo["label"],
                "params": combo["params"],
                "symbols_total": len(ok),
                "symbols_with_trades": len(with_trades),
                "trades": sum(row["trades"] for row in with_trades),
                "signal_bars": sum(row["signal_bars"] for row in rows),
                "account_return_median_pct": round(statistics.median(returns), 4) if returns else 0.0,
                "account_return_mean_pct": round(statistics.fmean(returns), 4) if returns else 0.0,
                "positive_symbols": int(sum(1 for value in returns if value > 0)),
                "max_drawdown_mean_pct": round(
                    statistics.fmean(
                        [row["max_drawdown_pct"] for row in with_trades if row["max_drawdown_pct"] is not None]
                    ),
                    4,
                )
                if with_trades
                else 0.0,
                "exposure_mean_pct": round(
                    statistics.fmean([row["exposure_pct"] for row in with_trades]), 4
                )
                if with_trades
                else 0.0,
                "avg_bars_held": round(
                    statistics.fmean([row["avg_bars_held"] for row in with_trades]), 2
                )
                if with_trades
                else 0.0,
                # 账户收益超过该币随机翻转分布 95 分位的币种占比
                "beats_random_ratio": round(
                    sum(
                        1
                        for row in with_trades
                        if row["random"]["p95_pct"] is not None
                        and (row["return_pct"] or 0.0) > row["random"]["p95_pct"]
                    )
                    / len(with_trades),
                    4,
                )
                if with_trades
                else 0.0,
                "random_mean_pct": round(statistics.fmean(random_means), 4) if random_means else 0.0,
                **pooled,
                "status_counts": _merge_counts(row["status_counts"] for row in with_trades),
                "blocked": _merge_counts(row["blocked"] for row in rows),
                "per_symbol": [
                    {
                        "symbol": row["symbol"],
                        "trades": entry["trades"],
                        "return_pct": entry["return_pct"],
                        "win_rate": entry["win_rate"],
                        "max_drawdown_pct": entry["max_drawdown_pct"],
                        "random_mean_pct": entry["random"]["mean_pct"],
                    }
                    for row, entry in zip(ok, rows)
                    if entry["trades"] > 0
                ],
            }
        )
    return summaries


# ---------------------------------------------------------------- 主流程


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Supertrend 趋势反转全币种回测")
    parser.add_argument("--symbols", default="", help="逗号分隔；优先于 --universe")
    parser.add_argument("--universe", default="all", choices=("all", "major"),
                        help="all = 全部有数据覆盖的永续合约；major = coin_universe.MAJOR_SYMBOLS")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=365, help="回测窗口天数")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 个币种（冒烟用）")
    parser.add_argument("--cash", type=float, default=10_000.0, help="每币独立账户初始资金")
    parser.add_argument("--fee", type=float, default=DEFAULT_STATE["fee_rate"])
    parser.add_argument("--slippage", type=float, default=DEFAULT_STATE["slippage"])
    parser.add_argument("--processes", type=int, default=0, help="进程数，0 = CPU 核数-2")
    parser.add_argument("--seeds", type=int, default=10, help="随机方向基线种子数")
    parser.add_argument("--min-bars", type=int, default=MIN_SEGMENT_BARS, help="单段最小K线数")
    parser.add_argument("--out", default="docs/supertrend-reversal-backtest")
    parser.add_argument("--grid-limit", type=int, default=0, help="只跑前 N 组参数（0 = 全部）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = get_spec(CODE)
    processes = args.processes or max(1, min(24, (os.cpu_count() or 4) - 2))

    base_params = resolve_params(
        CODE, {"fee_rate": args.fee, "slippage": args.slippage}, interval=args.interval
    )
    # 网格来自 spec.grid（策略自己声明的候选档位），基准组合一定在候选集里
    keys = list(spec.grid)
    combos = []
    for values in _iter_grid(spec.grid, keys):
        combo = dict(zip(keys, values))
        if combo == {key: base_params[key] for key in keys}:
            label = "基准"
        else:
            label = "_".join(f"{key}{value}" for key, value in combo.items())
        merged = {**base_params, **combo}
        try:
            resolve_params(CODE, {}, {k: v for k, v in merged.items()
                                      if k not in ("fee_rate", "slippage")})
        except ValueError:
            continue
        combos.append({"label": label, "params": combo, "resolved": merged})
    if args.grid_limit:
        combos = combos[: args.grid_limit]
    if not combos:
        raise SystemExit("没有可用的参数组合")

    engine = make_engine(read_only=True)
    from sqlalchemy.orm import sessionmaker as _sm

    db = _sm(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        if args.universe == "major":
            # 主流池是固定白名单，但覆盖度仍要逐个确认：某个币在该周期可能没有一年历史
            available = set(symbol_universe(db, args.interval, args.days))
            symbols = [item for item in MAJOR_SYMBOLS if item in available]
            missing = [item for item in MAJOR_SYMBOLS if item not in available]
            if missing:
                print(f"  [主流池] 无足够历史覆盖、已跳过: {', '.join(missing)}")
        else:
            symbols = symbol_universe(db, args.interval, args.days)
    if args.limit:
        symbols = symbols[: args.limit]
    end_ms = int(now_ms())
    print(
        f"[Supertrend 趋势反转] {args.interval} × {args.days} 天 | "
        f"{len(symbols)} 币 × {len(combos)} 组参数 | {processes} 进程 | "
        f"随机基线 {args.seeds} 种子"
    )

    tasks = [
        {
            "symbol": symbol,
            "interval": args.interval,
            "days": args.days,
            "end_ms": end_ms,
            "min_bars": args.min_bars,
            "cash": args.cash,
            "fee": args.fee,
            "slippage": args.slippage,
            "base": base_params,
            "combos": [{"label": item["label"], **item["params"]} for item in combos],
            "prepare_keys": list(spec.prepare_keys),
            "seeds": args.seeds,
            "base_seed": 20261025,
        }
        for symbol in symbols
    ]

    started = time.monotonic()
    results: list[dict] = []
    # 进程池全局建一次：Windows spawn 的 import 成本极高，不要每个币重建
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
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                print(f"  {done}/{len(tasks)} 币 · {elapsed:.0f}s · "
                      f"{done / elapsed * 60:.0f} 币/分", flush=True)
    elapsed = round(time.monotonic() - started, 1)

    # `as_completed` 的返回顺序与进程调度有关。不排序的话，池化后的成交数组每次
    # 排列都不同，固定种子的 bootstrap 会因此给出 ±0.02pp 漂移的 CI —— 报告就不
    # 可复现了。按币种名排序后，同一份数据重跑得到逐位相同的结果。
    results.sort(key=lambda row: row["symbol"])
    ok = [row for row in results if row["status"] == "ok"]
    skipped = [
        f"{row['symbol']}: {row.get('reason') or row.get('error')}"
        for row in results
        if row["status"] != "ok"
    ]
    summaries = summarise(results, combos)
    baseline = next((item for item in summaries if item["label"] == "基准"), summaries[0])

    payload = {
        "strategy": CODE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": args.interval,
        "days": args.days,
        "window": {
            "start": datetime.fromtimestamp((end_ms - args.days * DAY_MS) / 1000, timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(),
        },
        "universe": {
            "kind": args.universe if not args.symbols else "explicit",
            "requested": len(symbols),
            "evaluated": len(ok),
            "skipped": skipped,
        },
        "parameters": base_params,
        "grid": {key: list(value) for key, value in spec.grid.items()},
        "costs": {"fee_rate": args.fee, "slippage": args.slippage, "cash_per_symbol": args.cash},
        "random_baseline_seeds": args.seeds,
        "min_pooled_trades": MIN_POOLED_TRADES,
        "processes": processes,
        "elapsed_seconds": elapsed,
        "combos": summaries,
        "baseline": baseline,
    }

    out_md, out_json = Path(args.out + ".md"), Path(args.out + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")

    ci = baseline["expectancy_ci"]
    print(
        f"\n评估 {len(ok)} 币 | 基准成交 {baseline['trades']} 笔 | "
        f"池化每笔期望 {ci.get('mean_return_pct', 0.0):+.4f}% "
        f"CI [{ci.get('ci_low_pct', 0.0):+.4f}%, {ci.get('ci_high_pct', 0.0):+.4f}%] → {ci.get('verdict')}"
    )
    print(f"[输出] {out_md} / {out_json}　耗时 {elapsed}s")
    return 0


def _iter_grid(grid: dict, keys: list):
    """笛卡尔积展开（行优先，最后一维变化最快）。"""
    def walk(index: int):
        if index == len(keys):
            yield []
            return
        for value in grid[keys[index]]:
            for rest in walk(index + 1):
                yield [value, *rest]

    yield from walk(0)


# ---------------------------------------------------------------- 报告


def _ci_lines(ci: dict, indent: str = "- ") -> list[str]:
    if "mean_return_pct" not in ci:
        return [f"{indent}无成交样本，无法计算每笔期望的置信区间 → **no_trades**（不构成判据）"]
    return [
        f"{indent}每笔收益均值 **{ci['mean_return_pct']:+.4f}%**，95% CI "
        f"[{ci['ci_low_pct']:+.4f}%, {ci['ci_high_pct']:+.4f}%] → **{ci['verdict']}**"
        f"（{ci['trades']} 笔）",
    ]


def render_markdown(payload: dict) -> str:
    base = payload["baseline"]
    ci = base["expectancy_ci"]
    verdict = ci.get("verdict")
    total_exit = max(sum(base["status_counts"].values()), 1)
    lines = [
        f"# Supertrend 趋势反转 — {'主流币' if payload['universe'].get('kind') == 'major' else '全币种'}"
        f"一年回测报告",
        "",
        f"生成时间：{payload['generated_at']}　周期：**{payload['interval']}**　"
        f"窗口：{payload['days']} 天（{payload['window']['start'][:10]} → {payload['window']['end'][:10]}）",
        f"币种：{payload['universe']['evaluated']}/{payload['universe']['requested']} 个有可用连续K线　"
        f"进程：{payload['processes']}　耗时：{payload['elapsed_seconds']}s",
        "",
        "## 结论速览（基准组合 factor=2.5 / target=3R）",
        "",
        f"- 有成交的币种 **{base['symbols_with_trades']}/{base['symbols_total']}** 个，"
        f"合计 **{base['trades']} 笔**成交（信号根数 {base['signal_bars']}）",
        f"- 单币账户收益中位数 **{base['account_return_median_pct']:+.2f}%**，"
        f"均值 {base['account_return_mean_pct']:+.2f}%",
        f"- 正收益币种 {base['positive_symbols']}/{base['symbols_with_trades']}"
        f"（{base['positive_symbols'] / max(base['symbols_with_trades'], 1):.0%}）",
        f"- 平均最大回撤 {base['max_drawdown_mean_pct']:.2f}%，"
        f"平均持仓占比 {base['exposure_mean_pct']:.1f}%，平均每笔持仓 {base['avg_bars_held']:.1f} 根",
        "",
        f"### 池化判据（{base['trades']} 笔，最小样本门槛 {payload['min_pooled_trades']} 笔）",
        "",
        *_ci_lines(ci),
        f"- 重采样为正的比例 {ci.get('positive_share', 0.0):.1%}（>95% 才算统计上站得住）",
        f"- 随机方向基线（{payload['random_baseline_seeds']} 种子，同时点同仓位同成本只翻方向）"
        f"账户收益均值 {base['random_mean_pct']:+.3f}%",
        f"- 账户收益超过自身随机基线 95 分位的币种占比 {base['beats_random_ratio']:.0%}",
        "",
        "### 多空分解",
        "",
    ]
    for side, label in (("long", "做多（向上翻转入场）"), ("short", "做空（向下翻转入场）")):
        split = base["side_split"][side]
        if "mean_return_pct" in split:
            cell = (
                f"{split['mean_return_pct']:+.4f}%，95% CI [{split['ci_low_pct']:+.4f}%, "
                f"{split['ci_high_pct']:+.4f}%] → **{split['verdict']}**（{split['trades']} 笔）"
            )
        else:
            cell = "无成交"
        lines.append(f"- **{label}**：{cell}")
    lines.extend([
        "",
        "判读规则（与全项目一致）：CI 下界 > 0 才算有正 edge；跨 0 即不可判定；"
        "上界 < 0 说明这套规则在当前样本下是负贡献。",
        "",
        "随机基线在币种之间离散度极大（有的币随便翻方向都能赚，有的怎么翻都亏），"
        "所以「是否跑赢随机」只能逐币比它自己的分布——用 `超随机占比`，"
        "不要拿账户收益去比一个全市场平均出来的随机数。",
        "",
        "## 结论与处置",
        "",
    ])
    if verdict == "positive":
        lines.append(
            f"池化 CI 下界为正（{ci['ci_low_pct']:+.4f}%）→ 存在可判定的正 edge。"
            "下一步：换一段未参与的时间窗做 holdout 复检，通过后再考虑在巡检中启用。"
        )
    elif verdict == "negative":
        lines.append(
            f"池化 CI 上界为负（{ci['ci_high_pct']:+.4f}%）→ 在当前口径下这套规则是**负贡献**，"
            "不宜投入资金。不要再微调阈值救它：样本内变好大概率是噪声。"
        )
    else:
        lines.extend([
            "池化 CI 跨 0 → **不可判定**：既不能说有 edge，也不能说没有。"
            "这与本项目的判据一致——样本不足时的默认值是「不启用」，而不是「先跑跑看」。",
            "",
            "处置：",
            "",
            "1. 策略已完整实现并可随时手动开启（`supertrend_current`，`default_enabled=False`），"
            "便于对照观察与后续复检。",
            "2. **默认不启用**：没有 holdout 为正的证据，就不该在自动巡检里参与资金竞争。",
            "3. 若后续要再投入，先回答「是信号方向没信息，还是成本吃掉了毛利」——"
            "看随机基线均值与策略收益的差，而不是看收益本身。",
        ])
    if payload.get("cross_interval"):
        lines.extend([
            "",
            "## 周期对照（同一批币种、同一年、同一参数）",
            "",
            "| 周期 | 有成交币 | 成交 | 池化每笔期望 | 95% CI | 判定 | 做多 | 做空 | 随机基线 | 账户收益中位数 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for item in payload["cross_interval"]:
            item_ci = item["expectancy_ci"]
            lines.append(
                f"| **{item['interval']}** | {item['symbols_with_trades']}/{item['symbols_total']} | "
                f"{item['trades']} | {item_ci.get('mean_return_pct', 0.0):+.4f}% | "
                f"[{item_ci.get('ci_low_pct', 0.0):+.4f}%, {item_ci.get('ci_high_pct', 0.0):+.4f}%] | "
                f"{item_ci.get('verdict', '')} | "
                f"{item['side_split']['long'].get('mean_return_pct', 0.0):+.3f}% | "
                f"{item['side_split']['short'].get('mean_return_pct', 0.0):+.3f}% | "
                f"{item['random_mean_pct']:+.2f}% | {item['account_return_median_pct']:+.2f}% |"
            )
        note = (
            "\n换周期能改变的是「交易频次与成本摊销」，改变不了方向信号本身的信息量——"
            "各周期的多空分解同号，说明这不是某个周期的偶然。"
        )
        thin = [
            item for item in payload["cross_interval"]
            if item["expectancy_ci"].get("verdict") == "inconclusive"
        ]
        if thin:
            detail = "、".join(f"{item['interval']} 仅 {item['trades']} 笔" for item in thin)
            note += (
                f"\n\n{detail}，CI 跨 0 是**样本量不足**撑出来的宽区间，不是「更好」。"
                "笔数越少区间越宽——把「判不了」读成「有希望」是这类回测最常见的事故。"
            )
        lines.append(note)
    lines.extend([
        "",
        f"## 参数网格对比（spec.grid，"
        f"{'主流币' if payload['universe'].get('kind') == 'major' else '全市场'}口径）",
        "",
        "| 组合 | factor | target_r | 有成交币 | 成交 | 池化每笔期望 | 95% CI | 判定 | 账户收益中位数 | 超随机占比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for item in payload["combos"]:
        item_ci = item["expectancy_ci"]
        lines.append(
            f"| **{item['label']}** | {item['params'].get('supertrend_factor')} | "
            f"{item['params'].get('target_r')}R | {item['symbols_with_trades']}/{item['symbols_total']} | "
            f"{item['trades']} | {item_ci.get('mean_return_pct', 0.0):+.4f}% | "
            f"[{item_ci.get('ci_low_pct', 0.0):+.4f}%, {item_ci.get('ci_high_pct', 0.0):+.4f}%] | "
            f"{item_ci.get('verdict', '')} | {item['account_return_median_pct']:+.2f}% | "
            f"{item['beats_random_ratio']:.0%} |"
        )
    lines.extend([
        "",
        "网格只覆盖策略自己声明的候选档位。组数越多，「某组看起来更好」越可能是多重比较的"
        "产物——判据看的是池化 CI，不是哪一行数字最大。",
        "",
        "同一仓位的分批平仓合并成一笔，收益率按总净盈亏除以总保证金计算。",
        "",
        "## 出场原因分布（基准，池化）",
        "",
        "| 出场 | 笔数 | 占比 |",
        "| --- | --- | --- |",
    ])
    for reason, count in base["status_counts"].items():
        lines.append(f"| {reason} | {count} | {count / total_exit:.0%} |")
    lines.extend(["", "### 信号被拦截原因（基准，池化）", "", "| 原因 | 根数 |", "| --- | --- |"])
    for reason, count in list(base["blocked"].items())[:10]:
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "## 逐币种（基准，按账户收益排序）",
        "",
        "| 币种 | 成交 | 账户收益 | 胜率 | 最大回撤 | 随机基线均值 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    ranked = sorted(base["per_symbol"], key=lambda item: -(item["return_pct"] or 0.0))
    for item in ranked[:25]:
        lines.append(
            f"| {item['symbol']} | {item['trades']} | {item['return_pct']:+.2f}% | "
            f"{item['win_rate']:.1f}% | {item['max_drawdown_pct']:.2f}% | "
            f"{(item['random_mean_pct'] or 0.0):+.2f}% |"
        )
    if len(ranked) > 50:
        lines.append(f"| … | 省略 {len(ranked) - 50} 币 | | | | |")
        for item in ranked[-25:]:
            lines.append(
                f"| {item['symbol']} | {item['trades']} | {item['return_pct']:+.2f}% | "
                f"{item['win_rate']:.1f}% | {item['max_drawdown_pct']:.2f}% | "
                f"{(item['random_mean_pct'] or 0.0):+.2f}% |"
            )
    lines.extend([
        "",
        "## 口径与数据限制（必须先读）",
        "",
        "1. **撮合口径**：信号在收盘确认，次根开盘成交（含滑点与费率）；"
        "止损止盈以信号收盘价为基准推导，分批止盈。翻转根「先出场、不再进场」，"
        "所以反向仓位要等下一次翻转才建立，比「永远在市」的 Supertrend 保守一根 K 线。",
        "2. **跨缺口处理**：K 线上的数据缺口会把序列切成多段。各段**独立**撮合、"
        "账户权益按段复利结转（缺口处仓位已平，现金继续带着走）。"
        "若把各段拼成一整条再算 Supertrend，前一段的状态机会流进后一段，"
        "等于用缺口前的价格推断缺口后的方向。",
        f"3. **最短段长**：单段不足 {payload.get('min_bars', MIN_SEGMENT_BARS)} 根（指标预热需要）的片段直接丢弃，"
        "不参与统计。",
        "4. **账户口径**：每个币种是独立的 10,000 账户，全市场数字是这些账户的截面统计，"
        "不等于一个组合账户的收益——跨币同时持仓的资金占用与相关性不在本口径内。",
        "5. **随机基线的含义**：它保留了入场时点、仓位、杠杆、止损止盈与成本，只翻转方向。"
        "策略收益若不能稳定超过它，说明「什么时候进场」有价值而「往哪个方向」没有——"
        "这对趋势跟随策略是决定性的。",
        "",
        f"耗时 {payload['elapsed_seconds']}s，{payload['processes']} 进程。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
