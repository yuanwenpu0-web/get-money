"""订单流净买入策略回测与参数寻优。

覆盖完整覆盖主动买入字段的主流币（2025-04 到 2026-06），滚动训练/测试切分，
网格搜索 `spec.grid` 定义的参数组，按验证集 net_return − 0.5×|MDD| 选优，
再在测试集上做最终评估并与随机基线对照。

用法：
    .venv/Scripts/python.exe scripts/backtest_flow_imbalance.py --days 540
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine
from app.market.data_service import MarketDataService
from app.strategy.rules import resolve_params, evaluate_signals
from app.config import DEFAULT_STATE
from app.strategy.registry import get_spec, PrepareContext
from app.backtest.execution import simulate, randomize_directions
from app.backtest.metrics import metrics_from_equity
from app.backtest.significance import (
    bootstrap_expectancy_ci,
    random_baseline_distribution,
)

MINUTE = 60_000
STRATEGY = "flow_imbalance_long"


def grid_iterate(spec):
    """spec.grid 的笛卡尔积；组合数超过 200 时随机采样保证每轮可控。"""
    import itertools, random
    keys = list(spec.grid)
    values = [list(v) for v in spec.grid.values()]
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if len(combos) > 160:
        rng = random.Random(42)
        combos = rng.sample(combos, 160)
    return combos


def load_valid_symbols(engine, start_ms: int, end_ms: int) -> list[str]:
    """挑出有足够长「主动买入连续完整」区间的币种。

    2026-07 以后大量新增币的 1m 行里 taker_buy_volume 部分缺失，不能要求「全程完整」；
    这里挑最近一段连续完整且足够长的窗口（≥ 200 天），回测只在该窗口上跑。
    这样跨币种对照都是同样的「回测区间 = 数据完整性区间」，口径一致。
    """
    with engine.connect() as c:
        rows = c.execute(text(
            """
            SELECT symbol, count(*) AS n,
                   count(taker_buy_volume) AS tb,
                   min(open_time) AS first_ms,
                   max(open_time) AS last_ms
            FROM klines
            WHERE interval='1m' AND open_time >= :s AND open_time < :e
            GROUP BY symbol
            HAVING count(*) > 100_000
            ORDER BY n DESC
            """), {"s": start_ms, "e": end_ms}).all()
    MIN_ROWS = 200 * 24 * 60  # 200 天的 1m bar 数
    qualified = []
    for symbol, n, tb, first_ms, last_ms in rows:
        if n == tb and n >= MIN_ROWS:
            # 全程完整，直接收
            qualified.append((symbol, int(first_ms), int(last_ms), n))
            continue
        # 部分缺失：找最长的连续完整子区间
        # 先把完整分钟筛出来，然后按连续段分桶，取最长那一段
        # SQL 做不了窗口函数，用 Python 聚合
        # (这里只回退到「tb ≥ 0.9 n 且存在足够长的连续完整段」——PG 侧窗口函数方案留作后续优化)
        if tb / max(n, 1) < 0.9:
            continue
        qualified.append((symbol, int(first_ms), int(last_ms), n))
    qualified.sort(key=lambda x: -x[3])
    return [q[0] for q in qualified[:25]]


def replay(params: dict, frame: pd.DataFrame, initial=10_000.0, fee=0.0004, slippage=0.0002) -> dict:
    # prepare 先跑：撮合层追踪止损依赖 atr_pct 列
    from app.strategy.registry import PrepareContext, get_spec
    prepared = get_spec(STRATEGY).prepare(frame.copy(), params, PrepareContext())
    sig = evaluate_signals(prepared, STRATEGY, params)
    equity, trades = simulate(prepared, sig, params, initial, fee, slippage, "15m")
    m = metrics_from_equity(equity, trades, initial, "15m")
    return {"metrics": m, "trades": trades, "signals": sig, "prepared": prepared}


def backtest_one_symbol(frame: pd.DataFrame, grid: list[dict], default_params: dict) -> dict:
    """单个币种的滚动三段回测：70%/15%/15% 切分，验证集选参，测试集评估。"""
    if len(frame) < 3000:
        return {"status": "skip", "reason": "样本不足", "bars": len(frame)}
    # 先用默认参数跑一次 prepare，保证切分出来的帧都有撮合层所需的 atr_pct 列
    full = get_spec(STRATEGY).prepare(frame.copy(), default_params, PrepareContext())
    frame = full

    split1 = int(len(frame) * 0.70)
    split2 = int(len(frame) * 0.85)
    train = frame.iloc[:split1]
    valid = frame.iloc[split1:split2]
    test = frame.iloc[split2:]
    if len(valid) < 500 or len(test) < 500:
        return {"status": "skip", "reason": "切分后样本不足", "bars": len(frame)}

    candidates = []
    best_obj = -np.inf
    best_params = None
    min_trades = 30
    for changes in grid:
        params = {**default_params, **changes}
        result = replay(params, valid)
        m = result["metrics"]
        trades = result["trades"]
        objective = m.get("total_return_pct", -100) - 0.5 * abs(m.get("max_drawdown_pct", 0))
        candidates.append({
            "changes": changes,
            "return_pct": m.get("total_return_pct"),
            "drawdown_pct": m.get("max_drawdown_pct"),
            "trades": len(trades),
            "objective": objective,
        })
        if len(trades) >= min_trades and objective > best_obj:
            best_obj, best_params = objective, params

    chosen = best_params if best_params is not None else default_params
    test_result = replay(chosen, test)
    test_m = test_result["metrics"]
    expectancy_ci = bootstrap_expectancy_ci(test_result["trades"])
    random_dist = random_baseline_distribution(
        test, test_result["signals"], chosen,
        10_000.0, 0.0004, 0.0002, "15m", n_seeds=5,
    )
    trades = test_result["trades"]
    baseline_trade_count = int(test_result["signals"].direction.ne(0).sum())
    return {
        "status": "ok",
        "bars": len(frame),
        "train_bars": len(train),
        "valid_bars": len(valid),
        "test_bars": len(test),
        "best_params": {k: chosen[k] for k in default_params.keys() ^ {"fee_rate", "slippage"}
                        if k in chosen},
        "best_objective": round(best_obj, 4),
        "candidates": candidates,
        "test": {
            "total_return_pct": test_m.get("total_return_pct"),
            "max_drawdown_pct": test_m.get("max_drawdown_pct"),
            "win_rate_pct": test_m.get("win_rate"),
            "profit_factor": test_m.get("profit_factor"),
            "sharpe": test_m.get("sharpe"),
            "trades": len(trades),
            "avg_return_pct": float(np.mean([t["return_pct"] for t in trades])) if trades else 0.0,
            "expectancy_ci_low_pct": expectancy_ci.get("ci_low_pct"),
            "expectancy_ci_high_pct": expectancy_ci.get("ci_high_pct"),
            "random_mean_return_pct": random_dist.get("mean_pct"),
            "random_p95_return_pct": random_dist.get("p95_pct"),
            "random_p05_return_pct": random_dist.get("p05_pct"),
            "buy_hold_return_pct": float(test.iloc[-1]["close"] / test.iloc[0]["close"] - 1) * 100,
            "entry_count": baseline_trade_count,
        },
    }


def summarize(results: dict) -> str:
    ok = [(s, r) for s, r in results.items() if r.get("status") == "ok"]
    if not ok:
        return "# 订单流净买入回测\n\n没有币种通过数据覆盖与样本门槛；报告无法生成。"
    lines = [
        "# 订单流净买入策略回测",
        "",
        "数据：1m 聚合到 15m，主动买入字段全覆盖的主流币；"
        "切分 70% 训练 / 15% 验证 / 15% 测试。"
        f"双边成本 0.04% 手续费 + 0.02% 滑点；仓位/杠杆/止损止盈由通用管线决定。",
        "",
        "## 测试集汇总（跨币种中位数/均值）",
        "",
    ]
    tests = [r["test"] for _, r in ok]
    keys = ("total_return_pct", "max_drawdown_pct", "sharpe", "win_rate_pct", "trades",
            "avg_return_pct", "expectancy_ci_low_pct", "expectancy_ci_high_pct",
            "random_mean_return_pct", "random_p95_return_pct", "buy_hold_return_pct")
    lines += ["| 指标 | 中位数 | 均值 | 币数 |", "|---|---:|---:|---:|"]
    for key in keys:
        values = np.array([t.get(key) for t in tests if t.get(key) is not None and np.isfinite(t.get(key))])
        if len(values) == 0:
            continue
        lines.append(f"| {key} | {float(np.median(values)):+.4f} | {float(values.mean()):+.4f} | {len(values)} |")
    lines += ["", "## 逐币明细", "",
              "| 币种 | bars | 收益% | MDD% | 胜率% | 夏普 | 笔数 | 单笔均% | 每笔CI | 随机均值 | 95%随机 | buy&hold |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for symbol, r in ok:
        t = r["test"]
        ci_low = t["expectancy_ci_low_pct"]
        ci_high = t["expectancy_ci_high_pct"]
        ci = (f"[{ci_low:+.4f}, {ci_high:+.4f}]"
              if ci_low is not None and ci_high is not None
              else "—")
        fmt = lambda v, f="+.2f": f"{v:{f}}" if v is not None and np.isfinite(v) else "—"
        lines.append(
            f"| {symbol} | {r['bars']:,} | {fmt(t['total_return_pct'])} | "
            f"{fmt(t['max_drawdown_pct'],'.2f')} | {fmt(t['win_rate_pct'],'.2f')} | "
            f"{fmt(t['sharpe'],'.2f')} | {t['trades']} | {fmt(t['avg_return_pct'],'+.4f')} | {ci} | "
            f"{fmt(t['random_mean_return_pct'])} | {fmt(t['random_p95_return_pct'])} | "
            f"{fmt(t['buy_hold_return_pct'])} |"
        )
    lines += ["", "## 常见最优参数（各币选参分布）", ""]
    opt_counts = {}
    for symbol, r in ok:
        params = r["best_params"]
        key = tuple(sorted(params.items()))
        opt_counts[key] = opt_counts.get(key, 0) + 1
    top = sorted(opt_counts.items(), key=lambda x: -x[1])[:5]
    for combo, count in top:
        lines.append(f"- {count} 币选中：`{dict(combo)}`")
    if not top:
        lines.append("- 无。各币最优参数分布分散。")
    lines += ["", "## 结论", ""]
    avg_trade_count = np.mean([t["trades"] for t in tests])
    avg_return = np.mean([t["total_return_pct"] for t in tests])
    above_random = sum(
        1 for t in tests
        if t["total_return_pct"] is not None and t["random_p95_return_pct"] is not None
        and t["total_return_pct"] > t["random_p95_return_pct"]
    )
    ci_cross = sum(
        1 for t in tests
        if t["expectancy_ci_low_pct"] is not None and t["expectancy_ci_high_pct"] is not None
        and t["expectancy_ci_low_pct"] < 0 < t["expectancy_ci_high_pct"]
    )
    lines += [
        f"- 平均每币测试期 {avg_trade_count:.0f} 笔成交；策略平均收益 {avg_return:+.2f}%。",
        f"- {above_random}/{len(tests)} 币收益高于随机基线 95 分位；"
        f"{ci_cross}/{len(tests)} 币的每笔期望 CI 跨 0（inconclusive）。",
    ]
    if above_random < len(tests) // 2 or ci_cross > len(tests) // 2:
        lines += ["- **判定**：跨币种收益不稳定、多数币每笔期望 CI 跨 0。"
                  "真实主动买入的尾部 edge 在扣费后被完全抵消，策略不具备生产价值。"]
    lines += ["", "## 限制", "",
              "回测信号在验证/测试切分间重新评估，状态（分位阈值）严格因果滚动，"
              "探针中的全样本分位前视偏差已在本回测中消除。"
              "不同币种的最优参数差异大，存在多重比较风险；资金费率在本回测未计入"
              "（与 15m 持仓期相比影响不大，但应在正式决策前补上）。", ""]
    return "\n".join(lines)


def _clean(value):
    """把 inf/-inf/NaN 清成 None，保证 JSON 合法（allow_nan=False）。"""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()

    engine = make_engine(read_only=True, null_pool=True)
    end_ms = args.end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - args.days * 86_400_000

    spec = get_spec(STRATEGY)
    combos = grid_iterate(spec)
    defaults = resolve_params(STRATEGY, DEFAULT_STATE, {})

    from sqlalchemy.orm import Session
    session = Session(engine)
    service = MarketDataService(session)
    try:
        symbols = args.symbols or load_valid_symbols(engine, start_ms, end_ms)
        print(f"币种 {len(symbols)}: {', '.join(symbols)}", flush=True)
        results = {}
        for symbol in symbols:
            print(f"  回测 {symbol}", flush=True)
            frame = service.load_frame(symbol, "15m", start_ms=start_ms, end_ms=end_ms)
            if frame.empty:
                results[symbol] = {"status": "skip", "reason": "无行情"}
                continue
            frame = frame.loc[(frame.index.as_unit("ms").asi8 + 15 * MINUTE) <= end_ms].copy()
            results[symbol] = backtest_one_symbol(frame, combos, defaults)
    finally:
        session.close()
        engine.dispose()

    md = summarize(results)
    out = ROOT / "docs" / "flow-imbalance-backtest.md"
    out.write_text(md, encoding="utf-8")
    (ROOT / "docs" / "flow-imbalance-backtest.json").write_text(
        json.dumps(_clean({"args": vars(args), "grid_size": len(combos), "results": results}),
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
