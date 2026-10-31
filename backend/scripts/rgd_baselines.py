"""RGD 回测的证伪基线与池化汇总。

为什么不能用 `execution.randomize_directions`（契约 §7.4 / A24）：
它只翻 `direction` 一列（`execution.py:15-27`），而 RGD 的 `direction ≡ 0`
（专用 broker 早路由）。翻转后仍是全 0 —— no-op，随机基线会恒等于策略本身，
报告照常产出、结论全错，属于 generated-code-failure-modes 的「沉默逻辑错误」。

`randomize_rgd` 改为对整组 `rgd_*` 决策列做**段内循环位移**：保留信号数量与结构、
破坏时序对齐，让「什么时候开仓」变成随机变量。位移后的 `rgd_allow_*` / `rgd_state`
也跟着走，broker 的 flip 判定同样被随机化。
"""
from __future__ import annotations

import random
import statistics
from collections import Counter

import numpy as np

from app.backtest.significance import bootstrap_expectancy_ci
from app.features.rgd import DECISION_COLUMNS

# 池化判定的最小成交笔数：4 笔的 bootstrap 会给出「显著为正」的假精确读数。
MIN_POOLED_TRADES = 30


def randomize_rgd(signals, seed: int):
    """段内循环位移全部决策列；偏移量由种子决定且至少 1 根。"""
    rng = random.Random(seed)
    n = len(signals)
    if n < 2:
        return signals.copy()
    shift = rng.randrange(1, n)
    shuffled = signals.copy()
    for column in DECISION_COLUMNS:
        if column in signals.columns:
            shuffled[column] = np.roll(signals[column].to_numpy(), shift)
    return shuffled


def pool_trade_returns(rows: list[list]) -> dict:
    """把 [(return_pct, side)] 池化后算每笔期望的 CI，并做多空分解。"""
    pairs = [item for row in rows for item in row]
    ci = bootstrap_expectancy_ci(
        [{"return_pct": value} for value, _ in pairs], min_trades=MIN_POOLED_TRADES
    )
    longs = [value for value, side in pairs if side == "long"]
    shorts = [value for value, side in pairs if side == "short"]
    return {
        "trades": len(pairs),
        "expectancy_ci": ci,
        "side_split": {
            "long": bootstrap_expectancy_ci(
                [{"return_pct": value} for value in longs], min_trades=MIN_POOLED_TRADES
            ),
            "short": bootstrap_expectancy_ci(
                [{"return_pct": value} for value in shorts], min_trades=MIN_POOLED_TRADES
            ),
        },
    }


def merge_counts(sequence) -> dict:
    totals: Counter = Counter()
    for mapping in sequence:
        for key, value in (mapping or {}).items():
            totals[str(key)] += int(value)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def random_summary(samples: list[float]) -> dict:
    if not samples:
        return {"mean_pct": None, "p95_pct": None, "p05_pct": None}
    ordered = sorted(samples)
    return {
        "mean_pct": round(statistics.fmean(samples), 4),
        "p95_pct": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
        "p05_pct": round(ordered[int(len(ordered) * 0.05)], 4),
    }


def summarise_group(rows: list[dict], group: dict) -> dict:
    """把一组（G1..G4）的全币种结果聚合成报告口径。"""
    with_trades = [row for row in rows if row["trades"] > 0]
    returns = [row["return_pct"] for row in with_trades if row["return_pct"] is not None]
    random_means = [
        row["random"]["mean_pct"] for row in with_trades if row["random"]["mean_pct"] is not None
    ]
    drawdowns = [row["max_drawdown_pct"] for row in with_trades if row["max_drawdown_pct"] is not None]
    beat = [
        row for row in with_trades
        if row["random"]["p95_pct"] is not None and (row["return_pct"] or 0.0) > row["random"]["p95_pct"]
    ]
    slippages = sorted({round(float(row["slippage_used"]), 10) for row in with_trades})
    return {
        "id": group["id"], "label": group["label"], "dir_mode": group["dir_mode"],
        "fee_rate": group["fee_rate"], "slippage_mode": group["slippage"],
        "slippage_note": group["slippage_note"],
        "slippage_used": slippages[0] if len(slippages) == 1 else slippages,
        "symbols_total": len(rows), "symbols_with_trades": len(with_trades),
        "trades": sum(row["trades"] for row in with_trades),
        "signal_bars": sum(row["signal_bars"] for row in rows),
        "account_return_median_pct": round(statistics.median(returns), 4) if returns else 0.0,
        "account_return_mean_pct": round(statistics.fmean(returns), 4) if returns else 0.0,
        "positive_symbols": int(sum(1 for value in returns if value > 0)),
        "max_drawdown_mean_pct": round(statistics.fmean(drawdowns), 4) if drawdowns else 0.0,
        "exposure_mean_pct": round(statistics.fmean([row["exposure_pct"] for row in with_trades]), 4)
        if with_trades else 0.0,
        "beats_random_ratio": round(len(beat) / len(with_trades), 4) if with_trades else 0.0,
        "random_mean_pct": round(statistics.fmean(random_means), 4) if random_means else 0.0,
        **pool_trade_returns([row["trade_returns"] for row in rows]),
        "status_counts": merge_counts(row["status_counts"] for row in with_trades),
        # 腿级出场分布与 TP1 触发率：仓位级 status_counts 会被最后一腿覆盖，
        # 这两项才让读者看到实际发生过的部分止盈。
        "fill_status_counts": merge_counts(row["fill_status_counts"] for row in rows),
        "tp1_positions": sum(row["tp1_positions"] for row in rows),
        "tp1_position_ratio": round(
            sum(row["tp1_positions"] for row in rows)
            / max(sum(row["trades"] for row in rows), 1),
            4,
        ),
        "blocked": merge_counts(row["blocked"] for row in rows),
        "htf_degraded_symbols": [row["symbol"] for row in rows if row.get("htf_degraded")],
        "per_symbol": [
            {
                "symbol": row["symbol"], "trades": row["trades"],
                "return_pct": row["return_pct"], "win_rate": row["win_rate"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "random_mean_pct": row["random"]["mean_pct"],
            }
            for row in with_trades
        ],
    }
