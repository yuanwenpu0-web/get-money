"""评价指标与显著性工具。

口径约定（报告里必须原样复述）：

- 每笔固定名义本金 `stake`，收益按净额（已扣双边手续费与滑点）计。
- 最大并发 = 同时持仓的币种数上限；本金基数取 `stake × 最大并发`，
  这样不同候选的收益率与回撤处在同一「占用资金」口径下可比。
- 最大回撤按已实现盈亏曲线（按平仓时间排序）相对本金计算，不含浮亏。
- 显著性用单笔净收益的 t 统计量做正态近似，多候选再用 BH-FDR 校正。
"""

from __future__ import annotations

import math

import numpy as np


def _sweep_max_concurrent(trades: list) -> int:
    events: list[tuple[int, int]] = []
    for trade in trades:
        events.append((trade.entry_ts, 1))
        events.append((trade.exit_ts, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return max(peak, 1)


def equity_curve(trades: list, stake: float) -> tuple[np.ndarray, float, float]:
    """返回 (权益数组, 本金, 最大回撤百分比)。权益按平仓时间累加已实现盈亏。"""
    if not trades:
        return np.array([stake]), stake, 0.0
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    capital = stake * _sweep_max_concurrent(trades)
    pnl = np.cumsum([t.pnl for t in ordered])
    equity = capital + pnl
    peak = np.maximum.accumulate(equity)
    drawdown = float((equity / peak - 1.0).min() * 100.0)
    return equity, capital, drawdown


def summarize(trades: list, stake: float, symbols: int) -> dict:
    if not trades:
        return {
            "trades": 0, "win_rate": None, "payoff": None, "profit_factor": None,
            "total_pnl": 0.0, "avg_pnl": 0.0, "avg_net_pct": 0.0, "t_stat": 0.0,
            "p_value": 1.0, "max_drawdown_pct": 0.0, "capital": stake,
            "total_return_pct": 0.0, "max_concurrent": 0, "avg_bars": 0.0,
            "long_pnl": 0.0, "short_pnl": 0.0, "trades_per_symbol": 0.0,
            "exit_reasons": {}, "avg_fee": 0.0,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    pcts = np.array([t.net_pct for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    std = float(pcts.std(ddof=1)) if len(pcts) > 1 else 0.0
    t_stat = float(pcts.mean() / (std / math.sqrt(len(pcts)))) if std > 0 else 0.0
    p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))
    equity, capital, drawdown = equity_curve(trades, stake)
    reasons: dict[str, int] = {}
    for trade in trades:
        reasons[trade.reason] = reasons.get(trade.reason, 0) + 1
    return {
        "trades": len(trades),
        "win_rate": round(float(len(wins) / len(pnls)), 4),
        "payoff": round(float(wins.mean() / abs(losses.mean())), 3) if len(wins) and len(losses) else None,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "total_pnl": round(float(pnls.sum()), 4),
        "avg_pnl": round(float(pnls.mean()), 4),
        "avg_net_pct": round(float(pcts.mean()) * 100, 5),
        "t_stat": round(t_stat, 3),
        "p_value": p_value,
        "max_drawdown_pct": round(abs(drawdown), 3),
        "capital": round(capital, 2),
        "total_return_pct": round(float(pnls.sum() / capital) * 100, 3),
        "max_concurrent": int(_sweep_max_concurrent(trades)),
        "avg_bars": round(float(np.mean([t.bars for t in trades])), 2),
        "long_pnl": round(float(sum(t.pnl for t in trades if t.side == 1)), 4),
        "short_pnl": round(float(sum(t.pnl for t in trades if t.side == -1)), 4),
        "trades_per_symbol": round(len(trades) / max(symbols, 1), 2),
        "exit_reasons": reasons,
        "avg_fee": round(float(np.mean([t.fees for t in trades])), 5),
    }


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def bootstrap_mean_ci(values: np.ndarray, draws: int = 2000, seed: int = 20261101) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def percentile_rank(value: float, samples: list[float]) -> float:
    if not samples:
        return float("nan")
    arr = np.asarray(samples, dtype=float)
    return float((arr < value).mean())
