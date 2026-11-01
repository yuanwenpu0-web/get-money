"""滚动样本外的组合选型：把单信号、合成规则与等权组合放在同一资金口径下比。

与 `lab_experiment` 的单次切分不同，这里按**时间滚动**做多折检验：
每一折只用折前历史选规则，在紧随其后的一段样本外判定。这样能直接回答
「这套组合在没参与挑选的行情里还能不能赚钱」，而不是「历史上谁赚得多」。

资金口径的关键改动：单候选结果用「每笔固定名义本金」，不同候选并发数不同，
收益率不可比；这里统一改成**共享资金池**——每笔按当时在场腿数等分名义本金，
未成交的候选不占资金。这样单策略与组合策略能直接比较。
"""

from __future__ import annotations

import math

import numpy as np

from .lab_sim import SimConfig, Trade, to_ms
from .lab_stats import summarize


class PortfolioConfig:
    """共享资金池的撮合参数。"""

    def __init__(
        self,
        capital: float = 10000.0,
        fee: float = 0.0004,
        slippage: float = 0.0002,
        max_hold_bars: int = 240,
        side: int = 0,
        min_trades: int = 30,
        max_concurrent: int = 0,
        risk_per_trade_pct: float = 0.0,
        atr_mult: float = 0.0,
    ) -> None:
        self.capital = float(capital)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.max_hold_bars = int(max_hold_bars)
        self.side = int(side)
        self.min_trades = int(min_trades)
        # 同时在场的腿数上限（0 = 不限制）。没有它，回测会把资金切成几十份
        # 甚至透支，收益与回撤都会失真到不可解读。
        self.max_concurrent = int(max_concurrent)
        # 每笔风险预算（占总资金比例，0 = 用等分资金）。与 atr_mult 一起
        # 决定单笔名义本金，避免「满仓 × 高波动」把回撤放大到 -100% 以上。
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.atr_mult = float(atr_mult)

    def sim_config(self, stake: float) -> SimConfig:
        return SimConfig(stake, self.fee, self.slippage, self.max_hold_bars, self.side)


def _concurrency_map(per_leg: dict[str, list[Trade]]) -> dict[tuple[int, int], int]:
    """计算每笔成交在入场时点的并发腿数（含它自己）。"""
    events: list[tuple[int, int, tuple[int, int]]] = []
    for leg_index, trades in enumerate(per_leg.values()):
        for order, trade in enumerate(trades):
            events.append((trade.entry_ts, 0, (leg_index, order)))
            events.append((trade.exit_ts, 1, (leg_index, order)))
    # 同一 ts 先结算出场（kind=1 排在前），避免把已平腿算进并发
    events.sort(key=lambda item: (item[0], item[1]))
    concurrency: dict[tuple[int, int], int] = {}
    active = 0
    open_keys: dict[tuple[int, int], int] = {}
    for _ts, kind, key in events:
        if kind == 1:
            concurrency[key] = open_keys.pop(key, 1)
            active -= 1
            continue
        active += 1
        open_keys[key] = max(active, 1)
    for key, value in open_keys.items():
        concurrency.setdefault(key, value)
    return concurrency


def scale_to_pool(per_leg: dict[str, list[Trade]], max_concurrent: int = 0) -> list[Trade]:
    """把「每腿满仓」的成交按当时在场腿数缩放到共享资金池。

    原始 `simulate` 每条腿都用满额本金成交，直接相加等于透支。这里按
    entry 时点的并发腿数缩放每笔名义本金，盈亏同比例缩放。

    `max_concurrent > 0` 时按「资金 / 该上限」为每腿分配名义本金，并对
    超出上限的成交直接丢弃——真实账户容不下无限条腿同时开仓。
    """
    concurrency = _concurrency_map(per_leg)
    scaled: list[Trade] = []
    for leg_index, trades in enumerate(per_leg.values()):
        for order, trade in enumerate(trades):
            if max_concurrent and concurrency.get((leg_index, order), 1) > max_concurrent:
                continue
            share = 1.0 / float(concurrency.get((leg_index, order), 1) or 1)
            scaled.append(
                Trade(
                    trade.symbol,
                    trade.side,
                    trade.entry_ts,
                    trade.exit_ts,
                    trade.entry_price,
                    trade.exit_price,
                    trade.bars,
                    trade.pnl * share,
                    trade.net_pct * share,
                    trade.fees * share,
                    trade.reason,
                )
            )
    scaled.sort(key=lambda trade: trade.entry_ts)
    return scaled


def build_legs(ds, signals: dict[str, object], cfg: PortfolioConfig) -> list[Trade]:
    """按共享资金池撮合一组信号。

    每条腿独立用「资金 / 最大腿数」的名义本金撮合（同币同时一仓），
    再把所有腿的成交按时间合并。这样任一时点的名义敞口不超过总资金，
    不会像「每条腿都满仓再缩放」那样把回撤放大到不可解读。

    当 `risk_per_trade_pct > 0` 时改为「按 ATR 风险定仓」：单笔名义本金
    = 风险预算 / (atr_mult × ATR%)。波动大的币自动缩仓，这是把回撤压到
    可接受区间的必要条件，只靠等分资金做不到。
    """
    from .lab_sim import simulate

    slots = cfg.max_concurrent if cfg.max_concurrent > 0 else max(len(ds.frames), 1)
    stake = cfg.capital / float(slots)
    risk_cfg = cfg.risk_per_trade_pct > 0 and cfg.atr_mult > 0
    per_leg: dict[str, list[Trade]] = {}
    for symbol, frame in ds.frames.items():
        leg_stake = stake
        if risk_cfg:
            atr_pct = _entry_atr_pct(frame, signals[symbol], cfg)
            if atr_pct:
                leg_stake = min(stake, cfg.capital * cfg.risk_per_trade_pct / (cfg.atr_mult * atr_pct))
        per_leg[symbol] = simulate(
            symbol, frame, signals[symbol], ds.exits[symbol], cfg.sim_config(leg_stake)
        )
    merged: list[Trade] = []
    for trades in per_leg.values():
        merged.extend(trades)
    merged.sort(key=lambda trade: trade.entry_ts)
    return merged


def _entry_atr_pct(frame, signals, cfg: PortfolioConfig) -> float:
    """信号触发时点的平均 ATR%（用于把名义本金换算成风险预算）。"""
    from .lab_indicators import atr_wilder

    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr = atr_wilder(high, low, close, 14)
    longs = np.asarray(signals.long, dtype=bool)
    shorts = np.asarray(signals.short, dtype=bool)
    mask = longs | shorts
    if not mask.any():
        return 0.0
    ratio = atr[mask] / close[mask]
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return 0.0
    return float(np.median(ratio))


def _sharpe(trades: list[Trade], capital: float) -> float:
    if len(trades) < 2:
        return 0.0
    rets = np.array([t.pnl / capital for t in trades], dtype=float)
    std = float(rets.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(rets.mean() / std * math.sqrt(len(rets)))


def portfolio_metrics(trades: list[Trade], cfg: PortfolioConfig, bars: int, symbols: int) -> dict:
    """组合级指标：按平仓时间累计已实现盈亏，除以固定总资金。"""
    slots = cfg.max_concurrent if cfg.max_concurrent > 0 else max(symbols, 1)
    stake = cfg.capital / float(slots)
    stats = summarize(trades, stake, symbols)
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    equity = cfg.capital + np.cumsum([t.pnl for t in ordered]) if ordered else np.array([cfg.capital])
    peak = np.maximum.accumulate(equity)
    drawdown = float((equity / peak - 1.0).min() * 100.0)
    exposure = sum(max(t.bars, 0) for t in trades) / bars * 100.0 if bars > 0 else 0.0
    gross_win = float(sum(t.pnl for t in trades if t.pnl > 0))
    gross_loss = float(-sum(t.pnl for t in trades if t.pnl < 0))
    return {
        "trades": stats["trades"],
        "total_pnl": round(float(sum(t.pnl for t in trades)), 4),
        "return_pct": round(float(sum(t.pnl for t in trades) / cfg.capital * 100.0), 4),
        "max_drawdown_pct": round(abs(drawdown), 4),
        "win_rate": stats["win_rate"],
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "trades_per_symbol": stats["trades_per_symbol"],
        "avg_bars": stats["avg_bars"],
        "exposure_pct": round(exposure, 3),
        "sharpe": round(_sharpe(trades, cfg.capital), 4),
        "final_equity": round(float(cfg.capital + sum(t.pnl for t in trades)), 4),
    }


def _bars_between(ds, start: int | None, end: int | None) -> int:
    total = 0
    for frame in ds.frames.values():
        stamps = to_ms(frame.index)
        mask = np.ones(len(stamps), dtype=bool)
        if start is not None:
            mask &= stamps >= start
        if end is not None:
            mask &= stamps < end
        total += int(mask.sum())
    return max(total, 1)


def _window(trades: list[Trade], start: int | None, end: int | None) -> list[Trade]:
    return [
        t for t in trades
        if (start is None or t.exit_ts >= start) and (end is None or t.exit_ts < end)
    ]


def bootstrap_mean_ci(values: np.ndarray, draws: int = 2000, seed: int = 20261101) -> tuple[float, float]:
    """单笔收益均值的 bootstrap 置信区间；跨 0 即「无法判定有 edge」。"""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(draws, len(arr)))
    means = arr[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def expectancy(trades: list[Trade], capital: float) -> dict:
    """每笔期望（占总资金 %）与其 95% CI；判定只看 CI 是否跨 0。"""
    if not trades:
        return {"trades": 0, "mean_pct": 0.0, "ci_low": None, "ci_high": None, "verdict": "no_trades"}
    rets = np.array([t.pnl / capital * 100.0 for t in trades], dtype=float)
    low, high = bootstrap_mean_ci(rets)
    mean = float(rets.mean())
    if len(trades) < 30:
        verdict = "insufficient"
    elif low > 0:
        verdict = "positive"
    elif high < 0:
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {
        "trades": len(trades),
        "mean_pct": round(mean, 5),
        "ci_low": round(low, 5) if np.isfinite(low) else None,
        "ci_high": round(high, 5) if np.isfinite(high) else None,
        "verdict": verdict,
    }


def rolling_evaluate(
    ds,
    legs: dict[str, dict[str, object]],
    cfg: PortfolioConfig,
    folds: int = 4,
    train_ratio: float = 0.5,
    top_k: int = 3,
    select_metric: str = "return_pct",
) -> dict:
    """锚定式滚动检验：每折用折前历史选 top_k 条腿，在折内等权组合并判定。"""
    lo = min(int(to_ms(f.index)[0]) for f in ds.frames.values())
    hi = max(int(to_ms(f.index)[-1]) for f in ds.frames.values())
    first_test = lo + int((hi - lo) * train_ratio)
    bounds = [first_test]
    step = (hi - first_test) / max(folds, 1)
    for i in range(1, folds + 1):
        bounds.append(int(first_test + step * i))
    cached = {label: build_legs(ds, signals, cfg) for label, signals in legs.items()}
    fold_rows: list[dict] = []
    for k in range(len(bounds) - 1):
        start, end = bounds[k], bounds[k + 1]
        scored = []
        for label, trades in cached.items():
            history = _window(trades, None, start)
            metrics = portfolio_metrics(history, cfg, _bars_between(ds, lo, start), len(ds.frames))
            if metrics["trades"] < cfg.min_trades:
                continue
            scored.append((label, metrics))
        if not scored:
            fold_rows.append({"fold": k + 1, "start": start, "end": end, "selected": [], "test": None})
            continue
        scored.sort(key=lambda item: item[1][select_metric], reverse=True)
        picked = [label for label, _ in scored[:top_k]]
        merged: list[Trade] = []
        for label in picked:
            merged.extend(_window(cached[label], start, end))
        fold_rows.append(
            {
                "fold": k + 1,
                "start": start,
                "end": end,
                "selected": picked,
                "train": {label: metrics for label, metrics in scored[:top_k]},
                "test": portfolio_metrics(merged, cfg, _bars_between(ds, start, end), len(ds.frames)),
            }
        )
    return {"folds": fold_rows, "leg_trades": cached}
