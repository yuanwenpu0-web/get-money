"""多币种组合回测引擎：等权分配 + 曲线对齐 + 组合级/单币种指标与分散度。

上层封装 runner.run_backtest（不修改其现有行为，本模块只做编排与汇总）：
- 每个 (symbol, strategy) 为一条腿，资金等权分配 leg_cash = initial / (币种数 × 策略数)；
  单策略时即"每个币种 initial / 币种数"
- 各腿权益曲线按 bar 时间 outer join + 前向填充对齐；腿首个 bar 之前资金尚未投入，
  按该腿初始资金填充，组合期初权益 = initial_cash
- 等权下组合收益恒等于各币种收益均值，可用于校验"同时运行 vs 顺序运行"无差异
- 单币种回测失败（数据不足等）跳过该腿并记 errors，不阻塞整体；失败腿资金不计入组合
- 口径注意：本引擎允许同一币种的多策略腿同时存在（各占 1/(币种数×策略数) 资金），
  与实盘"同一币种同时只持一个仓位（评分择优）"不同，解读时需考虑该差异
"""

from __future__ import annotations

import pandas as pd

from ..market.metrics_history import metrics_as_of
from .execution import exposure_metrics
from .metrics import downsample, metrics_from_equity
from .runner import run_backtest
from .weighting import metric_weights, regime_weights, weighted_equity

WEIGHT_MODES = ("equal", "metric")
DEFAULT_REBALANCE_BARS = 24  # 1h 下约每日再平衡

BLACKLIST_MIN_TRADES = 20  # 负期望黑名单候选的最小成交笔数，样本不足不判死刑


def _curve_series(curve: list[dict]) -> pd.Series:
    """downsample 输出 [{"time","equity"}] → UTC 时间索引的权益 Series。"""
    if not curve:
        return pd.Series(dtype=float)
    index = pd.to_datetime([point["time"] for point in curve], utc=True)
    return pd.Series([float(point["equity"]) for point in curve], index=index).sort_index()


def _aligned_legs_frame(legs: list[dict]) -> pd.DataFrame:
    """legs → 对齐后的 DataFrame（列 leg0..legN），outer join + 前向填充；
    腿起点之前按该腿初始资金填充（资金未投入）。等权/加权两条路径共用同一对齐结果。"""
    columns: dict[str, pd.Series] = {}
    for position, leg in enumerate(legs):
        series = _curve_series(leg.get("curve") or [])
        if not series.empty:
            columns[f"leg{position}"] = series
    if not columns:
        return pd.DataFrame()
    aligned = pd.concat(columns, axis=1)
    for position, leg in enumerate(legs):
        name = f"leg{position}"
        if name in aligned.columns:
            aligned[name] = aligned[name].ffill().fillna(float(leg["initial"]))
    return aligned.sort_index()


def align_portfolio_equity(legs: list[dict]) -> pd.Series:
    """legs: [{"initial": float, "curve": [...]}] → 对齐求和后的组合权益 Series。

    outer join 对齐 bar 时间；前向填充；腿起点之前按该腿初始资金填充（资金未投入），
    保证组合期初权益 = 各腿初始资金之和。
    """
    aligned = _aligned_legs_frame(legs)
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.sum(axis=1).sort_index()


def _metric_weight_lookup(db, legs: list[dict], strategies: list[str], regime_weighting: bool):
    """构造 weight_lookup(ts_ms) → {legN: 权重}。指标一律经 metrics_as_of 取数（反前视）。"""
    symbols = list(dict.fromkeys(leg["symbol"] for leg in legs))
    fallback = 1.0 / max(len(strategies), 1)

    def lookup(ts_ms: int) -> dict[str, float]:
        metrics_by_symbol = {s: metrics_as_of(db, s, ts_ms) or {} for s in symbols}
        symbol_weights = metric_weights(symbols, metrics_by_symbol)
        # 只有全部币种都有指标时才倾斜：任一缺失 → 仓位与策略维度都退回等权，
        # 避免"部分币种无数据却仍按 regime 倾斜"的伪信号
        complete = all(metrics_by_symbol.values()) and len(strategies) > 1
        weights: dict[str, float] = {}
        for position, leg in enumerate(legs):
            symbol_w = symbol_weights.get(leg["symbol"], 0.0)
            if regime_weighting and complete:
                pair = regime_weights(metrics_by_symbol.get(leg["symbol"], {}))
                strategy_w = pair.get(leg["strategy"], fallback)
            else:
                strategy_w = fallback
            weights[f"leg{position}"] = max(symbol_w, 0.0) * max(strategy_w, 0.0)
        return weights

    return lookup


def _leg_summary(result: dict) -> dict:
    metrics = result.get("metrics", {})
    return {
        "return_pct": metrics.get("total_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "total_trades": metrics.get("total_trades"),
        "expectancy": metrics.get("expectancy"),
        "selected": (metrics.get("validation") or {}).get("selected"),
        "warnings": metrics.get("warnings") or [],
    }


def _run_legs(db, symbols, interval, strategies, days, fee, slippage, leg_cash, end_ms):
    """逐 (symbol, strategy) 跑单币种回测；失败跳过并记录，不阻塞整体。"""
    legs, errors = [], []
    for symbol in symbols:
        for strategy in strategies:
            try:
                result = run_backtest(
                    db, symbol, interval, strategy, days=days, initial_cash=leg_cash,
                    fee=fee, slippage=slippage, optimize=True, end_ms=end_ms,
                )
            except Exception as exc:  # noqa: BLE001 - 单币种失败只记录不抛出
                errors.append({"symbol": symbol, "strategy": strategy, "error": str(exc)[:300]})
                continue
            legs.append({
                "symbol": symbol, "strategy": strategy, "initial": leg_cash,
                "curve": result.get("equity_curve") or [], "result": result,
            })
    return legs, errors


def _symbol_rows(legs: list[dict], interval: str) -> list[dict]:
    """按币种汇总其策略腿：收益/回撤/成交/期望值，按收益降序。"""
    grouped: dict[str, list[dict]] = {}
    for leg in legs:
        grouped.setdefault(leg["symbol"], []).append(leg)
    rows = []
    for symbol, symbol_legs in grouped.items():
        series = align_portfolio_equity(symbol_legs)
        initial = float(sum(leg["initial"] for leg in symbol_legs))
        trades = [trade for leg in symbol_legs for trade in (leg["result"].get("trades") or [])]
        metrics = metrics_from_equity(series, trades, initial, interval)
        metrics.update(exposure_metrics(series, trades, interval))
        final = float(series.iloc[-1]) if not series.empty else initial
        rows.append({
            "symbol": symbol,
            "initial": round(initial, 4),
            "final_equity": round(final, 4),
            "pnl": round(final - initial, 4),
            "return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "sharpe": metrics.get("sharpe"),
            "total_trades": metrics.get("total_trades"),
            "expectancy": metrics.get("expectancy"),
            "win_rate": metrics.get("win_rate"),
            "total_fees": metrics.get("total_fees"),
            "exposure_pct": metrics.get("exposure_pct"),
            "strategies": {leg["strategy"]: _leg_summary(leg["result"]) for leg in symbol_legs},
        })
    rows.sort(key=lambda row: (row["return_pct"] is not None, row["return_pct"] or 0.0), reverse=True)
    return rows


def _diversification(rows: list[dict]) -> dict:
    """横截面分散度：收益标准差、正收益占比、按盈亏贡献的 Top3/Bottom3。"""
    returns = pd.Series([row["return_pct"] for row in rows], dtype=float)
    gross = float(sum(abs(row["pnl"]) for row in rows))
    for row in rows:
        row["contribution_pct"] = round(row["pnl"] / gross * 100, 2) if gross > 0 else None
        row["blacklist_candidate"] = bool(
            (row["expectancy"] or 0.0) < 0 and (row["total_trades"] or 0) >= BLACKLIST_MIN_TRADES
        )
    ranked = sorted(rows, key=lambda row: row["pnl"], reverse=True)

    def brief(row: dict) -> dict:
        return {"symbol": row["symbol"], "pnl": row["pnl"],
                "return_pct": row["return_pct"], "contribution_pct": row["contribution_pct"]}

    return {
        "symbol_count": len(rows),
        "return_std": round(float(returns.std()), 4) if len(returns) > 1 else 0.0,
        "positive_share_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else 0.0,
        "gross_pnl": round(gross, 4),
        "top3": [brief(row) for row in ranked[:3]],
        "bottom3": [brief(row) for row in reversed(ranked[-3:])],
        "mean_symbol_return_pct": round(float(returns.mean()), 4) if len(returns) else 0.0,
    }


def run_portfolio_backtest(db, symbols, interval, strategies, days, fee, slippage,
                           initial_cash, end_ms=None, weight_mode: str = "equal",
                           rebalance_bars: int = DEFAULT_REBALANCE_BARS,
                           regime_weighting: bool = False) -> dict:
    """多币种组合回测。返回组合级指标、对齐后组合权益曲线、单币种明细、分散度与 errors。

    weight_mode="equal"（默认）：与历史行为完全一致——各腿等权买入持有，无再平衡。
    weight_mode="metric"：每 rebalance_bars 根 K 线按指标重分配资金（指标经
    metrics_as_of 取数，反前视）；regime_weighting=True 时再叠加策略维度加权。
    """
    symbols = [str(item).strip().upper() for item in symbols if str(item).strip()]
    strategies = [str(item).strip() for item in strategies if str(item).strip()]
    if not symbols or not strategies:
        raise ValueError("币种与策略列表不能为空")
    if not float(initial_cash) > 0:
        raise ValueError("组合初始资金必须为正数")
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode 仅支持 {' / '.join(WEIGHT_MODES)}")
    leg_cash = float(initial_cash) / (len(symbols) * len(strategies))

    legs, errors = _run_legs(db, symbols, interval, strategies, days, fee, slippage, leg_cash, end_ms)
    if not legs:
        return {"ok": False, "errors": errors, "message": "全部币种回测失败，无组合结果"}

    portfolio_initial = float(sum(leg["initial"] for leg in legs))
    if weight_mode == "equal":
        portfolio_series = align_portfolio_equity(legs)
    else:
        aligned = _aligned_legs_frame(legs)
        portfolio_series = weighted_equity(
            aligned, portfolio_initial, rebalance_bars,
            _metric_weight_lookup(db, legs, strategies, regime_weighting),
        ).sort_index()
    all_trades = [trade for leg in legs for trade in (leg["result"].get("trades") or [])]
    portfolio_metrics = metrics_from_equity(portfolio_series, all_trades, portfolio_initial, interval)
    portfolio_metrics.update(exposure_metrics(portfolio_series, all_trades, interval))

    rows = _symbol_rows(legs, interval)
    return {
        "ok": True,
        "params": {
            "symbols": symbols, "strategies": strategies, "interval": interval,
            "days": days, "fee": fee, "slippage": slippage,
            "initial_cash": float(initial_cash), "leg_cash": round(leg_cash, 4),
            "portfolio_initial": round(portfolio_initial, 4), "end_ms": end_ms,
            "weight_mode": weight_mode, "rebalance_bars": int(rebalance_bars),
            "regime_weighting": bool(regime_weighting),
        },
        "portfolio": portfolio_metrics,
        "equity_curve": downsample(portfolio_series),
        "symbols": rows,
        "diversification": _diversification(rows),
        "errors": errors,
    }
