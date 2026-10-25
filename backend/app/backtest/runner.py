from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import FEATURES
from ..ml.dataset import build_dataset, purged_split
from ..ml.trainer import _build_estimator
from ..models import StrategyConfig
from ..state import all_state
from ..strategy.rules import evaluate_signals, resolve_params
from .execution import simulate

# Dependency information only; all runs use the same explicit execution semantics.
HAS_VBT = importlib.util.find_spec("vectorbt") is not None


def _metrics_from_equity(
    equity: pd.Series, trades: list[dict], initial: float, interval: str
) -> dict:
    equity = equity.dropna()
    if equity.empty:
        return {}
    drawdown = (equity / equity.cummax().clip(lower=initial) - 1) * 100
    daily = equity.resample("1D").last()
    daily = (
        pd.concat(
            [pd.Series([initial], index=[daily.index[0] - pd.Timedelta(days=1)]), daily]
        )
        .pct_change()
        .dropna()
    )
    sharpe = (
        float(daily.mean() / daily.std() * math.sqrt(365))
        if len(daily) > 2 and daily.std() > 0
        else 0.0
    )
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    return {
        "initial_cash": round(initial, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round((float(equity.iloc[-1]) / initial - 1) * 100, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "sharpe": round(sharpe, 2) if math.isfinite(sharpe) else 0.0,
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "profit_factor_infinite": bool(gross_profit > 0 and not gross_loss),
        "expectancy": round(sum(t["pnl"] for t in trades) / len(trades), 4)
        if trades
        else 0.0,
        "total_fees": round(sum(t["fees"] for t in trades), 2),
        "bars": len(equity),
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
    }


def _downsample(equity: pd.Series, limit: int = 1500) -> list[dict]:
    indices = np.unique(
        np.linspace(0, len(equity) - 1, min(limit, len(equity)), dtype=int)
    )
    return [
        {"time": str(equity.index[i]), "equity": round(float(equity.iloc[i]), 2)}
        for i in indices
    ]


def select_parameters(
    validation: pd.DataFrame,
    strategy: str,
    params: dict,
    probability: pd.Series | None,
    initial: float,
    fee: float,
    slippage: float,
    interval: str,
) -> tuple[dict, dict]:
    """Small predeclared search; this function never receives the test partition."""
    variants = [("current", {})]
    variants.extend(
        (f"score_{score:.2f}", {"min_score": max(params["min_score"], score)})
        for score in (0.60, 0.65, 0.70)
    )
    variants.append(
        ("cost_strict", {"min_atr_cost_ratio": max(0.75, params["min_atr_cost_ratio"])})
    )
    if strategy != "mean_reversion":
        variants.append(
            ("trend_strict", {"adx_filter": max(25.0, params["adx_filter"])})
        )
    candidates, seen = [], set()
    best_params, best_score, chosen = params, -math.inf, "current"
    for name, changes in variants:
        candidate = {**params, **changes}
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        signals = evaluate_signals(validation, strategy, candidate, probability)
        equity, trades = simulate(
            validation, signals, candidate, initial, fee, slippage, interval
        )
        metrics = _metrics_from_equity(equity, trades, initial, interval)
        objective = metrics["total_return_pct"] + 0.5 * metrics["max_drawdown_pct"]
        eligible = len(trades) >= 10
        candidates.append(
            {
                "name": name,
                "changes": changes,
                "objective": round(objective, 4),
                "eligible": eligible,
                **metrics,
            }
        )
        if eligible and objective > best_score:
            best_score, best_params, chosen = objective, candidate, name
    return best_params, {
        "enabled": True,
        "selected": chosen,
        "minimum_trades": 10,
        "status": "selected" if math.isfinite(best_score) else "insufficient_trades",
        "objective": "net_return_pct - 0.5 * abs(max_drawdown_pct)",
        "start": str(validation.index[0]),
        "end": str(validation.index[-1]),
        "candidates": candidates,
        "applied_to_paper": False,
    }


def run_backtest(
    db: Session,
    symbol: str,
    interval: str,
    strategy: str,
    algo: str = "lightgbm",
    days: int = 30,
    initial_cash: float | None = None,
    fee: float | None = None,
    slippage: float | None = None,
    train_ratio: float = 0.75,
    threshold: float | None = None,
    optimize: bool = False,
    end_ms: int | None = None,
) -> dict:
    state = all_state(db)
    initial = float(
        initial_cash
        if initial_cash is not None
        else state.get("initial_capital", 10000.0)
    )
    fee = float(fee if fee is not None else state["fee_rate"])
    slippage = float(slippage if slippage is not None else state["slippage"])
    if not math.isfinite(initial) or initial <= 0 or not 0.5 <= train_ratio <= 0.85:
        raise ValueError("初始资金或训练占比无效")
    if algo not in ("rule", "lightgbm", "xgboost"):
        raise ValueError("未知模型算法")
    config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
    overrides = json.loads(config.params or "{}") if config else {}
    if threshold is not None:
        overrides["min_score"] = threshold
    params = resolve_params(
        strategy, {**state, "fee_rate": fee, "slippage": slippage}, overrides
    )
    horizon = int(state.get("train_horizon", 5))
    data = build_dataset(
        db, symbol, interval, horizon=horizon, days=days, end_ms=end_ms
    )
    if len(data) < 800:
        raise ValueError(f"回测样本不足（{len(data)} 行），至少需要 800 根有效K线")
    split = int(len(data) * train_ratio)
    train, test = purged_split(data, split, horizon)
    if len(test) < 200:
        raise ValueError("测试集不足 200 根K线")
    validation = None
    if optimize:
        validation_start = int(len(data) * (train_ratio - 0.15))
        train, _ = purged_split(data, validation_start, horizon)
        validation = data.iloc[validation_start : split - horizon]
        if len(validation) < 100:
            raise ValueError("验证集不足 100 根K线")
    estimator, algo_used = None, "rule"
    if strategy == "ml_ensemble" and algo != "rule":
        if train["label"].nunique() < 2:
            raise ValueError("训练集只有单一方向标签，无法训练分类器")
        estimator = _build_estimator(algo)
        estimator.fit(train[FEATURES], train["label"])
        algo_used = algo

    def predict(frame):
        if estimator is None:
            return None
        return pd.Series(
            estimator.predict_proba(frame[FEATURES])[:, 1], index=frame.index
        )

    selection = {"enabled": False, "status": "disabled", "applied_to_paper": False}
    configured = params.copy()
    if optimize:
        params, selection = select_parameters(
            validation,
            strategy,
            params,
            predict(validation),
            initial,
            fee,
            slippage,
            interval,
        )
    probability = predict(test)
    signals = evaluate_signals(test, strategy, params, probability)
    equity, trades = simulate(test, signals, params, initial, fee, slippage, interval)
    # Identical fills, costs and risk caps; this is not the old vectorbt engine.
    baseline_params = {**configured, "regime_filter": False, "cost_filter": False}
    baseline_signals = evaluate_signals(test, strategy, baseline_params, probability)
    baseline_equity, baseline_trades = simulate(
        test, baseline_signals, baseline_params, initial, fee, slippage, interval
    )
    baseline = _metrics_from_equity(baseline_equity, baseline_trades, initial, interval)
    metrics = _metrics_from_equity(equity, trades, initial, interval)
    quality = data.attrs.get("data_quality", {})
    warnings = []
    if quality.get("gap_count"):
        warnings.append(
            f"历史数据存在 {quality['gap_count']} 处缺口；指标重新预热，跨缺口持仓在下一可见开盘平仓，结果受缺失行情影响。"
        )
    if quality.get("stale_hours", 0) > 1:
        warnings.append(
            f"行情已滞后 {quality['stale_hours']} 小时；本次仅评估已存历史。"
        )
    if len(trades) < 30:
        warnings.append("样本外成交不足 30 笔，收益和胜率的稳定性有限。")
    if optimize and selection["status"] == "insufficient_trades":
        warnings.append("验证集成交不足，保留当前参数。")
    metrics.update(
        {
            "engine": "sequential",
            "execution": "closed_signal_next_open",
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy,
            "algo": algo_used,
            "fee_rate": fee,
            "slippage": slippage,
            "threshold": params["min_score"],
            "buy_hold_return_pct": round(
                (float(test["close"].iloc[-1]) / float(test["open"].iloc[0]) - 1) * 100,
                4,
            ),
            "signal_bars": int(signals.direction.ne(0).sum()),
            "long_bars": int(signals.direction.eq(1).sum()),
            "short_bars": int(signals.direction.eq(-1).sum()),
            "blocked_signals": {
                str(k): int(v)
                for k, v in signals.block_reason[signals.block_reason.ne("")]
                .value_counts()
                .items()
            },
            "parameters": params,
            "baseline": baseline,
            "return_improvement_pct": round(
                metrics["total_return_pct"] - baseline["total_return_pct"], 4
            ),
            "validation": selection,
            "data_quality": quality,
            "warnings": warnings,
            "split": {
                "train_start": str(train.index[0]),
                "train_end": str(train.index[-1]),
                "train_rows": len(train),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "test_rows": len(test),
                "purge_bars": horizon,
            },
        }
    )
    return {
        "metrics": metrics,
        "equity_curve": _downsample(equity),
        "trades": trades[-500:],
    }
