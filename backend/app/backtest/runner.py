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
from ..strategy.coin_router import get_route
from ..strategy.local_filters import list_filters
from ..strategy.rules import evaluate_signals, resolve_params
from .execution import exposure_metrics, randomize_directions, simulate
from .metrics import downsample, metrics_from_equity
from .significance import (
    bootstrap_expectancy_ci,
    information_coefficient,
    percentile_rank,
    random_baseline_distribution,
)

# Dependency information only; all runs use the same explicit execution semantics.
HAS_VBT = importlib.util.find_spec("vectorbt") is not None


def with_entry_filters(signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Keep strategy direction for exits and attach independent entry permissions."""
    if signals is None:  # test doubles and callers without signals
        return signals
    for column in ("allow_long_entry", "allow_short_entry"):
        if column in data:
            signals[column] = data[column].reindex(signals.index).fillna(False).astype(bool)
    return signals


def minimum_sample_trades(validation_bars: int) -> int:
    """小样本惩罚：验证集每 50 根K线至少 1 笔成交，下限 **30 笔**。

    下限从 10 提到 30：单笔收益的方差极大，10 笔成交选出来的「最优参数」几乎
    必然是运气（实测 60 天窗口只成交 5~10 笔时，收益读数完全不可信）。
    2880 根验证集 → 57 笔门槛；验证集越短门槛越低，短窗口退回 30 笔保底。
    """
    return max(30, int(validation_bars) // 50)


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
    """Small predeclared search; this function never receives the test partition.

    候选参数按 minimum_sample_trades(len(validation)) 设成交笔数门槛，未达标
    一律不参与选择（零成交变体同样安全，不报错）。
    """
    variants = [("current", {})]
    variants.extend(
        (f"score_{score:.2f}", {"min_score": max(params["min_score"], score)})
        for score in (0.60, 0.65, 0.70)
    )
    variants.append(
        ("cost_strict", {"min_atr_cost_ratio": max(0.75, params["min_atr_cost_ratio"])})
    )
    if "adx_filter" in params:
        variants.append(
            ("trend_strict", {"adx_filter": max(25.0, params["adx_filter"])})
        )
    if "z_threshold" in params:
        variants.append(
            ("extreme_strict", {"z_threshold": params["z_threshold"] + 0.5})
        )
    if "target_r" in params:
        # 强回调专属网格：4h 趋势门槛更严 / 确认窗口更短（只承认快速转强）
        variants.append(
            ("htf_strict", {"htf_adx_filter": min(60.0, params["htf_adx_filter"] + 5.0)})
        )
        variants.append(
            (
                "confirm_strict",
                {"pullback_confirm_bars": max(2, int(params["pullback_confirm_bars"]) - 2)},
            )
        )
    candidates, seen = [], set()
    min_trades = minimum_sample_trades(len(validation))
    best_params, best_score, chosen = params, -math.inf, "current"
    for name, changes in variants:
        candidate = {**params, **changes}
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        signals = with_entry_filters(evaluate_signals(validation, strategy, candidate, probability), validation)
        equity, trades = simulate(
            validation, signals, candidate, initial, fee, slippage, interval
        )
        metrics = metrics_from_equity(equity, trades, initial, interval)
        objective = metrics["total_return_pct"] + 0.5 * metrics["max_drawdown_pct"]
        eligible = len(trades) >= min_trades
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
        "minimum_trades": min_trades,
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
    triple_barrier: bool = False,
    param_overrides: dict | None = None,
    use_route: bool = False,
) -> dict:
    """回测单个币种。

    use_route=True 时，若路由表对该 symbol 有精确或族群匹配，则覆盖 strategy
    并把路由 params 合并进 param_overrides。显式 param_overrides 优先级仍高于路由表。
    默认 use_route=False — UI 手动选策略的回测不应被自动路由覆盖。
    """
    # --- 币级路由（可选） ---
    if use_route:
        route = get_route(symbol)
        if route is not None:
            strategy = route["strategy"]
            route_params = route.get("params", {})
            if route_params:
                merged = {**route_params}
                if param_overrides:
                    merged.update(param_overrides)
                param_overrides = merged
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
    if param_overrides:
        if not isinstance(param_overrides, dict):
            raise ValueError("参数覆盖必须是 JSON 对象")
        overrides.update(param_overrides)
    if threshold is not None:
        overrides["min_score"] = threshold
    params = resolve_params(
        strategy, {**state, "fee_rate": fee, "slippage": slippage}, overrides,
        interval=interval,
    )
    horizon = int(state.get("train_horizon", 5))
    barrier = None
    if triple_barrier and strategy == "ml_ensemble" and algo != "rule":
        # triple-barrier 标签与实盘 sl/tp/max_hold 出场对齐
        barrier = {
            "sl_atr_mult": params["sl_atr_mult"],
            "tp_atr_mult": params["tp_atr_mult"],
            "max_hold_bars": params["max_hold_bars"],
            "horizon": horizon,
        }
    filters = [item for item in list_filters(db) if item["enabled"] and strategy in item["bound_strategies"]]
    dataset_options = {"entry_filters": filters} if filters else {}
    data = build_dataset(
        db, symbol, interval, horizon=horizon, days=days, end_ms=end_ms, barrier=barrier,
        **dataset_options,
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
        proba = estimator.predict_proba(frame[FEATURES])
        classes = getattr(estimator, "classes_", None)
        if classes is not None and len(classes) > 2:
            # 三分类：class==1 是「止盈」，用作做多概率（趋势策略只在止盈类做多）
            k = int(np.where(classes == 1)[0][0])
            return pd.Series(proba[:, k], index=frame.index)
        return pd.Series(proba[:, 1], index=frame.index)  # 类1 = 涨（二分类/未知）

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
    signals = with_entry_filters(evaluate_signals(test, strategy, params, probability), test)
    equity, trades = simulate(test, signals, params, initial, fee, slippage, interval)
    # Identical fills, costs and risk caps; this is not the old vectorbt engine.
    baseline_params = {**configured, "regime_filter": False, "cost_filter": False}
    baseline_signals = with_entry_filters(evaluate_signals(test, strategy, baseline_params, probability), test)
    baseline_equity, baseline_trades = simulate(
        test, baseline_signals, baseline_params, initial, fee, slippage, interval
    )
    baseline = metrics_from_equity(baseline_equity, baseline_trades, initial, interval)
    baseline.update(exposure_metrics(baseline_equity, baseline_trades, interval))
    # 随机入场证伪基线：方向 50/50 随机翻转（固定种子），其余与时点/仓位/风控/成本完全一致。
    # 若优化策略收益不显著高于本组，说明方向判断没有信息量，应淘汰。
    random_signals = randomize_directions(baseline_signals)
    random_equity, random_trades = simulate(test, random_signals, baseline_params, initial, fee, slippage, interval)
    random_baseline = metrics_from_equity(random_equity, random_trades, initial, interval)
    random_baseline.update(exposure_metrics(random_equity, random_trades, interval))
    metrics = metrics_from_equity(equity, trades, initial, interval)
    metrics.update(exposure_metrics(equity, trades, interval))
    quality = data.attrs.get("data_quality", {})
    warnings = []

    if filters:
        warnings.append("本地策略作为入场过滤器；策略、基线和随机方向均按各自方向检查过滤条件，平仓仍由系统策略管理。")

    # 统计显著性：收益读数必须与置信区间、IC、随机基线分布一起看，
    # 否则「+2%」和「0」在小样本下无法区分（见 significance 模块说明）。
    expectancy_ci = bootstrap_expectancy_ci(trades)
    ic = (
        information_coefficient(probability.reindex(test.index), test["future_return"])
        if probability is not None
        else {"pairs": 0, "ic": None, "verdict": "no_model"}
    )
    random_dist = random_baseline_distribution(
        test, baseline_signals, baseline_params, initial, fee, slippage, interval,
        n_seeds=10,
    )
    if expectancy_ci.get("verdict") == "inconclusive":
        warnings.append(
            f"每笔期望的 95% 置信区间 [{expectancy_ci['ci_low_pct']}, "
            f"{expectancy_ci['ci_high_pct']}] 跨 0，无法判定是否存在正 edge——不要据此下结论。"
        )
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
    if random_baseline and metrics.get("total_return_pct", 0) <= random_baseline.get("total_return_pct", 0):
        warnings.append("策略收益未超过随机入场基线（方向 50/50 随机翻转），方向判断可能没有信息量。")
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
            "local_entry_filters": [{"code": item["code"], "params": item["params"]} for item in filters],
            "local_filter_blocked_signals": int(
                ((signals.direction.eq(1) & ~signals.allow_long_entry)
                 | (signals.direction.eq(-1) & ~signals.allow_short_entry)).sum()
            ) if filters else 0,
            "baseline": baseline,
            "random_baseline": random_baseline,
            "significance": {
                "expectancy_ci": expectancy_ci,
                "information_coefficient": ic,
                "random_distribution": random_dist,
            },
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
        "equity_curve": downsample(equity),
        "trades": trades[-500:],
    }
