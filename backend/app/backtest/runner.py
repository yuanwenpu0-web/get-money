"""Sequential, single-strategy backtest runner.

整个仓库只剩一套规则（吸收反转），因此这里不再有模型训练/预测、多策略候选回放
或币级路由：同一份数据被切成训练/测试两段，规则只在测试段上按实盘口径回放。
"""
from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..ml.dataset import build_dataset, purged_split
from ..models import StrategyConfig
from ..market.data_service import MarketDataService
from ..strategy.bindings import binding_overrides
from ..state import all_state
from ..strategy.rules import evaluate_signals, resolve_params
from ..utils import interval_ms, now_ms
from .execution import exposure_metrics, randomize_directions, simulate
from .metrics import downsample, metrics_from_equity
from .significance import (
    bootstrap_expectancy_ci,
    random_baseline_distribution,
)

# Dependency information only; all runs use the same explicit execution semantics.
HAS_VBT = importlib.util.find_spec("vectorbt") is not None

# 只能由专用撮合表达的策略：它们的 `signals()` 输出方向恒为 0，
# 通用撮合会把它们的出场规则换成另一套，所以这里统一早路由。
OPENING_BREAKOUT = "opening_candle_breakout"
RGD = "rgd_donchian_v2"


def run_zigzag_backtest(db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool):
    """Replay the requested raw candle window with the Pine stop-order broker."""
    from .zigzag_execution import simulate_zigzag

    as_of = min(end_ms, now_ms()) if end_ms is not None else now_ms()
    step = interval_ms(interval)
    frame = MarketDataService(db).load_frame(
        symbol, interval, start_ms=as_of - days * 86_400_000, end_ms=as_of
    )
    if frame.empty:
        raise ValueError(f"{symbol} {interval} 本地无K线数据，请先拉取历史数据")
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= as_of].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 3:
        raise ValueError("已收盘K线不足 3 根")
    prices = frame[["open", "high", "low", "close"]]
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid &= frame.high.ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame.low.le(frame[["open", "close", "high"]].min(axis=1))
    if not valid.all():
        raise ValueError("行情包含无效 K 线，无法按原策略回放")
    diffs = frame.index.to_series().diff().dt.total_seconds().mul(1000)
    gap_count = int(diffs.gt(step).sum())
    signals = evaluate_signals(frame, "simple_zigzag_double", params)
    equity, trades = simulate_zigzag(frame, signals, params, initial, fee, slippage, interval)
    metrics = metrics_from_equity(equity, trades, initial, interval)
    metrics.update(exposure_metrics(equity, trades, interval))
    warnings = ["Pine 的指标状态从本地数据窗口首根开始计算；对照 TradingView 时须统一行情、历史起点和属性。"]
    if gap_count:
        warnings.append(f"行情存在 {gap_count} 处缺口；缺失时段的价格路径不可知。")
    if slippage:
        warnings.append("本地滑点按成交价百分比计算；TradingView 的 slippage 属性按最小跳动点数计算。")
    metrics.update({
        "engine": "pine_stop_orders", "execution": "pine_stop_orders",
        "symbol": symbol, "interval": interval, "strategy": "simple_zigzag_double",
        "pool": pool or "", "algo": "rule", "fee_rate": fee, "slippage": slippage,
        "threshold": None, "buy_hold_return_pct": round((float(frame.close.iloc[-1]) / float(frame.open.iloc[0]) - 1) * 100, 4),
        "signal_bars": int((signals.zz_long_entry | signals.zz_short_entry).sum()),
        "parameters": params, "baseline": None, "random_baseline": None,
        "validation": {"enabled": False}, "warnings": warnings,
        "data_quality": {"raw_bars": len(frame), "gap_count": gap_count},
        "split": {"test_start": str(frame.index[0]), "test_end": str(frame.index[-1]), "test_rows": len(frame), "purge_bars": 0},
    })
    return {"metrics": metrics, "equity_curve": downsample(equity), "trades": trades[-500:]}


def run_ocb_backtest(db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool):
    """回放开盘首根突破：会话买卖 + 按风险距离定手数，走专用撮合而不是通用信号撮合。"""
    from .ocb_execution import simulate_ocb

    as_of = min(end_ms, now_ms()) if end_ms is not None else now_ms()
    step = interval_ms(interval)
    frame = MarketDataService(db).load_frame(
        symbol, interval, start_ms=as_of - days * 86_400_000, end_ms=as_of
    )
    if frame.empty:
        raise ValueError(f"{symbol} {interval} 本地无K线数据，请先拉取历史数据")
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= as_of].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 3:
        raise ValueError("已收盘K线不足 3 根")
    prices = frame[["open", "high", "low", "close"]]
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid &= frame.high.ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame.low.le(frame[["open", "close", "high"]].min(axis=1))
    if not valid.all():
        raise ValueError("行情包含无效 K 线，无法按原策略回放")
    diffs = frame.index.to_series().diff().dt.total_seconds().mul(1000)
    gap_count = int(diffs.gt(step).sum())
    signals = evaluate_signals(frame, OPENING_BREAKOUT, params)
    equity, trades = simulate_ocb(frame, signals, params, initial, fee, slippage, interval)
    metrics = metrics_from_equity(equity, trades, initial, interval)
    metrics.update(exposure_metrics(equity, trades, interval))
    first_bars = int(signals["ocb_is_first"].sum())
    setups = int(signals["ocb_dir"].ne(0).sum())
    warnings = [
        "会话强平/每笔风险定手数只能由专用撮合表达；本回放是完整的窗口回放，没有训练/测试切分。",
        "Pine 的指标状态从本地数据窗口首根开始计算；对照 TradingView 时须统一行情、历史起点和属性。",
    ]
    if first_bars == 0:
        warnings.append(
            f"本周期没有任何K线的开盘时间落在开盘首窗（{params['first_session_start_min']}-"
            f"{params['first_session_end_min']} 分钟）内，零成交是这个原因而不是没有信号逻辑。"
        )
    if gap_count:
        warnings.append(f"行情存在 {gap_count} 处缺口；跨缺口的会话判定与 NaN 级 ATR 会让那些交易日直接跳过。")
    if slippage:
        warnings.append("本地滑点按成交价百分比计算；TradingView 的 slippage 属性按最小跳动点数计算。")
    if trades:
        open_end = equity.attrs.get("open_position")
        if open_end is not None:
            warnings.append("回放结束时有未平仓位（数据窗口在会话内截断），末根按盯市价计入净值。")
    metrics.update({
        "engine": "pine_session",
        "execution": "pine_session_orders",
        "symbol": symbol,
        "interval": interval,
        "strategy": OPENING_BREAKOUT,
        "pool": pool or "",
        "algo": "rule",
        "fee_rate": fee,
        "slippage": slippage,
        "threshold": None,
        "buy_hold_return_pct": round((float(frame.close.iloc[-1]) / float(frame.open.iloc[0]) - 1) * 100, 4),
        "signal_bars": setups,
        "opening_bars": first_bars,
        "blocked_signals": {
            str(k): int(v)
            for k, v in signals.block_reason[signals.block_reason.ne("")]
            .value_counts()
            .items()
        },
        "parameters": params,
        "baseline": None,
        "random_baseline": None,
        "validation": {"enabled": False},
        "warnings": warnings,
        "data_quality": {"raw_bars": len(frame), "gap_count": gap_count},
        "open_position": equity.attrs.get("open_position"),
        "split": {
            "test_start": str(frame.index[0]),
            "test_end": str(frame.index[-1]),
            "test_rows": len(frame),
            "purge_bars": 0,
        },
    })
    return {"metrics": metrics, "equity_curve": downsample(equity), "trades": trades[-500:]}


def run_rgd_backtest(db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool):
    """整窗回放 RGD：主周期帧 + 因果拼入的 HTF 电平，走专用 broker。

    与 `run_zigzag_backtest` 同一范式（不做 train/test split）：脚本 32 个参数是
    固定默认值检验，`spec.grid` 为空，没有可寻优的自由度。
    HTF attach 必须在 `evaluate_signals` 之前、对整帧做一次——`prepare` 幂等短路
    会让后置 attach 静默失效（契约 §7.3.1）。
    """
    from ..features.rgd import htf_interval_for
    from ..features.rgd_htf import attach_htf_levels
    from .rgd_execution import simulate_rgd

    as_of = min(end_ms, now_ms()) if end_ms is not None else now_ms()
    step = interval_ms(interval)
    htf_interval = htf_interval_for(interval, params["auto_htf"], int(params["htf_in"]))
    htf_step = interval_ms(htf_interval)
    start_ms = as_of - days * 86_400_000
    service = MarketDataService(db)
    frame = service.load_frame(symbol, interval, start_ms=start_ms, end_ms=as_of)
    if frame.empty:
        raise ValueError(f"{symbol} {interval} 本地无K线数据，请先拉取历史数据")
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + step) <= as_of].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 3:
        raise ValueError("已收盘K线不足 3 根")
    htf = service.load_frame(
        symbol, htf_interval, start_ms=start_ms - 2 * htf_step, end_ms=as_of
    )
    frame = attach_htf_levels(frame, htf, htf_interval, as_of)
    if not bool(frame["rgd_htf_ok"].any()):
        raise ValueError(
            f"{symbol} 无已完成的 {htf_interval} HTF bar，拒绝静默零成交（契约 §7.3.1）"
        )
    prices = frame[["open", "high", "low", "close"]]
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid &= frame.high.ge(frame[["open", "close", "low"]].max(axis=1))
    valid &= frame.low.le(frame[["open", "close", "high"]].min(axis=1))
    if not valid.all():
        raise ValueError("行情包含无效 K 线，无法按原策略回放")
    diffs = frame.index.to_series().diff().dt.total_seconds().mul(1000)
    gap_count = int(diffs.gt(step).sum())
    signals = evaluate_signals(frame, RGD, params)
    equity, trades = simulate_rgd(frame, signals, params, initial, fee, slippage, interval)
    metrics = metrics_from_equity(equity, trades, initial, interval)
    metrics.update(exposure_metrics(equity, trades, interval))
    warnings = [
        "TP1 限价部分平 + 保本滞后一根 + 接受度加权 trailing 只能由专用撮合表达；本回放是完整窗口回放，没有训练/测试切分。",
        "Pine 的指标状态从本地数据窗口首根开始计算；对照 TradingView 时须统一行情、历史起点和属性。",
    ]
    if gap_count:
        warnings.append(f"行情存在 {gap_count} 处缺口；缺失时段的价格路径不可知。")
    if slippage:
        warnings.append("本地滑点按成交价百分比计算；TradingView 的 slippage 属性按最小跳动点数计算。")
    open_end = equity.attrs.get("open_position")
    if open_end is not None:
        warnings.append("回放结束仍有未平仓位；末根按盯市价计入净值，不强制平仓（Pine 同口径）。")
    metrics.update({
        "engine": "pine_rgd_broker", "execution": "pine_rgd_orders",
        "symbol": symbol, "interval": interval, "strategy": RGD, "pool": pool or "",
        "algo": "rule", "fee_rate": fee, "slippage": slippage, "threshold": None,
        "buy_hold_return_pct": round(
            (float(frame.close.iloc[-1]) / float(frame.open.iloc[0]) - 1) * 100, 4
        ),
        "signal_bars": int((signals.rgd_long_break | signals.rgd_short_break).sum()),
        "pullback_bars": int(
            (signals.rgd_long_pull_core | signals.rgd_short_pull_core).sum()
        ),
        "htf_interval": htf_interval,
        "parameters": params, "baseline": None, "random_baseline": None,
        "validation": {"enabled": False}, "warnings": warnings,
        "data_quality": {"raw_bars": len(frame), "gap_count": gap_count},
        "open_position": open_end,
        "split": {
            "test_start": str(frame.index[0]), "test_end": str(frame.index[-1]),
            "test_rows": len(frame), "purge_bars": 0,
        },
    })
    return {"metrics": metrics, "equity_curve": downsample(equity), "trades": trades[-500:]}


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
    candidates, seen = [], set()
    min_trades = minimum_sample_trades(len(validation))
    best_params, best_score, chosen = params, -math.inf, "current"
    for name, changes in variants:
        candidate = {**params, **changes}
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        signals = evaluate_signals(validation, strategy, candidate)
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
    days: int = 30,
    initial_cash: float | None = None,
    fee: float | None = None,
    slippage: float | None = None,
    train_ratio: float = 0.75,
    threshold: float | None = None,
    optimize: bool = False,
    end_ms: int | None = None,
    param_overrides: dict | None = None,
    pool: str | None = None,
) -> dict:
    """回测单个币种：同一撮合口径下的「当前参数 / 关闭过滤基线 / 随机方向基线」三者对照。"""
    state = all_state(db)
    is_zigzag = strategy == "simple_zigzag_double"
    initial = float(
        initial_cash
        if initial_cash is not None
        else 1_000_000.0 if is_zigzag else state.get("initial_capital", 10000.0)
    )
    fee = float(fee if fee is not None else 0.001 if is_zigzag else state["fee_rate"])
    slippage = float(slippage if slippage is not None else 0.0 if is_zigzag else state["slippage"])
    if not math.isfinite(initial) or initial <= 0 or not 0.5 <= train_ratio <= 0.85:
        raise ValueError("初始资金或训练占比无效")
    # 指定池时用「策略 × 池」绑定的参数覆盖层，再叠该币种的覆盖，与实盘走同一套参数
    bound = binding_overrides(db, strategy, pool or "", interval, symbol=symbol)
    if bound is not None:
        overrides = dict(bound)
    else:
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
    if is_zigzag:
        if optimize:
            raise ValueError("Simple ZigZag Double 不支持通用验证集选参，请关闭选参后回测")
        return run_zigzag_backtest(db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool)
    if strategy == OPENING_BREAKOUT:
        if optimize:
            raise ValueError("开盘首根突破不支持通用验证集选参，请关闭选参后回测")
        return run_ocb_backtest(db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool)
    if strategy == RGD:
        if optimize:
            raise ValueError("RGD 只做固定参数检验（spec.grid 为空），请关闭选参后回测")
        return run_rgd_backtest(
            db, symbol, interval, params, initial, fee, slippage, days, end_ms, pool
        )
    horizon = int(state.get("train_horizon", 5))
    data = build_dataset(db, symbol, interval, horizon=horizon, days=days, end_ms=end_ms)
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
    selection = {"enabled": False, "status": "disabled", "applied_to_paper": False}
    configured = params.copy()
    if optimize:
        params, selection = select_parameters(
            validation,
            strategy,
            params,
            initial,
            fee,
            slippage,
            interval,
        )
    signals = evaluate_signals(test, strategy, params)
    equity, trades = simulate(test, signals, params, initial, fee, slippage, interval)
    # Identical fills, costs and risk caps; this is not the old vectorbt engine.
    baseline_params = {**configured, "regime_filter": False, "cost_filter": False}
    baseline_signals = evaluate_signals(test, strategy, baseline_params)
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

    # 统计显著性：收益读数必须与置信区间、随机基线分布一起看，
    # 否则「+2%」和「0」在小样本下无法区分（见 significance 模块说明）。
    expectancy_ci = bootstrap_expectancy_ci(trades)
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
            "pool": pool or "",
            "algo": "rule",
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
            "random_baseline": random_baseline,
            "significance": {
                "expectancy_ci": expectancy_ci,
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
