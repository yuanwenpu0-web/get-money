r"""分档模型 + 分档出入场参数的 walk-forward 回测。

对照组设计（决定是否真的有增益）
--------------------------------
同一批币种、同一时间段，跑两条完全对称的路径：
  A 分层组 tiered   ：每档一个 pooled 模型 + 每档独立寻优的出入场参数
  B 统一组 uniform  ：全部币种混合训一个模型 + 全局统一参数（同款寻优）
只有 A 显著优于 B，才能说明「按规模分档」带来的是真实增益，而非币种/时段的运气。

反前视三道防线
--------------
1. 分档依据取自回测窗口**之前** 30 天的成交额/波动（tier_asof = end_ms - days），
   不用回测窗口内的成交额给币种分档。
2. 模型训练只用各币种的 train 段，寻优只用 val 段，test 段全程不可见。
3. walk-forward 逐窗滚动，每窗独立重新分档/训练/寻优，无跨窗参数复用。

用法：
.\.venv\Scripts\python.exe backend/scripts/tier_backtest.py --windows 5 --per-tier 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import simulate  # noqa: E402
from app.backtest.metrics import downsample, metrics_from_equity  # noqa: E402
from app.backtest.portfolio import align_portfolio_equity  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.features.indicators import FEATURES  # noqa: E402
from app.ml.tier_model import (  # noqa: E402
    DEFAULT_N_ESTIMATORS,
    build_tier_panels,
    predict_tier_frame,
    split_symbol,
    train_tier_model,
)
from app.ml.tier_params import OPTIMIZE_ROUNDS, round_combos, tier_params  # noqa: E402
from app.ml.tiers import TIER_KEYS, assign_tiers, tier_summary  # noqa: E402
from app.ml.dataset import build_dataset  # noqa: E402
from app.strategy.rules import evaluate_signals, resolve_params  # noqa: E402

DAY_MS = 86_400_000
FEE = 0.0004
SLIPPAGE = 0.0002
INTERVAL = "1h"
INITIAL_CASH = 10000.0
STRATEGIES = ["ml_ensemble", "trend_follow"]
TIER_LOOKBACK_DAYS = 30   # 分档统计回看天数（在回测窗口之前）


# ---------------------------------------------------------------- 数据缓存
class DatasetCache:
    """缓存 build_dataset 结果：同一 (symbol, days, end_ms) 只算一次。

    分层组与对照组共用同一份 features —— 这是公平对比的前提，也省掉一半以上耗时。
    """

    def __init__(self) -> None:
        self._dataset: dict[tuple, pd.DataFrame | None] = {}
        self._misses = 0

    def get(self, db: Session, symbol: str, interval: str, horizon: int, days: int, end_ms: int):
        key = (symbol, interval, horizon, days, end_ms)
        if key in self._dataset:
            return self._dataset[key]
        try:
            data = build_dataset(
                db, symbol, interval, horizon=horizon, days=days, end_ms=end_ms
            )
        except Exception as exc:
            data = None
            self._misses += 1
            self._dataset[key] = None
            self._errors: list = getattr(self, "_errors", [])
            self._errors.append({"symbol": symbol, "reason": str(exc)[:160]})
        self._dataset[key] = data
        return data

    @property
    def errors(self) -> list:
        return getattr(self, "_errors", [])


# ---------------------------------------------------------------- 基础收益矿工
def _base_params(strategy: str, fee: float, slippage: float) -> dict:
    state = {**DEFAULT_STATE, "fee_rate": fee, "slippage": slippage}
    return resolve_params(strategy, state)


def minimum_trades_gate(frame_map: dict) -> int:
    """验证段成交笔数门槛：沿用 runner.minimum_sample_trades 的每 50 根 1 笔口径，
    按参与币种数放大（档内多币聚合后总笔数更多）。"""
    total_bars = int(sum(len(frame) for frame in frame_map.values()))
    return max(10, total_bars // 50)


def evaluate_split(
    frame_map: dict,
    strategy: str,
    params: dict,
    payload: dict | None,
    cash_per_symbol: dict[str, float],
    fee: float,
    slippage: float,
    interval: str,
) -> tuple[dict, list[dict], list[dict]] | None:
    """逐币种 simulate 后按 Portfolio 口径对齐聚合为组合曲线。

    必须用 per-symbol 跑：execution.simulate 是单币种顺序持仓模型，
    多币种数据混在同一时间索引会导致持仓互相覆盖。
    返回 (组合指标, 各腿, 全部成交明细) —— 成交明细必须带回，
    否则组合级的成交笔数/胜率/期望值会因为空列表而失真为 0。
    """
    legs: list[dict] = []
    trades_all: list[dict] = []
    for symbol, frame in frame_map.items():
        initial = float(cash_per_symbol.get(symbol, 0.0))
        if frame is None or frame.empty or initial <= 0:
            continue
        probability = None
        if payload is not None and strategy == "ml_ensemble":
            try:
                probability = predict_tier_frame(payload, frame)
            except Exception:
                probability = None
        signals = evaluate_signals(frame, strategy, params, probability)
        equity, trades = simulate(
            frame, signals, params, initial, fee, slippage, interval
        )
        if equity is None or len(equity) == 0:
            continue
        legs.append(
            {"symbol": symbol, "initial": initial, "curve": downsample(equity)}
        )
        trades_all.extend(trades)
    if not legs:
        return None
    series = align_portfolio_equity(legs)
    if series.empty:
        return None
    initial_total = float(sum(leg["initial"] for leg in legs))
    metrics = metrics_from_equity(series, trades_all, initial_total, interval)
    return metrics, legs, trades_all


def select_tier_params(
    val_map: dict,
    strategy: str,
    base_params: dict,
    payload: dict | None,
    cash_per_symbol: dict[str, float],
    fee: float,
    slippage: float,
    interval: str,
) -> tuple[dict, dict]:
    """坐标下降寻优：先 (min_score, sl_atr_mult)，再 (tp_atr_mult, max_hold_bars)。

    与 runner.select_parameters 一致：只在验证段选参，从未接触 test 段；
    未达成交笔数门槛的变体一律不参与选择，防止小样本幸运儿胜出。
    """
    min_trades = minimum_trades_gate(val_map)

    def score(params: dict) -> tuple[dict | None, int]:
        result = evaluate_split(
            val_map, strategy, params, payload, cash_per_symbol, fee, slippage, interval
        )
        if result is None:
            return None, 0
        metrics, _, _ = result
        return metrics, int(metrics.get("total_trades") or 0)

    current = dict(base_params)
    history: list[dict] = []
    for round_keys in OPTIMIZE_ROUNDS:
        # 每轮都用更新后的 current 重新生成候选（见 tier_params.round_combos 注释）
        combos = round_combos(round_keys, current)
        best_candidate, best_objective, best_trades = None, float("-inf"), 0
        for changes in combos:
            candidate = {**current, **dict(changes)}
            try:
                candidate = _sanitize(candidate)
            except Exception:
                continue
            metrics, trades = score(candidate)
            if metrics is None:
                continue
            objective = metrics["total_return_pct"] + 0.5 * metrics["max_drawdown_pct"]
            history.append({
                "changes": dict(changes),
                "objective": round(objective, 4),
                "return_pct": metrics["total_return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "trades": trades,
                "eligible": trades >= min_trades,
            })
            if trades >= min_trades and objective > best_objective:
                best_candidate, best_objective, best_trades = candidate, objective, trades
        if best_candidate is not None:
            current = best_candidate
    return current, {
        "selected_params": {k: current[k] for k in ("min_score", "sl_atr_mult", "tp_atr_mult", "max_hold_bars")},
        "minimum_trades": min_trades,
        "evaluated": len(history),
        "history_sample": history[:6],
    }


def _sanitize(params: dict) -> dict:
    """保证 heavy_score > min_score 且 base<=max_position，否则 resolve 会抛错/区间退化。"""
    params = dict(params)
    min_score = float(params.get("min_score", 0.55))
    heavy = float(params.get("heavy_score", 0.75))
    if heavy < min_score + 0.15:
        params["heavy_score"] = round(min(min_score + 0.15, 0.99), 4)
    if float(params["base_position_pct"]) > float(params["max_position_pct"]):
        params["base_position_pct"] = params["max_position_pct"]
    if int(params["max_hold_bars"]) < int(params["min_hold_bars"]):
        params["max_hold_bars"] = params["min_hold_bars"]
    return params


# ---------------------------------------------------------------- 主流程
def train_tier_models(db, args, active_tiers, train_map, val_map, test_map, strategy, horizon):
    """为每档训练独立 pooled 模型，返回 (tier -> payload, tier -> 模型指标)。

    非 ML 策略不需要模型，payload 留空由规则直接驱动信号。
    """
    payloads: dict = {}
    models: dict = {}
    for key in TIER_KEYS:
        members = active_tiers.get(key)
        if not members:
            continue
        if strategy != "ml_ensemble":
            payloads[key], models[key] = None, {}
            continue
        tier_train = {s: train_map[s] for s in members}
        tier_val = {s: val_map[s] for s in members}
        tier_test = {s: test_map[s] for s in members}
        info = {
            "n_symbols": len(members),
            "symbols": members,
            "horizon": horizon,
            "pooled_train_rows": int(sum(len(f) for f in tier_train.values())),
            "skipped": [],
        }
        try:
            payloads[key], models[key] = train_tier_model(
                db, key, tier_train, tier_val, tier_test, info, INTERVAL,
                n_estimators=args.n_estimators, persist=False,
            )
        except Exception as exc:
            payloads[key], models[key] = None, {"error": str(exc)[:200]}
    return payloads, models


def evaluate_tier_branch(
    args, active_tiers, train_map, val_map, test_map, base, strategy,
    fee, slippage, payloads, models,
):
    """按档独立寻优出入场参数并在 test 段回测，返回 (结果, 净值腿, 成交)。

    payloads/models 由调用方决定：传「每档自己的模型」就是分层组，
    传「统一模型的同一份引用」就是消融组——用于把「分档模型」与
    「分档参数」两种增益拆开单独度量。
    """
    results: list[dict] = []
    legs_all: list[dict] = []
    trades_all: list[dict] = []
    for key in TIER_KEYS:
        members = active_tiers.get(key)
        if not members:
            continue
        payload = payloads.get(key)
        model_metrics = models.get(key) or {}
        tier_val = {s: val_map[s] for s in members}
        tier_test = {s: test_map[s] for s in members}
        tier_cash_pool = args.initial_cash / len(active_tiers)
        cash_map = {s: tier_cash_pool / len(members) for s in members}

        tier_base = tier_params(base, key)
        params, selection = select_tier_params(
            tier_val, strategy, tier_base, payload, cash_map, fee, slippage, INTERVAL
        )
        evaluated = evaluate_split(
            tier_test, strategy, params, payload, cash_map, fee, slippage, INTERVAL
        )
        metrics = evaluated[0] if evaluated else None
        if evaluated:
            _, legs, trades = evaluated
            legs_all.extend(legs)
            trades_all.extend(trades)
        results.append({
            "tier": key,
            "symbols": members,
            "n_symbols": len(members),
            "pooled_train_rows": int(sum(len(train_map[s]) for s in members)),
            "model": {
                k: model_metrics.get(k)
                for k in ("train_auc", "val_auc", "test_auc", "test_accuracy",
                          "atr_mean_pct", "error")
            },
            "selected_params": selection["selected_params"],
            "baseline_params": {
                k: tier_base.get(k) for k in ("min_score", "sl_atr_mult", "tp_atr_mult", "max_hold_bars")
            },
            "optimization": {"minimum_trades": selection["minimum_trades"],
                             "evaluated": selection["evaluated"]},
            "test": {
                "total_return_pct": metrics["total_return_pct"] if evaluated else None,
                "max_drawdown_pct": metrics["max_drawdown_pct"] if evaluated else None,
                "sharpe": metrics["sharpe"] if evaluated else None,
                "total_trades": metrics["total_trades"] if evaluated else None,
                "expectancy": metrics["expectancy"] if evaluated else None,
                "win_rate": metrics["win_rate"] if evaluated else None,
            } if evaluated else None,
        })
    return results, legs_all, trades_all


def run_window(db: Session, args, end_ms: int, index: int, cache: DatasetCache) -> dict:
    strategy = args.strategy
    fee, slippage = args.fee, args.slippage
    window_days = args.window_days
    base = _base_params(strategy, fee, slippage)
    horizon = int(DEFAULT_STATE.get("train_horizon", 5))

    # 1) 反前视分档：用回测窗口开始前的成交额/波动给币种分档。
    #    候选集限定为已补齐 1m 历史的代表币种——否则会对全库 500+ 币种逐个做
    #    1m 聚合查询，慢且其中绝大多数没有足够历史参与回测。
    tier_asof = end_ms - window_days * DAY_MS
    frame, groups = assign_tiers(
        db, args.candidates or None, INTERVAL, tier_asof, TIER_LOOKBACK_DAYS, "auto"
    )
    summary = tier_summary(frame)

    selected: dict[str, list[str]] = {}
    for key in TIER_KEYS:
        members = groups.get(key) or []
        if members:
            selected[key] = members[: args.per_tier]

    symbols_all = sorted({s for members in selected.values() for s in members})

    # 2) 逐币种构建 features 并切分（缓存，分层组与对照组共用）
    train_map, val_map, test_map = {}, {}, {}
    skipped: list[dict] = []
    for symbol in symbols_all:
        data = cache.get(db, symbol, INTERVAL, horizon, window_days, end_ms)
        parts = split_symbol(data, args.train_ratio, args.val_ratio, horizon)
        if parts is None:
            skipped.append({"symbol": symbol, "reason": "样本不足或切分失败"})
            continue
        train_map[symbol], val_map[symbol], test_map[symbol] = parts
    if not test_map:
        return {"index": index, "ok": False, "reason": "无可用币种", "tier_summary": summary}

    active_tiers = {
        key: [s for s in members if s in test_map] for key, members in selected.items()
    }
    active_tiers = {k: v for k, v in active_tiers.items() if v}
    total_symbols = len(test_map)

    # ---- A 分层组：每档独立 pooled 模型 + 独立寻优参数 ----
    tier_payloads, tier_models = train_tier_models(
        db, args, active_tiers, train_map, val_map, test_map, strategy, horizon
    )
    tier_results, tier_legs, tier_trades = evaluate_tier_branch(
        args, active_tiers, train_map, val_map, test_map, base, strategy,
        fee, slippage, tier_payloads, tier_models,
    )

    cash_uniform = {s: args.initial_cash / total_symbols for s in test_map}

    # ---- B 统一组：全部币种混合训一个模型 + 全局参数（同款寻优）----
    uniform_payload, uniform_model = None, {}
    if strategy == "ml_ensemble":
        info_u = {
            "n_symbols": total_symbols,
            "symbols": list(test_map),
            "horizon": horizon,
            "pooled_train_rows": int(sum(len(f) for f in train_map.values())),
            "skipped": [],
        }
        try:
            uniform_payload, uniform_model = train_tier_model(
                db, "ALL", train_map, val_map, test_map, info_u, INTERVAL,
                n_estimators=args.n_estimators, persist=False,
            )
        except Exception as exc:
            uniform_model = {"error": str(exc)[:200]}

    uniform_params, uniform_selection = select_tier_params(
        val_map, strategy, base, uniform_payload, cash_uniform, fee, slippage, INTERVAL
    )
    uniform_eval = evaluate_split(
        test_map, strategy, uniform_params, uniform_payload, cash_uniform, fee, slippage, INTERVAL
    )

    # ---- C 消融组：统一模型 + 每档独立参数 ----
    #      与 B 只差「参数是否分档」，与 A 只差「模型是否分档」，
    #      用来判定观测到的增益究竟来自模型还是来自参数。
    cross_metrics = None
    cross_results: list[dict] = []
    if getattr(args, "cross_ablation", False) and uniform_payload is not None:
        shared = {k: uniform_payload for k in active_tiers}
        shared_models = {k: uniform_model for k in active_tiers}
        cross_results, cross_legs, cross_trades = evaluate_tier_branch(
            args, active_tiers, train_map, val_map, test_map, base, strategy,
            fee, slippage, shared, shared_models,
        )
        cross_series = align_portfolio_equity(cross_legs) if cross_legs else pd.Series(dtype=float)
        cross_initial = float(sum(leg["initial"] for leg in cross_legs)) if cross_legs else args.initial_cash
        cross_metrics = (
            metrics_from_equity(cross_series, cross_trades, cross_initial, INTERVAL)
            if not cross_series.empty
            else None
        )

    tiered_series = align_portfolio_equity(tier_legs) if tier_legs else pd.Series(dtype=float)
    initial_total = float(sum(leg["initial"] for leg in tier_legs)) if tier_legs else args.initial_cash
    tiered_metrics = (
        metrics_from_equity(tiered_series, tier_trades, initial_total, INTERVAL)
        if not tiered_series.empty
        else None
    )
    uniform_metrics = uniform_eval[0] if uniform_eval else None

    end_date = datetime.fromtimestamp(end_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    return {
        "index": index,
        "end_date": end_date,
        "end_ms": end_ms,
        "ok": True,
        "strategy": strategy,
        "total_symbols": total_symbols,
        "skipped": skipped,
        "tier_summary": summary,
        "tiers": tier_results,
        "cross": {
            "total_return_pct": cross_metrics["total_return_pct"] if cross_metrics else None,
            "max_drawdown_pct": cross_metrics["max_drawdown_pct"] if cross_metrics else None,
            "sharpe": cross_metrics["sharpe"] if cross_metrics else None,
            "total_trades": cross_metrics["total_trades"] if cross_metrics else None,
            "expectancy": cross_metrics["expectancy"] if cross_metrics else None,
            "tiers": cross_results,
        } if cross_metrics else None,
        "tiered": {
            "total_return_pct": tiered_metrics["total_return_pct"] if tiered_metrics else None,
            "max_drawdown_pct": tiered_metrics["max_drawdown_pct"] if tiered_metrics else None,
            "sharpe": tiered_metrics["sharpe"] if tiered_metrics else None,
            "total_trades": tiered_metrics["total_trades"] if tiered_metrics else None,
            "expectancy": tiered_metrics["expectancy"] if tiered_metrics else None,
        } if tiered_metrics else None,
        "uniform": {
            "total_return_pct": uniform_metrics["total_return_pct"] if uniform_metrics else None,
            "max_drawdown_pct": uniform_metrics["max_drawdown_pct"] if uniform_metrics else None,
            "sharpe": uniform_metrics["sharpe"] if uniform_metrics else None,
            "total_trades": uniform_metrics["total_trades"] if uniform_metrics else None,
            "expectancy": uniform_metrics["expectancy"] if uniform_metrics else None,
            "selected_params": uniform_selection["selected_params"],
            "model": {
                k: uniform_model.get(k) for k in ("train_auc", "val_auc", "test_auc", "error")
            },
        },
    }


def _fmt(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _pct(value):
    return _fmt(value, 2, "%")


def build_report(payload: dict) -> str:
    """生成 Markdown 报告。核心是「分层 vs 统一」的同口径对比。"""
    windows = [w for w in payload["windows"] if w.get("ok")]
    params = payload["params"]
    lines = [
        "# 分档模型 + 分档出入场参数回测", "",
        f"- as-of：{payload['as_of']}；策略：{params['strategy']}；interval：1h",
        f"- 窗口：{params['windows']} 个 × {params['window_days']} 天，步长 {params['step_days']} 天；"
        f"切分 train/val/test = {params['train_ratio']-params['val_ratio']:.0%}/"
        f"{params['val_ratio']:.0%}/{1-params['train_ratio']:.0%}",
        f"- 每档最多 {params['per_tier']} 个币种；组合初始资金 {float(params['initial_cash']):.0f} USDT，"
        f"fee={params['fee']}、slippage={params['slippage']}",
        "", "## 0. 数据与口径", "",
        "- 回测走标准引擎口径（从 1m 原始K线聚合 1h），币种已按档位补齐约 90 天分钟线，"
        "在此之前本地 1m 稀疏，1h 实际只取得到约 7 天数据。",
        "- **反前视三道防线**：① 分档依据取自回测窗口开始前 30 天的成交额/波动；"
        "② 训练只用 train 段、寻优只用 val 段，test 段全程不可见；③ walk-forward 逐窗独立重跑。",
        "- 对照组 **uniform**：同一批币种、同一模型算法、同一套寻优流程，但全部混合训一个模型、"
        "全局用一套参数。只有分层组显著优于它，才能说明分档是真增益。",
        "",
    ]
    if not windows:
        lines += ["未产出任何有效窗口，回测未成功。"]
        return "\n".join(lines)

    # ---- 1. 分档概况 ----
    latest = windows[0]
    lines += [
        "## 1. 规模 × 波动 分档概况（最新窗口）", "",
        "| 档位 | 币种数 | 日均成交额中位数 (USDT) | 平均波动率 atr_pct | 代表币种 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in latest.get("tier_summary", []):
        lines.append(
            f"| {row['tier']} | {row['symbols']} | {row['median_daily_usd']:,.0f} "
            f"| {row['median_atr_pct']:.3%} | {'、'.join(row['top_symbols'][:3])} |"
        )
    lines += [
        "",
        "分档是否有效，看「成交额递减 / 波动率递增」是否单调。若两条大多背离，"
        "说明该维度没抓到风险差异。", "",
    ]

    # ---- 2. 分层 vs 统一（核心）----
    has_cross = any((w.get("cross") or {}).get("total_return_pct") is not None for w in windows)
    head = "| 窗口 | 终点 | 参与币种 | 分层收益 % |"
    if has_cross:
        head += " 消融(统一模型+分档参数) % |"
    head += " 统一收益 % | 分层回撤 % | 统一回撤 % | 分层夏普 | 统一夏普 |"
    sep = "|---|---|---:|---:|"
    if has_cross:
        sep += "---:|"
    sep += "---:|---:|---:|---:|---:|"
    lines += ["## 2. 分层 vs 统一（组合口径，每窗口同批币种）", "", head, sep]

    deltas = []
    for w in windows:
        tiered = w.get("tiered") or {}
        uniform = w.get("uniform") or {}
        cross = w.get("cross") or {}
        t, u = tiered.get("total_return_pct"), uniform.get("total_return_pct")
        c = cross.get("total_return_pct")
        if t is not None and u is not None:
            deltas.append(t - u)
        row = (
            f"| {w['index']} | {w['end_date']} | {w.get('total_symbols')} "
            f"| {_pct(t)} |"
        )
        if has_cross:
            row += f" {_pct(c)} |"
        row += (
            f" {_pct(u)} | {_pct(tiered.get('max_drawdown_pct'))} "
            f"| {_pct(uniform.get('max_drawdown_pct'))} "
            f"| {_fmt(tiered.get('sharpe'))} | {_fmt(uniform.get('sharpe'))} |"
        )
        lines.append(row)
    if deltas:
        mean_delta = statistics.mean(deltas)
        positive = sum(1 for d in deltas if d > 0)
        lines += [
            "",
            f"- 分层相对统一的平均差值 **{mean_delta:+.2f} 个百分点**，"
            f"{positive}/{len(deltas)} 个窗口分层占优。",
        ]
        if positive > len(deltas) - positive and mean_delta > 0:
            lines.append("- 判定：分层多数窗口占优，分档带来正增益的证据成立。")
        elif positive == len(deltas) - positive:
            lines.append("- 判定：**两者打平，分档未见增益**，倾向于维持统一模型以简化维护。")
        else:
            lines.append("- 判定：分层在多数窗口不占优，**不支持分档**，应保留统一配置。")
        # 窗口数偏少或相邻窗口高度重叠时，单次配置的胜率不足以定论
        if len(deltas) < 3:
            lines.append(
                f"- **注意**：仅有 {len(deltas)} 个窗口，且相邻窗口存在重叠（"
                f"窗口 {params.get('window_days')} 天 / 步长 {params.get('step_days')} 天），"
                "不构成独立样本。**结论请合并多份配置后一起看，参见 `tier-backtest-summary.md`**。"
            )
    lines.append("")

    # ---- 3. 模型质量 ----
    lines += [
        "## 3. 各档共享模型（pooled）质量", "",
        "| 档位 | 币种数 | 训练行 | val AUC | test AUC | test 命中率 | 错误 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tier in windows[0].get("tiers", []):
        model = tier.get("model") or {}
        lines.append(
            f"| {tier['tier']} | {tier['n_symbols']} | {tier['pooled_train_rows']:,} "
            f"| {_fmt(model.get('val_auc'), 4)} | {_fmt(model.get('test_auc'), 4)} "
            f"| {_fmt(model.get('test_accuracy'), 4)} | {model.get('error') or '-'} |"
        )
    uniform_model = (windows[0].get("uniform") or {}).get("model") or {}
    lines += [
        f"| **uniform(对照)** | {windows[0].get('total_symbols')} | - "
        f"| {_fmt(uniform_model.get('val_auc'), 4)} | {_fmt(uniform_model.get('test_auc'), 4)} | - "
        f"| {uniform_model.get('error') or '-'} |",
        "",
        "AUC 显著 > 0.5 才说明模型有判别力；接近或低于 0.5 表示该档模型不可用。", "",
    ]

    # ---- 4. 各档寻优后的出入场参数（核心产出）----
    lines += [
        "## 4. 各档寻优后的出入场参数（核心产出）", "",
        "| 档位 | min_score | sl_atr_mult | tp_atr_mult | max_hold_bars | 测试收益 % | 回撤 % | 成交 | 期望值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in windows[0].get("tiers", []):
        selected = tier.get("selected_params") or {}
        test = tier.get("test") or {}
        lines.append(
            f"| {tier['tier']} | {_fmt(selected.get('min_score'), 3)} "
            f"| {_fmt(selected.get('sl_atr_mult'), 2)} | {_fmt(selected.get('tp_atr_mult'), 2)} "
            f"| {_fmt(selected.get('max_hold_bars'), 0)} "
            f"| {_pct(test.get('total_return_pct'))} | {_pct(test.get('max_drawdown_pct'))} "
            f"| {_fmt(test.get('total_trades'), 0)} | {_fmt(test.get('expectancy'), 4)} |"
        )
    lines += [
        "",
        "参数随规模变化的规律是否符合预期（止损/门槛随规模减小而放宽、"
        "持有时间与仓位随规模减小而收紧），是判断这套参数可信与否的关键："
        "若最优参数在各档间乱跳、无单调趋势，多半是寻优在小样本上过拟合。", "",
    ]

    # ---- 5. 结论 ----
    lines += [
        "## 5. 结论与边界", "",
        f"- 本次共 {len(windows)} 个窗口、{windows[0].get('total_symbols')} 个币种参与，"
        f"总耗时 {payload['elapsed_sec']}s。",
        "- **样本量边界**：单窗口 test 段约 10 天量级，成交笔数偏少，"
        "单档收益的统计显著性有限；结论应理解为方向性证据，不宜据此直接放大实盘仓位。",
        "- **历史宽度边界**：本地多币种行情深度约 90 天，窗口数与步长受此约束；"
        "待历史补齐后可重跑更多窗口以提升可信度。",
        "- 本报告只评估已存历史，不写订单、不改系统状态，不构成盈利证明。", "",
        "复现：", "", "```powershell",
        f'.\\.venv\\Scripts\\python.exe backend/scripts/tier_backtest.py --as-of "{payload["as_of"]}" '
        f"--windows {params['windows']} --per-tier {params['per_tier']}",
        "```", "",
    ]
    return "\n".join(lines)


def build_windows(end_ms: int, window_days: int, step_days: int, windows: int) -> list[int]:
    return [end_ms - i * step_days * DAY_MS for i in range(windows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", dest="as_of", default="2026-10-26T12:00:00+00:00")
    parser.add_argument("--strategy", default="ml_ensemble", choices=STRATEGIES)
    # 45 天窗口 + 15 天步长：受限于本地行情深度（补齐后每币种约 90 天），
    # 再长的窗口会让训练样本不足，再多的窗口会让末窗币种数骤降。
    parser.add_argument("--window-days", type=int, default=45)
    parser.add_argument("--step-days", type=int, default=15)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--per-tier", type=int, default=12, help="每档最多取几个币种")
    parser.add_argument(
        "--cross-ablation",
        action="store_true",
        help="追加「统一模型 + 分档参数」消融组，用于区分模型与参数的贡献",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="用已有 JSON 重新渲染 Markdown，不重跑回测",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--fee", type=float, default=FEE)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "tier-backtest.json")
    parser.add_argument(
        "--markdown-output", type=Path, default=ROOT / "docs" / "tier-backtest.md"
    )
    parser.add_argument(
        "--candidates-file", type=Path,
        default=ROOT / "docs" / "backfill-tier-minutes.json",
        help="回填脚本产出的代表币种清单（限制分档候选集，提升速度）",
    )
    args = parser.parse_args()

    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone")
    end_ms = int(as_of.timestamp() * 1000)

    candidates: list[str] | None = None
    if args.candidates_file and args.candidates_file.exists():
        try:
            raw = json.loads(args.candidates_file.read_text(encoding="utf-8"))
            candidates = sorted({str(s).upper() for s in (raw.get("symbols") or [])})
            print(f"候选币种集：{len(candidates)} 个（来自 {args.candidates_file.name}）")
        except Exception as exc:
            print(f"候选清单读取失败（{exc}），退化为全库扫描")
    args.candidates = candidates

    # 连接统一走 .env 里的 DATABASE_URL（SQLite/PostgreSQL 由 make_engine 按方言处理）
    engine = make_engine(read_only=True)

    # 只重渲染报告：改文案不必重跑整个回测（一次完整回测要 2~5 分钟）
    if args.report_only:
        if not args.output.exists():
            parser.error(f"--report-only 需要已存在的 JSON：{args.output}")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        args.markdown_output.write_text(build_report(payload), encoding="utf-8")
        print(f"已由 {args.output} 重新渲染报告 -> {args.markdown_output}")
        return

    cache = DatasetCache()
    windows_out: list[dict] = []
    started = time.time()
    with Session(engine) as db:
        for index, window_end in enumerate(
            build_windows(end_ms, args.window_days, args.step_days, args.windows)
        ):
            window_start = time.time()
            result = run_window(db, args, window_end, index, cache)
            windows_out.append(result)
            tiered = result.get("tiered") or {}
            uniform = result.get("uniform") or {}
            print(json.dumps({
                "window": index,
                "end": result.get("end_date"),
                "symbols": result.get("total_symbols"),
                "tiered_ret": tiered.get("total_return_pct"),
                "uniform_ret": uniform.get("total_return_pct"),
                "tiered_dd": tiered.get("max_drawdown_pct"),
                "uniform_dd": uniform.get("max_drawdown_pct"),
                "secs": round(time.time() - window_start, 1),
            }, ensure_ascii=False), flush=True)
    engine.dispose()

    payload = {
        "as_of": as_of.isoformat(),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_sec": round(time.time() - started, 1),
        "windows": windows_out,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    args.markdown_output.write_text(build_report(payload), encoding="utf-8")
    print(f"\n完成，耗时 {payload['elapsed_sec']}s")
    print(f"  JSON: {args.output}")
    print(f"  报告: {args.markdown_output}")


if __name__ == "__main__":
    main()
