"""组合级（策略 × 币种池）回测引擎：一个绑定只有一套参数。

与 `backtest.optimizer` 的区别
------------------------------
optimizer 回答「每个币各自该用哪套参数」，任务单位是 (币, 周期)，选参在单币内完成。
但「策略 × 币种池」绑定只有**一份** `params`，对整池所有币同时生效。所以在单币内
选出的冠军对绑定层不可用——必须先在池层面把成交聚合成一条序列再选参。

本模块因此只做三件事：
1. 逐币把 (参数组合) 跑成「每笔净收益序列」，收益一律按**名义本金**折算
   （`pnl / (entry*size)`），这样固定名义的口径（研究池 200U）与保证金口径
   （通用撮合）可以放进同一张表比；margin 口径读数不参与跨组合比较。
2. 按时间切训练 / 验证 / 测试三段，成交必须完整落在段内（跨段样本清除）。
3. 池化 bootstrap 判定 + 随机基线分位。

研究型策略（`research_*`）不走通用撮合：它们的 `direction≡0`、没有止损止盈、
按「持有 N 小时后下一开盘」出场，由模拟盘 `paper.open_research_position` 专用通道
执行。这里直接复用 `scripts/research_interval_exit_study.py` 的回放函数，保证
「回测的就是线上那一套」，不重写第二份持有期逻辑。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.execution import randomize_directions, simulate
from ..backtest.significance import bootstrap_expectancy_ci
from ..backtest.trades import aggregate_positions
from ..strategy.registry import get_spec
from ..strategy.rules import apply_exit_overlay

MIN_TRADES = 30
BOOT_SEED = 20261105
SPLIT = {"train": (0.0, 0.6), "valid": (0.6, 0.8), "test": (0.8, 1.0)}

_STUDY = None


def study_module():
    """惰性导入研究池回放脚本，复用其 hold_only 撮合语义（不复制一份）。

    注意：该模块的 WARMUP=200 决定「缺口后重新预热」的规则，与本引擎的预热
    保持一致；`simulate_exit` 的成本常量固定在 200U 名义 / 0.04% / 0.02%，
    正是研究策略在模拟盘上的真实口径。
    """
    global _STUDY
    if _STUDY is None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "research_interval_exit_study.py"
        spec = importlib.util.spec_from_file_location("_research_hold_replay", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STUDY = module
    return _STUDY


# ------------------------------------------------------------------ 逐币回放


def research_trades(frame: pd.DataFrame, channel_kind: int, hold_bars: int) -> list[dict]:
    """研究池策略：只做多、持有 N 根后下一开盘出场（沿用专用回放）。"""
    mod = study_module()
    kind = "boll" if int(channel_kind) == 0 else "atrch"
    entry = mod._entries(frame, kind)
    raw = mod.simulate_exit(frame, entry, int(hold_bars), "hold_only")
    index = frame.index
    return [
        {"t0": index[t["entry"]], "t1": index[t["exit"]], "ret": float(t["return_pct"]),
         "reason": t["reason"]}
        for t in raw
    ]


def generic_trades(
    prepared: pd.DataFrame, code: str, params: dict, *, fee: float, slippage: float,
    initial: float, interval: str, randomize: bool = False, seed: int = BOOT_SEED,
) -> list[dict]:
    """通用撮合：信号 t 收盘 → t+1 开盘成交，收益折算成名义本金口径。

    必须先按 `position_id` 合并分批止盈的腿：每条腿的 `return_pct` 只相对自己
    那一小份仓位，直接平均会把 1/6 仓位的 1R 止盈腿与 5/6 仓位的止损腿等权，
    尾部读数系统性偏正（实测 ETHUSDT 15m 180 天：腿级 +0.24% vs 仓位级 −0.09%）。
    """
    signals = apply_exit_overlay(prepared, get_spec(code).signals(prepared, params), params)
    if randomize:
        signals = randomize_directions(signals, seed=seed)
    _, trades = simulate(prepared, signals, params, initial, fee, slippage, interval)
    out = []
    for trade in aggregate_positions(trades):
        notional = float(trade["entry_price"]) * float(trade["size"])
        if not notional > 0:
            continue
        out.append({
            "t0": pd.Timestamp(trade["entry_time"]),
            "t1": pd.Timestamp(trade["exit_time"]),
            "ret": float(trade["pnl"]) / notional * 100.0,
            "reason": str(trade.get("status") or ""),
        })
    return out


def prepare_cached(cache: dict, frame: pd.DataFrame, code: str, params: dict, keys: tuple):
    """按 `spec.prepare_keys` 缓存已富化帧：滚动分位/指标预热是网格里最贵的一步。"""
    signature = tuple(params.get(key) for key in keys)
    prepared = cache.get(signature)
    if prepared is None:
        from ..strategy.rules import prepare_frame

        prepared = prepare_frame(frame, code, params)
        if len(cache) >= 6:
            cache.clear()
        cache[signature] = prepared
    return prepared


# ------------------------------------------------------------------ 分段与判定


def segment_bounds(index: pd.DatetimeIndex) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """按 bar 数比例切三段；返回每段的墙钟边界（含左不含右）。"""
    n = len(index)
    out = {}
    for name, (lo, hi) in SPLIT.items():
        start = int(round(n * lo))
        end = int(round(n * hi))
        out[name] = (index[min(start, n - 1)], index[min(end, n - 1)])
    return out


def by_segment(trades: list[dict], bounds: dict) -> dict[str, list[float]]:
    """把成交按段归类：跨段的样本清除，避免一棵仓同时算进训练和验证。"""
    out: dict[str, list[float]] = {name: [] for name in bounds}
    for trade in trades:
        for name, (start, end) in bounds.items():
            if trade["t0"] >= start and trade["t1"] < end:
                out[name].append(trade["ret"])
                break
    return out


def pooled(values: list[float]) -> dict:
    """池化判定：字段名统一成 short form，报告与随机基线都读这一份。"""
    raw = bootstrap_expectancy_ci([{"return_pct": v} for v in values], min_trades=MIN_TRADES)
    return {
        "trades": int(raw.get("trades") or 0),
        "mean": raw.get("mean_return_pct"),
        "ci_low": raw.get("ci_low_pct"),
        "ci_high": raw.get("ci_high_pct"),
        "positive_share": raw.get("positive_share"),
        "verdict": raw.get("verdict"),
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def pick(rows: list[dict], min_trades: int = MIN_TRADES, share: float = 0.4) -> dict | None:
    """训练段选参：池化每笔期望最高者。

    稀疏格子先剔除：网格里只要有一档「开 ADX/开 EMA200 过滤」，成交会掉到中位数的
    百分之几，而几个幸运的样本就能把均值顶到最高——那是小样本假精确，不是 edge。
    门槛取「中位成交数的 40%」（结构差异如持有 24h 与 168h 落在中位数附近不会
    被误杀，filter 型坍塌则远低于门槛）。`min_trades` 仍是 H4 的绝对底线。
    """
    counts = sorted(len(row["train"]) for row in rows)
    if not counts or counts[-1] == 0:
        return None
    median = counts[len(counts) // 2]
    floor = max(min_trades, int(median * share))
    eligible = [row for row in rows if len(row["train"]) >= floor]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(np.mean(row["train"])), len(row["train"])))


def random_entry_returns(
    frames: list, hold_bars: int, n_trades: int, *, n_seeds: int = 20,
    base_seed: int = 20261105, warmup: int = 200,
) -> list[float]:
    """研究型策略的随机基线：同样笔数、同样持有期，入场时点随机。

    原策略只做多，方向翻转没有意义（翻转等于做空，那是另一个 beta），所以基线
    建在「随机择时」这一维上：如果随便什么时候买、持有同样久也能拿到同样的每笔
    期望，edge 就不在择时上。多币一起池化，避免单个币的波动冒充基线。
    """
    mod = study_module()
    values = []
    for offset in range(n_seeds):
        rng = np.random.default_rng(base_seed + offset * 7919)
        marks: list[float] = []
        for opens in frames:
            span = len(opens) - hold_bars - 1
            if span <= warmup:
                continue
            for _ in range(max(1, n_trades)):
                entry = int(rng.integers(warmup, span))
                fill = opens[entry] * (1 + mod.SLIP)
                exit_at = min(entry + hold_bars, len(opens) - 1)
                marks.append((opens[exit_at] * (1 - mod.SLIP) - fill) / fill * 100 - 2 * mod.FEE * 100)
        if marks:
            values.append(float(np.mean(marks)))
    return values


def research_baseline(
    service, symbols: list[str], interval: str, hold_bars: int, start_ms: int, end_ms: int,
    per_symbol: int, *, n_seeds: int = 20, warmup: int = 420,
) -> tuple[list[float], int]:
    """随机入场基线的取数与采样一步到位，返回 (每种子池化均值, 实际参与币数)。

    只取 open 列：随机择时对照只需要「何时买、持有多久、何时卖」的价格，指标预热
    那 200 根对随机入场没有意义，但保留 `warmup` 让采样区间与原策略一致。
    """
    from ..market.data_service import MarketDataService

    service = service or MarketDataService(_session_fallback())
    frames = []
    for symbol in symbols:
        frame = service.load_frame(symbol, interval, start_ms=start_ms - warmup * _step(interval),
                                   end_ms=end_ms)
        if len(frame) > hold_bars + 300:
            frames.append(frame["open"].to_numpy(float))
    return random_entry_returns(frames, hold_bars, per_symbol, n_seeds=n_seeds,
                                warmup=max(200, warmup // 2)), len(frames)


def _step(interval: str) -> int:
    from ..utils import interval_ms

    return interval_ms(interval)


def _session_fallback():
    from ..db import SessionLocal

    return SessionLocal()


def taker_ready_symbols(
    db, symbols: list[str], start: int, end: int, share: float = 0.9,
) -> tuple[list[str], list[dict]]:
    """15m/1m 的主动买入量只来自 1m 聚合，原生 15m 一行 taker 都没有（2026-11-05 核实）。

    所以 `flow_imbalance_long` 的可用宇宙 = 「有 1m 且 taker 覆盖达标」的币，
    不是池成员表里的 494 个。覆盖率不足的币必须显式列进 skip 清单（H8）。
    """
    from sqlalchemy import text

    rows = db.execute(text(
        "select symbol, count(*) as n, sum(case when taker_buy_volume is not null then 1 else 0 end) as tk "
        "from klines where interval='1m' and open_time>=:s and open_time<:e group by symbol"
    ), {"s": start, "e": end}).all()
    ready = {row[0] for row in rows if row[1] and (row[2] or 0) / row[1] >= share}
    keep = [symbol for symbol in symbols if symbol in ready]
    skip = [{"symbol": symbol, "reason": f"1m taker 覆盖不足 {share:.0%}（或全无 1m）"}
            for symbol in symbols if symbol not in ready]
    return keep, skip


def percentile_rank(value: float, samples: list[float]) -> float:
    array = np.asarray(samples, dtype=float)
    if not len(array):
        return 0.0
    return round(float((array < value).mean()), 4)


def buy_hold(frame: pd.DataFrame, bounds: dict) -> dict[str, float]:
    """样本期基准涨跌（beta 归因用）：段内 close 首尾变化。"""
    out = {}
    for name, (start, end) in bounds.items():
        window = frame[(frame.index >= start) & (frame.index < end)]
        if len(window) >= 2:
            out[name] = round(float(window["close"].iloc[-1] / window["close"].iloc[0] - 1) * 100, 2)
    return out
