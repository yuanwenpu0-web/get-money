"""RGD 回测的单币 worker：取数 + HTF 因果拼帧 + 4 组矩阵逐组回放。

拆出本模块只为守住「单文件 ≤300 行」：主脚本负责参数解析、进程池编排、汇总与
输出；这里负责「一个币怎么跑完 4 组」。

HTF 铁律（契约 §7.3.1，最高危的沉默零成交）：`load_segments` 必须
`load_frame -> attach_htf_levels(整帧一次) -> 切段`。attach 放到 `prepare` 之后会
静默失效：`prepare` 幂等短路（列齐直接 return），而缺 HTF 列时的降级分支会把
`rgd_htf_ok` 全设 False -> `rg=0` -> 永不入场，四组跑完零成交且不抛任何异常。
所以切段后逐个断言，缺列/全 False 直接 raise 带 symbol。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

from app.backtest.execution import exposure_metrics
from app.backtest.metrics import metrics_from_equity
from app.backtest.rgd_execution import simulate_rgd
from app.backtest.trades import aggregate_positions
from app.db import make_engine
from app.features.rgd import htf_interval_for
from app.features.rgd_htf import HTF_COLUMNS, attach_htf_levels
from app.market.data_service import MarketDataService
from app.strategy.rules import evaluate_signals, prepare_frame
from app.utils import interval_ms
from rgd_baselines import random_summary, randomize_rgd

CODE = "rgd_donchian_v2"
DAY_MS = 86_400_000
MIN_SEGMENT_BARS = 800
GROUPS = (
    {"id": "G1", "label": "Long & Short · 项目全局成本", "dir_mode": 0,
     "fee_rate": 0.0004, "slippage": 0.0002, "slippage_note": "比例滑点（项目全局）"},
    {"id": "G2", "label": "Long Only · 项目全局成本", "dir_mode": 1,
     "fee_rate": 0.0004, "slippage": 0.0002, "slippage_note": "比例滑点（项目全局）"},
    {"id": "G3", "label": "Long & Short · 脚本原样成本", "dir_mode": 0,
     "fee_rate": 0.0005, "slippage": "2ticks", "slippage_note": "2 ticks 每币标量近似（R15）"},
    {"id": "G4", "label": "Long Only · 脚本原样成本", "dir_mode": 1,
     "fee_rate": 0.0005, "slippage": "2ticks", "slippage_note": "2 ticks 每币标量近似（R15）"},
)
_SESSION = None


def _db():
    """worker 进程自己的只读会话（NullPool：一条连接、用完即关）。"""
    global _SESSION
    if _SESSION is None:
        engine = make_engine(read_only=True, null_pool=True)
        _SESSION = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False, future=True
        )()
    return _SESSION


def tick_size_for(price: float) -> float:
    """`utils.round_price` 的 decimals 约定作为每币 tick 代理（契约 §8 R15）。"""
    if price >= 1000:
        return 1e-2
    if price >= 1:
        return 1e-4
    return 1e-6


def two_tick_slippage(closes: np.ndarray) -> float:
    """`2 * tick_size(median_close) / median_close`，逐币标量。"""
    median = float(np.nanmedian(closes))
    if not math.isfinite(median) or median <= 0:
        return 0.0
    return 2.0 * tick_size_for(median) / median


def load_segments(symbol: str, interval: str, days: int, end_ms: int,
                  min_bars: int, as_of: int) -> list[pd.DataFrame]:
    """取一个币的连续 K 线段；HTF 必须在切段之前对整帧 attach 一次（§7.3.1）。"""
    service = MarketDataService(_db())
    start_ms = as_of - days * DAY_MS
    step = interval_ms(interval)
    htf_interval = htf_interval_for(interval, True, 1)  # D1：15m -> 4h
    htf_step = interval_ms(htf_interval)
    full = service.load_frame(symbol, interval, start_ms=start_ms, end_ms=as_of)
    if full.empty:
        return []
    # 排除未收盘的桶：半截 OHLCV 当信号等于用未来数据。
    full = full.loc[(full.index.as_unit("ms").asi8 + step) <= as_of].copy()
    full = full.loc[~full.index.duplicated(keep="last")].sort_index()
    valid = full[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    mask = valid.notna().all(axis=1)
    mask &= (valid[["open", "high", "low", "close"]] > 0).all(axis=1)
    mask &= valid["volume"].ge(0)
    mask &= valid["high"].ge(valid[["open", "close", "low"]].max(axis=1))
    mask &= valid["low"].le(valid[["open", "close", "high"]].min(axis=1))
    full = full.loc[mask]
    if full.empty:
        return []
    htf = service.load_frame(
        symbol, htf_interval, start_ms=start_ms - 2 * htf_step, end_ms=end_ms
    )
    full = attach_htf_levels(full, htf, htf_interval, as_of)
    diffs = full.index.to_series().diff().dt.total_seconds() * 1000
    segments = [seg for _, seg in full.groupby(diffs.ne(step).cumsum()) if len(seg) >= min_bars]
    for segment in segments:
        missing = [name for name in HTF_COLUMNS if name not in segment.columns]
        if missing or not bool(segment["rgd_htf_ok"].any()):
            raise ValueError(
                f"{symbol}: HTF 列缺失或全 False（缺 {missing}），拒绝静默零成交（契约 §7.3.1）"
            )
    return segments


def evaluate_symbol(task: dict) -> dict:
    """一个币 = 一个任务：取数 + HTF 一次，跑完 4 组矩阵。"""
    symbol = task["symbol"]
    try:
        segments = load_segments(
            symbol, task["interval"], task["days"], task["end_ms"],
            task["min_bars"], task["as_of"],
        )
    except Exception as exc:  # 无数据/查询异常 — 跳过而不是中断整轮
        return {"symbol": symbol, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    if not segments:
        return {"symbol": symbol, "status": "skipped", "reason": "无满足长度的连续K线段"}

    cash, interval = task["cash"], task["interval"]
    closes = np.concatenate([seg["close"].to_numpy(dtype=float) for seg in segments])
    prepared: dict[tuple, tuple] = {}
    group_rows = []
    for group in GROUPS:
        params = {**task["base"], "dir_mode": group["dir_mode"]}
        key = tuple(params[name] for name in task["prepare_keys"])
        if key not in prepared:
            frames = [prepare_frame(seg, CODE, params) for seg in segments]
            prepared[key] = (frames, [evaluate_signals(f, CODE, params) for f in frames])
        frame_list, signal_list = prepared[key]
        fee = float(group["fee_rate"])
        slippage = (
            two_tick_slippage(closes) if group["slippage"] == "2ticks"
            else float(group["slippage"])
        )

        equity_parts: list[pd.Series] = []
        # `fills` 保留每条实际成交腿（TP1 + 最终 trail/stop）；`trades` 只给账户指标与
        # bootstrap 用，按 position_id 合并，避免小的 partial fill 被收益统计高估。
        # 两者的出场分布必须分别统计：合并仓位只留最后一腿 status，会把 TP1 掩盖成 stop。
        fills_all: list[dict] = []
        trades: list[dict] = []
        fill_status: Counter = Counter()
        tp1_ids: set = set()
        factor = 1.0
        signal_bars = 0
        blocked: Counter = Counter()
        for segment, signals in zip(frame_list, signal_list):
            equity, fills = simulate_rgd(segment, signals, params, cash, fee, slippage, interval)
            equity_parts.append(equity * factor)
            factor *= float(equity.iloc[-1]) / cash   # 段间复利：缺口处仓已平，现金结转
            fills_all.extend(fills)
            fill_status.update(item["status"] for item in fills)
            tp1_ids.update(
                item["position_id"] for item in fills if item["status"] == "take_profit_1"
            )
            trades.extend(aggregate_positions(fills))
            signal_bars += int(signals["rgd_long_break"].sum() + signals["rgd_short_break"].sum())
            blocked.update({
                str(name): int(count)
                for name, count in signals.block_reason[
                    signals.block_reason.ne("")
                ].value_counts().items()
            })
        combined = pd.concat(equity_parts)
        metrics = metrics_from_equity(combined, trades, cash, interval)

        samples = []
        for offset in range(task["seeds"]):
            seed = task["base_seed"] + offset * 7919
            random_factor = 1.0
            for segment, signals in zip(frame_list, signal_list):
                shuffled = randomize_rgd(signals, seed)
                equity, _ = simulate_rgd(segment, shuffled, params, cash, fee, slippage, interval)
                random_factor *= float(equity.iloc[-1]) / cash
            samples.append((random_factor - 1) * 100)

        group_rows.append({
            "symbol": symbol,
            "trades": len(trades),
            "signal_bars": signal_bars,
            "return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "win_rate": metrics.get("win_rate"),
            "slippage_used": slippage,
            "trade_returns": [[round(float(item["return_pct"]), 4), item["side"]] for item in trades],
            # 仓位级：合并多腿后的最终出场（TP1 会被后一腿 stop 覆盖，属正常）
            "status_counts": dict(Counter(item["status"] for item in trades)),
            # 腿级：每条真实成交腿，TP1 在这里可见
            "fill_status_counts": dict(fill_status),
            "tp1_positions": len(tp1_ids),
            "tp1_position_ratio": round(len(tp1_ids) / len(trades), 4) if trades else 0.0,
            "blocked": dict(blocked),
            "random": random_summary(samples),
            "htf_degraded": any(bool(f.attrs.get("rgd_htf_degraded")) for f in frame_list),
            **exposure_metrics(combined, trades, interval),
        })
    return {
        "symbol": symbol, "status": "ok",
        "bars": int(sum(len(seg) for seg in segments)),
        "segments": len(segments), "groups": group_rows,
    }
