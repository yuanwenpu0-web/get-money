"""按「规模 × 波动率」二维切分币种到不同 tier，为每档独立建模与参数提供分组依据。

设计要点
--------
1. **规模（size）用绝对成交额阈值**：日均成交额（USDT）有明确的经济含义
   （流动性深度、滑点、可承载仓位），阈值稳定可解释，跨窗口可比。
2. **波动率（vol）用「档内中位数」二次切分**：实测币圈 1h atr_pct 中位数高达 1.46%，
   绝对阈值会把几乎全部币种塞进 volatile（0.6% 阈值下 stable 仅 9/519），
   严重偏斜。改为在 size 档内部按 atr_pct 中位数切稳定/高波两组，
   既实现"二次修正"，又保证每档样本均衡、且不同规模档各自适配其波动基准。
3. **严格反前视**：所有统计量只取自 [end_ms - lookback, end_ms] 闭区间，
   绝不读取 end_ms 之后的任何一根 K 线；ATR 预热区间虽回溯更早，同样止于 end_ms。

因此每个 walk-forward 窗口都会重新分档——币种会因流动性变化在不同窗口迁徙到不同档，
这是刻意为之：用「当时可见的信息」分档，而非事后用全样本信息，避免前视泄漏。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Kline
from ..utils import interval_ms

# ---- 阈值（基于全库 519 币种实测分布校准，见 docs/tier-backtest.md 第 0 节）----
SIZE_TIERS: tuple[str, ...] = ("mega", "large", "mid", "small")
VOL_TIERS: tuple[str, ...] = ("stable", "volatile")

# 由大到小匹配：(日均成交额下限 USDT, 档位名)
SIZE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (5e8, "mega"),     # BTC/ETH 级别
    (1e8, "large"),    # 主流大盘
    (2e7, "mid"),      # 中盘
    (5e6, "small"),    # 小盘；低于此视为流动性不足/死币，排除
)
MIN_DAILY_USD = 5e6
# 排除稳定币与锚定资产：实测 USDCUSDT atr_pct 仅 0.00002，稳定币无交易价值且污染模型
MIN_ATR_PCT = 0.003
MIN_BARS = 600        # 窗口内至少 600 根，保证统计量可靠
ATR_WINDOW = 14
ATR_WARMUP_DAYS = 20  # ATR14 预热（只取 end_ms 之前）
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}USDT$")


def size_tier(daily_usd: float) -> str | None:
    """按日均成交额返回规模档；低于最低门槛返回 None（应排除）。"""
    if not np.isfinite(daily_usd) or daily_usd < MIN_DAILY_USD:
        return None
    for threshold, name in SIZE_THRESHOLDS:
        if daily_usd >= threshold:
            return name
    return None


def tier_key(size: str, vol: str) -> str:
    return f"{size}_{vol}"


TIER_KEYS: tuple[str, ...] = tuple(tier_key(s, v) for s in SIZE_TIERS for v in VOL_TIERS)


def is_clean_symbol(symbol: str) -> bool:
    """过滤脏数据：库中存在「龙虾USDT」等非标准合约名。"""
    return bool(SYMBOL_PATTERN.match(symbol or ""))


def _load_bars(
    session: Session,
    symbols: list[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    source: str = "auto",
) -> pd.DataFrame:
    """读取 OHLC + quote_volume，用于算成交额与波动率。

    source="derived"（默认 auto 的首选）：走 MarketDataService._merged_candle_rows，
    从 1m 原始K线按目标周期聚合，并在 1m 未覆盖的时段用同周期原生行补齐 ——
    **与回测引擎完全同源**（回测走 load_frame 的同一条混合取数路径），
    这是分档口径正确性的前提。补齐只在聚合缺失处发生，重叠桶一律取聚合值。
    若整批币种在该模式下无数据，再回落读原生 interval 行（"native"）。
    """
    from ..config import MARKET_NATIVE_FALLBACK
    from ..market.data_service import MarketDataService

    columns = ["symbol", "open_time", "high", "low", "close", "quote_volume"]
    frames: list[pd.DataFrame] = []

    if source in ("auto", "derived"):
        try:
            service = MarketDataService(session)
        except Exception:  # 读取 app_state 失败时回落到原生行，不让分档链路整体崩掉
            service = None
        for symbol in (symbols if service is not None else []):
            try:
                rows = service._merged_candle_rows(
                    symbol, interval, start_ms, end_ms,
                    allow_native=MARKET_NATIVE_FALLBACK,
                )[0]
            except Exception:
                continue
            if rows:
                frames.append(
                    pd.DataFrame(
                        [
                            [symbol, r["open_time"], r["high"], r["low"],
                             r["close"], r["quote_volume"]]
                            for r in rows
                        ],
                        columns=columns,
                    )
                )

    if (source == "native") or (source == "auto" and not frames):
        rows = session.execute(
            select(
                Kline.symbol, Kline.open_time, Kline.high, Kline.low,
                Kline.close, Kline.quote_volume,
            ).where(
                Kline.interval == interval,
                Kline.open_time >= start_ms,
                Kline.open_time <= end_ms,
                Kline.symbol.in_(symbols),
            ).order_by(Kline.symbol, Kline.open_time)
        ).all()
        if rows:
            frames.append(pd.DataFrame(rows, columns=columns))

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def symbol_liquidity_stats(
    session: Session,
    symbols: Sequence[str] | None,
    interval: str,
    end_ms: int,
    lookback_days: int = 30,
    source: str = "auto",
) -> pd.DataFrame:
    """统计每个币种在 [end_ms - lookback, end_ms] 内的日均成交额与平均波动率。

    返回列：symbol / daily_usd / atr_pct / bars。反前视：不使用 end_ms 之后的数据。
    source 控制取数口径，详见 _load_bars。
    """
    step = interval_ms(interval)
    lookback_start = end_ms - lookback_days * 86_400_000
    # ATR 需要预热，但读数窗口仍严格止于 end_ms
    query_start = lookback_start - ATR_WARMUP_DAYS * 86_400_000

    wanted = list(symbols) if symbols else None
    if wanted:
        cleaned = [s for s in wanted if is_clean_symbol(s)]
    else:
        distinct = session.execute(
            select(func.distinct(Kline.symbol)).where(
                Kline.interval == interval,
                Kline.open_time >= query_start,
                Kline.open_time <= end_ms,
            )
        ).scalars()
        cleaned = [s for s in distinct if is_clean_symbol(s)]
    if not cleaned:
        return pd.DataFrame(columns=["symbol", "daily_usd", "atr_pct", "bars"])

    frame = _load_bars(session, cleaned, interval, query_start, end_ms, source)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "daily_usd", "atr_pct", "bars"])

    bars_per_day = 86_400_000 / step

    records = []
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("open_time")
        previous = group["close"].shift(1)
        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - previous).abs(),
                (group["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(ATR_WINDOW).mean()
        atr_pct = (atr / group["close"]).replace([np.inf, -np.inf], np.nan)

        # 只取实际回测窗口 [lookback_start, end_ms] 做统计
        window = group["open_time"] >= lookback_start
        if window.sum() < MIN_BARS:
            continue
        quoted = float(group.loc[window, "quote_volume"].sum())
        # 用实际覆盖的 bar 数换算日均，避免缺失小时拖低成交额
        covered = int(window.sum())
        daily_usd = quoted / covered * bars_per_day
        mean_atr = float(atr_pct[window].mean())
        if not np.isfinite(daily_usd) or not np.isfinite(mean_atr):
            continue
        records.append(
            {"symbol": symbol, "daily_usd": daily_usd, "atr_pct": mean_atr, "bars": covered}
        )

    stats = pd.DataFrame(records)
    if stats.empty:
        return pd.DataFrame(columns=["symbol", "daily_usd", "atr_pct", "bars"])
    # 过滤流动性不足与疑似稳定币
    stats = stats[(stats["daily_usd"] >= MIN_DAILY_USD) & (stats["atr_pct"] >= MIN_ATR_PCT)]
    return stats.reset_index(drop=True)


def classify_tiers(stats: pd.DataFrame) -> pd.DataFrame:
    """给统计表加上 size / vol / tier 三列。

    size：绝对成交额阈值；vol：size 档内 atr_pct 中位数二次切分
    （< 中位数 → stable，>= 中位数 → volatile）。档内不足 2 个币种时全归 stable。
    """
    if stats.empty:
        return pd.DataFrame(columns=["symbol", "daily_usd", "atr_pct", "bars", "size", "vol", "tier"])

    frame = stats.copy()
    frame["size"] = frame["daily_usd"].map(size_tier)
    frame = frame[frame["size"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "daily_usd", "atr_pct", "bars", "size", "vol", "tier"])

    volumes = []
    for size_value, group in frame.groupby("size", sort=False):
        if len(group) < 2:
            volumes.append(pd.Series("stable", index=group.index))
            continue
        median = float(group["atr_pct"].median())
        # 恰好等于中位数时归入 volatile，保证二分不丢样本
        volumes.append(
            pd.Series(np.where(group["atr_pct"] < median, "stable", "volatile"), index=group.index)
        )
    frame["vol"] = pd.concat(volumes).reindex(frame.index)
    frame["tier"] = frame["size"] + "_" + frame["vol"]
    return frame.reset_index(drop=True)


def group_by_tier(frame: pd.DataFrame) -> dict[str, list[str]]:
    """tier -> 该档币种列表（按日均成交额降序，便于取代表性样本）。"""
    result: dict[str, list[str]] = {key: [] for key in TIER_KEYS}
    if frame.empty:
        return result
    for key, group in frame.groupby("tier", sort=False):
        ordered = group.sort_values("daily_usd", ascending=False)["symbol"].tolist()
        result.setdefault(str(key), []).extend(ordered)
    return result


def cap_symbols(mapping: dict[str, list[str]], per_tier: int) -> dict[str, list[str]]:
    """每档最多保留 per_tier 个币种（已按成交额降序），控制训练/回测耗时。"""
    return {key: list(value[:per_tier]) for key, value in mapping.items()}


def tier_summary(frame: pd.DataFrame) -> list[dict]:
    """每档的摘要：币种数、成交额中位数、波动率中位数。"""
    if frame.empty:
        return []
    rows = []
    for key in TIER_KEYS:
        group = frame[frame["tier"] == key]
        if group.empty:
            continue
        rows.append(
            {
                "tier": key,
                "symbols": int(len(group)),
                "median_daily_usd": round(float(group["daily_usd"].median()), 2),
                "median_atr_pct": round(float(group["atr_pct"].median()), 6),
                "top_symbols": group.sort_values("daily_usd", ascending=False)["symbol"].head(5).tolist(),
            }
        )
    return rows


def assign_tiers(
    session: Session,
    symbols: Sequence[str] | None,
    interval: str,
    end_ms: int,
    lookback_days: int = 30,
    source: str = "auto",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """一步到位：统计 → 分档 → 分组。返回 (带 tier 列的 DataFrame, tier->symbols)。"""
    stats = symbol_liquidity_stats(session, symbols, interval, end_ms, lookback_days, source)
    frame = classify_tiers(stats)
    return frame, group_by_tier(frame)
