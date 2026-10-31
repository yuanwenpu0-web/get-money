"""Bayesian Trend Factor [WavesUnchained] — 对数几率融合的概率化趋势质量引擎。

输出 **Trend Factor**（−100..+100）与 **Confidence**（0..100）。五组证据：

1. **方向**：`dir = (linreg(close)[0] − linreg(close)[1]) / max(ATR, tick)`，按 0.18 归一化并钳到 ±1。
2. **强度**：`0.40×norm(ADX,15,35) + 0.35×效率比 + 0.25×R²`；
   效率比 = |close−close[len]| / Σ|Δclose|，R² = corr(close, bar_index)²。
   只有 `|dir| >= strength_dir_gate` 时强度才带方向（否则记 0）。
3. **结构**：pivot 追踪最近两个同侧枢轴 → HH/LH/HL/LL，多结构 0.5/项、空结构 0.5/项，
   差值 × 新鲜度（`1 − age/fresh_bars`，钳在 0.25..1）；再按「close 在最近低点上方
   且方向为正 +0.15」（空头镜像）。
4. **波动率**：ATR 在 100 根内的分位 `atrRank`；最优区 0.45–0.75 取 1，<0.20 线性上升，
   0.75–1.00 线性下降到 0 → `vol_optimal`；质量乘子
   `(1−w) + w×(0.65 + 0.35×vol_optimal)`。
5. **衰竭**（对当前方向的惩罚）：过度延伸（|dist|>2 ATR）、影线拒斥（|dist|>1.25 且
   影线占比 >0.45 且收盘位置反向）、动量衰减（|dist|>1.5 且加速度反向），
   三项各 0.45/0.35/0.20，取负号乘方向。

**融合**：各证据映射为伪概率 `0.5 + e×k`（k = 0.45/0.35/0.40/0.35）→ logit × 权重求和
→ × 一致性刹车（`0.75 + 0.25×平均绝对证据`）→ × 质量乘子 → sigmoid → `factor = (pb−1+pb)*100`。
**置信度** = `|dir|×24 + 强度×26 + |结构|×24 + vol_optimal×16 − 衰竭×16`（钳 0..100）；
低于 `conf_threshold` 时按比例缩水，再经 `factor_smooth_len` 的 EMA 平滑。

**状态机**：`factor >= trend_threshold` → +1；`<= −trend_threshold` → −1；
否则保持原状态直到跌破 `min(trend_exit, trend_threshold−1)` 的对称阈值。

**回撤/续势**（可选）：趋势 + 置信度 + 未衰竭 → 允许方向。价格从 EMA/Fair Path 上方
回落到 `[fair−0.75ATR, fair+0.25ATR]` 区间且触及 EMA → Armed；持续 `min_bars` 根且
深度落在 `[min_depth, max_depth]×ATR` → Mature；随后 EMA 回收 / 单根 micro-BOS /
影线拒斥任一确认 → 续势信号（受冷却与超时约束）。深度超上限或结构破坏或趋势失效 → 失效。

**平滑 K 线**（可选）：开/高/低/收各自均线（开用 (open[1]+close[1])/2），
高/低再与开收取极值；MA 线按 BTF 梯度着色。

丢弃：全部 plot/plotcandle/barcolor/label/table 绘图、tooltip、告警配置、渐变颜色计算。

Source: pine-collection/indicators/trend_strength/bayesian_trend_factor/bayesian_trend_factor.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    correlation,
    dmi,
    ema,
    flags,
    highest,
    integer,
    linreg,
    lowest,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
    sma,
    wma,
)

__all__ = ["bayesian_trend_factor"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm01(value: float, low: float, high: float) -> float:
    return _clamp((value - low) / max(high - low, 0.000001), 0.0, 1.0)


def _logit(probability: float) -> float:
    p = _clamp(probability, 0.01, 0.99)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _ma(series: pd.Series, length: int, kind: str) -> pd.Series:
    if kind == "SMA":
        return sma(series, length)
    if kind == "WMA":
        return wma(series, length)
    return ema(series, length)


def bayesian_trend_factor(
    frame: pd.DataFrame,
    *,
    reg_len: int = 34,
    atr_len: int = 14,
    adx_len: int = 14,
    er_len: int = 20,
    pivot_len: int = 4,
    w_direction: float = 1.25,
    w_strength: float = 1.10,
    w_structure: float = 1.30,
    w_volatility: float = 0.85,
    w_exhaustion: float = 1.00,
    trend_threshold: float = 55.0,
    trend_exit_threshold: float = 45.0,
    strong_threshold: float = 80.0,
    conf_threshold: float = 60.0,
    factor_smooth_len: int = 2,
    strength_dir_gate: float = 0.15,
    structure_fresh_bars: int = 50,
    enable_pullback: bool = True,
    pb_ema_len: int = 13,
    pb_rsi_len: int = 14,
    pb_min_bars: int = 2,
    pb_timeout: int = 30,
    pb_cooldown: int = 5,
    pb_impulse_lookback: int = 20,
    pb_min_depth_atr: float = 0.25,
    pb_max_depth_atr: float = 1.75,
    pb_min_score: float = 6.5,
    pb_rsi_bull_floor: float = 42.0,
    pb_rsi_bear_ceil: float = 58.0,
    pb_wick_min: float = 0.35,
    show_trend_candles: bool = True,
    candle_open_len: int = 25,
    candle_open_type: str = "EMA",
    candle_close_len: int = 20,
    candle_close_type: str = "EMA",
    candle_high_len: int = 20,
    candle_low_len: int = 20,
    candle_ma_len: int = 55,
    candle_ma_type: str = "WMA",
) -> pd.DataFrame:
    """返回 BTF 因子、置信度、状态、回撤状态机与平滑 K 线。"""
    reg_len = integer("reg_len", reg_len, 10)
    atr_len = integer("atr_len", atr_len, 5)
    adx_len = integer("adx_len", adx_len, 5)
    er_len = integer("er_len", er_len, 5)
    pivot_len = integer("pivot_len", pivot_len, 2, 20)
    w_direction = number("w_direction", w_direction, 0.0)
    w_strength = number("w_strength", w_strength, 0.0)
    w_structure = number("w_structure", w_structure, 0.0)
    w_volatility = number("w_volatility", w_volatility, 0.0, 1.0)
    w_exhaustion = number("w_exhaustion", w_exhaustion, 0.0)
    trend_threshold = number("trend_threshold", trend_threshold, 20.0, 90.0)
    trend_exit_threshold = number("trend_exit_threshold", trend_exit_threshold, 0.0, 55.0)
    strong_threshold = number("strong_threshold", strong_threshold, 50.0, 100.0)
    conf_threshold = number("conf_threshold", conf_threshold, 0.0, 100.0)
    factor_smooth_len = integer("factor_smooth_len", factor_smooth_len, 1, 10)
    strength_dir_gate = number("strength_dir_gate", strength_dir_gate, 0.0, 1.0)
    structure_fresh_bars = integer("structure_fresh_bars", structure_fresh_bars, 10, 300)
    enable_pullback = boolean("enable_pullback", enable_pullback)
    pb_ema_len = integer("pb_ema_len", pb_ema_len, 3)
    pb_rsi_len = integer("pb_rsi_len", pb_rsi_len, 3)
    pb_min_bars = integer("pb_min_bars", pb_min_bars, 0, 20)
    pb_timeout = integer("pb_timeout", pb_timeout, 5, 100)
    pb_cooldown = integer("pb_cooldown", pb_cooldown, 0, 50)
    pb_impulse_lookback = integer("pb_impulse_lookback", pb_impulse_lookback, 5, 100)
    pb_min_depth_atr = number("pb_min_depth_atr", pb_min_depth_atr, 0.0)
    pb_max_depth_atr = number("pb_max_depth_atr", pb_max_depth_atr, 0.25)
    pb_min_score = number("pb_min_score", pb_min_score, 1.0, 10.0)
    pb_rsi_bull_floor = number("pb_rsi_bull_floor", pb_rsi_bull_floor, 20.0, 55.0)
    pb_rsi_bear_ceil = number("pb_rsi_bear_ceil", pb_rsi_bear_ceil, 45.0, 80.0)
    pb_wick_min = number("pb_wick_min", pb_wick_min, 0.10, 0.80)
    show_trend_candles = boolean("show_trend_candles", show_trend_candles)
    candle_open_len = integer("candle_open_len", candle_open_len, 1)
    candle_open_type = choice("candle_open_type", candle_open_type, ("EMA", "SMA", "WMA"))
    candle_close_len = integer("candle_close_len", candle_close_len, 1)
    candle_close_type = choice("candle_close_type", candle_close_type, ("EMA", "SMA", "WMA"))
    candle_high_len = integer("candle_high_len", candle_high_len, 1)
    candle_low_len = integer("candle_low_len", candle_low_len, 1)
    candle_ma_len = integer("candle_ma_len", candle_ma_len, 1)
    candle_ma_type = choice("candle_ma_type", candle_ma_type, ("EMA", "SMA", "WMA"))

    high, low, close, open_ = ohlcv(frame, "high", "low", "close", "open")
    size = len(frame)
    highs, lows = high.to_numpy(), low.to_numpy()
    opens, closes = open_.to_numpy(), close.to_numpy()

    atr_values = atr(frame, atr_len)
    safe_atr = atr_values.clip(lower=1e-12)
    bar_range = (high - low).clip(lower=1e-12)
    close_position = (close - low) / bar_range
    upper_wick_ratio = (high - np.maximum(open_, close)) / bar_range
    lower_wick_ratio = (np.minimum(open_, close) - low) / bar_range

    # --- 1. direction -------------------------------------------------------
    reg_now = linreg(close, reg_len, 0)
    reg_prev = linreg(close, reg_len, 1)
    reg_slope = reg_now - reg_prev
    direction_raw = reg_slope / safe_atr
    direction_evidence = (direction_raw / 0.18).clip(-1.0, 1.0)
    dir_abs = direction_evidence.abs()

    # --- 2. strength --------------------------------------------------------
    _, _, adx = dmi(frame, adx_len)
    net_move = (close - close.shift(er_len)).abs()
    sum_move = close.diff().abs().rolling(er_len, min_periods=er_len).sum()
    efficiency = (net_move / sum_move.replace(0, np.nan)).fillna(0.0)
    corr = correlation(close, pd.Series(np.arange(size, dtype=float), index=frame.index), reg_len)
    r2 = corr.fillna(0.0) ** 2
    adx_score = ((adx - 15.0) / 20.0).clip(0.0, 1.0)
    strength_score = adx_score * 0.40 + efficiency.clip(0.0, 1.0) * 0.35 + r2.clip(0.0, 1.0) * 0.25
    gate = dir_abs >= strength_dir_gate
    strength_evidence = np.where(gate.to_numpy(), strength_score.to_numpy()
                                 * np.sign(direction_evidence.to_numpy()), 0.0)

    # --- 3. structure -------------------------------------------------------
    pivot_high = pivothigh(high, pivot_len, pivot_len).to_numpy()
    pivot_low = pivotlow(low, pivot_len, pivot_len).to_numpy()
    last_high = np.full(size, np.nan)
    prev_high = np.full(size, np.nan)
    last_low = np.full(size, np.nan)
    prev_low = np.full(size, np.nan)
    structure_bar = np.full(size, np.nan)
    current_high = current_low = math.nan
    previous_high = previous_low = math.nan
    last_structure_bar = math.nan
    for bar in range(size):
        if not np.isnan(pivot_high[bar]):
            previous_high, current_high = current_high, pivot_high[bar]
            last_structure_bar = bar - pivot_len
        if not np.isnan(pivot_low[bar]):
            previous_low, current_low = current_low, pivot_low[bar]
            last_structure_bar = bar - pivot_len
        last_high[bar] = current_high
        prev_high[bar] = previous_high
        last_low[bar] = current_low
        prev_low[bar] = previous_low
        structure_bar[bar] = last_structure_bar

    hh = np.isfinite(last_high) & np.isfinite(prev_high) & (last_high > prev_high)
    lh = np.isfinite(last_high) & np.isfinite(prev_high) & (last_high < prev_high)
    hl = np.isfinite(last_low) & np.isfinite(prev_low) & (last_low > prev_low)
    ll = np.isfinite(last_low) & np.isfinite(prev_low) & (last_low < prev_low)
    bull_structure = np.where(hh, 0.5, 0.0) + np.where(hl, 0.5, 0.0)
    bear_structure = np.where(ll, 0.5, 0.0) + np.where(lh, 0.5, 0.0)
    structure_age = np.where(np.isfinite(structure_bar),
                             np.arange(size, dtype=float) - structure_bar, 0.0)
    structure_freshness = np.clip(1.0 - structure_age / structure_fresh_bars, 0.25, 1.0)
    structure_evidence = (bull_structure - bear_structure) * structure_freshness
    above_last_low = np.isfinite(last_low) & (closes > last_low)
    below_last_high = np.isfinite(last_high) & (closes < last_high)
    structure_evidence = structure_evidence + np.where(
        (direction_evidence > 0) & above_last_low, 0.15, 0.0)
    structure_evidence = structure_evidence - np.where(
        (direction_evidence < 0) & below_last_high, 0.15, 0.0)
    structure_evidence = np.clip(structure_evidence, -1.0, 1.0)

    # --- 4. volatility regime ----------------------------------------------
    atr_rank_len = 100
    atr_low = lowest(atr_values, atr_rank_len)
    atr_high = highest(atr_values, atr_rank_len)
    atr_rank = ((atr_values - atr_low) / (atr_high - atr_low).replace(0, np.nan)).fillna(0.0)
    atr_rank = atr_rank.clip(0.0, 1.0)
    rank = atr_rank.to_numpy()
    vol_optimal = np.where(rank < 0.20, rank / 0.20,
                           np.where(rank <= 0.75, 1.0,
                                    np.where(rank <= 1.00, 1.0 - (rank - 0.75) / 0.25, 0.0)))
    vol_optimal = np.clip(vol_optimal, 0.0, 1.0)
    volatility_weight = _clamp(w_volatility, 0.0, 1.0)
    quality_multiplier = (1.0 - volatility_weight) + volatility_weight * (0.65 + vol_optimal * 0.35)

    # --- 5. exhaustion -----------------------------------------------------
    fair_path = reg_now
    distance_atr = (close - fair_path) / safe_atr
    fast_len = max(10, int(math.floor(reg_len / 2.0)))
    slope_fast = linreg(close, fast_len, 0) - linreg(close, fast_len, 1)
    acceleration = (slope_fast - reg_slope) / safe_atr

    dirs = direction_evidence.to_numpy()
    dist = distance_atr.to_numpy()
    accel = acceleration.to_numpy()
    up_wick = upper_wick_ratio.to_numpy()
    down_wick = lower_wick_ratio.to_numpy()
    pos = close_position.to_numpy()
    bull_over = (dirs > 0) & (dist > 2.0)
    bear_over = (dirs < 0) & (dist < -2.0)
    bull_rejection = (dirs > 0) & (dist > 1.25) & (up_wick > 0.45) & (pos < 0.65)
    bear_rejection = (dirs < 0) & (dist < -1.25) & (down_wick > 0.45) & (pos > 0.35)
    bull_fade = (dirs > 0) & (dist > 1.50) & (accel < -0.05)
    bear_fade = (dirs < 0) & (dist < -1.50) & (accel > 0.05)
    bull_exhaustion = bull_over & ((up_wick > 0.35) | (accel < 0.0))
    bear_exhaustion = bear_over & ((down_wick > 0.35) | (accel > 0.0))
    exhaustion_raw = (np.where(bull_exhaustion | bear_exhaustion, 0.45, 0.0)
                      + np.where(bull_rejection | bear_rejection, 0.35, 0.0)
                      + np.where(bull_fade | bear_fade, 0.20, 0.0))
    exhaustion_score = np.clip(exhaustion_raw, 0.0, 1.0)
    exhaustion_evidence = -exhaustion_score * np.sign(dirs)

    # --- Bayesian fusion ---------------------------------------------------
    p_direction = 0.5 + dirs * 0.45
    p_strength = 0.5 + strength_evidence * 0.35
    p_structure = 0.5 + structure_evidence * 0.40
    p_exhaustion = 0.5 + exhaustion_evidence * 0.35
    log_odds = (np.array([_logit(value) for value in p_direction]) * w_direction
                + np.array([_logit(value) for value in p_strength]) * w_strength
                + np.array([_logit(value) for value in p_structure]) * w_structure
                + np.array([_logit(value) for value in p_exhaustion]) * w_exhaustion)
    agreement = (np.abs(dirs) + np.abs(strength_evidence) + np.abs(structure_evidence)
                 + np.abs(exhaustion_evidence)) / 4.0
    brake = 0.75 + agreement * 0.25
    adjusted = log_odds * brake * quality_multiplier
    bull_prob = np.array([_sigmoid(value) for value in adjusted])
    trend_factor = (bull_prob - (1.0 - bull_prob)) * 100.0

    strength_series = strength_score.to_numpy()
    confidence_raw = (np.abs(dirs) * 24.0 + strength_series * 26.0
                      + np.abs(structure_evidence) * 24.0 + vol_optimal * 16.0
                      - exhaustion_score * 16.0)
    confidence = np.clip(confidence_raw, 0.0, 100.0)

    if conf_threshold <= 0:
        gated = trend_factor
    else:
        gated = np.where(confidence >= conf_threshold, trend_factor,
                         trend_factor * (confidence / conf_threshold))
    factor_final = (ema(pd.Series(gated, index=frame.index), factor_smooth_len)
                    if factor_smooth_len > 1 else pd.Series(gated, index=frame.index))
    bull_final = ((factor_final + 100.0) / 200.0).clip(0.0, 1.0)

    # --- trend state machine ------------------------------------------------
    exit_level = min(trend_exit_threshold, trend_threshold - 1.0)
    factors = factor_final.to_numpy()
    trend_state = np.zeros(size, dtype=int)
    state = 0
    for bar in range(size):
        value = factors[bar]
        if value >= trend_threshold:
            state = 1
        elif value <= -trend_threshold:
            state = -1
        elif state == 1 and value > exit_level:
            state = 1
        elif state == -1 and value < -exit_level:
            state = -1
        else:
            state = 0
        trend_state[bar] = state
    is_bull = trend_state == 1
    is_bear = trend_state == -1
    is_strong_bull = factors >= strong_threshold
    is_strong_bear = factors <= -strong_threshold
    is_neutral = ~is_bull & ~is_bear
    state_text = np.where(
        is_strong_bull, "Strong Bull",
        np.where(is_bull, "Bull Trend",
                 np.where(is_strong_bear, "Strong Bear",
                          np.where(is_bear, "Bear Trend", "Neutral / Transition"))))

    # --- pullback / continuation -------------------------------------------
    pb_ema = ema(close, pb_ema_len)
    pb_rsi = rsi(close, pb_rsi_len)
    pb_impulse_high = highest(high.shift(1), pb_impulse_lookback)
    pb_impulse_low = lowest(low.shift(1), pb_impulse_lookback)
    conf_ok = np.ones(size, dtype=bool) if conf_threshold <= 0 else confidence >= conf_threshold
    bull_trend_ok = enable_pullback & is_bull & conf_ok & (bull_final.to_numpy() > (1 - bull_final).to_numpy()) \
        & (exhaustion_score < 0.80)
    bear_trend_ok = enable_pullback & is_bear & conf_ok & ((1 - bull_final).to_numpy() > bull_final.to_numpy()) \
        & (exhaustion_score < 0.80)

    previous_close = close.shift(1).to_numpy()
    previous_ema = pb_ema.shift(1).to_numpy()
    previous_fair = fair_path.shift(1).to_numpy()
    ema_now = pb_ema.to_numpy()
    fair_now = fair_path.to_numpy()
    atr_now = safe_atr.to_numpy()
    bull_was_above = (previous_close > previous_ema) & (previous_close > previous_fair)
    bear_was_below = (previous_close < previous_ema) & (previous_close < previous_fair)
    bull_zone = (lows <= ema_now) & (lows <= fair_now + atr_now * 0.25) \
        & (lows >= fair_now - atr_now * 0.75)
    bear_zone = (highs >= ema_now) & (highs >= fair_now - atr_now * 0.25) \
        & (highs <= fair_now + atr_now * 0.75)
    touch_bull = bull_trend_ok & bull_was_above & bull_zone
    touch_bear = bear_trend_ok & bear_was_below & bear_zone

    out = {name: np.zeros(size, dtype=bool) for name in
           ("bull_pb_start", "bear_pb_start", "bull_continuation", "bear_continuation",
            "bull_pb_invalidated", "bear_pb_invalidated")}
    out["bull_pb_state"] = np.zeros(size, dtype=int)
    out["bear_pb_state"] = np.zeros(size, dtype=int)
    out["pb_depth_atr"] = np.zeros(size)
    out["pb_bars"] = np.zeros(size, dtype=int)
    out["pb_cont_score"] = np.zeros(size)
    out["pb_state"] = np.full(size, "Idle", dtype=object)
    out["pb_reason"] = np.full(size, "", dtype=object)

    bull_state = bear_state = 0
    start_bar = math.nan
    extreme = math.nan
    start_high = math.nan
    start_low = math.nan
    last_bull_cont = None
    last_bear_cont = None
    previous_touch_bull = False
    previous_touch_bear = False
    previous_raw_bull = False
    previous_raw_bear = False
    rsi_values = pb_rsi.to_numpy()

    for bar in range(size):
        # terminal states reset at the start of the next bar
        if bull_state in (3, 4):
            bull_state = 0
        if bear_state in (3, 4):
            bear_state = 0

        bull_start = bool(enable_pullback and bull_state == 0 and touch_bull[bar]
                          and not previous_touch_bull)
        bear_start = bool(enable_pullback and bear_state == 0 and touch_bear[bar]
                          and not previous_touch_bear)
        if bull_start:
            bull_state = 1
            bear_state = 0
            start_bar = bar
            extreme = lows[bar]
            start_high = pb_impulse_high.iloc[bar] if np.isfinite(pb_impulse_high.iloc[bar]) else highs[bar]
            start_low = lows[bar]
        if bear_start:
            bear_state = 1
            bull_state = 0
            start_bar = bar
            extreme = highs[bar]
            start_high = highs[bar]
            start_low = pb_impulse_low.iloc[bar] if np.isfinite(pb_impulse_low.iloc[bar]) else lows[bar]

        in_bull = bull_state in (1, 2)
        in_bear = bear_state in (1, 2)
        if in_bull and np.isfinite(extreme):
            extreme = min(extreme, lows[bar])
        if in_bear and np.isfinite(extreme):
            extreme = max(extreme, highs[bar])

        bars_in = 0 if not np.isfinite(start_bar) else bar - int(start_bar)
        bull_depth = max(0.0, (start_high - extreme) / atr_now[bar]) if in_bull and np.isfinite(extreme) else 0.0
        bear_depth = max(0.0, (extreme - start_low) / atr_now[bar]) if in_bear and np.isfinite(extreme) else 0.0
        depth = bull_depth if in_bull else bear_depth if in_bear else 0.0

        bull_struct_ok = bool(in_bull and (not np.isfinite(last_low[bar]) or closes[bar] > last_low[bar]))
        bear_struct_ok = bool(in_bear and (not np.isfinite(last_high[bar]) or closes[bar] < last_high[bar]))
        bull_timeout = bool(in_bull and bars_in >= pb_timeout)
        bear_timeout = bool(in_bear and bars_in >= pb_timeout)
        bull_too_deep = bool(in_bull and bull_depth > pb_max_depth_atr)
        bear_too_deep = bool(in_bear and bear_depth > pb_max_depth_atr)

        trend_score = _norm01(abs(factors[bar]), trend_threshold, strong_threshold) * 3.0
        depth_score = 2.0 if pb_min_depth_atr <= depth <= pb_max_depth_atr else (
            1.0 if 0.0 < depth < pb_min_depth_atr else 0.0)
        structure_score = 2.0 if (bull_struct_ok or bear_struct_ok) else 0.0
        weakness_score = ((1.0 if ((in_bull and rsi_values[bar] >= pb_rsi_bull_floor)
                                   or (in_bear and rsi_values[bar] <= pb_rsi_bear_ceil)) else 0.0)
                          + (1.0 if exhaustion_score[bar] <= 0.60 else 0.0)
                          + (1.0 if vol_optimal[bar] >= 0.35 else 0.0))
        continuation_score = (_clamp(trend_score + depth_score + structure_score
                                     + weakness_score, 0.0, 10.0) if (in_bull or in_bear) else 0.0)

        bull_mature = bool(in_bull and bars_in >= pb_min_bars
                           and pb_min_depth_atr <= bull_depth <= pb_max_depth_atr)
        bear_mature = bool(in_bear and bars_in >= pb_min_bars
                           and pb_min_depth_atr <= bear_depth <= pb_max_depth_atr)
        if bull_state == 1 and bull_mature:
            bull_state = 2
        if bear_state == 1 and bear_mature:
            bear_state = 2

        bull_reclaim = bool(bull_mature and closes[bar] > ema_now[bar]
                            and previous_close[bar] <= previous_ema[bar])
        bear_reclaim = bool(bear_mature and closes[bar] < ema_now[bar]
                            and previous_close[bar] >= previous_ema[bar])
        bull_micro_bos = bool(bull_mature and bar > 0 and closes[bar] > highs[bar - 1]
                              and closes[bar] > opens[bar])
        bear_micro_bos = bool(bear_mature and bar > 0 and closes[bar] < lows[bar - 1]
                              and closes[bar] < opens[bar])
        bull_reject = bool(bull_mature and lower_wick_ratio.iloc[bar] >= pb_wick_min
                           and closes[bar] > opens[bar] and close_position.iloc[bar] > 0.55)
        bear_reject = bool(bear_mature and upper_wick_ratio.iloc[bar] >= pb_wick_min
                           and closes[bar] < opens[bar] and close_position.iloc[bar] < 0.45)

        bull_cooldown_ok = last_bull_cont is None or bar - last_bull_cont >= pb_cooldown
        bear_cooldown_ok = last_bear_cont is None or bar - last_bear_cont >= pb_cooldown
        raw_bull = bool(enable_pullback and bull_trend_ok[bar] and bull_struct_ok
                        and not bull_timeout and bull_cooldown_ok
                        and continuation_score >= pb_min_score
                        and (bull_reclaim or bull_micro_bos or bull_reject))
        raw_bear = bool(enable_pullback and bear_trend_ok[bar] and bear_struct_ok
                        and not bear_timeout and bear_cooldown_ok
                        and continuation_score >= pb_min_score
                        and (bear_reclaim or bear_micro_bos or bear_reject))
        # Every replayed bar is closed, so Pine's `barstate.isconfirmed` gate always passes.
        bull_cont = bool(raw_bull and not previous_raw_bull)
        bear_cont = bool(raw_bear and not previous_raw_bear)
        out["bull_pb_start"][bar] = bull_start
        out["bear_pb_start"][bar] = bear_start
        out["bull_continuation"][bar] = bull_cont
        out["bear_continuation"][bar] = bear_cont
        out["pb_reason"][bar] = ("Pullback EMA reclaim" if (bull_reclaim or bear_reclaim)
                                 else "One-bar micro-BOS" if (bull_micro_bos or bear_micro_bos)
                                 else "Rejection wick from pullback zone"
                                 if (bull_reject or bear_reject) else "")

        if bull_cont:
            last_bull_cont = bar
            bull_state = 3
            start_bar = extreme = start_high = start_low = math.nan
        if bear_cont:
            last_bear_cont = bar
            bear_state = 3
            start_bar = extreme = start_high = start_low = math.nan

        if bull_state in (1, 2) and (bull_timeout or bull_too_deep or not bull_trend_ok[bar]
                                     or not bull_struct_ok):
            bull_state = 4
            out["bull_pb_invalidated"][bar] = True
            start_bar = extreme = start_high = start_low = math.nan
        if bear_state in (1, 2) and (bear_timeout or bear_too_deep or not bear_trend_ok[bar]
                                     or not bear_struct_ok):
            bear_state = 4
            out["bear_pb_invalidated"][bar] = True
            start_bar = extreme = start_high = start_low = math.nan

        out["bull_pb_state"][bar] = bull_state
        out["bear_pb_state"][bar] = bear_state
        out["pb_depth_atr"][bar] = depth
        out["pb_bars"][bar] = bars_in
        out["pb_cont_score"][bar] = continuation_score
        out["pb_state"][bar] = (
            "Bull Cont" if bull_state == 3 else "Bear Cont" if bear_state == 3
            else "PB Invalidated" if bull_state == 4 or bear_state == 4
            else "Bull PB Mature" if bull_state == 2 else "Bear PB Mature" if bear_state == 2
            else "Bull PB Armed" if bull_state == 1 else "Bear PB Armed" if bear_state == 1
            else "Idle")
        previous_touch_bull = bool(touch_bull[bar])
        previous_touch_bear = bool(touch_bear[bar])
        previous_raw_bull = raw_bull
        previous_raw_bear = raw_bear

    # --- smoothed trend candles --------------------------------------------
    candle_open = _ma((open_.shift(1) + close.shift(1)) / 2.0, candle_open_len, candle_open_type)
    candle_high_raw = _ma(pd.concat([high, close], axis=1).max(axis=1), candle_high_len, "EMA")
    candle_low_raw = _ma(pd.concat([low, close], axis=1).min(axis=1), candle_low_len, "EMA")
    candle_close = _ma((high + low + close + open_) / 4.0, candle_close_len, candle_close_type)
    candle_high = pd.concat([candle_high_raw, candle_open, candle_close], axis=1).max(axis=1)
    candle_low = pd.concat([candle_low_raw, candle_open, candle_close], axis=1).min(axis=1)
    candle_ma = _ma((high + low) / 2.0, candle_ma_len, candle_ma_type)

    index = frame.index
    data: dict[str, pd.Series] = {
        "trend_factor": factor_final,
        "confidence": pd.Series(confidence, index=index),
        "bull_prob": bull_final,
        "bear_prob": 1.0 - bull_final,
        "raw_bull_prob": pd.Series(bull_prob, index=index),
        "trend_state": pd.Series(trend_state, index=index).astype("int64"),
        "state": pd.Series(state_text, index=index, dtype=object),
        "is_bull": pd.Series(is_bull, index=index, dtype=bool),
        "is_bear": pd.Series(is_bear, index=index, dtype=bool),
        "is_strong_bull": pd.Series(is_strong_bull, index=index, dtype=bool),
        "is_strong_bear": pd.Series(is_strong_bear, index=index, dtype=bool),
        "is_neutral": pd.Series(is_neutral, index=index, dtype=bool),
        "direction_evidence": direction_evidence,
        "strength_score": strength_score,
        "strength_evidence": pd.Series(strength_evidence, index=index),
        "structure_evidence": pd.Series(structure_evidence, index=index),
        "structure_freshness": pd.Series(structure_freshness, index=index),
        "vol_optimal": pd.Series(vol_optimal, index=index),
        "vol_quality": pd.Series(vol_optimal, index=index),
        "quality_multiplier": pd.Series(quality_multiplier, index=index),
        "exhaustion_score": pd.Series(exhaustion_score, index=index),
        "atr_rank": atr_rank,
        "fair_path": fair_path,
        "fair_band_upper": fair_path + atr_values * 2.0,
        "fair_band_lower": fair_path - atr_values * 2.0,
        "distance_atr": distance_atr,
        "acceleration": acceleration,
        "efficiency": efficiency,
        "r2": r2,
        "adx": adx,
        "pb_ema": pb_ema,
        "pb_rsi": pb_rsi,
        "pb_state": pd.Series(out["pb_state"], index=index, dtype=object),
        "bull_pb_state": pd.Series(out["bull_pb_state"], index=index).astype("int64"),
        "bear_pb_state": pd.Series(out["bear_pb_state"], index=index).astype("int64"),
        "bull_pb_start": pd.Series(out["bull_pb_start"], index=index, dtype=bool),
        "bear_pb_start": pd.Series(out["bear_pb_start"], index=index, dtype=bool),
        "bull_continuation": pd.Series(out["bull_continuation"], index=index, dtype=bool),
        "bear_continuation": pd.Series(out["bear_continuation"], index=index, dtype=bool),
        "bull_pb_invalidated": pd.Series(out["bull_pb_invalidated"], index=index, dtype=bool),
        "bear_pb_invalidated": pd.Series(out["bear_pb_invalidated"], index=index, dtype=bool),
        "pb_depth_atr": pd.Series(out["pb_depth_atr"], index=index),
        "pb_bars": pd.Series(out["pb_bars"], index=index).astype("int64"),
        "pb_cont_score": pd.Series(out["pb_cont_score"], index=index),
        "pb_reason": pd.Series(out["pb_reason"], index=index, dtype=object),
        "trend_candle_open": candle_open,
        "trend_candle_high": candle_high,
        "trend_candle_low": candle_low,
        "trend_candle_close": candle_close,
        "trend_candle_ma": candle_ma,
        "show_trend_candles": pd.Series(show_trend_candles, index=index),
    }
    return pd.DataFrame(data, index=index)
