"""JMA Cluster Entries with Market Structure v2.5.1 的 Python 移植。

核心思路：**不预测方向，只在结构极端处等反转。** 把 10 条不同周期的 JMA（20…600）
当成一个「均线簇」，用「收盘价在多少条 JMA 之上」得到一个 0–100 的簇分数；分数在
极端时出现**分数反转**（而价格还在创新高/新低）就是衰竭的证据。

三条独立的信号类
----------------
* **MOMENTUM** —— 簇分数从极端回落（``revExtreme`` → ``revReturn``）且价格偏离
  慢速 JMA 超过 ``devMult`` 个标准差。
* **EXHAUSTION** —— 分数仍高但**不再创新高**（背离）+ 效率下滑 + 拒绝影线，
  需要下一根确认。
* **STRUCTURE** —— 吸收（放量但走不出距离）配合拒绝影线。

再加一条独立的 **TREND** 路径：簇翻多/翻空 + 斜率/斜率方向/分数跳变/ATR 距离
四道过滤，用于趋势模式。

关键设计点
----------
* **簇分数是 10 条 JMA 的跨界比例**，不是加权平均 —— 它天然是 0/10…10/10 的阶梯，
  所以「跳变步数」（``score_jump_steps``）等于**这一根有几条 JMA 被穿过**。这正是
  过滤器想要的量：一步＝一条 JMA，与启用了几条无关。
* **斜率用线性回归**而不是首尾差 —— 首尾差会被两端的单根噪声主导。
* **吸收（effort vs result）必须有真实成交量**：``range/volume`` 下降 = 放量滞涨。
  无量能时该信号一律不出（不是恒真也不是恒假，而是明确不判定）。
* **两级冷却**：告警冷却（10 根）与入场冷却（``signal_cooldown``，默认 20 根）分开，
  预警可以比入场频繁。

未移植：Wyckoff 相位与事件（SC/BC/Spring/Upthrust 检测，约 450 行）、SMC 订单块 /
FVG / Breaker、双枢轴 ZigZag 与预测区、质量加分/扣分模型（``calculateEntryQuality``，
约 300 行的实验分支）、绘图与日志。因此 ``structure_long/short`` 只走**吸收**路径，
不包含 Wyckoff spring/upthrust 成因。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    ema,
    highestbars,
    integer,
    lowestbars,
    number,
    ohlcv,
    rsi,
    sma,
    stdev,
    stoch,
)

__all__ = ["jma_cluster_entries_with_market_structure"]

_JMA_LENGTHS = (20, 50, 100, 150, 200, 250, 300, 400, 500, 600)
_SLOW_OPTIONS = ("20", "50", "100", "150", "200", "250", "300", "400", "500", "600")
_MODES = ("Trend", "Reversion")

_ENTRY_NONE, _ENTRY_TREND, _ENTRY_STRUCTURE, _ENTRY_EXHAUSTION, _ENTRY_MOMENTUM = 0, 1, 2, 3, 4


def _jma(values: np.ndarray, length: int, phase: float, power: float) -> np.ndarray:
    """``calculateJMA``：JMA 的闭式近似（phase 映射 + beta 递推）。"""
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    out = np.full(len(values), np.nan)
    e0 = e1 = e2 = jma = math.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            out[i] = jma
            continue
        prev_e0 = e0 if np.isfinite(e0) else x
        prev_e1 = e1 if np.isfinite(e1) else 0.0
        prev_jma = jma if np.isfinite(jma) else x
        prev_e2 = e2 if np.isfinite(e2) else 0.0
        e0 = (1.0 - alpha) * x + alpha * prev_e0
        e1 = (x - e0) * (1.0 - beta) + beta * prev_e1
        e2 = (e0 + phase_ratio * e1 - prev_jma) * (1.0 - alpha) ** 2 + alpha ** 2 * prev_e2
        jma = e2 + prev_jma
        out[i] = jma
    return out


def _linreg_slope(values: np.ndarray, length: int) -> np.ndarray:
    """滚动线性回归斜率（首尾差会被两端单根噪声主导，回归不会）。"""
    out = np.full(len(values), np.nan)
    if length < 2:
        raise ValueError("slope_length 至少为 2")
    for i in range(length - 1, len(values)):
        window = values[i - length + 1:i + 1]
        if not np.isfinite(window).all():
            continue
        x = np.arange(length, dtype=float)
        sum_x = x.sum()
        sum_y = window.sum()
        sum_xy = float((x * window).sum())
        sum_x2 = float((x * x).sum())
        denominator = length * sum_x2 - sum_x * sum_x
        if denominator != 0:
            out[i] = (length * sum_xy - sum_x * sum_y) / denominator
    return out


def _bars_since_extreme(values: np.ndarray, length: int, highest: bool) -> np.ndarray:
    """``-ta.highestbars/lowestbars``：距窗口内极值的 bar 数（0 = 本根即极值）。"""
    series = pd.Series(values)
    if highest:
        return (-highestbars(series, length)).to_numpy(dtype=float)
    return (-lowestbars(series, length)).to_numpy(dtype=float)


def jma_cluster_entries_with_market_structure(
    frame: pd.DataFrame,
    *,
    source: str = "close",
    phase: float = 0.0,
    power: float = 2.0,
    mode: str = "Trend",
    enabled_jmas: tuple[bool, ...] | None = None,
    cluster_pct: int = 75,
    slow_length: str = "200",
    slope_length: int = 5,
    use_min_slope: bool = True,
    min_slope_pct: float = 0.02,
    use_score_jump: bool = True,
    min_score_jump: int = 2,
    signal_cooldown: int = 20,
    use_atr_filter: bool = True,
    atr_length: int = 14,
    atr_max_mult: float = 3.5,
    require_slope_direction: bool = False,
    rev_extreme: int = 92,
    rev_return: int = 78,
    dev_length: int = 100,
    dev_mult: float = 1.5,
    rsi_length: int = 14,
    wt_channel_length: int = 9,
    wt_average_length: int = 12,
    alert_cooldown: int = 10,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """JMA Cluster Entries：簇分数 + 动量/衰竭/结构三类信号 + 冷却门。"""
    mode = choice("mode", mode, _MODES)
    slow_length = choice("slow_length", slow_length, _SLOW_OPTIONS)
    phase = number("phase", phase, -100.0, 100.0)
    power = number("power", power, 1.0, 4.0)
    cluster_pct = integer("cluster_pct", cluster_pct, 50, 100)
    slope_length = integer("slope_length", slope_length, 1, 50)
    min_slope_pct = number("min_slope_pct", min_slope_pct, 0.0, 1.0)
    min_score_jump = integer("min_score_jump", min_score_jump, 1, 10)
    signal_cooldown = integer("signal_cooldown", signal_cooldown, 1, 500)
    atr_length = integer("atr_length", atr_length, 1, 200)
    atr_max_mult = number("atr_max_mult", atr_max_mult, 0.1, 50.0)
    rev_extreme = integer("rev_extreme", rev_extreme, 80, 99)
    rev_return = integer("rev_return", rev_return, 50, 95)
    dev_length = integer("dev_length", dev_length, 10, 500)
    dev_mult = number("dev_mult", dev_mult, 0.1, 20.0)
    rsi_length = integer("rsi_length", rsi_length, 5, 50)
    wt_channel_length = integer("wt_channel_length", wt_channel_length, 5, 20)
    wt_average_length = integer("wt_average_length", wt_average_length, 5, 30)
    alert_cooldown = integer("alert_cooldown", alert_cooldown, 1, 200)

    open_s, high_s, low_s, close_s, volume_s = ohlcv(frame, "open", "high", "low", "close", "volume")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    sources = {"close": close, "open": open_a, "hl2": (high + low) / 2.0,
               "hlc3": (high + low + close) / 3.0, "ohlc4": (open_a + high + low + close) / 4.0}
    if source not in sources:
        raise ValueError(f"未知的 source：{source!r}")
    src = sources[source]

    if enabled_jmas is None:
        enabled = tuple(True for _ in _JMA_LENGTHS)
    else:
        if len(enabled_jmas) != len(_JMA_LENGTHS):
            raise ValueError("enabled_jmas 必须提供 10 个布尔值")
        enabled = tuple(bool(flag) for flag in enabled_jmas)
    active_count = sum(enabled)
    if active_count == 0:
        raise ValueError("至少需要启用一条 JMA")

    # --- JMA 簇 -------------------------------------------------------------
    jma_values = [_jma(src, length, phase, power) for length in _JMA_LENGTHS]
    bull_count = np.zeros(n, dtype=np.int64)
    for flag, values in zip(enabled, jma_values):
        if flag:
            bull_count += (close >= values).astype(np.int64)
    score = bull_count * 100.0 / active_count
    score_prev = np.r_[np.nan, score[:-1]]

    cluster_bull = score >= cluster_pct
    cluster_bear = score <= (100 - cluster_pct)
    flip_bull = cluster_bull & ~np.r_[False, cluster_bull[:-1]]
    flip_bear = cluster_bear & ~np.r_[False, cluster_bear[:-1]]
    score_jump = np.abs(score - np.nan_to_num(score_prev, nan=score[0]))
    score_jump_steps = np.round(score_jump * active_count / 100.0)

    slow_j = jma_values[_SLOW_OPTIONS.index(slow_length)]
    linreg = _linreg_slope(slow_j, slope_length)
    slope_pct = np.where(slow_j != 0, linreg / np.where(slow_j != 0, slow_j, 1.0) * 100.0, 0.0)
    slope_pct = np.where(np.isfinite(linreg), slope_pct, 0.0)
    slope_delta = slope_pct - np.r_[0.0, slope_pct[:-1]]
    slope_up = slope_pct > 0
    slope_down = slope_pct < 0

    # --- 市场指标 -----------------------------------------------------------
    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    atr1 = atr(frame, 14).to_numpy(dtype=float)
    volume_average20 = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), 20).to_numpy(dtype=float)
    # 无量能时（均量为 0）相对量能无定义 —— 保持 na 而不是 0，让下游比较自然为假
    rel_volume = np.where(volume_average20 > 0, volume / np.where(volume_average20 > 0, volume_average20, 1.0), np.nan)
    if volume_is_real is None:
        volume_is_real = bool(np.isfinite(volume).any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)

    range_size = high - low
    body_size = np.abs(close - open_a)
    lower_wick = np.minimum(open_a, close) - low
    upper_wick = high - np.maximum(open_a, close)
    safe_range = np.where(range_size > 0, range_size, np.nan)
    lower_wick_pct = np.where(range_size > 0, lower_wick / safe_range * 100.0, 0.0)
    upper_wick_pct = np.where(range_size > 0, upper_wick / safe_range * 100.0, 0.0)
    clv = np.where(range_size > 0, ((close - low) - (high - close)) / safe_range, 0.0)
    effort_result = np.where(volume > 0, range_size / np.where(volume > 0, volume, 1.0), 0.0)

    rejection_long = lower_wick_pct > 60.0
    rejection_short = upper_wick_pct > 60.0
    bar_efficiency = np.where(atr_a > 0, body_size / np.where(atr_a > 0, atr_a, 1.0), 0.0)
    bar_efficiency_ma = sma(pd.Series(bar_efficiency, index=idx), 5).to_numpy(dtype=float)
    bar_efficiency_falling = bar_efficiency < bar_efficiency_ma

    # ta.falling(effortResult, 3)：连续 3 根都比各自的前一根更低
    effort_falling = np.ones(n, dtype=bool)
    for step in range(1, 4):
        shifted = np.r_[np.full(step, np.nan), effort_result[:-step]] if step < n else np.full(n, np.nan)
        effort_falling &= effort_result < shifted
    absorption_active = np.asarray(volume_is_real) & effort_falling & (rel_volume > 1.2)

    bars_since_high = _bars_since_extreme(high, 20, True)
    bars_since_low = _bars_since_extreme(low, 20, False)

    previous_close = np.r_[np.nan, close[:-1]]
    progress = np.abs(close - previous_close)
    efficiency = np.where(atr_a > 0, progress / np.where(atr_a > 0, atr_a, 1.0), 0.0)
    # Pine 按图表周期选 3/5/7/10；批量调用只有一种周期，用中位档 5
    efficiency_length = 5
    efficiency_sma = sma(pd.Series(efficiency, index=idx), efficiency_length).to_numpy(dtype=float)
    efficiency_falling = efficiency < efficiency_sma

    dev = stdev(pd.Series(close - slow_j, index=idx), dev_length).to_numpy(dtype=float)
    dev_ok_long = (close - slow_j) < -dev_mult * dev
    dev_ok_short = (close - slow_j) > dev_mult * dev

    rsi_value = rsi(close_s, rsi_length).to_numpy(dtype=float)
    rsi_for_stoch = rsi(close_s, 14)
    stoch_rsi_k_arr = stoch(rsi_for_stoch, rsi_for_stoch, rsi_for_stoch, 14).to_numpy(dtype=float)
    ap = (high + low + close) / 3.0
    esa = ema(pd.Series(ap, index=idx), wt_channel_length).to_numpy(dtype=float)
    wt_dev = ema(pd.Series(np.abs(ap - esa), index=idx), wt_channel_length).to_numpy(dtype=float)
    wt_ci = np.where(wt_dev != 0, (ap - esa) / (0.015 * np.where(wt_dev != 0, wt_dev, 1.0)), 0.0)
    wt1 = ema(pd.Series(wt_ci, index=idx), wt_average_length).to_numpy(dtype=float)

    # --- 冲击区 & 确认 -------------------------------------------------------
    shock_bar = np.asarray(volume_is_real) & (range_size > atr1 * 1.8) & (rel_volume > 1.4)
    previous_low = np.r_[np.nan, low[:-1]]
    previous_high = np.r_[np.nan, high[:-1]]
    confirm_long = (low >= previous_low) & (clv > 0.1)
    confirm_short = (high <= previous_high) & (clv < -0.1)

    # --- 逐 bar 信号 --------------------------------------------------------
    flags = {name: np.zeros(n, dtype=np.int64) for name in (
        "trend_long", "trend_short", "rev_long", "rev_short",
        "exhaustion_long_raw", "exhaustion_short_raw",
        "exhaustion_long", "exhaustion_short",
        "failed_continue_long", "failed_continue_short",
        "structure_long", "structure_short",
        "entry_long", "entry_short", "alert_long", "alert_short",
        "momentum_long", "momentum_short",
        "exhaustion_long_trigger", "exhaustion_short_trigger",
    )}
    score_delta_long = np.zeros(n)
    score_delta_short = np.zeros(n)
    quality_long = np.zeros(n)
    quality_short = np.zeros(n)
    class_long = np.zeros(n, dtype=np.int64)
    class_short = np.zeros(n, dtype=np.int64)

    bars_since_long_entry = 999
    bars_since_short_entry = 999
    bars_since_long_alert = 999
    bars_since_short_alert = 999
    exhaustion_long_raw = False
    exhaustion_short_raw = False
    exhaustion_long_bar = -1
    exhaustion_short_bar = -1
    failure_threshold = 5
    buildup_bearish_active = False
    buildup_bullish_active = False
    buildup_bearish_bar = -1
    buildup_bullish_bar = -1
    buildup_timeout = 40
    alert_bar_short = -1
    alert_bar_long = -1
    score_at_alert_short = math.nan
    score_at_alert_long = math.nan

    score_highest5 = pd.Series(score, index=idx).rolling(5, min_periods=1).max().to_numpy(dtype=float)
    score_lowest5 = pd.Series(score, index=idx).rolling(5, min_periods=1).min().to_numpy(dtype=float)
    prior_highest10 = pd.Series(high, index=idx).rolling(10, min_periods=10).max().shift(1).to_numpy(dtype=float)
    prior_lowest10 = pd.Series(low, index=idx).rolling(10, min_periods=10).min().shift(1).to_numpy(dtype=float)

    for i in range(n):
        bars_since_long_entry += 1
        bars_since_short_entry += 1
        bars_since_long_alert += 1
        bars_since_short_alert += 1
        entry_ok_long = bars_since_long_entry >= signal_cooldown
        entry_ok_short = bars_since_short_entry >= signal_cooldown
        alert_ok_long = bars_since_long_alert >= alert_cooldown
        alert_ok_short = bars_since_short_alert >= alert_cooldown

        slope_ok_long = (not use_min_slope) or slope_pct[i] >= min_slope_pct
        slope_ok_short = (not use_min_slope) or slope_pct[i] <= -min_slope_pct
        jump_ok = (not use_score_jump) or score_jump_steps[i] >= min_score_jump
        dir_ok_long = (not require_slope_direction) or slope_up[i]
        dir_ok_short = (not require_slope_direction) or slope_down[i]
        dist_ok = (not use_atr_filter) or abs(close[i] - slow_j[i]) <= atr_max_mult * atr_a[i]

        stoch_rsi_k = stoch_rsi_k_arr[i]
        buildup_bearish_cond = bool(score[i] >= 60.0 and (score[i] >= 90.0 or stoch_rsi_k > 70.0
                                                          or rsi_value[i] > 65.0 or wt1[i] > 50.0))
        buildup_bullish_cond = bool(score[i] <= 40.0 and (score[i] <= 10.0 or stoch_rsi_k < 30.0
                                                          or rsi_value[i] < 35.0 or wt1[i] < -50.0))

        trend_long = bool(flip_bull[i]) and dir_ok_long and slope_ok_long and jump_ok and dist_ok and entry_ok_long
        trend_short = bool(flip_bear[i]) and dir_ok_short and slope_ok_short and jump_ok and dist_ok and entry_ok_short

        extreme_bull = score[i] >= rev_extreme
        extreme_bear = score[i] <= (100 - rev_extreme)
        previous_extreme_bull = i > 0 and score[i - 1] >= rev_extreme
        previous_extreme_bear = i > 0 and score[i - 1] <= (100 - rev_extreme)
        rev_short = previous_extreme_bull and score[i] < rev_return and bool(dev_ok_short[i])
        rev_long = previous_extreme_bear and score[i] > (100 - rev_return) and bool(dev_ok_long[i])

        # buildup 状态机：条件成立且未在 buildup 中 → 进入并记 bar；仅超时才复位
        if buildup_bearish_cond and not buildup_bearish_active and alert_ok_short:
            buildup_bearish_active = True
            buildup_bearish_bar = i
        if buildup_bullish_cond and not buildup_bullish_active and alert_ok_long:
            buildup_bullish_active = True
            buildup_bullish_bar = i
        if buildup_bearish_active and buildup_bearish_bar >= 0 and i - buildup_bearish_bar > buildup_timeout:
            buildup_bearish_active = False
        if buildup_bullish_active and buildup_bullish_bar >= 0 and i - buildup_bullish_bar > buildup_timeout:
            buildup_bullish_active = False

        # 告警 = 「正在走向极端」的 buildup 条件 + 冷却，不是「已经在极端」。
        # Pine 的原意是给入场留出提前量：分数 >= 60 且振荡器确认超买（或已 >= 90）。
        buildup_bearish = buildup_bearish_cond
        buildup_bullish = buildup_bullish_cond
        if buildup_bearish and alert_ok_short:
            bars_since_short_alert = 0
            flags["alert_short"][i] = 1
            alert_bar_short = i
            score_at_alert_short = score[i]
        if buildup_bullish and alert_ok_long:
            bars_since_long_alert = 0
            flags["alert_long"][i] = 1
            alert_bar_long = i
            score_at_alert_long = score[i]

        # 入场窗口：告警后 15 根内（不含告警那根），且分数相对告警时反向移动 >= 15
        in_window_short = (alert_bar_short >= 0 and 0 < i - alert_bar_short <= 15)
        in_window_long = (alert_bar_long >= 0 and 0 < i - alert_bar_long <= 15)
        delta_from_alert_short = (score[i] - score_at_alert_short) if alert_bar_short >= 0 else 0.0
        delta_from_alert_long = (score[i] - score_at_alert_long) if alert_bar_long >= 0 else 0.0
        reversal_short = (buildup_bearish_active and delta_from_alert_short < -15.0
                          and in_window_short and entry_ok_short)
        reversal_long = (buildup_bullish_active and delta_from_alert_long > 15.0
                         and in_window_long and entry_ok_long)
        if reversal_short:
            buildup_bearish_active = False
        if reversal_long:
            buildup_bullish_active = False

        # 衰竭：分数仍高但不再创新高 + 效率下滑 + 拒绝影线
        exhaustion_short = (score[i] >= 65 and high[i] > previous_high[i]
                            and score[i] < score_highest5[i] and efficiency_falling[i]
                            and rejection_short[i] and not shock_bar[i])
        if exhaustion_short:
            exhaustion_short_raw = True
            exhaustion_short_bar = i
            flags["exhaustion_short_trigger"][i] = 1
        exhaustion_long = (score[i] <= 35 and low[i] < previous_low[i]
                           and score[i] > score_lowest5[i] and efficiency_falling[i]
                           and rejection_long[i] and not shock_bar[i])
        if exhaustion_long:
            exhaustion_long_raw = True
            exhaustion_long_bar = i
            flags["exhaustion_long_trigger"][i] = 1

        failed_continue_short = bool(score[i] >= 70 and np.isfinite(bars_since_high[i])
                                     and bars_since_high[i] > failure_threshold
                                     and np.isfinite(prior_highest10[i]) and high[i] <= prior_highest10[i]
                                     and not shock_bar[i])
        failed_continue_long = bool(score[i] <= 30 and np.isfinite(bars_since_low[i])
                                    and bars_since_low[i] > failure_threshold
                                    and np.isfinite(prior_lowest10[i]) and low[i] >= prior_lowest10[i]
                                    and not shock_bar[i])

        # 上一根触发过 raw、本根确认
        exhaustion_short_confirmed = False
        exhaustion_long_confirmed = False
        if i > 0:
            if flags["exhaustion_short_raw"][i - 1] == 1 and confirm_short[i] and entry_ok_short:
                exhaustion_short_confirmed = True
            if flags["exhaustion_long_raw"][i - 1] == 1 and confirm_long[i] and entry_ok_long:
                exhaustion_long_confirmed = True

        flags["exhaustion_short_raw"][i] = 1 if exhaustion_short_raw else 0
        flags["exhaustion_long_raw"][i] = 1 if exhaustion_long_raw else 0
        if exhaustion_short_confirmed or (exhaustion_short_bar >= 0 and i - exhaustion_short_bar > 3):
            exhaustion_short_raw = False
        if exhaustion_long_confirmed or (exhaustion_long_bar >= 0 and i - exhaustion_long_bar > 3):
            exhaustion_long_raw = False

        exhaustion_short_signal = bool(exhaustion_short_confirmed
                                       or (failed_continue_short and confirm_short[i] and entry_ok_short))
        exhaustion_long_signal = bool(exhaustion_long_confirmed
                                      or (failed_continue_long and confirm_long[i] and entry_ok_long))

        structure_long = bool(absorption_active[i] and rejection_long[i] and score[i] >= 60)
        structure_short = bool(absorption_active[i] and rejection_short[i] and score[i] <= 40)

        momentum_long = bool(reversal_long)
        momentum_short = bool(reversal_short)

        # 原始（斜率）质量模型
        def _quality(has_signal: bool, is_short: bool) -> float:
            if not has_signal:
                return 0.0
            value = 50.0
            if is_short:
                if slope_pct[i] < -0.05:
                    value += 20.0
                if slope_pct[i] < -0.2:
                    value += 30.0
                if slope_pct[i] > 0.1:
                    value -= 20.0
            else:
                if slope_pct[i] > 0.05:
                    value += 20.0
                if slope_pct[i] > 0.2:
                    value += 30.0
                if slope_pct[i] < -0.1:
                    value -= 20.0
            return float(min(100.0, max(0.0, value)))

        long_hit = momentum_long or exhaustion_long_signal or structure_long
        short_hit = momentum_short or exhaustion_short_signal or structure_short
        quality_long[i] = _quality(long_hit, False)
        quality_short[i] = _quality(short_hit, True)

        if mode == "Trend":
            entry_long = trend_long and entry_ok_long
            entry_short = trend_short and entry_ok_short
            class_long[i] = _ENTRY_TREND if entry_long else _ENTRY_NONE
            class_short[i] = _ENTRY_TREND if entry_short else _ENTRY_NONE
        else:
            entry_long = long_hit and entry_ok_long
            entry_short = short_hit and entry_ok_short
            class_long[i] = (_ENTRY_STRUCTURE if structure_long else
                             _ENTRY_EXHAUSTION if exhaustion_long_signal else
                             _ENTRY_MOMENTUM if momentum_long else _ENTRY_NONE)
            class_short[i] = (_ENTRY_STRUCTURE if structure_short else
                              _ENTRY_EXHAUSTION if exhaustion_short_signal else
                              _ENTRY_MOMENTUM if momentum_short else _ENTRY_NONE)

        flags["trend_long"][i] = 1 if trend_long else 0
        flags["trend_short"][i] = 1 if trend_short else 0
        flags["rev_long"][i] = 1 if rev_long else 0
        flags["rev_short"][i] = 1 if rev_short else 0
        flags["exhaustion_long"][i] = 1 if exhaustion_long_signal else 0
        flags["exhaustion_short"][i] = 1 if exhaustion_short_signal else 0
        flags["failed_continue_long"][i] = 1 if failed_continue_long else 0
        flags["failed_continue_short"][i] = 1 if failed_continue_short else 0
        flags["structure_long"][i] = 1 if structure_long else 0
        flags["structure_short"][i] = 1 if structure_short else 0
        flags["entry_long"][i] = 1 if entry_long else 0
        flags["entry_short"][i] = 1 if entry_short else 0
        flags["momentum_long"][i] = 1 if momentum_long else 0
        flags["momentum_short"][i] = 1 if momentum_short else 0
        score_delta_long[i] = delta_from_alert_long
        score_delta_short[i] = delta_from_alert_short
        if entry_long:
            bars_since_long_entry = 0
        if entry_short:
            bars_since_short_entry = 0

    def as_int(values) -> pd.Series:
        return pd.Series(np.asarray(values).astype(np.int64), index=idx).astype("Int64")

    data = {
        "jma_score": score,
        "jma_bull_count": as_int(bull_count),
        "jma_active_count": pd.Series(np.full(n, active_count), index=idx).astype("Int64"),
        "jma_slow": slow_j,
        "jma_slope_pct": slope_pct,
        "jma_slope_delta": slope_delta,
        "jma_score_jump": score_jump,
        "jma_score_jump_steps": score_jump_steps,
        "jma_cluster_bull": as_int(cluster_bull),
        "jma_cluster_bear": as_int(cluster_bear),
        "jma_cluster_flip_bull": as_int(flip_bull),
        "jma_cluster_flip_bear": as_int(flip_bear),
        "jma_atr": atr_a,
        "jma_atr14": atr1,
        "jma_rel_volume": rel_volume,
        "jma_lower_wick_pct": lower_wick_pct,
        "jma_upper_wick_pct": upper_wick_pct,
        "jma_clv": clv,
        "jma_effort_result": effort_result,
        "jma_rejection_long": as_int(rejection_long),
        "jma_rejection_short": as_int(rejection_short),
        "jma_bar_efficiency": bar_efficiency,
        "jma_bar_efficiency_falling": as_int(bar_efficiency_falling),
        "jma_absorption_active": as_int(absorption_active),
        "jma_bars_since_high": bars_since_high,
        "jma_bars_since_low": bars_since_low,
        "jma_progress": progress,
        "jma_efficiency": efficiency,
        "jma_efficiency_falling": as_int(efficiency_falling),
        "jma_deviation": dev,
        "jma_rsi": rsi_value,
        "jma_wavetrend": wt1,
        "jma_shock_bar": as_int(shock_bar),
        "jma_confirm_long": as_int(confirm_long),
        "jma_confirm_short": as_int(confirm_short),
        "jma_quality_long": quality_long,
        "jma_quality_short": quality_short,
        "jma_entry_class_long": as_int(class_long),
        "jma_entry_class_short": as_int(class_short),
    }
    for name, values in flags.items():
        data["jma_" + name] = as_int(values)
    for k, length in enumerate(_JMA_LENGTHS):
        data[f"jma_value_{length}"] = jma_values[k]
    return pd.DataFrame(data, index=idx)
