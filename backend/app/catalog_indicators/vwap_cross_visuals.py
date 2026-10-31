"""VWAP Cross Visuals [WavesUnchained] v2.3.1 的 Python 移植。

原脚本是 4 个内联模块的合成体（VWAP 核心 / Zone Management / Volume Profile /
HTF Stack）。这里移植其中**全部数值计算**，去掉纯绘图：

* **VWAP 核心** —— 会话 VWAP（每日重置）、枢轴锚定 VWAP High/Low、价格×VWAP 交叉、
  VWAP 结构交叉（会话 vs 锚定）、多 VWAP 聚集区、信号节流、三种强度、偏置评分。
* **偏置带与目标尺度** —— VWAP ± (ATR 或 σ) × 倍数 → 三态偏置；ATR/σ 目标档位。
* **成交量分布** —— 滚动窗口分箱的 POC / VAH / VAL / HVN / LVN。
* **HTF 栈** —— 按图表周期自动选 4 个高周期，各自 EMA50 斜率（取**上一根已完成**
  HTF bar，避免未来函数）→ 加权一致率、震荡指数、市场状态、栈信号。

未移植（纯视觉层，需 `request.security` 逐 TF 取枢轴/区间）
--------------------------------------------------------
Zone Management 的**箱体生命周期**：4 条建区路径（枢轴 / 订单块 / VWAP 聚集 /
FVG）+ 打分排序 + 裁剪 + 触摸/失效着色 + 最近区选择 + 入场确认链。这层本质是
「画框 + 排序」，且 HTF 建区依赖逐 TF 的枢轴，单帧无法复现；其偏置带与目标档位
（数值部分）已单独移植。

数据契约：VWAP 必须有真实成交量。``volume_is_real=False``（或成交量缺失/全 0）时，
整个 VWAP / 分布 / 偏置层一律输出 NaN 而不是在假数据上算出「看起来像信号」的曲线。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, ema, integer, number, ohlcv, pivothigh, pivotlow, stdev

__all__ = ["vwap_cross_visuals"]

_DISPLAY_MODES = ("Smart", "Price Only", "Cluster Only")
_SIGNAL_SCOPES = ("VWAP Cross only", "All signals")
_PROFILE_MODES = ("FIXED", "ROLLING", "SESSIONS")
_TARGET_MODES = ("ATR", "SIGMA", "LIQUIDITY")

_HTF_WEIGHTS = (0.20, 0.25, 0.30, 0.25)


def _tf_seconds(token: str) -> int:
    """把 Pine 的周期记号（"15" / "60" / "240" / "D" / "W" / "M"）换算成秒。"""
    text = str(token).strip().upper()
    table = {"D": 86400, "W": 604800, "M": 2592000, "3M": 7776000}
    if text in table:
        return table[text]
    return int(float(text)) * 60


def _auto_timeframes(bar_seconds: float) -> tuple[int, ...]:
    """``getOptimalTimeframes``：按图表周期选 4 个更高周期（秒）。"""
    if bar_seconds <= 300:
        return (300, 900, 3600, 14400)
    if bar_seconds <= 900:
        return (900, 3600, 14400, 86400)
    if bar_seconds <= 3600:
        return (3600, 14400, 86400, 604800)
    if bar_seconds <= 14400:
        return (14400, 86400, 604800, 2592000)
    return (86400, 604800, 2592000, 7776000)


def _bar_seconds(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 3600.0
    deltas = pd.Series(index).diff().dt.total_seconds().dropna()
    median = float(deltas.median()) if len(deltas) else 3600.0
    return median if median > 0 else 3600.0


def _safe_div(numerator, denominator, fallback: float = math.nan) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.isfinite(den) & (den != 0.0)
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _htf_slope(frame: pd.DataFrame, target_seconds: int, length: int = 50, lookback: int = 5) -> np.ndarray:
    """把收盘价重采样到目标周期后算 EMA 斜率，并取**上一根已完成** bar 的值。"""
    idx = pd.DatetimeIndex(frame.index)
    if target_seconds >= 28 * 86400:
        key = pd.PeriodIndex(idx, freq="M")
    elif target_seconds >= 6 * 86400:
        key = pd.PeriodIndex(idx, freq="W")
    elif target_seconds >= 86400:
        key = pd.PeriodIndex(idx, freq="D")
    else:
        key = idx.floor(pd.Timedelta(seconds=int(target_seconds)))
    grouped = frame.groupby(key)["close"].last()
    line = ema(grouped, length)
    prior = line.shift(lookback)
    slope = (line - prior) / prior.replace(0, np.nan) * 100.0
    slope = slope.shift(1)  # 只用上一根已完成 HTF bar
    return pd.Series(np.asarray(key)).map(slope).to_numpy(dtype=float)


def _volume_profile(
    price: np.ndarray,
    volume: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    rolling_length: int,
    bin_count: int,
    value_area_percent: float,
) -> dict[str, np.ndarray]:
    """滚动窗口成交量分布：POC / VAH / VAL / HVN 与 LVN 数量。"""
    n = len(price)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    hvn_count = np.zeros(n, dtype=np.int64)
    lvn_count = np.zeros(n, dtype=np.int64)

    for i in range(n):
        start = max(0, i - rolling_length + 1)
        window_high = np.nanmax(high[start:i + 1])
        window_low = np.nanmin(low[start:i + 1])
        if not (np.isfinite(window_high) and np.isfinite(window_low)):
            continue
        span = max(window_high - window_low, 1e-12)
        bin_size = max(span / bin_count, 1e-12)
        centers = window_low + (np.arange(bin_count) + 0.5) * bin_size
        px = price[start:i + 1]
        vol = np.nan_to_num(volume[start:i + 1], nan=0.0)
        bins = np.clip(np.floor((px - window_low) / bin_size).astype(np.int64), 0, bin_count - 1)
        bin_volumes = np.bincount(bins, weights=vol, minlength=bin_count).astype(float)
        if bin_volumes.sum() <= 0:
            continue

        poc_idx = int(np.argmax(bin_volumes))
        poc[i] = centers[poc_idx]

        # Value Area：从成交量最大的箱往外扩，直到覆盖目标比例
        order = np.argsort(-bin_volumes)
        target = bin_volumes.sum() * value_area_percent
        accumulated = 0.0
        chosen: list[int] = []
        for idx_bin in order:
            accumulated += bin_volumes[idx_bin]
            chosen.append(int(idx_bin))
            if accumulated >= target:
                break
        vah[i] = centers[max(chosen)]
        val[i] = centers[min(chosen)]

        # 3 点平滑后找峰/谷（HVN / LVN）
        smoothed = np.convolve(np.r_[bin_volumes[0], bin_volumes, bin_volumes[-1]], np.ones(3) / 3.0, mode="valid")
        average = bin_volumes.sum() / bin_count
        threshold = average * 0.50
        hvn = 0
        lvn = 0
        for k in range(1, bin_count - 1):
            left, mid_v, right = smoothed[k - 1], smoothed[k], smoothed[k + 1]
            if mid_v > left and mid_v > right and (mid_v - max(left, right)) > threshold * 0.10:
                hvn += 1
            if mid_v < left and mid_v < right and (min(left, right) - mid_v) > threshold * 0.10:
                lvn += 1
        hvn_count[i] = hvn
        lvn_count[i] = lvn

    return {"poc": poc, "vah": vah, "val": val, "hvn_count": hvn_count, "lvn_count": lvn_count}


def vwap_cross_visuals(
    frame: pd.DataFrame,
    *,
    display_mode: str = "Smart",
    cluster_pct: float = 0.3,
    vwap_pivot_left: int = 5,
    vwap_pivot_right: int = 5,
    min_bars_between_signals: int = 50,
    min_signal_strength: float = 50.0,
    signal_scope: str = "VWAP Cross only",
    emit_signals: bool = False,
    use_zone_management: bool = False,
    bias_length: int = 50,
    bias_band_multiplier: float = 1.5,
    target_mode: str = "ATR",
    atr_steps: str = "0.5,1,1.5,2",
    sigma_steps: str = "1,1.5,2",
    use_volume_profile: bool = False,
    profile_mode: str = "ROLLING",
    rolling_length: int = 250,
    bin_count: int = 64,
    value_area_percent: float = 0.70,
    retest_distance_atr: float = 0.25,
    use_htf_stack: bool = False,
    htf_stack_threshold: float = 0.5,
    htf_timeframes: tuple[int, ...] | None = None,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """VWAP Cross Visuals：逐 bar 的 VWAP / 交叉 / 聚集 / 亏损带 / 成交量分布 / HTF 栈。"""
    display_mode = choice("display_mode", display_mode, _DISPLAY_MODES)
    cluster_pct = number("cluster_pct", cluster_pct, 0.1, 2.0)
    vwap_pivot_left = integer("vwap_pivot_left", vwap_pivot_left, 1, 20)
    vwap_pivot_right = integer("vwap_pivot_right", vwap_pivot_right, 1, 20)
    min_bars_between_signals = integer("min_bars_between_signals", min_bars_between_signals, 10, 200)
    min_signal_strength = number("min_signal_strength", min_signal_strength, 0.0, 100.0)
    signal_scope = choice("signal_scope", signal_scope, _SIGNAL_SCOPES)
    emit_signals = boolean("emit_signals", emit_signals)
    use_zone_management = boolean("use_zone_management", use_zone_management)
    bias_length = integer("bias_length", bias_length, 10, 200)
    bias_band_multiplier = number("bias_band_multiplier", bias_band_multiplier, 0.5, 5.0)
    target_mode = choice("target_mode", target_mode, _TARGET_MODES)
    use_volume_profile = boolean("use_volume_profile", use_volume_profile)
    profile_mode = choice("profile_mode", profile_mode, _PROFILE_MODES)
    rolling_length = integer("rolling_length", rolling_length, 50, 1000)
    bin_count = integer("bin_count", bin_count, 24, 200)
    value_area_percent = number("value_area_percent", value_area_percent, 0.5, 0.99)
    retest_distance_atr = number("retest_distance_atr", retest_distance_atr, 0.0, 5.0)
    use_htf_stack = boolean("use_htf_stack", use_htf_stack)
    htf_stack_threshold = number("htf_stack_threshold", htf_stack_threshold, 0.5, 1.0)

    high_s, low_s, close_s, volume_s = ohlcv(frame, "high", "low", "close", "volume")
    open_s = ohlcv(frame, "open")[0]
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    open_a = open_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    # --- 数据有效性 ---------------------------------------------------------
    if volume_is_real is None:
        volume_is_real = bool(np.isfinite(volume).any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)
    vol = volume if volume_is_real else np.full(n, np.nan)

    atr14 = atr(frame, 14).to_numpy(dtype=float)
    hlc3 = (high + low + close) / 3.0

    # --- 会话 VWAP（按日重置） ----------------------------------------------
    session_vwap = np.full(n, np.nan)
    if volume_is_real:
        pv = close * vol
        day_key = pd.Series(idx).dt.floor("D").to_numpy() if isinstance(idx, pd.DatetimeIndex) else None
        if day_key is None:
            session_vwap = _safe_div(np.cumsum(pv), np.cumsum(vol))
        else:
            starts = np.r_[0, np.flatnonzero(day_key[1:] != day_key[:-1]) + 1]
            ends = np.r_[starts[1:], n]
            for start, end in zip(starts, ends):
                cumulative_pv = np.nancumsum(pv[start:end])
                cumulative_v = np.nancumsum(vol[start:end])
                session_vwap[start:end] = _safe_div(cumulative_pv, cumulative_v)
    session_vwap_prev = np.r_[np.nan, session_vwap[:-1]]

    # --- 枢轴锚定 VWAP ------------------------------------------------------
    pivot_high = pivothigh(high_s, vwap_pivot_left, vwap_pivot_right).to_numpy(dtype=float)
    pivot_low = pivotlow(low_s, vwap_pivot_left, vwap_pivot_right).to_numpy(dtype=float)
    pv_series = pd.Series(close * vol, index=idx)
    v_series = pd.Series(vol, index=idx)
    cumulative_pv = pv_series.fillna(0.0).cumsum().to_numpy(dtype=float)
    cumulative_v = v_series.fillna(0.0).cumsum().to_numpy(dtype=float)
    pv_at_pivot = cumulative_pv - pv_series.rolling(vwap_pivot_right, min_periods=vwap_pivot_right).sum().to_numpy()
    v_at_pivot = cumulative_v - v_series.rolling(vwap_pivot_right, min_periods=vwap_pivot_right).sum().to_numpy()
    if not volume_is_real:
        cumulative_pv = np.full(n, np.nan)
        cumulative_v = np.full(n, np.nan)
        pv_at_pivot = np.full(n, np.nan)
        v_at_pivot = np.full(n, np.nan)

    high_mask = np.isfinite(pivot_high)
    low_mask = np.isfinite(pivot_low)
    anchor_pv_high = pd.Series(np.where(high_mask, pv_at_pivot, np.nan)).ffill().to_numpy(dtype=float)
    anchor_v_high = pd.Series(np.where(high_mask, v_at_pivot, np.nan)).ffill().to_numpy(dtype=float)
    anchor_pv_low = pd.Series(np.where(low_mask, pv_at_pivot, np.nan)).ffill().to_numpy(dtype=float)
    anchor_v_low = pd.Series(np.where(low_mask, v_at_pivot, np.nan)).ffill().to_numpy(dtype=float)
    has_high_anchor = np.isfinite(anchor_pv_high)
    has_low_anchor = np.isfinite(anchor_pv_low)

    anchored_high = _safe_div(cumulative_pv - anchor_pv_high, cumulative_v - anchor_v_high)
    anchored_low = _safe_div(cumulative_pv - anchor_pv_low, cumulative_v - anchor_v_low)
    anchored_high = np.where(has_high_anchor, anchored_high, np.nan)
    anchored_low = np.where(has_low_anchor, anchored_low, np.nan)

    # --- 交叉检测 -----------------------------------------------------------
    def _cross_over(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        prev_left = np.r_[np.nan, left[:-1]]
        prev_right = np.r_[np.nan, right[:-1]]
        return (left > right) & (prev_left <= prev_right)

    def _cross_under(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        prev_left = np.r_[np.nan, left[:-1]]
        prev_right = np.r_[np.nan, right[:-1]]
        return (left < right) & (prev_left >= prev_right)

    price_cross_up = _cross_over(close, session_vwap)
    price_cross_down = _cross_under(close, session_vwap)
    structure_cross_up = _cross_over(session_vwap, anchored_high) | _cross_over(session_vwap, anchored_low)
    structure_cross_down = _cross_under(session_vwap, anchored_high) | _cross_under(session_vwap, anchored_low)

    # --- 聚集区检测 ---------------------------------------------------------
    cluster_tolerance = (cluster_pct / 100.0) * close
    vwaps_in_cluster = np.isfinite(anchored_high) & np.isfinite(anchored_low) & \
        (np.abs(anchored_high - anchored_low) <= cluster_tolerance)
    cluster_center = (anchored_high + anchored_low) / 2.0
    session_in_cluster = vwaps_in_cluster & np.isfinite(session_vwap) & \
        (np.abs(session_vwap - cluster_center) <= cluster_tolerance)
    price_in_cluster = vwaps_in_cluster & (np.abs(close - cluster_center) <= cluster_tolerance)
    cluster_active = vwaps_in_cluster & price_in_cluster & np.isfinite(session_vwap)

    # --- 信号节流 -----------------------------------------------------------
    can_price = np.ones(n, dtype=bool)
    can_structure = np.ones(n, dtype=bool)
    can_cluster = np.ones(n, dtype=bool)
    last_price = last_structure = last_cluster = -10 ** 9
    price_cross_shown = np.zeros(n, dtype=np.int64)
    structure_cross_shown = np.zeros(n, dtype=np.int64)
    cluster_shown = np.zeros(n, dtype=np.int64)
    for i in range(n):
        can_price[i] = (i - last_price) >= min_bars_between_signals
        can_structure[i] = (i - last_structure) >= min_bars_between_signals
        can_cluster[i] = (i - last_cluster) >= min_bars_between_signals
        if (price_cross_up[i] or price_cross_down[i]) and can_price[i]:
            last_price = i
            price_cross_shown[i] = 1 if price_cross_up[i] else -1
        if (structure_cross_up[i] or structure_cross_down[i]) and can_structure[i]:
            last_structure = i
            structure_cross_shown[i] = 1 if structure_cross_up[i] else -1
        if cluster_active[i] and can_cluster[i]:
            last_cluster = i
            cluster_shown[i] = 1

    # --- 强度 ---------------------------------------------------------------
    atr_safe = np.where(np.isfinite(atr14) & (atr14 > 0), atr14, 1.0)
    cross_strength = np.where(price_cross_up | price_cross_down,
                              np.where(atr14 > 0,
                                       np.minimum(100.0, np.abs(close - session_vwap) / atr_safe * 50.0), 50.0),
                              0.0)
    spread = np.abs(anchored_high - anchored_low)
    tightness = np.where(cluster_tolerance > 0, np.maximum(0.0, 1.0 - spread / cluster_tolerance), 0.0)
    cluster_strength = np.where(vwaps_in_cluster & price_in_cluster,
                                np.minimum(100.0, 75.0 + tightness * 25.0 + np.where(session_in_cluster, 10.0, 0.0)),
                                0.0)
    structure_high = np.where(np.isfinite(anchored_high) & np.isfinite(session_vwap) & (atr14 > 0),
                              np.minimum(100.0, np.abs(session_vwap - anchored_high) / atr_safe * 30.0 + 40.0), 0.0)
    structure_low = np.where(np.isfinite(anchored_low) & np.isfinite(session_vwap) & (atr14 > 0),
                             np.minimum(100.0, np.abs(session_vwap - anchored_low) / atr_safe * 30.0 + 40.0), 0.0)
    structure_strength = np.maximum(structure_high, structure_low)

    # --- 趋势位置 / 距离 -----------------------------------------------------
    position = np.where(close > session_vwap, 1, np.where(close < session_vwap, -1, 0)).astype(np.int64)
    position = np.where(np.isfinite(session_vwap), position, 0)
    distance = np.abs(close - session_vwap)
    distance_pct = np.where(session_vwap > 0, distance / session_vwap * 100.0, 0.0)

    # --- 偏置评分 -----------------------------------------------------------
    cross_term = np.where(price_cross_up, 25.0, np.where(price_cross_down, -25.0, 0.0))
    structure_term = np.where(structure_cross_up, 20.0, np.where(structure_cross_down, -20.0, 0.0))
    distance_bonus = np.where(atr14 > 0, np.minimum(2.0, np.nan_to_num(distance) / atr_safe) * 10.0, 0.0)
    bias_score = np.clip(position * 10.0 + cross_term + structure_term + position * distance_bonus, -100.0, 100.0)
    bias_band_multiplier_arr = bias_band_multiplier

    # --- 偏置带（Zone 模块的 calculateBias） --------------------------------
    deviation = stdev(close_s - pd.Series(session_vwap, index=idx), bias_length).to_numpy(dtype=float)
    band_unit = deviation * bias_band_multiplier_arr if target_mode == "SIGMA" else atr14 * bias_band_multiplier_arr
    band_upper = session_vwap + band_unit
    band_lower = session_vwap - band_unit
    bias_value = np.where(close > band_upper, 1, np.where(close < band_lower, -1, 0)).astype(np.int64)
    bias_pct = np.where(session_vwap > 0, (close - session_vwap) / session_vwap * 100.0, 0.0)

    # --- 目标档位 -----------------------------------------------------------
    steps_text = sigma_steps if target_mode == "SIGMA" else atr_steps
    try:
        steps = [float(part) for part in steps_text.split(",") if part.strip()]
    except ValueError as exc:  # pragma: no cover - 防御性
        raise ValueError(f"目标档位无法解析：{steps_text!r}") from exc
    unit = band_unit / bias_band_multiplier_arr if bias_band_multiplier_arr else np.zeros(n)
    target_up_1 = session_vwap + unit * (steps[0] if steps else 1.0)
    target_dn_1 = session_vwap - unit * (steps[0] if steps else 1.0)
    target_up_max = session_vwap + unit * (steps[-1] if steps else 1.0)
    target_dn_max = session_vwap - unit * (steps[-1] if steps else 1.0)

    # --- 汇合 ---------------------------------------------------------------
    confluence_count = ((price_cross_up | price_cross_down).astype(np.int64)
                        + (structure_cross_up | structure_cross_down).astype(np.int64)
                        + cluster_active.astype(np.int64))
    confluence_sum = (np.where(price_cross_up | price_cross_down, cross_strength, 0.0)
                      + np.where(structure_cross_up | structure_cross_down, structure_strength, 0.0)
                      + np.where(cluster_active, cluster_strength, 0.0))
    confluence_avg = np.where(confluence_count > 0, confluence_sum / np.maximum(confluence_count, 1), 0.0)

    # --- 成交量分布 ---------------------------------------------------------
    if use_volume_profile and volume_is_real:
        profile = _volume_profile(hlc3, vol, high, low, rolling_length, bin_count, value_area_percent)
    else:
        profile = {name: np.full(n, np.nan) for name in ("poc", "vah", "val")}
        profile["hvn_count"] = np.zeros(n, dtype=np.int64)
        profile["lvn_count"] = np.zeros(n, dtype=np.int64)

    # LVN 突破 / VA 回踩（需要 ATR 距离门）
    va_reentry_up = (np.isfinite(profile["vah"]) & (close > profile["vah"])
                     & (np.r_[np.nan, close[:-1]] <= np.r_[np.nan, profile["vah"][:-1]]))
    va_reentry_dn = (np.isfinite(profile["val"]) & (close < profile["val"])
                     & (np.r_[np.nan, close[:-1]] >= np.r_[np.nan, profile["val"][:-1]]))
    near_poc = np.isfinite(profile["poc"]) & (np.abs(close - profile["poc"]) <= atr14 * retest_distance_atr)

    # --- HTF 栈 -------------------------------------------------------------
    if use_htf_stack:
        timeframes = htf_timeframes if htf_timeframes is not None else _auto_timeframes(_bar_seconds(idx))
        if len(timeframes) > 4:
            timeframes = timeframes[:4]
        slopes = [_htf_slope(frame, int(tf)) for tf in timeframes]
        strengths = []
        for slope in slopes:
            raw = np.where(np.abs(slope) > 0.6, slope / 10.0, 0.0)
            strengths.append(np.clip(raw, -1.0, 1.0))
        bull_weight = np.zeros(n)
        bear_weight = np.zeros(n)
        total_weight = 0.0
        for k, strength in enumerate(strengths):
            weight = _HTF_WEIGHTS[k] if k < len(_HTF_WEIGHTS) else 0.25
            valid = np.isfinite(strength)
            bull_weight += np.where(valid & (strength > 0), strength * weight, 0.0)
            bear_weight += np.where(valid & (strength < 0), np.abs(strength) * weight, 0.0)
            total_weight += weight
        bull_confluence = (bull_weight / total_weight * 100.0) if total_weight > 0 else np.zeros(n)
        bear_confluence = (bear_weight / total_weight * 100.0) if total_weight > 0 else np.zeros(n)
        stack_strength = np.maximum(bull_confluence, bear_confluence)
        stack_bias = np.where(bull_confluence > bear_confluence + 20.0, 1,
                              np.where(bear_confluence > bull_confluence + 20.0, -1, 0)).astype(np.int64)

        true_range = np.maximum(high - low, np.maximum(np.abs(high - np.r_[np.nan, close[:-1]]),
                                                       np.abs(low - np.r_[np.nan, close[:-1]])))
        chop_len = 14
        atr_sum = pd.Series(np.nan_to_num(true_range)).rolling(chop_len, min_periods=chop_len).sum().to_numpy()
        range_high = high_s.rolling(chop_len, min_periods=chop_len).max().to_numpy()
        range_low = low_s.rolling(chop_len, min_periods=chop_len).min().to_numpy()
        span = range_high - range_low
        with np.errstate(divide="ignore", invalid="ignore"):
            chop = np.where(span > 0, 100.0 * np.log10(atr_sum / span) / math.log10(chop_len), 50.0)
        choppiness = np.clip(np.nan_to_num(chop, nan=50.0), 0.0, 100.0)
        regime = np.where(choppiness > 61.8, 2, np.where((choppiness < 38.2) & (stack_strength > 60.0), 1, 0))
        prev_bull = np.r_[np.nan, bull_confluence[:-1]]
        prev_bear = np.r_[np.nan, bear_confluence[:-1]]
        bull_stack_signal = (bull_confluence > htf_stack_threshold * 100.0) & \
            (~np.isfinite(prev_bull) | (prev_bull <= htf_stack_threshold * 100.0))
        bear_stack_signal = (bear_confluence > htf_stack_threshold * 100.0) & \
            (~np.isfinite(prev_bear) | (prev_bear <= htf_stack_threshold * 100.0))
        prev_max = np.fmax(prev_bull, prev_bear)
        curr_max = np.fmax(bull_confluence, bear_confluence)
        stack_break = (prev_max > 60.0) & (curr_max < 40.0)
    else:
        bull_confluence = np.full(n, np.nan)
        bear_confluence = np.full(n, np.nan)
        stack_strength = np.full(n, np.nan)
        stack_bias = np.zeros(n, dtype=np.int64)
        choppiness = np.full(n, np.nan)
        regime = np.zeros(n, dtype=np.int64)
        bull_stack_signal = np.zeros(n, dtype=bool)
        bear_stack_signal = np.zeros(n, dtype=bool)
        stack_break = np.zeros(n, dtype=bool)

    # --- 展示模式过滤 -------------------------------------------------------
    price_mode = display_mode in ("Smart", "Price Only")
    cluster_mode = display_mode in ("Smart", "Cluster Only")
    strong_enough = np.where(price_cross_up | price_cross_down, cross_strength, 0.0) >= min_signal_strength

    show_price_up = price_mode & price_cross_up & (price_cross_shown == 1) & (strong_enough | (min_signal_strength <= 0))
    show_price_down = price_mode & price_cross_down & (price_cross_shown == -1) & (strong_enough | (min_signal_strength <= 0))
    show_structure_up = (emit_signals and signal_scope == "All signals" and price_mode
                         and structure_cross_up and (structure_cross_shown == 1))
    show_structure_down = (emit_signals and signal_scope == "All signals" and price_mode
                           and structure_cross_down and (structure_cross_shown == -1))
    show_cluster = emit_signals and signal_scope == "All signals" and cluster_mode and (cluster_shown == 1)

    long_signal = show_price_up | show_structure_up
    short_signal = show_price_down | show_structure_down

    def as_int(values) -> pd.Series:
        return pd.Series(np.asarray(values).astype(np.int64), index=idx).astype("Int64")

    data = {
        "vwap_session": session_vwap,
        "vwap_session_prev": session_vwap_prev,
        "vwap_anchored_high": anchored_high,
        "vwap_anchored_low": anchored_low,
        "vwap_has_high_anchor": as_int(has_high_anchor),
        "vwap_has_low_anchor": as_int(has_low_anchor),
        "vwap_price_position": as_int(position),
        "vwap_price_distance": distance,
        "vwap_price_distance_pct": distance_pct,
        "vwap_bias_score": bias_score,
        "vwap_bias_value": as_int(bias_value),
        "vwap_bias_pct": bias_pct,
        "vwap_band_upper": band_upper,
        "vwap_band_lower": band_lower,
        "vwap_target_up_1": target_up_1,
        "vwap_target_dn_1": target_dn_1,
        "vwap_target_up_max": target_up_max,
        "vwap_target_dn_max": target_dn_max,
        "vwap_price_cross_up": as_int(price_cross_up),
        "vwap_price_cross_down": as_int(price_cross_down),
        "vwap_structure_cross_up": as_int(structure_cross_up),
        "vwap_structure_cross_down": as_int(structure_cross_down),
        "vwap_can_show_price_cross": as_int(can_price),
        "vwap_can_show_structure_cross": as_int(can_structure),
        "vwap_can_show_cluster": as_int(can_cluster),
        "vwap_cluster_active": as_int(cluster_active),
        "vwap_vwaps_in_cluster": as_int(vwaps_in_cluster),
        "vwap_price_in_cluster": as_int(price_in_cluster),
        "vwap_session_in_cluster": as_int(session_in_cluster),
        "vwap_cluster_tolerance": cluster_tolerance,
        "vwap_cross_strength": cross_strength,
        "vwap_cluster_strength": cluster_strength,
        "vwap_structure_strength": structure_strength,
        "vwap_confluence_count": as_int(confluence_count),
        "vwap_confluence_avg": confluence_avg,
        "vwap_poc": profile["poc"],
        "vwap_vah": profile["vah"],
        "vwap_val": profile["val"],
        "vwap_hvn_count": as_int(profile["hvn_count"]),
        "vwap_lvn_count": as_int(profile["lvn_count"]),
        "vwap_va_reentry_up": as_int(va_reentry_up),
        "vwap_va_reentry_dn": as_int(va_reentry_dn),
        "vwap_near_poc": as_int(near_poc),
        "vwap_htf_bull_confluence": bull_confluence,
        "vwap_htf_bear_confluence": bear_confluence,
        "vwap_htf_stack_strength": stack_strength,
        "vwap_htf_stack_bias": as_int(stack_bias),
        "vwap_htf_choppiness": choppiness,
        "vwap_htf_regime": as_int(regime),
        "vwap_htf_bull_stack_signal": as_int(bull_stack_signal),
        "vwap_htf_bear_stack_signal": as_int(bear_stack_signal),
        "vwap_htf_stack_break": as_int(stack_break),
        "vwap_show_price_cross_up": as_int(show_price_up),
        "vwap_show_price_cross_down": as_int(show_price_down),
        "vwap_show_cluster": as_int(show_cluster),
        "vwap_long_signal": as_int(long_signal),
        "vwap_short_signal": as_int(short_signal),
    }
    return pd.DataFrame(data, index=idx)
