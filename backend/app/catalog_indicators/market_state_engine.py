"""Market State Engine [WavesUnchained] v2.1.2 的 Python 移植。

这是一个**上下文引擎**，不是买卖信号：把市场拆成三条正交轴，再从轴的组合里
用「分数仲裁 + 领先优势」推导九个状态。

三条轴
------
* **STRUCTURE REGIME** —— 0 冲突 / 1 平衡 / 2 多头 / 3 空头
* **ENERGY PHASE** —— 0 中性 / 1 收缩 / 2 扩张 / 3 衰竭
* **AUCTION LOCATION** —— 归一化位置（折价 / 价值 / 溢价）+ 方向化拍卖压力

九个状态
--------
0 BALANCE / 1 COMPRESSION / 2 BULL_EXPANSION / 3 BEAR_EXPANSION /
4 BULL_PULLBACK / 5 BEAR_PULLBACK / 6 BULL_EXHAUSTION / 7 BEAR_EXHAUSTION /
8 TRANSITION

核心机制
--------
* **分数仲裁**：八个候选状态各有一个分数，取最高者为 `rawState`；但还要过
  ``minimum_state_score`` 与 ``state_dominance_margin``（最高分减亚军分），
  否则一律落到 TRANSITION —— 这样「两个状态都很高」不会被误判成其中一个。
* **状态机**：候选连续计数 + 最小停留时长滞回；反向扩张在置信度 ≥
  ``opposite_break_threshold`` 时可以**紧急越级**切换。
* **脉冲记忆**：扩张状态被确认时开始记一条「脉冲」（起点取候选人运行段的
  真实起点，不是确认那根），结束时结算净进度 / 总幅 / 效率 / 质量；此后
  回撤深度 + 结构是否仍支持 + HTF 是否反对共同决定「脉冲趋势是否完好」。
* **权重再归一**：吸收（effort-vs-result）项只在有真实成交量时参与；无量能
  时该项权重被**剔除并把剩余权重归一化**，避免方向压力与衰竭分结构性偏低。

移植说明
--------
``syminfo.mintick`` 不可得 → ``mintick`` 参数替代；``syminfo.volumetype`` 不可得 →
``volume_is_real`` 参数替代。HTF 用重采样近似，取**上一根已完成** HTF bar
（原脚本用 ``lookahead_on``，等价于读已完成 bar，这里显式 shift 避免未来函数）。

未移植：Balance/Compression 区箱体（raw zone 起始/宽限/冲击柱守卫/接受式突破/
跟随窗口）、全部绘图与仪表盘、告警。这些是视觉与箱体管理层；状态机的数值判定
不依赖它们。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, dmi, ema, integer, number, ohlcv, roc, sma, stdev, vwma

__all__ = ["market_state_engine", "STATE_NAMES", "REGIME_NAMES", "ENERGY_NAMES",
           "TRANSITION_NAMES", "CONTEXT_NAMES"]

STATE_BALANCE, STATE_COMPRESSION = 0, 1
STATE_BULL_EXPANSION, STATE_BEAR_EXPANSION = 2, 3
STATE_BULL_PULLBACK, STATE_BEAR_PULLBACK = 4, 5
STATE_BULL_EXHAUSTION, STATE_BEAR_EXHAUSTION = 6, 7
STATE_TRANSITION = 8

STATE_NAMES = (
    "BALANCE", "COMPRESSION", "BULL_EXPANSION", "BEAR_EXPANSION",
    "BULL_PULLBACK", "BEAR_PULLBACK", "BULL_EXHAUSTION", "BEAR_EXHAUSTION",
    "TRANSITION",
)

REGIME_CONFLICTED, REGIME_BALANCED, REGIME_BULLISH, REGIME_BEARISH = 0, 1, 2, 3
REGIME_NAMES = ("CONFLICTED", "BALANCED", "BULLISH", "BEARISH")

ENERGY_NEUTRAL, ENERGY_CONTRACTING, ENERGY_EXPANDING, ENERGY_EXHAUSTING = 0, 1, 2, 3
ENERGY_NAMES = ("NEUTRAL", "CONTRACTING", "EXPANDING", "EXHAUSTING")

TRANSITION_NONE, TRANSITION_NO_DATA, TRANSITION_CONFLICT = 0, 1, 2
TRANSITION_TREND_BUILD, TRANSITION_BREAK, TRANSITION_CHAOTIC = 3, 4, 5
TRANSITION_NAMES = ("NONE", "NO_DATA", "CONFLICT", "TREND_BUILD", "BREAK", "CHAOTIC")

CONTEXT_UNFAVORABLE, CONTEXT_NEUTRAL, CONTEXT_COMPATIBLE, CONTEXT_PREFERRED = 0, 1, 2, 3
CONTEXT_NAMES = ("UNFAVORABLE", "NEUTRAL", "COMPATIBLE", "PREFERRED")

_PRESETS = {
    # (fast, slow, range, swing, value, energy, compression, volume, confirm)
    "Position": (34, 89, 80, 7, 89, 30, 100, 60, 4),
    "Swing": (21, 55, 50, 5, 55, 20, 60, 40, 3),
    "Active": (13, 34, 30, 3, 34, 14, 40, 25, 2),
}
_SMOOTHINGS = ("EMA", "DEMA", "KAMA", "JMA Approx")
_VALUE_METHODS = ("VWMA", "EMA", "Hybrid")


def _clamp(values, low: float, high: float) -> np.ndarray:
    return np.maximum(low, np.minimum(high, np.asarray(values, dtype=float)))


def _safe_div(numerator, denominator, fallback: float) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.abs(den) > 0.0000001
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _normalize(values, low: float, high: float) -> np.ndarray:
    return _clamp(_safe_div(np.asarray(values, dtype=float) - low, high - low, 0.0), 0.0, 1.0)


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if 0 < steps < len(values):
        out[steps:] = values[:-steps]
    elif steps == 0:
        out[:] = values
    return out


def _kama(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    previous = math.nan
    for i in range(len(values)):
        x = values[i]
        if not np.isfinite(x):
            out[i] = previous
            continue
        if i < length:
            previous = x if not np.isfinite(previous) else previous
            out[i] = previous
            continue
        change = abs(x - values[i - length])
        volatility = float(np.nansum(np.abs(np.diff(values[i - length:i + 1]))))
        efficiency = change / volatility if volatility != 0 else 0.0
        smoothing = (efficiency * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2
        base = previous if np.isfinite(previous) else x
        previous = base + smoothing * (x - base)
        out[i] = previous
    return out


def _jma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    phase_ratio = 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** 2.0
    e0 = e1 = e2 = jma_val = math.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            out[i] = jma_val
            continue
        prev_e0 = e0 if np.isfinite(e0) else x
        prev_e1 = e1 if np.isfinite(e1) else 0.0
        prev_jma = jma_val if np.isfinite(jma_val) else x
        prev_e2 = e2 if np.isfinite(e2) else 0.0
        e0 = (1.0 - alpha) * x + alpha * prev_e0
        e1 = (x - e0) * (1.0 - beta) + beta * prev_e1
        e2 = (e0 + phase_ratio * e1 - prev_jma) * (1.0 - alpha) ** 2 + alpha ** 2 * prev_e2
        jma_val = prev_jma + e2
        out[i] = jma_val
    return out


def _ema_na(series: pd.Series, length: int) -> pd.Series:
    values = series.astype(float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    positions = np.flatnonzero(values.notna().to_numpy())
    if positions.size == 0:
        return out
    start = int(positions[0])
    tail = values.iloc[start:]
    if tail.isna().any():
        tail = tail.ffill()
    out.iloc[start:] = ema(tail.reset_index(drop=True), length).to_numpy()
    return out


def _structure_average(values: np.ndarray, length: int, smoothing: str) -> np.ndarray:
    series = pd.Series(values)
    if smoothing == "DEMA":
        first = _ema_na(series, length)
        return (2.0 * first - _ema_na(first, length)).to_numpy(dtype=float)
    if smoothing == "KAMA":
        return _kama(values, length)
    if smoothing == "JMA Approx":
        return _jma(values, length)
    return ema(series, length).to_numpy(dtype=float)


def _htf_bar_seconds(token: str) -> int:
    text = str(token).strip().upper()
    fixed = {"D": 86400, "W": 604800, "M": 2592000}
    if text in fixed:
        return fixed[text]
    if not text:
        return 86400
    return int(float(text)) * 60


def _htf_snapshot(frame: pd.DataFrame, target_seconds: int, fast_len: int, slow_len: int,
                  atr_len: int, smoothing: str) -> dict[str, np.ndarray]:
    """在目标周期上算结构快照，并取**上一根已完成** HTF bar 的值。"""
    idx = pd.DatetimeIndex(frame.index)
    if target_seconds >= 28 * 86400:
        key = pd.PeriodIndex(idx, freq="M")
    elif target_seconds >= 6 * 86400:
        key = pd.PeriodIndex(idx, freq="W")
    elif target_seconds >= 86400:
        key = pd.PeriodIndex(idx, freq="D")
    else:
        key = idx.floor(pd.Timedelta(seconds=int(target_seconds)))
    grouped = frame.groupby(key).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    close_h = grouped["close"].to_numpy(dtype=float)
    fast = pd.Series(_structure_average(close_h, fast_len, smoothing), index=grouped.index)
    slow = pd.Series(_structure_average(close_h, slow_len, smoothing), index=grouped.index)
    true_range = np.maximum(
        grouped["high"].to_numpy(dtype=float) - grouped["low"].to_numpy(dtype=float),
        np.maximum(np.abs(grouped["high"].to_numpy(dtype=float) - np.r_[np.nan, close_h[:-1]]),
                   np.abs(grouped["low"].to_numpy(dtype=float) - np.r_[np.nan, close_h[:-1]])))
    atr_h = pd.Series(true_range, index=grouped.index).rolling(atr_len, min_periods=atr_len).mean()

    # f_htfStructureSnapshot: [close[1], fast[1], slow[1], slow[2], atr[1]]
    snapshot = {
        "close": grouped["close"].shift(1),
        "fast": fast.shift(1),
        "slow": slow.shift(1),
        "slow_previous": slow.shift(2),
        "atr": atr_h.shift(1),
    }
    bars = pd.Series(np.asarray(key))
    return {name: bars.map(series).to_numpy(dtype=float) for name, series in snapshot.items()}


def market_state_engine(
    frame: pd.DataFrame,
    *,
    preset: str = "Swing",
    custom_fast_len: int = 21,
    custom_slow_len: int = 55,
    custom_range_len: int = 40,
    custom_swing_len: int = 5,
    custom_value_len: int = 55,
    value_width: float = 1.25,
    custom_energy_len: int = 20,
    custom_compression_len: int = 50,
    custom_volume_len: int = 40,
    custom_confirm_bars: int = 3,
    structure_smoothing: str = "JMA Approx",
    atr_len: int = 14,
    di_len: int = 14,
    adx_smooth: int = 14,
    trend_adx_threshold: float = 21.0,
    strong_trend_adx: float = 28.0,
    efficiency_trend_threshold: float = 0.34,
    value_method: str = "VWMA",
    discount_threshold: float = -0.65,
    premium_threshold: float = 0.65,
    extreme_location_threshold: float = 1.50,
    compression_threshold: float = 0.62,
    expansion_threshold: float = 0.55,
    exhaustion_threshold: float = 0.58,
    use_volume: bool = True,
    directional_bias_threshold: float = 0.18,
    use_higher_timeframe: bool = True,
    higher_timeframe: str = "D",
    require_htf_agreement: bool = False,
    minimum_state_bars: int = 3,
    opposite_break_threshold: float = 0.75,
    minimum_state_score: float = 0.55,
    state_dominance_margin: float = 0.06,
    mintick: float = 1e-8,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Market State Engine：三轴 + 九态 + 状态机 + 脉冲记忆 + 上下文就绪度。"""
    preset = choice("preset", preset, tuple(_PRESETS) + ("Custom",))
    structure_smoothing = choice("structure_smoothing", structure_smoothing, _SMOOTHINGS)
    value_method = choice("value_method", value_method, _VALUE_METHODS)
    custom_fast_len = integer("custom_fast_len", custom_fast_len, 2, 200)
    custom_slow_len = integer("custom_slow_len", custom_slow_len, 3, 400)
    custom_range_len = integer("custom_range_len", custom_range_len, 5, 300)
    custom_swing_len = integer("custom_swing_len", custom_swing_len, 1, 50)
    custom_value_len = integer("custom_value_len", custom_value_len, 2, 400)
    custom_energy_len = integer("custom_energy_len", custom_energy_len, 2, 200)
    custom_compression_len = integer("custom_compression_len", custom_compression_len, 5, 400)
    custom_volume_len = integer("custom_volume_len", custom_volume_len, 2, 200)
    custom_confirm_bars = integer("custom_confirm_bars", custom_confirm_bars, 1, 20)
    value_width = number("value_width", value_width, 0.1, 5.0)
    atr_len = integer("atr_len", atr_len, 1, 100)
    di_len = integer("di_len", di_len, 1, 100)
    adx_smooth = integer("adx_smooth", adx_smooth, 1, 100)
    trend_adx_threshold = number("trend_adx_threshold", trend_adx_threshold, 0.0, 100.0)
    strong_trend_adx = number("strong_trend_adx", strong_trend_adx, 0.0, 100.0)
    efficiency_trend_threshold = number("efficiency_trend_threshold", efficiency_trend_threshold, 0.0, 1.0)
    discount_threshold = number("discount_threshold", discount_threshold, -5.0, 0.0)
    premium_threshold = number("premium_threshold", premium_threshold, 0.0, 5.0)
    extreme_location_threshold = number("extreme_location_threshold", extreme_location_threshold, 0.0, 5.0)
    compression_threshold = number("compression_threshold", compression_threshold, 0.0, 1.0)
    expansion_threshold = number("expansion_threshold", expansion_threshold, 0.0, 1.0)
    exhaustion_threshold = number("exhaustion_threshold", exhaustion_threshold, 0.0, 1.0)
    use_volume = boolean("use_volume", use_volume)
    directional_bias_threshold = number("directional_bias_threshold", directional_bias_threshold, 0.0, 1.0)
    use_higher_timeframe = boolean("use_higher_timeframe", use_higher_timeframe)
    require_htf_agreement = boolean("require_htf_agreement", require_htf_agreement)
    minimum_state_bars = integer("minimum_state_bars", minimum_state_bars, 1, 50)
    opposite_break_threshold = number("opposite_break_threshold", opposite_break_threshold, 0.0, 1.0)
    minimum_state_score = number("minimum_state_score", minimum_state_score, 0.0, 1.0)
    state_dominance_margin = number("state_dominance_margin", state_dominance_margin, 0.0, 1.0)

    if preset == "Custom":
        fast_len, slow_len = custom_fast_len, custom_slow_len
        range_len, swing_len, value_len = custom_range_len, custom_swing_len, custom_value_len
        energy_len, compression_len = custom_energy_len, custom_compression_len
        volume_len, confirm_bars = custom_volume_len, custom_confirm_bars
    else:
        fast_len, slow_len, range_len, swing_len, value_len, energy_len, compression_len, volume_len, confirm_bars = _PRESETS[preset]

    open_s, high_s, low_s, close_s, volume_s = ohlcv(frame, "open", "high", "low", "close", "volume")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    floor_tick = mintick if mintick and mintick > 0 else 1e-8

    # --- 基础 ---------------------------------------------------------------
    atr_a = atr(frame, atr_len).to_numpy(dtype=float)
    safe_atr = np.fmax(atr_a, floor_tick)
    safe_atr = np.where(np.isfinite(atr_a), safe_atr, np.nan)

    structure_fast = _structure_average(close, fast_len, structure_smoothing)
    structure_slow = _structure_average(close, slow_len, structure_smoothing)
    fast_slope = _safe_div(structure_fast - _shift(structure_fast, 3), safe_atr * 3.0, 0.0)
    slow_slope = _safe_div(structure_slow - _shift(structure_slow, 5), safe_atr * 5.0, 0.0)
    structure_separation = _safe_div(np.abs(structure_fast - structure_slow), safe_atr, 0.0)

    plus_di, minus_di, adx_s = dmi(frame, di_len)
    plus_di = plus_di.to_numpy(dtype=float)
    minus_di = minus_di.to_numpy(dtype=float)
    adx = adx_s.to_numpy(dtype=float)
    directional_di = _safe_div(plus_di - minus_di, plus_di + minus_di, 0.0)

    # --- 效率 ---------------------------------------------------------------
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    path_travel = pd.Series(abs_change, index=idx).rolling(energy_len, min_periods=energy_len).sum().to_numpy()
    net_travel = np.abs(close - _shift(close, energy_len))
    efficiency_ratio = _clamp(_safe_div(net_travel, path_travel, 0.0), 0.0, 1.0)
    previous_close_lag = _shift(close, energy_len)
    efficiency_direction = np.where(close > previous_close_lag, 1.0,
                                    np.where(close < previous_close_lag, -1.0, 0.0))
    signed_efficiency = efficiency_ratio * efficiency_direction

    overlap_top = np.minimum(high, _shift(high, 1))
    overlap_bottom = np.maximum(low, _shift(low, 1))
    overlap_amount = np.maximum(overlap_top - overlap_bottom, 0.0)
    previous_range = np.fmax(_shift(high, 1) - _shift(low, 1), floor_tick)
    current_range = np.fmax(high - low, floor_tick)
    overlap_ratio = _clamp(_safe_div(overlap_amount, np.fmin(previous_range, current_range), 0.0), 0.0, 1.0)
    average_overlap = ema(pd.Series(overlap_ratio, index=idx), energy_len).to_numpy(dtype=float)

    # --- 结构方向 -----------------------------------------------------------
    structure_bull_score = (
        0.30 * _normalize(structure_fast - structure_slow, -safe_atr, safe_atr)
        + 0.25 * _normalize(fast_slope, -0.20, 0.20)
        + 0.20 * _normalize(slow_slope, -0.12, 0.12)
        + 0.15 * _normalize(directional_di, -0.35, 0.35)
        + 0.10 * _normalize(signed_efficiency, -0.40, 0.40)
    )
    structure_bear_score = 1.0 - structure_bull_score
    structure_direction = structure_bull_score - structure_bear_score
    structural_bull = (structure_fast > structure_slow) & (slow_slope > -0.02) & (structure_direction > 0.08)
    structural_bear = (structure_fast < structure_slow) & (slow_slope < 0.02) & (structure_direction < -0.08)

    # --- HTF 上下文 ---------------------------------------------------------
    htf_enabled = use_higher_timeframe
    if htf_enabled:
        snapshot = _htf_snapshot(frame, _htf_bar_seconds(higher_timeframe), fast_len, slow_len,
                                 atr_len, structure_smoothing)
        htf_close, htf_fast, htf_slow = snapshot["close"], snapshot["fast"], snapshot["slow"]
        htf_slow_previous, htf_atr = snapshot["slow_previous"], snapshot["atr"]
        htf_slow_slope = _safe_div(htf_slow - htf_slow_previous, np.fmax(htf_atr, floor_tick), 0.0)
        htf_bull = (htf_fast > htf_slow) & (htf_close > htf_slow) & (htf_slow_slope >= 0)
        htf_bear = (htf_fast < htf_slow) & (htf_close < htf_slow) & (htf_slow_slope <= 0)
    else:
        htf_close = htf_fast = htf_slow = htf_slow_previous = htf_atr = np.full(n, np.nan)
        htf_slow_slope = np.zeros(n)
        htf_bull = np.ones(n, dtype=bool)
        htf_bear = np.ones(n, dtype=bool)

    # --- 自适应价值 ---------------------------------------------------------
    hlc3_s = (high_s + low_s + close_s) / 3.0
    ema_value = ema(hlc3_s, value_len).to_numpy(dtype=float)
    vwma_value = vwma(hlc3_s, volume_s, value_len).to_numpy(dtype=float)
    if volume_is_real is None:
        volume_is_real = bool(np.isfinite(volume).any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)
    volume_average_len = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), volume_len).to_numpy(dtype=float)
    has_usable_volume = np.asarray(volume_is_real, dtype=bool) & (np.nan_to_num(volume_average_len) > 0)
    effective_vwma = np.where(has_usable_volume, vwma_value, ema_value)
    fair_value = {
        "EMA": ema_value,
        "Hybrid": (ema_value + effective_vwma) / 2.0,
    }.get(value_method, effective_vwma)

    value_residual = close - fair_value
    residual_deviation = stdev(pd.Series(value_residual, index=idx), value_len).to_numpy(dtype=float)
    adaptive_deviation = np.fmax(residual_deviation, safe_atr * 0.60)
    value_upper = fair_value + adaptive_deviation * value_width
    value_lower = fair_value - adaptive_deviation * value_width
    normalized_location = _safe_div(close - fair_value, adaptive_deviation * value_width, 0.0)

    rolling_high = high_s.rolling(range_len, min_periods=range_len).max().to_numpy(dtype=float)
    rolling_low = low_s.rolling(range_len, min_periods=range_len).min().to_numpy(dtype=float)
    rolling_width = np.fmax(rolling_high - rolling_low, floor_tick)
    range_location = _clamp(_safe_div(close - rolling_low, rolling_width, 0.5), 0.0, 1.0)

    # --- 波动与压缩 ---------------------------------------------------------
    atr_percent = _safe_div(atr_a, close, 0.0)
    atr_baseline = sma(pd.Series(atr_percent, index=idx), compression_len).to_numpy(dtype=float)
    atr_compression = 1.0 - _clamp(_safe_div(atr_percent, atr_baseline * 1.25, 1.0), 0.0, 1.0)
    basis = sma(close_s, energy_len).to_numpy(dtype=float)
    deviation = stdev(close_s, energy_len).to_numpy(dtype=float)
    bb_width = _safe_div(deviation * 4.0, basis, 0.0)
    bb_width_baseline = sma(pd.Series(bb_width, index=idx), compression_len).to_numpy(dtype=float)
    band_compression = 1.0 - _clamp(_safe_div(bb_width, bb_width_baseline * 1.20, 1.0), 0.0, 1.0)
    range_atr_ratio = _safe_div(rolling_width, safe_atr * math.sqrt(range_len), 0.0)
    range_containment = 1.0 - _normalize(range_atr_ratio, 1.50, 4.50)
    compression_score = _clamp(
        0.32 * atr_compression + 0.28 * band_compression
        + 0.22 * average_overlap + 0.18 * range_containment, 0.0, 1.0)

    # --- 扩张 ---------------------------------------------------------------
    candle_body = np.abs(close - open_a)
    body_efficiency = _clamp(_safe_div(candle_body, current_range, 0.0), 0.0, 1.0)
    close_location = _clamp(_safe_div(close - low, current_range, 0.5), 0.0, 1.0)
    bull_body_quality = _clamp(_safe_div(close - open_a, current_range, 0.0), 0.0, 1.0)
    bear_body_quality = _clamp(_safe_div(open_a - close, current_range, 0.0), 0.0, 1.0)
    range_expansion = _normalize(current_range, safe_atr * 0.70, safe_atr * 1.80)
    atr_acceleration = _normalize(_safe_div(atr_a, _shift(atr_a, 5), 1.0), 0.90, 1.25)
    adx_expansion = _normalize(adx, trend_adx_threshold - 5.0, strong_trend_adx + 8.0)
    directional_progress = _safe_div(close - _shift(close, energy_len), safe_atr * math.sqrt(energy_len), 0.0)
    bull_progress = _normalize(directional_progress, 0.0, 2.0)
    bear_progress = _normalize(-directional_progress, 0.0, 2.0)
    bull_expansion_score = _clamp(
        0.22 * efficiency_ratio + 0.18 * adx_expansion + 0.16 * range_expansion
        + 0.12 * atr_acceleration + 0.12 * bull_body_quality
        + 0.12 * close_location + 0.08 * bull_progress, 0.0, 1.0)
    bear_expansion_score = _clamp(
        0.22 * efficiency_ratio + 0.18 * adx_expansion + 0.16 * range_expansion
        + 0.12 * atr_acceleration + 0.12 * bear_body_quality
        + 0.12 * (1.0 - close_location) + 0.08 * bear_progress, 0.0, 1.0)
    expansion_score = np.fmax(bull_expansion_score, bear_expansion_score)

    # --- 拍卖压力 -----------------------------------------------------------
    volume_average = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), volume_len).to_numpy(dtype=float)
    volume_deviation = stdev(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), volume_len).to_numpy(dtype=float)
    auction_volume_usable = np.asarray(has_usable_volume, dtype=bool) & bool(use_volume)
    volume_z = _safe_div(volume - volume_average, volume_deviation, 0.0) if volume_is_real else np.zeros(n)
    volume_strength = np.where(auction_volume_usable, _normalize(volume_z, -0.50, 2.50), 0.50)

    signed_close_position = close_location * 2.0 - 1.0
    directional_result = _safe_div(np.abs(close - _shift(close, 1)), safe_atr, 0.0)
    effort_result_mismatch = np.where(
        auction_volume_usable,
        _clamp(_normalize(volume_z, 0.50, 2.50) * (1.0 - _normalize(directional_result, 0.20, 1.20)), 0.0, 1.0),
        0.0)

    prior_structure_low = low_s.shift(1).rolling(swing_len * 2, min_periods=swing_len * 2).min().to_numpy(dtype=float)
    prior_structure_high = high_s.shift(1).rolling(swing_len * 2, min_periods=swing_len * 2).max().to_numpy(dtype=float)
    failed_breakdown = (low < prior_structure_low) & (close > prior_structure_low) & (close_location > 0.55)
    failed_breakout = (high > prior_structure_high) & (close < prior_structure_high) & (close_location < 0.45)
    failed_auction_bias = np.where(failed_breakdown, 1.0, np.where(failed_breakout, -1.0, 0.0))

    value_response_bias = np.where(
        (normalized_location < discount_threshold) & (signed_close_position > 0), signed_close_position,
        np.where((normalized_location > premium_threshold) & (signed_close_position < 0), signed_close_position, 0.0))

    pressure_raw = signed_close_position * body_efficiency * (0.50 + volume_strength * 0.50)
    pressure_smoothed = ema(pd.Series(pressure_raw, index=idx), energy_len).to_numpy(dtype=float)
    absorption_direction = np.where((range_location < 0.35) & (signed_close_position > 0), 1.0,
                                    np.where((range_location > 0.65) & (signed_close_position < 0), -1.0, 0.0))
    absorption_bias = effort_result_mismatch * absorption_direction
    bias_weight_sum = 0.40 + 0.25 + 0.15 + np.where(auction_volume_usable, 0.20, 0.0)
    directional_bias = _clamp(_safe_div(
        0.40 * pressure_smoothed + 0.25 * failed_auction_bias
        + np.where(auction_volume_usable, 0.20 * absorption_bias, 0.0)
        + 0.15 * value_response_bias, bias_weight_sum, 0.0), -1.0, 1.0)
    smoothed_directional_bias = ema(pd.Series(directional_bias, index=idx), 3).to_numpy(dtype=float)

    # --- 衰竭 ---------------------------------------------------------------
    bull_extension = _normalize(normalized_location, 0.70, 2.20)
    bear_extension = _normalize(-normalized_location, 0.70, 2.20)
    directional_extension = np.where(structural_bull, bull_extension,
                                     np.where(structural_bear, bear_extension, 0.0))

    momentum_fast_len = max(3, int(round(energy_len * 0.35)))
    momentum_fast = roc(close_s, momentum_fast_len).to_numpy(dtype=float)
    momentum_slow = roc(close_s, energy_len).to_numpy(dtype=float)
    momentum_delta = momentum_fast - _shift(momentum_fast, 3)
    momentum_delta_dev = stdev(pd.Series(momentum_delta, index=idx), energy_len).to_numpy(dtype=float)
    normalized_momentum_delta = _safe_div(momentum_delta, np.fmax(np.nan_to_num(momentum_delta_dev), 0.0001), 0.0)
    bull_deceleration = np.where(momentum_slow > 0, _normalize(-normalized_momentum_delta, 0.0, 2.5), 0.0)
    bear_deceleration = np.where(momentum_slow < 0, _normalize(normalized_momentum_delta, 0.0, 2.5), 0.0)
    directional_deceleration = np.where(structural_bull, bull_deceleration,
                                        np.where(structural_bear, bear_deceleration, 0.0))

    structure_flip = (structural_bull != _shift(structural_bull.astype(float), 1)) | \
                     (structural_bear != _shift(structural_bear.astype(float), 1))
    trend_age = np.full(n, np.nan)
    counter = 0
    for i in range(n):
        if i == 0 or structure_flip[i]:
            counter = 0
        else:
            counter += 1
        trend_age[i] = counter
    age_score = _normalize(np.nan_to_num(trend_age), energy_len, energy_len * 4.0)
    failed_progress = np.where(efficiency_ratio < _shift(efficiency_ratio, 5),
                               _normalize(_shift(efficiency_ratio, 5) - efficiency_ratio, 0.0, 0.25), 0.0)
    exhaustion_weight_sum = 0.30 + 0.25 + 0.18 + 0.17 + np.where(auction_volume_usable, 0.10, 0.0)
    exhaustion_score = _safe_div(
        0.30 * directional_extension + 0.25 * directional_deceleration + 0.18 * age_score
        + 0.17 * failed_progress
        + np.where(auction_volume_usable, 0.10 * effort_result_mismatch, 0.0),
        exhaustion_weight_sum, 0.0)
    directional_expansion_persistence = np.where(structural_bull, bull_expansion_score,
                                                 np.where(structural_bear, bear_expansion_score, 0.0))
    exhaustion_score = _clamp(exhaustion_score * (1.0 - directional_expansion_persistence * 0.30), 0.0, 1.0)
    has_exhaustion_trigger = (directional_deceleration >= 0.30) | (failed_progress >= 0.35) | \
        (effort_result_mismatch >= 0.55)

    # --- 平衡 / 趋势质量 ----------------------------------------------------
    balance_score = _clamp(
        0.28 * (1.0 - efficiency_ratio) + 0.24 * average_overlap + 0.20 * (1.0 - adx_expansion)
        + 0.16 * (1.0 - _normalize(structure_separation, 0.20, 1.50))
        + 0.12 * range_containment, 0.0, 1.0)
    balance_candidate = (balance_score >= 0.56) & (adx < strong_trend_adx) & (np.abs(structure_direction) < 0.48)
    compression_candidate = (compression_score >= compression_threshold) & (balance_score >= 0.44)

    directional_agreement = np.where(structural_bull, _normalize(structure_direction, 0.0, 0.65),
                                     np.where(structural_bear, _normalize(-structure_direction, 0.0, 0.65), 0.0))
    htf_agreement_score = np.full(n, 0.65 if not htf_enabled else 0.45)
    if htf_enabled:
        htf_agreement_score = np.where(
            (structural_bull & htf_bull) | (structural_bear & htf_bear), 1.0,
            np.where((structural_bull & htf_bear) | (structural_bear & htf_bull), 0.0, 0.45))
    trend_quality = _clamp(
        0.30 * efficiency_ratio + 0.22 * adx_expansion + 0.18 * directional_agreement
        + 0.15 * (1.0 - average_overlap) + 0.15 * htf_agreement_score, 0.0, 1.0)
    balance_quality = _clamp(0.40 * balance_score + 0.35 * compression_score + 0.25 * average_overlap, 0.0, 1.0)

    ready_components = [structure_fast, structure_slow, atr_a, adx, atr_baseline, bb_width_baseline,
                        residual_deviation, average_overlap, momentum_slow, momentum_delta_dev]
    local_ready = np.ones(n, dtype=bool)
    for component in ready_components:
        local_ready &= np.isfinite(component)
    if htf_enabled:
        htf_ready = np.isfinite(htf_close) & np.isfinite(htf_fast) & np.isfinite(htf_slow) & \
            np.isfinite(htf_slow_previous) & np.isfinite(htf_atr)
    else:
        htf_ready = np.ones(n, dtype=bool)
    engine_ready = local_ready & htf_ready

    # --- 正交轴 -------------------------------------------------------------
    structure_regime = np.zeros(n, dtype=np.int64)
    structure_regime = np.where(engine_ready & balance_candidate, REGIME_BALANCED, structure_regime)
    structure_regime = np.where(engine_ready & ~balance_candidate & structural_bull, REGIME_BULLISH, structure_regime)
    structure_regime = np.where(engine_ready & ~balance_candidate & ~structural_bull & structural_bear,
                                REGIME_BEARISH, structure_regime)
    structure_regime = np.where(engine_ready & ~balance_candidate & ~structural_bull & ~structural_bear
                                & (balance_score >= 0.44), REGIME_BALANCED, structure_regime)

    contracting = compression_score
    expanding = np.fmax(bull_expansion_score, bear_expansion_score)
    exhausting = np.where(has_exhaustion_trigger, exhaustion_score, 0.0)
    strongest = np.fmax(contracting, np.fmax(expanding, exhausting))
    energy_phase = np.zeros(n, dtype=np.int64)
    active = engine_ready & (strongest >= 0.45)
    energy_phase = np.where(active & (strongest == contracting), ENERGY_CONTRACTING, energy_phase)
    energy_phase = np.where(active & (strongest == exhausting) & (strongest != contracting), ENERGY_EXHAUSTING, energy_phase)
    energy_phase = np.where(active & (strongest == expanding) & (strongest != contracting) & (strongest != exhausting),
                            ENERGY_EXPANDING, energy_phase)

    # --- 候选分数 -----------------------------------------------------------
    if require_htf_agreement:
        # 注意：不能写 `(~flag) | array` —— Python 的 ~True 是 -2（按位），
        # 恒为真值，会把「必须 HTF 同向」这条门静默吃掉。
        bull_expansion_allowed = structural_bull & htf_bull
        bear_expansion_allowed = structural_bear & htf_bear
    else:
        bull_expansion_allowed = structural_bull
        bear_expansion_allowed = structural_bear
    bull_exhausted = structural_bull & (exhaustion_score >= exhaustion_threshold) & has_exhaustion_trigger
    bear_exhausted = structural_bear & (exhaustion_score >= exhaustion_threshold) & has_exhaustion_trigger
    bull_expanding = bull_expansion_allowed & (bull_expansion_score >= expansion_threshold) & \
        (signed_efficiency >= efficiency_trend_threshold)
    bear_expanding = bear_expansion_allowed & (bear_expansion_score >= expansion_threshold) & \
        (signed_efficiency <= -efficiency_trend_threshold)

    balance_candidate_score = np.where(balance_candidate, balance_score * (1.0 - 0.20 * compression_score), 0.0)
    compression_candidate_score = np.where(compression_candidate, 0.72 * compression_score + 0.28 * balance_score, 0.0)
    bull_expansion_candidate_score = np.where(bull_expanding, 0.72 * bull_expansion_score + 0.28 * structure_bull_score, 0.0)
    bear_expansion_candidate_score = np.where(bear_expanding, 0.72 * bear_expansion_score + 0.28 * structure_bear_score, 0.0)
    bull_exhaustion_candidate_score = np.where(bull_exhausted, 0.72 * exhaustion_score + 0.28 * structure_bull_score, 0.0)
    bear_exhaustion_candidate_score = np.where(bear_exhausted, 0.72 * exhaustion_score + 0.28 * structure_bear_score, 0.0)

    # --- 脉冲记忆（顺序） ---------------------------------------------------
    candidate_states_scores = [
        (STATE_BALANCE, balance_candidate_score),
        (STATE_COMPRESSION, compression_candidate_score),
        (STATE_BULL_EXPANSION, bull_expansion_candidate_score),
        (STATE_BEAR_EXPANSION, bear_expansion_candidate_score),
        (STATE_BULL_EXHAUSTION, bull_exhaustion_candidate_score),
        (STATE_BEAR_EXHAUSTION, bear_exhaustion_candidate_score),
    ]

    raw_state = np.full(n, STATE_TRANSITION, dtype=np.int64)
    best_score_arr = np.zeros(n)
    dominance_arr = np.zeros(n)
    for i in range(n):
        if not engine_ready[i]:
            continue
        best, second, best_state = 0.0, 0.0, STATE_TRANSITION
        for code, score_arr in candidate_states_scores:
            value = score_arr[i]
            if value > best:
                second, best, best_state = best, value, code
            elif value > second:
                second = value
        best_score_arr[i] = best
        dominance_arr[i] = best - second
        if best >= minimum_state_score and (best - second) >= state_dominance_margin:
            raw_state[i] = best_state

    raw_confidence = np.where(
        raw_state == STATE_TRANSITION,
        np.where(engine_ready,
                 _clamp(np.where(best_score_arr < minimum_state_score,
                                 1.0 - best_score_arr, 1.0 - dominance_arr), 0.0, 1.0), 0.0),
        best_score_arr)
    state_quality_raw = np.where(
        raw_state == STATE_TRANSITION,
        _clamp(np.fmax(trend_quality, balance_quality) * (1.0 - dominance_arr), 0.0, 1.0),
        best_score_arr)

    # --- 状态机 -------------------------------------------------------------
    confirmed_state = np.full(n, STATE_TRANSITION, dtype=np.int64)
    bars_in_state = np.zeros(n, dtype=np.int64)
    transition_reason = np.zeros(n, dtype=np.int64)
    confirmed_reason = TRANSITION_NO_DATA

    state = STATE_TRANSITION
    candidate = STATE_TRANSITION
    candidate_bars = 0
    bars_confirmed = 0
    candidate_start_close = math.nan
    candidate_high = math.nan
    candidate_low = math.nan

    # 脉冲记忆
    active_dir = 0
    active_start = active_end = active_high = active_low = math.nan
    active_atr_sum = active_quality_sum = 0.0
    active_bars = 0
    last_dir = 0
    last_end_bar = -1
    last_start = last_end = last_high = last_low = math.nan
    last_counter_extreme = last_counter_close = math.nan
    last_size_atr = last_gross_atr = last_efficiency = last_quality = math.nan
    last_bars = 0
    market_quality_now = 0.0

    correction_memory = max(20, compression_len)
    pullback_memory = max(20, compression_len)
    min_state_bars_eff = minimum_state_bars

    impulse_dir_out = np.zeros(n, dtype=np.int64)
    impulse_intact_out = np.zeros(n, dtype=np.int64)
    correction_depth_out = np.full(n, np.nan)
    long_ctx = np.full(n, CONTEXT_NEUTRAL, dtype=np.int64)
    short_ctx = np.full(n, CONTEXT_NEUTRAL, dtype=np.int64)
    long_ready = np.zeros(n, dtype=np.int64)
    short_ready = np.zeros(n, dtype=np.int64)
    bull_pullback_out = np.zeros(n, dtype=np.int64)
    bear_pullback_out = np.zeros(n, dtype=np.int64)
    market_quality_out = np.zeros(n)

    for i in range(n):
        # 迁移原因
        if not engine_ready[i]:
            reason = TRANSITION_NO_DATA
        elif raw_state[i] != STATE_TRANSITION:
            reason = TRANSITION_TREND_BUILD if state == STATE_TRANSITION else TRANSITION_NONE
        else:
            if best_score_arr[i] >= minimum_state_score and dominance_arr[i] < state_dominance_margin:
                reason = TRANSITION_CONFLICT
            elif ((state in (STATE_BULL_EXPANSION, STATE_BULL_PULLBACK, STATE_BULL_EXHAUSTION) and not structural_bull[i])
                  or (state in (STATE_BEAR_EXPANSION, STATE_BEAR_PULLBACK, STATE_BEAR_EXHAUSTION) and not structural_bear[i])):
                reason = TRANSITION_BREAK
            elif structure_regime[i] in (REGIME_BULLISH, REGIME_BEARISH):
                reason = TRANSITION_TREND_BUILD
            else:
                reason = TRANSITION_CHAOTIC

        opposite = ((state in (STATE_BULL_EXPANSION, STATE_BULL_PULLBACK, STATE_BULL_EXHAUSTION)
                     and raw_state[i] == STATE_BEAR_EXPANSION)
                    or (state in (STATE_BEAR_EXPANSION, STATE_BEAR_PULLBACK, STATE_BEAR_EXHAUSTION)
                        and raw_state[i] == STATE_BULL_EXPANSION))
        emergency = opposite and raw_confidence[i] >= opposite_break_threshold

        if engine_ready[i]:
            if raw_state[i] == candidate:
                candidate_bars += 1
                candidate_high = max(candidate_high if np.isfinite(candidate_high) else high[i], high[i])
                candidate_low = min(candidate_low if np.isfinite(candidate_low) else low[i], low[i])
            else:
                candidate = raw_state[i]
                candidate_bars = 1
                candidate_start_close = close[i - 1] if i > 0 else close[i]
                candidate_high = max(high[i], candidate_start_close)
                candidate_low = min(low[i], candidate_start_close)
            bars_confirmed += 1
            normal_allowed = bars_confirmed >= min_state_bars_eff and candidate_bars >= confirm_bars
            if candidate != state and (normal_allowed or emergency):
                state = candidate
                bars_confirmed = 0
            if state == STATE_TRANSITION:
                confirmed_reason = reason

        # 脉冲结算
        current_dir = 1 if state == STATE_BULL_EXPANSION else (-1 if state == STATE_BEAR_EXPANSION else 0)
        if active_dir != 0 and current_dir != active_dir:
            avg_atr = active_atr_sum / active_bars if active_bars > 0 else safe_atr[i]
            gross = max(active_high - active_low, floor_tick)
            net = (active_end - active_start) if active_dir == 1 else (active_start - active_end)
            last_dir = active_dir
            last_end_bar = i - 1
            last_start, last_end = active_start, active_end
            last_high, last_low = active_high, active_low
            last_counter_extreme = active_high if active_dir == 1 else active_low
            last_counter_close = active_end
            last_size_atr = max(net, 0.0) / avg_atr if avg_atr else 0.0
            last_gross_atr = gross / avg_atr if avg_atr else 0.0
            last_efficiency = float(_clamp(max(net, 0.0) / gross, 0.0, 1.0))
            last_quality = active_quality_sum / active_bars if active_bars > 0 else 0.0
            last_bars = active_bars
            active_dir = 0
            active_start = active_end = active_high = active_low = math.nan
            active_atr_sum = active_quality_sum = 0.0
            active_bars = 0
        if current_dir != 0:
            if active_dir == 0:
                active_dir = current_dir
                active_start = candidate_start_close if np.isfinite(candidate_start_close) else close[i - 1 if i else 0]
                active_end = close[i]
                active_high = max(candidate_high if np.isfinite(candidate_high) else high[i], high[i])
                active_low = min(candidate_low if np.isfinite(candidate_low) else low[i], low[i])
                active_atr_sum = safe_atr[i]
                active_quality_sum = market_quality_now
                active_bars = 1
            else:
                active_end = close[i]
                active_high = max(active_high, high[i])
                active_low = min(active_low, low[i])
                active_atr_sum += safe_atr[i]
                active_quality_sum += market_quality_now
                active_bars += 1
        if last_dir == 1 and current_dir == 0:
            last_counter_extreme = min(last_counter_extreme if np.isfinite(last_counter_extreme) else last_high, low[i])
            last_counter_close = min(last_counter_close if np.isfinite(last_counter_close) else last_end, close[i])
        elif last_dir == -1 and current_dir == 0:
            last_counter_extreme = max(last_counter_extreme if np.isfinite(last_counter_extreme) else last_low, high[i])
            last_counter_close = max(last_counter_close if np.isfinite(last_counter_close) else last_end, close[i])

        # 拉回检测
        prior_age = (pullback_memory + 1) if last_end_bar < 0 else (i - last_end_bar)
        recent_bull_impulse = active_dir == 1 or (last_dir == 1 and prior_age <= pullback_memory
                                                  and np.nan_to_num(last_efficiency) >= 0.45)
        recent_bear_impulse = active_dir == -1 or (last_dir == -1 and prior_age <= pullback_memory
                                                   and np.nan_to_num(last_efficiency) >= 0.45)
        bull_extreme = active_high if active_dir == 1 else last_high
        bull_start = active_start if active_dir == 1 else last_start
        bear_extreme = active_low if active_dir == -1 else last_low
        bear_start = active_start if active_dir == -1 else last_start
        bull_depth = (float(_clamp(_safe_div(bull_extreme - close[i], bull_extreme - bull_start, 0.0), 0.0, 1.50))
                      if recent_bull_impulse and np.isfinite(bull_extreme) and np.isfinite(bull_start) else math.nan)
        bear_depth = (float(_clamp(_safe_div(close[i] - bear_extreme, bear_start - bear_extreme, 0.0), 0.0, 1.50))
                      if recent_bear_impulse and np.isfinite(bear_extreme) and np.isfinite(bear_start) else math.nan)
        location_score = 1.0 - float(_normalize(abs(close[i] - structure_slow[i]), safe_atr[i] * 0.10, safe_atr[i] * 1.20))

        bull_pullback_score = 0.0
        if recent_bull_impulse and structural_bull[i] and np.isfinite(bull_depth):
            bull_pullback_score = float(_clamp(
                0.30 * structure_bull_score[i] + 0.25 * location_score
                + 0.20 * (1.0 - bear_expansion_score[i])
                + 0.15 * (1.0 - float(_normalize(bull_depth, 0.55, 0.85)))
                + 0.10 * float(_normalize(bull_depth, 0.08, 0.30)), 0.0, 1.0))
        bear_pullback_score = 0.0
        if recent_bear_impulse and structural_bear[i] and np.isfinite(bear_depth):
            bear_pullback_score = float(_clamp(
                0.30 * structure_bear_score[i] + 0.25 * location_score
                + 0.20 * (1.0 - bull_expansion_score[i])
                + 0.15 * (1.0 - float(_normalize(bear_depth, 0.55, 0.85)))
                + 0.10 * float(_normalize(bear_depth, 0.08, 0.30)), 0.0, 1.0))

        bull_pullback = (bull_pullback_score >= minimum_state_score and close[i] < structure_fast[i]
                         and close[i] >= structure_slow[i] - safe_atr[i] * 0.35
                         and np.isfinite(bull_depth) and 0.08 < bull_depth < 0.80 and fast_slope[i] > -0.12)
        bear_pullback = (bear_pullback_score >= minimum_state_score and close[i] > structure_fast[i]
                         and close[i] <= structure_slow[i] + safe_atr[i] * 0.35
                         and np.isfinite(bear_depth) and 0.08 < bear_depth < 0.80 and fast_slope[i] < 0.12)
        bull_pullback_out[i] = 1 if bull_pullback else 0
        bear_pullback_out[i] = 1 if bear_pullback else 0

        # 脉冲上下文
        impulse_age = (correction_memory + 1) if last_end_bar < 0 else (i - last_end_bar)
        has_recent_impulse = (last_dir != 0 and 1 <= impulse_age <= correction_memory and last_bars >= 1
                              and np.nan_to_num(last_size_atr) >= 0.80
                              and np.nan_to_num(last_efficiency) >= 0.45
                              and np.nan_to_num(last_quality) >= 0.35)
        correction_depth = math.nan
        impulse_intact = False
        if has_recent_impulse:
            wick_range = (max(np.nan_to_num(last_high - last_start), floor_tick) if last_dir == 1
                          else max(np.nan_to_num(last_start - last_low), floor_tick))
            close_range = max(abs(np.nan_to_num(last_end - last_start)), floor_tick)
            wick_depth = (float(_safe_div(last_high - last_counter_extreme, wick_range, 0.0)) if last_dir == 1
                          else float(_safe_div(last_counter_extreme - last_low, wick_range, 0.0)))
            close_depth = (float(_safe_div(last_end - last_counter_close, close_range, 0.0)) if last_dir == 1
                           else float(_safe_div(last_counter_close - last_end, close_range, 0.0)))
            correction_depth = float(_clamp(0.35 * wick_depth + 0.65 * close_depth, 0.0, 1.50))
            opposite_structure = (last_dir == 1 and structural_bear[i]) or (last_dir == -1 and structural_bull[i])
            if last_dir == 1:
                structure_held = structure_fast[i] >= structure_slow[i] - safe_atr[i] * 0.20 and slow_slope[i] > -0.08
            elif last_dir == -1:
                structure_held = structure_fast[i] <= structure_slow[i] + safe_atr[i] * 0.20 and slow_slope[i] < 0.08
            else:
                structure_held = False
            htf_compatible = (not htf_enabled) or (last_dir == 1 and not htf_bear[i]) or \
                (last_dir == -1 and not htf_bull[i])
            impulse_intact = (correction_depth < 0.75) and (not opposite_structure) and structure_held and htf_compatible
        correction_depth_out[i] = correction_depth
        impulse_dir_out[i] = last_dir
        impulse_intact_out[i] = 1 if impulse_intact else 0

        # 上下文等级
        lvl, slv = CONTEXT_NEUTRAL, CONTEXT_NEUTRAL
        if engine_ready[i]:
            if state == STATE_BULL_EXPANSION:
                lvl = CONTEXT_NEUTRAL if normalized_location[i] >= extreme_location_threshold else CONTEXT_COMPATIBLE
                slv = CONTEXT_UNFAVORABLE
            elif state == STATE_BEAR_EXPANSION:
                lvl = CONTEXT_UNFAVORABLE
                slv = CONTEXT_NEUTRAL if normalized_location[i] <= -extreme_location_threshold else CONTEXT_COMPATIBLE
            elif state == STATE_BULL_PULLBACK:
                bull_reaction = (close[i] > open_a[i] and close_location[i] >= 0.60
                                 and (low[i] <= structure_slow[i] + safe_atr[i] * 0.20 or low[i] <= value_upper[i])
                                 and directional_bias[i] > 0)
                lvl = CONTEXT_PREFERRED if bull_reaction else CONTEXT_COMPATIBLE
                slv = CONTEXT_UNFAVORABLE
            elif state == STATE_BEAR_PULLBACK:
                bear_reaction = (close[i] < open_a[i] and close_location[i] <= 0.40
                                 and (high[i] >= structure_slow[i] - safe_atr[i] * 0.20 or high[i] >= value_lower[i])
                                 and directional_bias[i] < 0)
                lvl = CONTEXT_UNFAVORABLE
                slv = CONTEXT_PREFERRED if bear_reaction else CONTEXT_COMPATIBLE
            elif state == STATE_BULL_EXHAUSTION:
                lvl = CONTEXT_NEUTRAL
                slv = CONTEXT_COMPATIBLE if normalized_location[i] >= premium_threshold else CONTEXT_NEUTRAL
            elif state == STATE_BEAR_EXHAUSTION:
                lvl = CONTEXT_COMPATIBLE if normalized_location[i] <= discount_threshold else CONTEXT_NEUTRAL
                slv = CONTEXT_NEUTRAL
            elif state in (STATE_BALANCE, STATE_COMPRESSION):
                if has_recent_impulse and impulse_intact:
                    lvl = CONTEXT_COMPATIBLE if last_dir == 1 else CONTEXT_UNFAVORABLE
                    slv = CONTEXT_COMPATIBLE if last_dir == -1 else CONTEXT_UNFAVORABLE
                else:
                    lvl = CONTEXT_UNFAVORABLE if normalized_location[i] >= premium_threshold else CONTEXT_NEUTRAL
                    slv = CONTEXT_UNFAVORABLE if normalized_location[i] <= discount_threshold else CONTEXT_NEUTRAL
                if (normalized_location[i] <= discount_threshold and directional_bias[i] > 0
                        and not (has_recent_impulse and impulse_intact and last_dir == -1)):
                    lvl = CONTEXT_PREFERRED
                if (normalized_location[i] >= premium_threshold and directional_bias[i] < 0
                        and not (has_recent_impulse and impulse_intact and last_dir == 1)):
                    slv = CONTEXT_PREFERRED
                if state == STATE_COMPRESSION and directional_bias[i] > directional_bias_threshold and lvl >= CONTEXT_NEUTRAL:
                    lvl = max(lvl, CONTEXT_COMPATIBLE)
                if state == STATE_COMPRESSION and directional_bias[i] < -directional_bias_threshold and slv >= CONTEXT_NEUTRAL:
                    slv = max(slv, CONTEXT_COMPATIBLE)
            if htf_enabled and htf_bear[i]:
                lvl = min(lvl, CONTEXT_NEUTRAL)
            if htf_enabled and htf_bull[i]:
                slv = min(slv, CONTEXT_NEUTRAL)
        long_ctx[i], short_ctx[i] = lvl, slv

        # 就绪度
        l_ready = (lvl >= CONTEXT_COMPATIBLE and (
            (state == STATE_BULL_PULLBACK and close[i] > open_a[i] and close_location[i] >= 0.60
             and directional_bias[i] > 0)
            or (state == STATE_COMPRESSION and directional_bias[i] > directional_bias_threshold)
            or (state == STATE_BALANCE and normalized_location[i] <= discount_threshold and directional_bias[i] > 0)))
        s_ready = (slv >= CONTEXT_COMPATIBLE and (
            (state == STATE_BEAR_PULLBACK and close[i] < open_a[i] and close_location[i] <= 0.40
             and directional_bias[i] < 0)
            or (state == STATE_COMPRESSION and directional_bias[i] < -directional_bias_threshold)
            or (state == STATE_BALANCE and normalized_location[i] >= premium_threshold and directional_bias[i] < 0)))
        long_ready[i] = 1 if l_ready else 0
        short_ready[i] = 1 if s_ready else 0

        # 质量映射
        quality_map = {
            STATE_BALANCE: balance_candidate_score[i],
            STATE_COMPRESSION: compression_candidate_score[i],
            STATE_BULL_EXPANSION: bull_expansion_candidate_score[i],
            STATE_BEAR_EXPANSION: bear_expansion_candidate_score[i],
            STATE_BULL_PULLBACK: bull_pullback_score,
            STATE_BEAR_PULLBACK: bear_pullback_score,
            STATE_BULL_EXHAUSTION: bull_exhaustion_candidate_score[i],
            STATE_BEAR_EXHAUSTION: bear_exhaustion_candidate_score[i],
        }
        market_quality_now = quality_map.get(state, state_quality_raw[i])
        market_quality_out[i] = market_quality_now

        confirmed_state[i] = state
        bars_in_state[i] = bars_confirmed
        transition_reason[i] = reason

    state_changed = np.zeros(n, dtype=np.int64)
    state_changed[1:] = (confirmed_state[1:] != confirmed_state[:-1]).astype(np.int64)

    def as_int(values) -> pd.Series:
        return pd.Series(np.asarray(values).astype(np.int64), index=idx).astype("Int64")

    data = {
        "mse_engine_ready": as_int(engine_ready),
        "mse_structure_fast": structure_fast,
        "mse_structure_slow": structure_slow,
        "mse_fast_slope": fast_slope,
        "mse_slow_slope": slow_slope,
        "mse_structure_separation": structure_separation,
        "mse_structure_bull_score": structure_bull_score,
        "mse_structure_direction": structure_direction,
        "mse_structural_bull": as_int(structural_bull),
        "mse_structural_bear": as_int(structural_bear),
        "mse_structure_regime": as_int(structure_regime),
        "mse_energy_phase": as_int(energy_phase),
        "mse_htf_bull": as_int(htf_bull),
        "mse_htf_bear": as_int(htf_bear),
        "mse_htf_slow_slope": htf_slow_slope,
        "mse_adx": adx,
        "mse_plus_di": plus_di,
        "mse_minus_di": minus_di,
        "mse_adx_expansion": adx_expansion,
        "mse_range_expansion": range_expansion,
        "mse_atr_acceleration": atr_acceleration,
        "mse_close_location": close_location,
        "mse_body_efficiency": body_efficiency,
        "mse_bull_body_quality": bull_body_quality,
        "mse_bear_body_quality": bear_body_quality,
        "mse_directional_progress": directional_progress,
        "mse_efficiency_ratio": efficiency_ratio,
        "mse_signed_efficiency": signed_efficiency,
        "mse_average_overlap": average_overlap,
        "mse_fair_value": fair_value,
        "mse_value_upper": value_upper,
        "mse_value_lower": value_lower,
        "mse_normalized_location": normalized_location,
        "mse_range_location": range_location,
        "mse_compression_score": compression_score,
        "mse_balance_score": balance_score,
        "mse_bull_expansion_score": bull_expansion_score,
        "mse_bear_expansion_score": bear_expansion_score,
        "mse_expansion_score": expansion_score,
        "mse_directional_bias": directional_bias,
        "mse_smoothed_directional_bias": smoothed_directional_bias,
        "mse_signed_close_position": signed_close_position,
        "mse_volume_z": volume_z,
        "mse_effort_result_mismatch": effort_result_mismatch,
        "mse_exhaustion_score": exhaustion_score,
        "mse_has_exhaustion_trigger": as_int(has_exhaustion_trigger),
        "mse_trend_quality": trend_quality,
        "mse_balance_quality": balance_quality,
        "mse_raw_state": as_int(raw_state),
        "mse_raw_confidence": raw_confidence,
        "mse_best_state_score": best_score_arr,
        "mse_state_dominance": dominance_arr,
        "mse_state": as_int(confirmed_state),
        "mse_state_changed": as_int(state_changed),
        "mse_bars_in_state": as_int(bars_in_state),
        "mse_state_quality": market_quality_out,
        "mse_transition_reason": as_int(transition_reason),
        "mse_impulse_direction": as_int(impulse_dir_out),
        "mse_impulse_intact": as_int(impulse_intact_out),
        "mse_correction_depth": correction_depth_out,
        "mse_bull_pullback": as_int(bull_pullback_out),
        "mse_bear_pullback": as_int(bear_pullback_out),
        "mse_long_context": as_int(long_ctx),
        "mse_short_context": as_int(short_ctx),
        "mse_long_readiness": as_int(long_ready),
        "mse_short_readiness": as_int(short_ready),
    }
    return pd.DataFrame(data, index=idx)
