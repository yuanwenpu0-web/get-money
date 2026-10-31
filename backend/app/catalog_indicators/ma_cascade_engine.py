"""MA Cascade Engine [WavesUnchained] v1.5.0 — 均线级联引擎。

Port of ``data/pine-collection/indicators/trend_direction/ma_cascade_engine/ma_cascade_engine.pine``.

Pine 原脚本把均线当成一个四层结构，而不是孤立的交叉信号：

1. **Visual**：Fast / Base / Anchor 三条均线（8 种平滑可选）
2. **Feature**：方向、ATR 归一化斜率、曲率、分离度、价格延伸、交叉频率、均线尊重度
3. **State**：扩张 / 回调 / 中性压缩 / 过渡 / 过度延伸
4. **Decision**：permission（方向许可）→ setup（回调到位）→ trigger（重新夺回均线）→ exit

本移植保留 **全部四层** 的数值与布尔输出，丢弃绘图、状态标签与告警（``alertcondition``
里的每一条都已能由输出列组合出来）。

``f_jma`` 是开源的 Jurik 风格近似，不是 Mark Jurik 的专利原算法 —— 与 Pine 一致。

输出 ``mce_`` 前缀列：三条均线及其斜率/曲率/分离度、波动率比、交叉计数、
状态码、总分与各分项、尊重率、以及 permission/setup/trigger/exit 的布尔标志。
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
    highest,
    hma,
    integer,
    lowest,
    number,
    ohlcv,
    rma,
    sma,
    wma,
)

STATE_BULL_EXPANSION = 1
STATE_BULL_PULLBACK = 2
STATE_NEUTRAL = 3
STATE_BEAR_PULLBACK = 4
STATE_BEAR_EXPANSION = 5
STATE_OVEREXTENDED = 6
STATE_TRANSITION = 7

_PRESETS = ("Responsive", "Balanced", "Structural", "Custom")
_MA_TYPES = ("JMA", "EMA", "SMA", "RMA", "WMA", "HMA", "DEMA", "KAMA")


def _clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Pine ``f_clamp``：``math.max(low, math.min(high, value))``，na 传播。"""
    return np.maximum(low, np.minimum(high, values))


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Pine ``f_safeDivide`` 的除零守卫。

    分母为 0 时返回 0.0（Pine 的原语义）；分母或分子为 na 时保持 na ——
    这样暖机期的斜率/分离度是“暂无值”，不会退化成 0 而伪造出“走平”的假象。
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    zero = np.isfinite(den) & (den == 0.0)
    safe_den = np.where(den == 0.0, 1.0, den)
    return np.where(zero, 0.0, num / safe_den)


def _sign(values: np.ndarray) -> np.ndarray:
    """Pine ``f_sign``：+1 / -1 / 0。"""
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 0.0, 1.0, np.where(arr < 0.0, -1.0, 0.0))


def _shift(values: np.ndarray, length: int) -> np.ndarray:
    """``series[length]``：前 ``length`` 根补 na。"""
    arr = np.asarray(values, dtype=float)
    if length <= 0:
        return arr.copy()
    pad = np.full(min(length, len(arr)), np.nan)
    return np.concatenate([pad, arr[: len(arr) - length]]) if len(arr) > length else np.full(len(arr), np.nan)


def _shift_bool(values: np.ndarray, length: int) -> np.ndarray:
    """布尔序列的 ``[length]``，补 False（Pine 里 na 视作假）。"""
    arr = np.asarray(values, dtype=bool)
    if length <= 0:
        return arr.copy()
    if len(arr) <= length:
        return np.zeros(len(arr), dtype=bool)
    return np.concatenate([np.zeros(length, dtype=bool), arr[: len(arr) - length]])


def _jma(src: np.ndarray, length: int, phase: int, power: float) -> np.ndarray:
    """Pine ``f_jma``：开源 Jurik 风格近似（非专利原算法）。"""
    safe_length = max(length, 1)
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (safe_length - 1.0) / (0.45 * (safe_length - 1.0) + 2.0)
    alpha = beta ** power

    n = len(src)
    out = np.full(n, np.nan)
    e0 = e1 = e2 = result = math.nan
    for i in range(n):
        price = src[i]
        prev_result = price if not np.isfinite(result) else result
        e0 = (1.0 - alpha) * price + alpha * (price if not np.isfinite(e0) else e0)
        e1 = (price - e0) * (1.0 - beta) + beta * (0.0 if not np.isfinite(e1) else e1)
        e2 = (e0 + phase_ratio * e1 - prev_result) * (1.0 - alpha) ** 2 + alpha ** 2 * (
            0.0 if not np.isfinite(e2) else e2
        )
        result = e2 + prev_result
        out[i] = result
    return out


def _dema(series: pd.Series, length: int) -> pd.Series:
    """Pine ``f_dema``：2·EMA − EMA(EMA)。"""
    first = ema(series, length)
    return 2.0 * first - ema(first, length)


def _kama(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ``f_kama``：Kaufman 自适应均线（ER 驱动平滑常数）。"""
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    series = pd.Series(src)
    change = np.abs(src - _shift(src, length))
    volatility = sma((series - series.shift(1)).abs(), length).to_numpy(dtype=float) * length
    efficiency = _safe_div(change, volatility)
    smoothing = np.power(efficiency * (fast_sc - slow_sc) + slow_sc, 2.0)

    n = len(src)
    out = np.full(n, np.nan)
    result = math.nan
    for i in range(n):
        previous = src[i] if not np.isfinite(result) else result
        result = previous + smoothing[i] * (src[i] - previous)
        out[i] = result
    return out


def _ma(src: np.ndarray, length: int, kind: str, phase: int, power: float) -> np.ndarray:
    """Pine ``f_ma``：八种平滑的派发（未知类型回落 EMA）。"""
    series = pd.Series(src)
    if kind == "JMA":
        return _jma(src, length, phase, power)
    if kind == "EMA":
        return ema(series, length).to_numpy(dtype=float)
    if kind == "SMA":
        return sma(series, length).to_numpy(dtype=float)
    if kind == "RMA":
        return rma(series, length).to_numpy(dtype=float)
    if kind == "WMA":
        return wma(series, length).to_numpy(dtype=float)
    if kind == "HMA":
        return hma(series, length).to_numpy(dtype=float)
    if kind == "DEMA":
        return _dema(series, length).to_numpy(dtype=float)
    if kind == "KAMA":
        return _kama(src, length)
    return ema(series, length).to_numpy(dtype=float)


def _rolling_count(values: np.ndarray, length: int) -> np.ndarray:
    """``ta.sma(cond ? 1 : 0, length) * length``：窗口内事件计数（暖机期为 na）。"""
    return sma(pd.Series(np.asarray(values, dtype=float)), length).to_numpy(dtype=float) * length


def _crossover(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pine ``ta.crossover``：本根在上方、上一根在下方或相等（na 一律假）。"""
    return (left > right) & (_shift(left, 1) <= _shift(right, 1))


def _crossunder(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pine ``ta.crossunder``：本根在下方、上一根在上方或相等。"""
    return (left < right) & (_shift(left, 1) >= _shift(right, 1))


def _rate(count: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Pine 的 ``total > 0 ? count / total : 0``。"""
    return np.where(np.asarray(total) > 0.0, np.asarray(count) / np.where(total > 0, total, 1.0), 0.0)


def ma_cascade_engine(
    frame: pd.DataFrame,
    *,
    preset: str = "Balanced",
    source: str = "close",
    jma_phase: int = 0,
    jma_power: float = 2.0,
    custom_fast_type: str = "JMA",
    custom_fast_length: int = 21,
    custom_base_type: str = "EMA",
    custom_base_length: int = 50,
    custom_anchor_type: str = "SMA",
    custom_anchor_length: int = 200,
    atr_length: int = 14,
    slope_lookback: int = 5,
    minimum_fast_slope: float = 0.015,
    minimum_base_slope: float = 0.008,
    curvature_lookback: int = 3,
    minimum_curvature: float = 0.005,
    entanglement_spread: float = 0.15,
    cross_frequency_lookback: int = 30,
    maximum_crosses: int = 6,
    pullback_distance: float = 0.70,
    pullback_spread: float = 0.75,
    maximum_extension: float = 1.75,
    extreme_volatility_ratio: float = 1.80,
    volatility_baseline: int = 100,
    use_respect: bool = True,
    use_anchor_respect: bool = True,
    touch_tolerance: float = 0.15,
    reaction_bars: int = 5,
    reaction_move: float = 0.50,
    respect_invalidation: float = 0.35,
    respect_lookback: int = 150,
    require_price_confirmation: bool = True,
    require_curvature: bool = True,
    strong_bias_threshold: float = 60.0,
    directional_bias_threshold: float = 25.0,
    mintick: float | None = None,
) -> pd.DataFrame:
    """MA Cascade Engine —— 均线级联的方向/状态/决策输出。

    Args:
        frame: 含 ``open`` / ``high`` / ``low`` / ``close`` 的 OHLC，索引任意。
        preset: ``Responsive`` / ``Balanced`` / ``Structural`` / ``Custom``。
            前三个会覆盖三条均线的类型与长度；``Custom`` 才读下面的 ``custom_*``。
        source: 均线输入序列，可以是列名（``close``/``open``/``hl2``/``hlc3``/``ohlc4``）。
        jma_phase: JMA 相位，-100..100。
        jma_power: JMA 幂次，1.0..8.0。
        custom_fast_type: 快线类型（Custom 预设）。
        custom_fast_length: 快线长度。
        custom_base_type: 基线类型。
        custom_base_length: 基线长度。
        custom_anchor_type: 锚线类型。
        custom_anchor_length: 锚线长度（至少 2）。
        atr_length: ATR 长度。
        slope_lookback: 斜率回看 bar 数。
        minimum_fast_slope: 触发器要求的每 bar 最小快线斜率（ATR 归一化）。
        minimum_base_slope: 方向性 regime 要求的每 bar 最小基线斜率。
        curvature_lookback: 曲率回看 bar 数。
        minimum_curvature: 触发所需的最小曲率。
        entanglement_spread: 两条均线间距都小于该 ATR 值时视为缠绕。
        cross_frequency_lookback: 交叉频率统计窗口。
        maximum_crosses: 窗口内允许的最大交叉次数。
        pullback_distance: setup 允许的价格距基线最大 ATR 距离。
        pullback_spread: setup 允许的快/基线最大间距（ATR）。
        maximum_extension: 价格距快线的 ATR 延伸上限（超过即禁止新入场）。
        extreme_volatility_ratio: ATR / 长期 ATR 超过该值视为极端波动（扣分）。
        volatility_baseline: ATR 基线长度。
        use_respect: 是否把基线尊重度计入总分。
        use_anchor_respect: 是否把锚线尊重度计入总分。
        touch_tolerance: 触摸基/锚线的 ATR 容差。
        reaction_bars: 触摸后评估反应的水平 bar 数。
        reaction_move: 要求的反应幅度（ATR）。
        respect_invalidation: 允许的不利穿透深度（ATR）。
        respect_lookback: 尊重度统计窗口。
        require_price_confirmation: 触发是否要求突破前一根的高/低。
        require_curvature: 触发是否要求曲率为正/负。
        strong_bias_threshold: 强偏置阈值（>= 该值为 Strong）。
        directional_bias_threshold: 方向偏置阈值。
        mintick: 最小跳动，``None`` 时按价格量级推断。

    Returns:
        ``pd.DataFrame``，索引与 ``frame`` 一致，列为 ``mce_*``。

    Raises:
        ValueError: 参数越界或取值不在允许集合内。
    """
    preset = choice("preset", preset, _PRESETS)
    custom_fast_type = choice("custom_fast_type", custom_fast_type, _MA_TYPES)
    custom_base_type = choice("custom_base_type", custom_base_type, _MA_TYPES)
    custom_anchor_type = choice("custom_anchor_type", custom_anchor_type, _MA_TYPES)
    jma_phase = integer("jma_phase", jma_phase, -100, 100)
    jma_power = number("jma_power", jma_power, 1.0, 8.0)
    custom_fast_length = integer("custom_fast_length", custom_fast_length, 1, 5000)
    custom_base_length = integer("custom_base_length", custom_base_length, 1, 5000)
    custom_anchor_length = integer("custom_anchor_length", custom_anchor_length, 2, 5000)
    atr_length = integer("atr_length", atr_length, 2, 500)
    slope_lookback = integer("slope_lookback", slope_lookback, 1, 100)
    minimum_fast_slope = number("minimum_fast_slope", minimum_fast_slope, 0.0, 10.0)
    minimum_base_slope = number("minimum_base_slope", minimum_base_slope, 0.0, 10.0)
    curvature_lookback = integer("curvature_lookback", curvature_lookback, 1, 50)
    minimum_curvature = number("minimum_curvature", minimum_curvature, 0.0, 10.0)
    entanglement_spread = number("entanglement_spread", entanglement_spread, 0.01, 100.0)
    cross_frequency_lookback = integer("cross_frequency_lookback", cross_frequency_lookback, 5, 300)
    maximum_crosses = integer("maximum_crosses", maximum_crosses, 1, 50)
    pullback_distance = number("pullback_distance", pullback_distance, 0.05, 100.0)
    pullback_spread = number("pullback_spread", pullback_spread, 0.05, 100.0)
    maximum_extension = number("maximum_extension", maximum_extension, 0.25, 100.0)
    extreme_volatility_ratio = number("extreme_volatility_ratio", extreme_volatility_ratio, 1.0, 100.0)
    volatility_baseline = integer("volatility_baseline", volatility_baseline, 20, 500)
    touch_tolerance = number("touch_tolerance", touch_tolerance, 0.01, 100.0)
    reaction_bars = integer("reaction_bars", reaction_bars, 1, 30)
    reaction_move = number("reaction_move", reaction_move, 0.05, 100.0)
    respect_invalidation = number("respect_invalidation", respect_invalidation, 0.05, 100.0)
    respect_lookback = integer("respect_lookback", respect_lookback, 20, 1000)
    strong_bias_threshold = number("strong_bias_threshold", strong_bias_threshold, 25.0, 100.0)
    directional_bias_threshold = number(
        "directional_bias_threshold", directional_bias_threshold, 5.0, 75.0
    )
    use_respect = boolean("use_respect", use_respect)
    use_anchor_respect = boolean("use_anchor_respect", use_anchor_respect)
    require_price_confirmation = boolean("require_price_confirmation", require_price_confirmation)
    require_curvature = boolean("require_curvature", require_curvature)

    high_s, low_s, close_s = ohlcv(frame, "high", "low", "close")
    n = len(frame)
    close = close_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)

    src = _resolve_source(frame, source, high, low, close)

    # --- 预设解析 -----------------------------------------------------------
    if preset == "Responsive":
        fast_type, fast_length = "JMA", 9
        base_type, base_length = "JMA", 21
        anchor_type, anchor_length = "JMA", 50
    elif preset == "Balanced":
        fast_type, fast_length = "JMA", 21
        base_type, base_length = "EMA", 50
        anchor_type, anchor_length = "SMA", 200
    elif preset == "Structural":
        fast_type, fast_length = "EMA", 20
        base_type, base_length = "SMA", 50
        anchor_type, anchor_length = "SMA", 200
    else:
        fast_type, fast_length = custom_fast_type, custom_fast_length
        base_type, base_length = custom_base_type, custom_base_length
        anchor_type, anchor_length = custom_anchor_type, custom_anchor_length

    fast_ma = _ma(src, fast_length, fast_type, jma_phase, jma_power)
    base_ma = _ma(src, base_length, base_type, jma_phase, jma_power)
    anchor_ma = _ma(src, anchor_length, anchor_type, jma_phase, jma_power)

    # --- 波动率归一化 -------------------------------------------------------
    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    if mintick is None:
        magnitude = np.nanmedian(np.abs(close)) if n else 1.0
        magnitude = magnitude if np.isfinite(magnitude) and magnitude > 0 else 1.0
        tick = 10.0 ** (np.floor(np.log10(magnitude)) - 5)
    else:
        tick = number("mintick", mintick, 1e-12, 1e15)
    safe_atr = np.where(np.isnan(atr_a), np.nan, np.fmax(atr_a, tick))

    atr_baseline = sma(pd.Series(atr_a), volatility_baseline).to_numpy(dtype=float)
    volatility_ratio = _safe_div(atr_a, atr_baseline)
    extreme_volatility = np.isfinite(volatility_ratio) & (volatility_ratio >= extreme_volatility_ratio)

    # --- 方向 ---------------------------------------------------------------
    bullish_order = (fast_ma > base_ma) & (base_ma > anchor_ma)
    bearish_order = (fast_ma < base_ma) & (base_ma < anchor_ma)
    price_above_fast = close > fast_ma
    price_below_fast = close < fast_ma
    price_above_base = close > base_ma
    price_below_base = close < base_ma

    # --- ATR 归一化斜率与曲率 ------------------------------------------------
    denominator = safe_atr * slope_lookback
    fast_slope = _safe_div(fast_ma - _shift(fast_ma, slope_lookback), denominator)
    base_slope = _safe_div(base_ma - _shift(base_ma, slope_lookback), denominator)
    anchor_slope = _safe_div(anchor_ma - _shift(anchor_ma, slope_lookback), denominator)

    fast_rising = fast_slope > minimum_fast_slope
    fast_falling = fast_slope < -minimum_fast_slope
    base_rising = base_slope > minimum_base_slope
    base_falling = base_slope < -minimum_base_slope

    fast_curvature = fast_slope - _shift(fast_slope, curvature_lookback)
    base_curvature = base_slope - _shift(base_slope, curvature_lookback)
    bullish_curvature = fast_curvature > minimum_curvature
    bearish_curvature = fast_curvature < -minimum_curvature

    # --- 分离度 -------------------------------------------------------------
    fast_base_spread = _safe_div(fast_ma - base_ma, safe_atr)
    base_anchor_spread = _safe_div(base_ma - anchor_ma, safe_atr)
    absolute_fast_base = np.abs(fast_base_spread)
    absolute_base_anchor = np.abs(base_anchor_spread)
    fast_base_spread_change = fast_base_spread - _shift(fast_base_spread, 1)
    base_anchor_spread_change = base_anchor_spread - _shift(base_anchor_spread, 1)

    bullish_spread_expansion = (
        (fast_base_spread > 0.0)
        & (base_anchor_spread > 0.0)
        & (fast_base_spread_change > 0.0)
        & (base_anchor_spread_change > 0.0)
    )
    bearish_spread_expansion = (
        (fast_base_spread < 0.0)
        & (base_anchor_spread < 0.0)
        & (fast_base_spread_change < 0.0)
        & (base_anchor_spread_change < 0.0)
    )
    fast_base_compression = absolute_fast_base < _shift(absolute_fast_base, 1)
    base_anchor_compression = absolute_base_anchor < _shift(absolute_base_anchor, 1)
    spread_compression = fast_base_compression | base_anchor_compression

    # --- 价格延伸 -----------------------------------------------------------
    price_extension = _safe_div(close - fast_ma, safe_atr)
    price_base_distance = _safe_div(close - base_ma, safe_atr)
    bullish_overextension = price_extension > maximum_extension
    bearish_overextension = price_extension < -maximum_extension

    # --- 交叉频率与缠绕 -----------------------------------------------------
    price_base_cross = _crossover(close, base_ma) | _crossunder(close, base_ma)
    fast_base_cross = _crossover(fast_ma, base_ma) | _crossunder(fast_ma, base_ma)
    price_base_cross_count = _rolling_count(price_base_cross.astype(float), cross_frequency_lookback)
    fast_base_cross_count = _rolling_count(fast_base_cross.astype(float), cross_frequency_lookback)
    combined_cross_count = price_base_cross_count + fast_base_cross_count

    spread_entanglement = (absolute_fast_base < entanglement_spread) & (
        absolute_base_anchor < entanglement_spread
    )
    cross_entanglement = combined_cross_count >= maximum_crosses
    flat_entanglement = (np.abs(fast_slope) < minimum_fast_slope) & (
        np.abs(base_slope) < minimum_base_slope
    )
    entangled = spread_entanglement | cross_entanglement | (
        flat_entanglement & (absolute_fast_base < entanglement_spread * 1.5)
    )

    # --- 基线 / 锚线尊重度 --------------------------------------------------
    base_respect = _respect_block(
        close, high, low, base_ma, safe_atr, base_ma,
        touch_tolerance, reaction_bars, reaction_move, respect_invalidation, respect_lookback,
    )
    anchor_respect = _respect_block(
        close, high, low, anchor_ma, safe_atr, anchor_ma,
        touch_tolerance, reaction_bars, reaction_move, respect_invalidation, respect_lookback,
        exclude_touch=(low <= base_ma + touch_tolerance * safe_atr)
        & (high >= base_ma - touch_tolerance * safe_atr),
    )

    # --- Regime / 权限 / setup / 触发 ---------------------------------------
    long_regime = (base_ma > anchor_ma) & (base_slope > minimum_base_slope)
    short_regime = (base_ma < anchor_ma) & (base_slope < -minimum_base_slope)

    long_permission = long_regime & (~entangled) & (~bullish_overextension)
    short_permission = short_regime & (~entangled) & (~bearish_overextension)

    price_near_base = np.abs(price_base_distance) <= pullback_distance
    fast_near_base = absolute_fast_base <= pullback_spread
    long_setup = long_regime & (~entangled) & (~bullish_overextension) & price_near_base
    short_setup = short_regime & (~entangled) & (~bearish_overextension) & price_near_base

    previous_close = _shift(close, 1)
    previous_fast = _shift(fast_ma, 1)
    long_price_reclaim = (close > fast_ma) & (previous_close <= previous_fast)
    short_price_reclaim = (close < fast_ma) & (previous_close >= previous_fast)

    previous_high = _shift(high, 1)
    previous_low = _shift(low, 1)
    if require_price_confirmation:
        long_price_confirmation = close > previous_high
        short_price_confirmation = close < previous_low
    else:
        long_price_confirmation = np.ones(n, dtype=bool)
        short_price_confirmation = np.ones(n, dtype=bool)

    if require_curvature:
        long_curvature_confirmation = bullish_curvature
        short_curvature_confirmation = bearish_curvature
    else:
        long_curvature_confirmation = np.ones(n, dtype=bool)
        short_curvature_confirmation = np.ones(n, dtype=bool)

    fast_slope_previous = _shift(fast_slope, 1)
    long_trigger = (
        long_price_reclaim
        & (fast_slope > minimum_fast_slope)
        & (fast_slope > fast_slope_previous)
        & long_curvature_confirmation
        & long_price_confirmation
    )
    short_trigger = (
        short_price_reclaim
        & (fast_slope < -minimum_fast_slope)
        & (fast_slope < fast_slope_previous)
        & short_curvature_confirmation
        & short_price_confirmation
    )

    enter_long = long_setup & long_trigger & (~bullish_overextension) & (~entangled)
    enter_short = short_setup & short_trigger & (~bearish_overextension) & (~entangled)

    exit_long = (close < base_ma) & (
        (base_slope <= 0.0) | _crossunder(fast_ma, base_ma)
    )
    exit_short = (close > base_ma) & (
        (base_slope >= 0.0) | _crossover(fast_ma, base_ma)
    )

    # --- 状态机 -------------------------------------------------------------
    bull_expansion = (
        bullish_order
        & fast_rising
        & base_rising
        & bullish_spread_expansion
        & price_above_fast
        & (~bullish_overextension)
        & (~entangled)
    )
    bear_expansion = (
        bearish_order
        & fast_falling
        & base_falling
        & bearish_spread_expansion
        & price_below_fast
        & (~bearish_overextension)
        & (~entangled)
    )
    bull_pullback = (
        long_regime
        & (~bull_expansion)
        & (~entangled)
        & (~bullish_overextension)
        & (price_near_base | fast_near_base | spread_compression)
    )
    bear_pullback = (
        short_regime
        & (~bear_expansion)
        & (~entangled)
        & (~bearish_overextension)
        & (price_near_base | fast_near_base | spread_compression)
    )
    overextended = (long_regime & bullish_overextension) | (short_regime & bearish_overextension)
    transition = (
        (~entangled)
        & (~long_regime)
        & (~short_regime)
        & ((np.abs(fast_slope) >= minimum_fast_slope) | (np.abs(base_slope) >= minimum_base_slope))
    )

    state = np.full(n, STATE_NEUTRAL, dtype=np.int64)
    state = np.where(transition, STATE_TRANSITION, state)
    state = np.where(bear_pullback, STATE_BEAR_PULLBACK, state)
    state = np.where(bull_pullback, STATE_BULL_PULLBACK, state)
    state = np.where(bear_expansion, STATE_BEAR_EXPANSION, state)
    state = np.where(bull_expansion, STATE_BULL_EXPANSION, state)
    state = np.where(overextended, STATE_OVEREXTENDED, state)

    # --- 衰竭 ---------------------------------------------------------------
    long_fading = (close < fast_ma) & (fast_curvature < 0.0) & fast_base_compression
    short_fading = (close > fast_ma) & (fast_curvature > 0.0) & fast_base_compression
    bull_regime_fading = long_regime & long_fading
    bear_regime_fading = short_regime & short_fading
    state_fading = bull_regime_fading | bear_regime_fading

    # --- 评分 ---------------------------------------------------------------
    direction_score = (
        10.0 * _sign(fast_ma - base_ma)
        + 15.0 * _sign(base_ma - anchor_ma)
        + 5.0 * _sign(close - base_ma)
    )
    fast_slope_score = 10.0 * _clamp(
        _safe_div(fast_slope, np.full(n, max(minimum_fast_slope, 0.001))), -1.0, 1.0
    )
    base_slope_score = 10.0 * _clamp(
        _safe_div(base_slope, np.full(n, max(minimum_base_slope, 0.001))), -1.0, 1.0
    )
    anchor_slope_score = 5.0 * _clamp(
        _safe_div(anchor_slope, np.full(n, max(minimum_base_slope * 0.5, 0.001))), -1.0, 1.0
    )
    slope_score = fast_slope_score + base_slope_score + anchor_slope_score

    fast_base_expansion_score = 10.0 * _clamp(_safe_div(fast_base_spread_change, 0.05), -1.0, 1.0)
    base_anchor_expansion_score = 10.0 * _clamp(_safe_div(base_anchor_spread_change, 0.03), -1.0, 1.0)
    expansion_score = fast_base_expansion_score + base_anchor_expansion_score

    curvature_denominator = np.full(n, max(minimum_curvature, 0.001))
    curvature_score = 6.0 * _clamp(_safe_div(fast_curvature, curvature_denominator), -1.0, 1.0) + 4.0 * _clamp(
        _safe_div(base_curvature, curvature_denominator), -1.0, 1.0
    )

    pre_quality_score = direction_score + slope_score + expansion_score + curvature_score

    respect_score = (
        8.0 * _clamp(base_respect["directional"], -1.0, 1.0)
        if use_respect
        else np.zeros(n)
    )
    anchor_respect_score = (
        5.0 * _clamp(anchor_respect["directional"], -1.0, 1.0) if use_anchor_respect else np.zeros(n)
    )

    cross_quality = 1.0 - _clamp(_safe_div(combined_cross_count, np.full(n, maximum_crosses)), 0.0, 1.0)
    cross_quality_score = 7.0 * cross_quality * _sign(pre_quality_score)

    raw_score = pre_quality_score + respect_score + anchor_respect_score + cross_quality_score

    penalty_fraction = (
        np.where(overextended, 0.20, 0.0)
        + np.where(entangled, 0.25, 0.0)
        + np.where(extreme_volatility, 0.10, 0.0)
    )
    score = _clamp(raw_score * np.maximum(0.0, 1.0 - penalty_fraction), -100.0, 100.0)

    # --- 偏置与许可变化 -----------------------------------------------------
    bias = np.zeros(n, dtype=np.int64)
    bias = np.where(score >= directional_bias_threshold, 1, bias)
    bias = np.where(score >= strong_bias_threshold, 2, bias)
    bias = np.where(score <= -directional_bias_threshold, -1, bias)
    bias = np.where(score <= -strong_bias_threshold, -2, bias)

    long_permission_start = long_permission & (~_shift_bool(long_permission, 1))
    short_permission_start = short_permission & (~_shift_bool(short_permission, 1))

    # --- 组装 ---------------------------------------------------------------
    out: dict[str, object] = {
        "mce_fast_ma": fast_ma,
        "mce_base_ma": base_ma,
        "mce_anchor_ma": anchor_ma,
        "mce_fast_slope": fast_slope,
        "mce_base_slope": base_slope,
        "mce_anchor_slope": anchor_slope,
        "mce_fast_curvature": fast_curvature,
        "mce_base_curvature": base_curvature,
        "mce_fast_base_spread": fast_base_spread,
        "mce_base_anchor_spread": base_anchor_spread,
        "mce_price_extension": price_extension,
        "mce_price_base_distance": price_base_distance,
        "mce_volatility_ratio": volatility_ratio,
        "mce_cross_count": combined_cross_count,
        "mce_state": pd.array(state, dtype="Int64"),
        "mce_score": score,
        "mce_bias": pd.array(bias, dtype="Int64"),
        "mce_direction_score": direction_score,
        "mce_slope_score": slope_score,
        "mce_expansion_score": expansion_score,
        "mce_curvature_score": curvature_score,
        "mce_respect_score": respect_score,
        "mce_anchor_respect_score": anchor_respect_score,
        "mce_cross_quality_score": cross_quality_score,
        "mce_base_respect_touches": base_respect["valid_count"],
        "mce_anchor_respect_touches": anchor_respect["valid_count"],
        "mce_base_respect_directional": base_respect["directional"],
        "mce_bullish_respect_rate": base_respect["bullish"],
        "mce_bearish_respect_rate": base_respect["bearish"],
        "mce_anchor_respect_directional": anchor_respect["directional"],
        "mce_bullish_anchor_respect_rate": anchor_respect["bullish"],
        "mce_bearish_anchor_respect_rate": anchor_respect["bearish"],
        "mce_bullish_order": _flag(bullish_order),
        "mce_bearish_order": _flag(bearish_order),
        "mce_entangled": _flag(entangled),
        "mce_extreme_volatility": _flag(extreme_volatility),
        "mce_overextended": _flag(overextended),
        "mce_fading": _flag(state_fading),
        "mce_bull_regime_fading": _flag(bull_regime_fading),
        "mce_bear_regime_fading": _flag(bear_regime_fading),
        "mce_long_permission": _flag(long_permission),
        "mce_short_permission": _flag(short_permission),
        "mce_long_permission_start": _flag(long_permission_start),
        "mce_short_permission_start": _flag(short_permission_start),
        "mce_long_setup": _flag(long_setup),
        "mce_short_setup": _flag(short_setup),
        "mce_long_trigger": _flag(long_trigger),
        "mce_short_trigger": _flag(short_trigger),
        "mce_enter_long": _flag(enter_long),
        "mce_enter_short": _flag(enter_short),
        "mce_exit_long": _flag(exit_long),
        "mce_exit_short": _flag(exit_short),
        "mce_bullish_anchor_rejection": _flag(anchor_respect["bullish_success"]),
        "mce_bearish_anchor_rejection": _flag(anchor_respect["bearish_success"]),
    }
    return pd.DataFrame(out, index=frame.index)


def _flag(values: np.ndarray) -> pd.Series:
    """布尔序列 → Int64 的 0/1 列。"""
    return pd.array(np.asarray(values, dtype=bool).astype(np.int64), dtype="Int64")


def _resolve_source(frame: pd.DataFrame, source: str, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
    """``input.source`` 的取值：列名或 Pine 惯用的合成价。"""
    if source == "hl2":
        return (high + low) / 2.0
    if source == "hlc3":
        return (high + low + close) / 3.0
    if source == "ohlc4":
        open_ = frame["open"].to_numpy(dtype=float)
        return (open_ + high + low + close) / 4.0
    if source not in frame.columns:
        raise ValueError(f"source {source!r} is not a column of the frame")
    return frame[source].to_numpy(dtype=float)


def _respect_block(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    line: np.ndarray,
    safe_atr: np.ndarray,
    touch_line: np.ndarray,
    touch_tolerance: float,
    reaction_bars: int,
    reaction_move: float,
    respect_invalidation: float,
    respect_lookback: int,
    *,
    exclude_touch: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pine 的“尊重度”统计：触摸 → 反应窗口 → 有效/无效 → 比率。

    与 Pine 一致：一次“贴线”只记一次（只有连续贴线的第一根算一次），并且要求
    贴线前一根收盘在线的**同侧**（穿越不算尊重，而是突破）。
    """
    n = len(close)
    touching = (low <= touch_line + touch_tolerance * safe_atr) & (
        high >= touch_line - touch_tolerance * safe_atr
    )
    if exclude_touch is not None:
        touching = touching & (~exclude_touch)
    touch_start = touching & (~_shift_bool(touching, 1))

    eligible = _shift_bool(touch_start, reaction_bars)
    approach_above = close > line
    approach_below = close < line
    historic_above = _shift_bool(approach_above, reaction_bars)
    historic_below = _shift_bool(approach_below, reaction_bars)
    historic_line = _shift(line, reaction_bars)
    historic_atr = _shift(safe_atr, reaction_bars)

    reaction_low = lowest(pd.Series(low), reaction_bars + 1).to_numpy(dtype=float)
    reaction_high = highest(pd.Series(high), reaction_bars + 1).to_numpy(dtype=float)

    bullish_success = (
        eligible
        & historic_above
        & (close >= historic_line + reaction_move * historic_atr)
        & (reaction_low >= historic_line - respect_invalidation * historic_atr)
    )
    bearish_success = (
        eligible
        & historic_below
        & (close <= historic_line - reaction_move * historic_atr)
        & (reaction_high <= historic_line + respect_invalidation * historic_atr)
    )

    valid_count = _rolling_count(eligible.astype(float), respect_lookback)
    bullish_count = _rolling_count(bullish_success.astype(float), respect_lookback)
    bearish_count = _rolling_count(bearish_success.astype(float), respect_lookback)

    bullish_rate = _rate(bullish_count, valid_count)
    bearish_rate = _rate(bearish_count, valid_count)
    return {
        "bullish": bullish_rate,
        "bearish": bearish_rate,
        "directional": bullish_rate - bearish_rate,
        "valid_count": valid_count,
        "bullish_success": bullish_success,
        "bearish_success": bearish_success,
    }
