"""Trade Permission Engine [WavesUnchained] v1.3.5 — 交易许可引擎。

Port of ``data/pine-collection/indicators/composite/trade_permission_engine/
trade_permission_engine_v1.pine``.

回答的不是“方向会不会涨”，而是“**此刻这个位置是否允许做多/做空**”。第 1 版
是 ``Directional Probability Engine``（已归档），这里把它的因子结构重建为
``Context / Maturity / Exhaustion / Trigger / Target-Invalidation`` 五段，并加了
两个机制（缠论式的参考腿失效门槛、Scenario 感知的目标几何）。

Pine 的流水线（本移植逐一保留）：

1. **参考腿**：交替确认的摆动枢轴；反向枢轴只有把前一条腿回撤到
   ``leg_invalidation_retracement``（0.90）才取代它 —— 普通回调不得篡位。
2. **腿质量**：效率（净幅/路径和）×（1−重叠率）；再叠加最近 N 条腿的
   递近加权得到 *Live Trend Quality*（趋势是否正在退化）。
3. **Maturity**：以回撤比例 r 为自变量，Continuation 用 bell 曲线落在典型
   回调深度、Reversal 用 bell 曲线落在满回撤附近；再用“腿时长比”的 bell
   曲线做时间成熟度，并新增 Endpoint Reversal（腿相对 ATR 过度延伸）。
4. **Exhaustion**：反扑腿自身效率相对其均值的衰减 + 振荡器加速度 + 背离加成。
5. **Trigger**：UNTRIGGERED → BROKEN（极值拒绝，需跟随）→ CONFIRMED（收在
   破位价之外）→ ACCEPTED（保持 ``acceptance_bars`` 根），带 epoch 失效、
   有效窗口与冷却。
6. **Permission**：``floor-multiplicative`` 门乘法（腿质量 × 趋势质量 × 成熟度
   × 衰竭），Continuation 与 Reversal 用 ``max()`` 竞争而非 noisy-OR，HTF
   影响按场景非对称施加，dominant 场景带滞回。
7. **Target/Invalidation**：T1 在最近确认枢轴里找最近合格目标（找不到用 ATR
   兜底），T2/失效位按场景取不同的极值跟踪器。
8. **CRV / 衰减 / 最终分**：冻结交易的 CRV 用冻结几何，停滞 bar 按
   ``decay_per_bar`` 衰减，最终分 = 许可 ×（确认地板 + 确认率）× CRV × 衰减，
   再 EMA 平滑。
9. **持仓健康**（HOLD/PROTECT/REDUCE/EXIT）：六维加权平均，与 Permission
   完全解耦（不受 CRV/Trigger/生命周期影响）。

未移植：绘图、表格、``alertcondition``（都能由输出列组合）、多周期
``request.security`` 之外的部分图元；HTF 用“重采样到目标周期的**上一根已完成**
bar”复刻（EMA/收盘/ATR 都取 ``[1]``，与 Pine 的 ``lookahead_on`` 语义一致）。

关于 ``confirmed_bar_only``：它只影响“显示中的分数是否冻结到确认 bar”，历史
数据里每根 bar 都是确认 bar，因此该开关对回测无影响（保留参数以对齐签名）。
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
    integer,
    lowest,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rma,
    rsi,
    sma,
    stdev,
)

EPS = 1e-10

_PRESETS = ("Auto", "Scalping", "Intraday", "Swing", "Position", "Custom")
_OSCILLATORS = ("Stochastic RSI", "RSI", "WaveTrend", "CMO")


# ---------------------------------------------------------------------------
# Pine 辅助函数
# ---------------------------------------------------------------------------


def _clamp(values, low: float, high: float):
    """Pine ``f_clamp``：``max(low, min(high, x))``，na 传播。"""
    return np.maximum(low, np.minimum(high, values))


def _safe_div(numerator, denominator, fallback: float):
    """Pine ``f_safeDiv``：|分母| > EPS 才做除法，否则用 ``fallback``。"""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.abs(den) > EPS
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _lerp(a, b, t):
    """Pine ``f_lerp``：t 先夹到 [0,1]。"""
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _gaussian_like(x, center: float, width: float):
    """Pine ``f_gaussianLike``：以 center 为峰、width 为尺度的钟形。"""
    values = np.asarray(x, dtype=float)
    if width <= EPS:
        return np.ones_like(values, dtype=float)
    z = (values - center) / width
    return np.exp(-0.5 * z * z)


def _weighted_avg2(v1, w1: float, v2, w2: float):
    """Pine ``f_weightedAvg2``。"""
    total = max(w1 + w2, EPS)
    return (np.asarray(v1, dtype=float) * w1 + np.asarray(v2, dtype=float) * w2) / total


def _gated_factor(value100, floor: float):
    """Pine ``f_gatedFactor``：把 0-100 归一到 [floor, 1]。"""
    return floor + (1.0 - floor) * _clamp(value100, 0.0, 100.0) / 100.0


def _shift(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 0:
        return arr.copy()
    if len(arr) <= length:
        return np.full(len(arr), np.nan)
    return np.concatenate([np.full(length, np.nan), arr[: len(arr) - length]])


def _resolve_preset(preset: str, bar_seconds: float) -> str:
    """Pine 的 ``Auto`` 按图上有几秒一根 bar 落到四档之一。"""
    if preset != "Auto":
        return preset
    if bar_seconds <= 300:
        return "Scalping"
    if bar_seconds <= 1800:
        return "Intraday"
    if bar_seconds <= 14400:
        return "Swing"
    return "Position"


def _preset_value(preset: str, table: dict[str, int | float], fallback: int | float):
    """四档预设的查表（Custom 用调用方给的 fallback）。"""
    return table.get(preset, fallback)


def _htf_trend(
    frame: pd.DataFrame,
    target_seconds: float,
    chart_seconds: float,
    fast_len: int,
    slow_len: int,
    atr_len: int,
    tick: float,
) -> np.ndarray:
    """HTF 上下文：把图表重采样到目标周期，取**上一根已完成** bar 的
    EMA(快)/EMA(慢)/收盘/ATR，映射回图表 bar。

    对应 Pine 的 ``request.security(..., [emaFast[1], emaSlow[1], close[1], atr[1]],
    lookahead_on)``；目标周期不高于图表周期时返回全 0（Pine 的
    ``validHigherTimeframe`` 门）。
    """
    n = len(frame)
    if n == 0 or target_seconds <= chart_seconds:
        return np.zeros(n)
    idx = pd.DatetimeIndex(frame.index)
    if target_seconds >= 28 * 86400:
        key = pd.PeriodIndex(idx, freq="M")
    elif target_seconds >= 6 * 86400:
        key = pd.PeriodIndex(idx, freq="W")
    elif target_seconds >= 86400:
        key = pd.PeriodIndex(idx, freq="D")
    else:
        # 分钟级目标周期：把时间 floor 到目标时长，天然得到周期键
        key = idx.floor(pd.Timedelta(seconds=target_seconds))

    grouped = frame.groupby(key).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    htf_close = grouped["close"]
    fast = ema(htf_close, fast_len).shift(1)
    slow = ema(htf_close, slow_len).shift(1)
    tr = pd.concat(
        [
            grouped["high"] - grouped["low"],
            (grouped["high"] - htf_close.shift(1)).abs(),
            (grouped["low"] - htf_close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    htf_atr = rma(tr, atr_len).shift(1)
    htf_close = htf_close.shift(1)

    bars = pd.Series(np.asarray(key))
    fast_a = bars.map(fast).to_numpy(dtype=float)
    slow_a = bars.map(slow).to_numpy(dtype=float)
    close_a = bars.map(htf_close).to_numpy(dtype=float)
    atr_a = bars.map(htf_atr).to_numpy(dtype=float)

    atr_safe = np.fmax(np.nan_to_num(atr_a, nan=tick), tick)
    spread = fast_a - slow_a
    scale = np.abs(spread) + atr_safe
    trend = _clamp(0.70 * _safe_div(spread, scale, 0.0) + 0.30 * _safe_div(close_a - slow_a, scale, 0.0), -1.0, 1.0)
    return np.nan_to_num(trend, nan=0.0)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def trade_permission_engine(
    frame: pd.DataFrame,
    *,
    preset: str = "Auto",
    smoothing: int = 3,
    confirmed_bar_only: bool = False,
    custom_pivot_len: int = 8,
    custom_atr_len: int = 14,
    pivot_atr_tolerance: float = 0.12,
    min_swing_atr: float = 1.0,
    leg_invalidation_retracement: float = 0.90,
    context_floor: float = 0.25,
    max_leg_history: int = 4,
    live_trend_quality_floor: float = 0.25,
    decay_per_bar: float = 0.5,
    decay_atr_lookback: int = 10,
    shallow_center: float = 0.382,
    normal_center: float = 0.500,
    weak_center: float = 0.618,
    fib_tolerance_atr: float = 0.20,
    correction_presence_min: float = 0.10,
    reversal_center: float = 1.272,
    reversal_width: float = 0.28,
    time_maturity_center: float = 0.35,
    time_maturity_width: float = 0.35,
    reversal_time_maturity_center: float = 0.80,
    reversal_time_maturity_width: float = 0.60,
    overextension_atr: float = 3.0,
    endpoint_presence_max: float = 0.30,
    maturity_floor: float = 0.15,
    oscillator: str = "Stochastic RSI",
    custom_osc_len: int = 14,
    custom_osc_smooth: int = 3,
    exhaustion_recent_bars: int = 5,
    divergence_enabled: bool = True,
    divergence_strength: float = 18.0,
    divergence_max_bars: int = 120,
    exhaustion_floor: float = 0.15,
    micro_pivot_len_auto: bool = True,
    custom_micro_pivot_len: int = 3,
    rejection_close_loc: float = 0.6,
    acceptance_bars: int = 3,
    trigger_weight: float = 70.0,
    acceptance_weight: float = 30.0,
    confirmation_floor: float = 0.35,
    cooldown_bars: int = 5,
    trigger_validity_bars: int = 8,
    use_t2: bool = True,
    extension_mult: float = 1.272,
    invalidation_atr_buffer: float = 0.5,
    min_target_distance_atr: float = 0.5,
    min_crv: float = 1.0,
    crv_floor: float = 0.30,
    max_pivot_history: int = 20,
    htf_enabled: bool = True,
    htf_mode: str = "Auto",
    manual_htf_seconds: int = 14400,
    htf_influence: float = 15.0,
    reversal_htf_aligned_factor: float = 0.5,
    reversal_htf_conflict_factor: float = 1.5,
    htf_fast_ema: int = 21,
    htf_slow_ema: int = 55,
    htf_atr_len: int = 14,
    scenario_hysteresis_margin: float = 10.0,
    permission_threshold: float = 12.0,
    signal_retention: float = 50.0,
    w_live_quality: float = 25.0,
    w_pullback_health: float = 20.0,
    w_trend_progress: float = 20.0,
    w_fatigue: float = 15.0,
    w_structure_intact: float = 10.0,
    w_regime: float = 10.0,
    structural_progress_lookback: int = 20,
    trend_progress_full_credit_atr: float = 2.0,
    regime_lookback: int = 20,
    trend_fatigue_growth_scale: float = 3.0,
    health_hold_threshold: float = 68.0,
    health_protect_threshold: float = 54.0,
    health_reduce_threshold: float = 36.0,
    mintick: float | None = None,
) -> pd.DataFrame:
    """Trade Permission Engine —— 逐 bar 的许可 / 触发 / 目标 / 持仓健康。

    Args:
        frame: OHLC 数据（``open``/``high``/``low``/``close``），索引为时间。
        preset: ``Auto`` / ``Scalping`` / ``Intraday`` / ``Swing`` / ``Position`` /
            ``Custom``。``Auto`` 按图表周期自动落到四档之一。
        smoothing: 最终分数的 EMA 平滑长度。
        confirmed_bar_only: Pine 只在确认 bar 上冻结显示分数；历史数据全是确认
            bar，因此本移植里是无效开关（保留以对齐签名）。
        custom_pivot_len: Custom 预设的摆动枢轴腿长。
        custom_atr_len: Custom 预设的 ATR 长度。
        pivot_atr_tolerance: 枢轴“等价”容差（ATR 倍数），用于端点延伸判定。
        min_swing_atr: 有效摆动腿的最小长度（ATR 倍数）。
        leg_invalidation_retracement: 反向枢轴取代前腿所需的回撤比例。
        context_floor: 腿质量的门地板。
        max_leg_history: 参与 Live Trend Quality 的历史腿数。
        live_trend_quality_floor: Live Trend Quality 的门地板。
        decay_per_bar: 每个停滞 bar 的许可衰减百分比。
        decay_atr_lookback: ATR 扩张判定的回看长度。
        shallow_center: 强腿偏好的回调深度。
        normal_center: 普通腿偏好的回调深度。
        weak_center: 弱腿偏好的回调深度。
        fib_tolerance_atr: 回调区的软度（ATR）。
        correction_presence_min: 计为“真实回调”的最小回撤比例。
        reversal_center: Reversal 偏好的满回撤倍数。
        reversal_width: Reversal 区的软度。
        time_maturity_center: 偏好的回调时长（占腿时长比例）。
        time_maturity_width: 时长区的软度。
        reversal_time_maturity_center: Reversal 偏好的时长中心。
        reversal_time_maturity_width: Reversal 时长区的软度。
        overextension_atr: 参考腿超过该 ATR 倍数视为过度延伸。
        endpoint_presence_max: Endpoint Reversal 的存续回撤窗口。
        maturity_floor: 成熟度的门地板。
        oscillator: ``Stochastic RSI`` / ``RSI`` / ``WaveTrend`` / ``CMO``。
        custom_osc_len: Custom 预设的振荡器长度。
        custom_osc_smooth: Custom 预设的振荡器平滑长度。
        exhaustion_recent_bars: 衰竭判定用的近期窗口。
        divergence_enabled: 是否计入常规/隐藏背离。
        divergence_strength: 背离的最大加分。
        divergence_max_bars: 两个背离枢轴之间的最大间隔。
        exhaustion_floor: 衰竭的门地板。
        micro_pivot_len_auto: 微型摆动长度是否自动取 pivot_len/3。
        custom_micro_pivot_len: 手动微型摆动长度。
        rejection_close_loc: 极值拒绝所需的收盘位置（占 bar 区间）。
        acceptance_bars: 从 CONFIRMED 到 ACCEPTED 需要的保持 bar 数。
        trigger_weight: Confirmation 里 Trigger 分的权重。
        acceptance_weight: Confirmation 里 Acceptance 分的权重。
        confirmation_floor: 确认因子的地板。
        cooldown_bars: 同方向两次触发之间的冷却。
        trigger_validity_bars: 触发保持可用的 bar 数。
        use_t2: 是否启用 T2 扩展目标。
        extension_mult: T2 扩展倍数。
        invalidation_atr_buffer: 失效位的 ATR 缓冲。
        min_target_distance_atr: T1 的最小距离（ATR）。
        min_crv: 满分 Target Feasibility 所需的盈亏比。
        crv_floor: CRV 因子的地板。
        max_pivot_history: T1 搜索的枢轴历史长度。
        htf_enabled: 是否启用 HTF 上下文。
        htf_mode: ``Auto`` / ``Manual``。
        manual_htf_seconds: 手动 HTF 的周期秒数。
        htf_influence: HTF 影响的最大加减分。
        reversal_htf_aligned_factor: Reversal 顺 HTF 时的加成折扣。
        reversal_htf_conflict_factor: Reversal 逆 HTF 时的惩罚放大。
        htf_fast_ema: HTF 快 EMA 长度。
        htf_slow_ema: HTF 慢 EMA 长度。
        htf_atr_len: HTF ATR 长度。
        scenario_hysteresis_margin: Continuation/Reversal 主导权的滞回余量。
        permission_threshold: Setup 的许可门槛。
        signal_retention: 合格信号要求 Active Permission 保留的比例（%）。
        w_live_quality: 持仓健康里 Live Quality 的权重。
        w_pullback_health: Pullback Health 权重。
        w_trend_progress: Trend Progress 权重。
        w_fatigue: Trend Fatigue 权重。
        w_structure_intact: Structure Intact 权重。
        w_regime: Regime/Sideways-Risk 权重。
        structural_progress_lookback: Trend Progress 的回看 bar 数。
        trend_progress_full_credit_atr: Trend Progress 拿满分所需的 ATR 距离。
        regime_lookback: Regime 效率比的回看长度。
        trend_fatigue_growth_scale: Trend Fatigue 增长侧的压缩系数。
        health_hold_threshold: HOLD 阈值。
        health_protect_threshold: PROTECT 阈值。
        health_reduce_threshold: REDUCE 阈值。
        mintick: 最小跳动，``None`` 时按价格量级推断。

    Returns:
        ``pd.DataFrame``，索引与 ``frame`` 一致，列为 ``tpe_*``。

    Raises:
        ValueError: 参数越界或取值不在允许集合内。
    """
    preset = choice("preset", preset, _PRESETS)
    oscillator = choice("oscillator", oscillator, _OSCILLATORS)
    htf_mode = choice("htf_mode", htf_mode, ("Auto", "Manual"))
    smoothing = integer("smoothing", smoothing, 1, 20)
    custom_pivot_len = integer("custom_pivot_len", custom_pivot_len, 2, 50)
    custom_atr_len = integer("custom_atr_len", custom_atr_len, 5, 100)
    pivot_atr_tolerance = number("pivot_atr_tolerance", pivot_atr_tolerance, 0.0, 1.0)
    min_swing_atr = number("min_swing_atr", min_swing_atr, 0.25, 10.0)
    leg_invalidation_retracement = number(
        "leg_invalidation_retracement", leg_invalidation_retracement, 0.50, 1.50
    )
    context_floor = number("context_floor", context_floor, 0.0, 1.0)
    max_leg_history = integer("max_leg_history", max_leg_history, 4, 8)
    live_trend_quality_floor = number("live_trend_quality_floor", live_trend_quality_floor, 0.0, 1.0)
    decay_per_bar = number("decay_per_bar", decay_per_bar, 0.0, 3.0)
    decay_atr_lookback = integer("decay_atr_lookback", decay_atr_lookback, 3, 50)
    shallow_center = number("shallow_center", shallow_center, 0.20, 0.65)
    normal_center = number("normal_center", normal_center, 0.25, 0.75)
    weak_center = number("weak_center", weak_center, 0.35, 0.90)
    fib_tolerance_atr = number("fib_tolerance_atr", fib_tolerance_atr, 0.0, 2.0)
    correction_presence_min = number("correction_presence_min", correction_presence_min, 0.02, 0.30)
    reversal_center = number("reversal_center", reversal_center, 0.90, 2.00)
    reversal_width = number("reversal_width", reversal_width, 0.10, 0.60)
    time_maturity_center = number("time_maturity_center", time_maturity_center, 0.10, 1.00)
    time_maturity_width = number("time_maturity_width", time_maturity_width, 0.10, 1.00)
    reversal_time_maturity_center = number(
        "reversal_time_maturity_center", reversal_time_maturity_center, 0.10, 2.00
    )
    reversal_time_maturity_width = number(
        "reversal_time_maturity_width", reversal_time_maturity_width, 0.10, 1.50
    )
    overextension_atr = number("overextension_atr", overextension_atr, 0.5, 15.0)
    endpoint_presence_max = number("endpoint_presence_max", endpoint_presence_max, 0.05, 0.60)
    maturity_floor = number("maturity_floor", maturity_floor, 0.0, 1.0)
    custom_osc_len = integer("custom_osc_len", custom_osc_len, 5, 100)
    custom_osc_smooth = integer("custom_osc_smooth", custom_osc_smooth, 1, 20)
    exhaustion_recent_bars = integer("exhaustion_recent_bars", exhaustion_recent_bars, 2, 20)
    divergence_strength = number("divergence_strength", divergence_strength, 0.0, 40.0)
    divergence_max_bars = integer("divergence_max_bars", divergence_max_bars, 10, 500)
    exhaustion_floor = number("exhaustion_floor", exhaustion_floor, 0.0, 1.0)
    custom_micro_pivot_len = integer("custom_micro_pivot_len", custom_micro_pivot_len, 1, 20)
    rejection_close_loc = number("rejection_close_loc", rejection_close_loc, 0.3, 0.95)
    acceptance_bars = integer("acceptance_bars", acceptance_bars, 1, 20)
    trigger_weight = number("trigger_weight", trigger_weight, 0.0, 100.0)
    acceptance_weight = number("acceptance_weight", acceptance_weight, 0.0, 100.0)
    confirmation_floor = number("confirmation_floor", confirmation_floor, 0.0, 1.0)
    cooldown_bars = integer("cooldown_bars", cooldown_bars, 0, 100)
    trigger_validity_bars = integer("trigger_validity_bars", trigger_validity_bars, 1, 50)
    extension_mult = number("extension_mult", extension_mult, 1.0, 3.0)
    invalidation_atr_buffer = number("invalidation_atr_buffer", invalidation_atr_buffer, 0.0, 3.0)
    min_target_distance_atr = number("min_target_distance_atr", min_target_distance_atr, 0.1, 3.0)
    min_crv = number("min_crv", min_crv, 0.25, 5.0)
    crv_floor = number("crv_floor", crv_floor, 0.0, 1.0)
    max_pivot_history = integer("max_pivot_history", max_pivot_history, 5, 50)
    htf_influence = number("htf_influence", htf_influence, 0.0, 40.0)
    reversal_htf_aligned_factor = number(
        "reversal_htf_aligned_factor", reversal_htf_aligned_factor, 0.0, 1.0
    )
    reversal_htf_conflict_factor = number(
        "reversal_htf_conflict_factor", reversal_htf_conflict_factor, 1.0, 3.0
    )
    htf_fast_ema = integer("htf_fast_ema", htf_fast_ema, 2, 200)
    htf_slow_ema = integer("htf_slow_ema", htf_slow_ema, 3, 400)
    htf_atr_len = integer("htf_atr_len", htf_atr_len, 5, 100)
    scenario_hysteresis_margin = number(
        "scenario_hysteresis_margin", scenario_hysteresis_margin, 0.0, 50.0
    )
    permission_threshold = number("permission_threshold", permission_threshold, 0.0, 100.0)
    signal_retention = number("signal_retention", signal_retention, 0.0, 100.0)
    w_live_quality = number("w_live_quality", w_live_quality, 0.0, 100.0)
    w_pullback_health = number("w_pullback_health", w_pullback_health, 0.0, 100.0)
    w_trend_progress = number("w_trend_progress", w_trend_progress, 0.0, 100.0)
    w_fatigue = number("w_fatigue", w_fatigue, 0.0, 100.0)
    w_structure_intact = number("w_structure_intact", w_structure_intact, 0.0, 100.0)
    w_regime = number("w_regime", w_regime, 0.0, 100.0)
    structural_progress_lookback = integer(
        "structural_progress_lookback", structural_progress_lookback, 5, 100
    )
    trend_progress_full_credit_atr = number(
        "trend_progress_full_credit_atr", trend_progress_full_credit_atr, 0.25, 10.0
    )
    regime_lookback = integer("regime_lookback", regime_lookback, 5, 100)
    trend_fatigue_growth_scale = number(
        "trend_fatigue_growth_scale", trend_fatigue_growth_scale, 1.0, 10.0
    )
    health_hold_threshold = number("health_hold_threshold", health_hold_threshold, 0.0, 100.0)
    health_protect_threshold = number("health_protect_threshold", health_protect_threshold, 0.0, 100.0)
    health_reduce_threshold = number("health_reduce_threshold", health_reduce_threshold, 0.0, 100.0)
    divergence_enabled = boolean("divergence_enabled", divergence_enabled)
    micro_pivot_len_auto = boolean("micro_pivot_len_auto", micro_pivot_len_auto)
    use_t2 = boolean("use_t2", use_t2)
    htf_enabled = boolean("htf_enabled", htf_enabled)
    confirmed_bar_only = boolean("confirmed_bar_only", confirmed_bar_only)

    high_s, low_s, close_s = ohlcv(frame, "high", "low", "close")
    close = close_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    hlc3 = (high + low + close) / 3.0
    n = len(frame)

    bar_seconds = _bar_seconds(frame)
    resolved = _resolve_preset(preset, bar_seconds)
    pivot_len = _preset_value(resolved, {"Scalping": 4, "Intraday": 6, "Swing": 9, "Position": 13}, custom_pivot_len)
    atr_len = _preset_value(resolved, {"Scalping": 10, "Intraday": 14, "Swing": 21, "Position": 34}, custom_atr_len)
    osc_len = _preset_value(resolved, {"Scalping": 9, "Intraday": 14, "Swing": 21, "Position": 34}, custom_osc_len)
    osc_smooth = _preset_value(resolved, {"Scalping": 2, "Intraday": 3, "Swing": 4, "Position": 5}, custom_osc_smooth)
    micro_len = max(2, pivot_len // 3) if micro_pivot_len_auto else custom_micro_pivot_len

    # --- 基础序列 -----------------------------------------------------------
    atr_a = atr(frame, atr_len).to_numpy(dtype=float)
    if mintick is None:
        magnitude = np.nanmedian(np.abs(close)) if n else 1.0
        magnitude = magnitude if np.isfinite(magnitude) and magnitude > 0 else 1.0
        tick = 10.0 ** (math.floor(math.log10(magnitude)) - 5)
    else:
        tick = number("mintick", mintick, 1e-12, 1e15)
    atr_safe = np.fmax(np.nan_to_num(atr_a, nan=tick), tick)

    pivot_high = pivothigh(high_s, pivot_len, pivot_len).to_numpy(dtype=float)
    pivot_low = pivotlow(low_s, pivot_len, pivot_len).to_numpy(dtype=float)
    micro_high_raw = pivothigh(high_s, micro_len, micro_len).to_numpy(dtype=float)
    micro_low_raw = pivotlow(low_s, micro_len, micro_len).to_numpy(dtype=float)
    micro_last_high = _forward_fill(micro_high_raw)
    micro_last_low = _forward_fill(micro_low_raw)

    rolling_low_n = lowest(low_s, pivot_len + 1).to_numpy(dtype=float)
    rolling_high_n = highest(high_s, pivot_len + 1).to_numpy(dtype=float)

    abs_delta = np.abs(np.diff(close, prepend=close[0]))
    abs_delta[0] = 0.0
    cur_smaller = np.minimum(high - low, np.r_[np.nan, (high - low)[:-1]])
    bar_overlap = np.where(
        cur_smaller > EPS,
        np.fmax(
            0.0,
            np.fmin(high, np.r_[np.nan, high[:-1]]) - np.fmax(low, np.r_[np.nan, low[:-1]]),
        )
        / np.where(cur_smaller > EPS, cur_smaller, 1.0),
        0.0,
    )

    def rolling_sum(values: np.ndarray, length: int) -> np.ndarray:
        """``math.sum(values, length)``：窗口内求和，窗口不完整时为 na。"""
        return (
            pd.Series(np.asarray(values, dtype=float))
            .rolling(length, min_periods=length)
            .sum()
            .to_numpy(dtype=float)
        )

    seed_path = rolling_sum(abs_delta, pivot_len)
    seed_overlap = rolling_sum(bar_overlap, pivot_len)
    recent_path = rolling_sum(abs_delta, exhaustion_recent_bars)
    regime_path = rolling_sum(abs_delta, regime_lookback)
    recent_net = close - _shift(close, exhaustion_recent_bars)
    regime_net = np.abs(close - _shift(close, regime_lookback))

    # --- 振荡器 -------------------------------------------------------------
    rsi_a = rsi(close_s, osc_len).to_numpy(dtype=float)
    rsi_lowest = lowest(pd.Series(rsi_a), osc_len).to_numpy(dtype=float)
    rsi_highest = highest(pd.Series(rsi_a), osc_len).to_numpy(dtype=float)
    stoch_raw = 100.0 * _safe_div(rsi_a - rsi_lowest, rsi_highest - rsi_lowest, 0.5)
    stoch_rsi = ema(pd.Series(stoch_raw), osc_smooth).to_numpy(dtype=float)

    wt_esa = ema(pd.Series(hlc3), max(5, osc_len)).to_numpy(dtype=float)
    wt_dev = ema(pd.Series(np.abs(hlc3 - wt_esa)), max(5, osc_len)).to_numpy(dtype=float)
    wt_ci = _safe_div(hlc3 - wt_esa, 0.015 * wt_dev, 0.0)
    wt_raw = ema(pd.Series(wt_ci), max(3, osc_len * 2)).to_numpy(dtype=float)
    wt_norm = _clamp(50.0 + wt_raw * 0.65, 0.0, 100.0)

    delta = np.diff(close, prepend=close[0])
    up_mean = sma(pd.Series(np.where(delta > 0, delta, 0.0)), osc_len).to_numpy(dtype=float)
    down_mean = sma(pd.Series(np.where(delta < 0, -delta, 0.0)), osc_len).to_numpy(dtype=float)
    cmo_raw = 100.0 * _safe_div(up_mean - down_mean, up_mean + down_mean, 0.0)
    cmo_norm = _clamp(50.0 + cmo_raw * 0.5, 0.0, 100.0)

    if oscillator == "RSI":
        osc_raw = rsi_a
    elif oscillator == "WaveTrend":
        osc_raw = wt_norm
    elif oscillator == "CMO":
        osc_raw = cmo_norm
    else:
        osc_raw = stoch_rsi
    osc_smoothed = ema(pd.Series(osc_raw), osc_smooth).to_numpy(dtype=float)
    osc_velocity = osc_smoothed - _shift(osc_smoothed, 1)
    osc_accel = osc_velocity - _shift(osc_velocity, 1)
    osc_vol = stdev(pd.Series(osc_velocity), max(5, osc_len)).to_numpy(dtype=float)
    accel_norm = _clamp(_safe_div(osc_accel, np.fmax(osc_vol, 0.5), 0.0), -2.0, 2.0)

    # --- HTF 上下文 ---------------------------------------------------------
    target_seconds = (
        _auto_htf_seconds(bar_seconds) if htf_mode == "Auto" else float(manual_htf_seconds)
    )
    if htf_enabled:
        htf_trend_signed = _htf_trend(
            frame, target_seconds, bar_seconds, htf_fast_ema, htf_slow_ema, htf_atr_len, tick
        )
    else:
        htf_trend_signed = np.zeros(n)
    valid_htf = target_seconds > bar_seconds

    # --- 输出容器 -----------------------------------------------------------
    cols: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in (
        "swing_direction", "last_high", "last_low", "leg_range_atr", "reference_leg_quality",
        "live_trend_quality", "live_leg_quality", "bullish_retracement", "bearish_retracement",
        "preferred_fib_center", "continuation_maturity_long", "continuation_maturity_short",
        "reversal_maturity_long", "reversal_maturity_short", "exhaustion_bull", "exhaustion_bear",
        "impulse_exhaustion_bull", "impulse_exhaustion_bear", "divergence_bull_bonus",
        "divergence_bear_bonus", "cont_permission_long", "cont_permission_short",
        "rev_permission_long", "rev_permission_short", "permission_long", "permission_short",
        "t1_long", "t2_long", "inv_long", "t1_short", "t2_short", "inv_short",
        "trigger_score_long", "trigger_score_short", "acceptance_long", "acceptance_short",
        "confirmation_long", "confirmation_short", "crv_factor_long", "crv_factor_short",
        "decay_factor_long", "decay_factor_short", "stale_bars_long", "stale_bars_short",
        "score_raw_long", "score_raw_short",
        "live_quality_long", "live_quality_short", "pullback_health_long", "pullback_health_short",
        "trend_progress_long", "trend_progress_short", "trend_fatigue", "structure_intact_long",
        "structure_intact_short", "regime_score", "health_long", "health_short",
        "atr", "oscillator", "acceleration_norm", "fib_softness", "time_maturity_ratio",
        "htf_trend_signed",
    )}
    flags: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.int64) for name in (
        "new_leg_bull", "new_leg_bear", "regular_bull_div", "hidden_bull_div",
        "regular_bear_div", "hidden_bear_div", "trigger_state_long", "trigger_state_short",
        "long_is_continuation", "setup_long", "setup_short", "new_setup_long", "new_setup_short",
        "trigger_long", "trigger_short", "signal_long", "signal_short",
        "t1_hit_long", "t2_hit_long", "inv_hit_long", "cancelled_long",
        "t1_hit_short", "t2_hit_short", "inv_hit_short", "cancelled_short",
        "health_state_long", "health_state_short",
    )}

    # 状态机变量
    swing_dir = 0
    origin_price = math.nan
    origin_bar = -1
    end_price = math.nan
    end_bar = -1
    reference_epoch = 0
    atr_at_leg_end = math.nan
    low_since_last_high = math.nan
    high_since_last_low = math.nan
    low_since_last_low = math.nan
    high_since_last_high = math.nan
    leg_path_sum = 0.0
    leg_overlap_sum = 0.0
    leg_overlap_count = 0
    ref_eff = math.nan
    ref_overlap = math.nan
    ref_bars = math.nan
    ref_quality = 50.0
    quality_hist: list[float] = []
    range_hist: list[float] = []
    live_path_sum = 0.0
    live_overlap_sum = 0.0
    live_overlap_count = 0

    last_low_osc = math.nan
    prev_low_osc = math.nan
    last_low_price = math.nan
    prev_low_price = math.nan
    last_low_div_bar = math.nan
    prev_low_div_bar = math.nan
    last_high_osc = math.nan
    prev_high_osc = math.nan
    last_high_price = math.nan
    prev_high_price = math.nan
    last_high_div_bar = math.nan
    prev_high_div_bar = math.nan
    bull_div_bar = math.nan
    bear_div_bar = math.nan
    bull_div_strength = 0.0
    bear_div_strength = 0.0

    long_state = 0
    long_level = math.nan
    long_state_bar = -1
    long_epoch = math.nan
    long_consumed = False
    short_state = 0
    short_level = math.nan
    short_state_bar = -1
    short_epoch = math.nan
    short_consumed = False
    last_long_trigger_bar = -10**9
    last_short_trigger_bar = -10**9

    long_sticky = True
    long_sticky_epoch = math.nan
    short_sticky = True
    short_sticky_epoch = math.nan

    stale_long = 0.0
    stale_short = 0.0

    t1_long_f = t2_long_f = inv_long_f = math.nan
    t1_short_f = t2_short_f = inv_short_f = math.nan
    long_t1_hit = long_t2_hit = long_inv_hit = long_cancelled = False
    short_t1_hit = short_t2_hit = short_inv_hit = short_cancelled = False

    pivot_high_hist: list[float] = []
    pivot_low_hist: list[float] = []

    score_raw_long = np.zeros(n)
    score_raw_short = np.zeros(n)
    long_setup_prev = False
    short_setup_prev = False

    for i in range(n):
        price = close[i]
        atr_i = atr_safe[i]
        pivot_tolerance = atr_i * pivot_atr_tolerance

        # --- 参考腿 ---------------------------------------------------------
        new_high = not np.isnan(pivot_high[i])
        new_low = not np.isnan(pivot_low[i])
        if new_high:
            pivot_high_hist.append(float(pivot_high[i]))
            if len(pivot_high_hist) > max_pivot_history:
                pivot_high_hist.pop(0)
        if new_low:
            pivot_low_hist.append(float(pivot_low[i]))
            if len(pivot_low_hist) > max_pivot_history:
                pivot_low_hist.pop(0)

        new_reference_leg = False
        high_endpoint_updated = False
        low_endpoint_updated = False
        if new_high:
            if swing_dir == -1:
                current_range = abs(end_price - origin_price)
                if np.isnan(origin_price) or (
                    current_range > EPS
                    and (pivot_high[i] - end_price) / current_range >= leg_invalidation_retracement
                ):
                    origin_price, origin_bar = end_price, end_bar
                    end_price, end_bar = float(pivot_high[i]), i - pivot_len
                    swing_dir = 1
                    reference_epoch += 1
                    new_reference_leg = True
                    high_endpoint_updated = True
            else:
                if np.isnan(end_price) or pivot_high[i] > end_price + pivot_tolerance:
                    end_price, end_bar = float(pivot_high[i]), i - pivot_len
                    high_endpoint_updated = True
                swing_dir = 1
        if new_low:
            if swing_dir == 1:
                current_range = abs(end_price - origin_price)
                if np.isnan(origin_price) or (
                    current_range > EPS
                    and (end_price - pivot_low[i]) / current_range >= leg_invalidation_retracement
                ):
                    origin_price, origin_bar = end_price, end_bar
                    end_price, end_bar = float(pivot_low[i]), i - pivot_len
                    swing_dir = -1
                    reference_epoch += 1
                    new_reference_leg = True
                    low_endpoint_updated = True
            else:
                if np.isnan(end_price) or pivot_low[i] < end_price - pivot_tolerance:
                    end_price, end_bar = float(pivot_low[i]), i - pivot_len
                    low_endpoint_updated = True
                swing_dir = -1

        last_high = last_low = math.nan
        last_high_bar = last_low_bar = -1
        if swing_dir == 1:
            last_high, last_high_bar = end_price, end_bar
            last_low, last_low_bar = origin_price, origin_bar
        elif swing_dir == -1:
            last_low, last_low_bar = end_price, end_bar
            last_high, last_high_bar = origin_price, origin_bar

        has_leg = not np.isnan(origin_price)
        leg_span = last_high - last_low if (has_leg and not np.isnan(last_high) and not np.isnan(last_low)) else math.nan
        latest_bull = swing_dir == 1 and has_leg and leg_span / atr_i >= min_swing_atr
        latest_bear = swing_dir == -1 and has_leg and leg_span / atr_i >= min_swing_atr

        if high_endpoint_updated or low_endpoint_updated:
            atr_at_leg_end = atr_i

        if high_endpoint_updated:
            low_since_last_high = rolling_low_n[i]
        elif not np.isnan(last_high):
            low_since_last_high = low[i] if np.isnan(low_since_last_high) else min(low_since_last_high, low[i])
        if low_endpoint_updated:
            high_since_last_low = rolling_high_n[i]
        elif not np.isnan(last_low):
            high_since_last_low = high[i] if np.isnan(high_since_last_low) else max(high_since_last_low, high[i])
        if low_endpoint_updated:
            low_since_last_low = rolling_low_n[i]
        elif not np.isnan(last_low):
            low_since_last_low = low[i] if np.isnan(low_since_last_low) else min(low_since_last_low, low[i])
        if high_endpoint_updated:
            high_since_last_high = rolling_high_n[i]
        elif not np.isnan(last_high):
            high_since_last_high = high[i] if np.isnan(high_since_last_high) else max(high_since_last_high, high[i])

        # --- 腿累积器（参考腿） ---------------------------------------------
        if new_reference_leg:
            leg_bars = (end_bar - origin_bar) if (origin_bar >= 0 and end_bar >= 0) else None
            if leg_bars is not None and leg_bars > 0 and leg_path_sum > EPS:
                ref_eff = float(_clamp(_safe_div(abs(end_price - origin_price), leg_path_sum, 0.0), 0.0, 1.0))
                ref_overlap = (
                    float(_clamp(leg_overlap_sum / leg_overlap_count, 0.0, 1.0))
                    if leg_overlap_count > 0
                    else 0.5
                )
                ref_bars = float(leg_bars)
            leg_path_sum = 0.0
            leg_overlap_sum = 0.0
            leg_overlap_count = 0
        leg_path_sum += abs(price - close[i - 1]) if i > 0 else 0.0
        leg_overlap_sum += float(bar_overlap[i]) if np.isfinite(bar_overlap[i]) else 0.0
        leg_overlap_count += 1

        if not np.isnan(ref_eff):
            ref_quality = float(_clamp(100.0 * (ref_eff * 0.55 + (1.0 - ref_overlap) * 0.45), 0.0, 100.0))
        else:
            ref_quality = 50.0

        if new_reference_leg and not np.isnan(ref_eff):
            quality_hist.append(ref_quality)
            denom = max(atr_at_leg_end if np.isfinite(atr_at_leg_end) else atr_i, tick)
            range_hist.append(abs(end_price - origin_price) / denom)
            if len(quality_hist) > max_leg_history + 1:
                quality_hist.pop(0)
                range_hist.pop(0)

        prior_quality = quality_hist[:-1] if quality_hist else []
        prior_range = range_hist[:-1] if range_hist else []
        recent_range = range_hist[-max_leg_history:] if range_hist else []
        level_q = _recency_weighted_avg(prior_quality)
        trend_q = _recent_half_vs_older_half(prior_quality)
        range_trend = _recent_half_vs_older_half(prior_range)
        range_ratio = _recent_vs_older_ratio(recent_range)
        if level_q is None:
            live_trend_quality = 50.0
        else:
            live_trend_quality = float(
                _clamp(level_q + 0.3 * trend_q + 8.0 * float(_clamp(range_trend, -2.0, 2.0)), 0.0, 100.0)
            )

        live_leg_bars = (i - end_bar) if end_bar >= 0 else None
        current_endpoint_updated = (swing_dir == 1 and high_endpoint_updated) or (
            swing_dir == -1 and low_endpoint_updated
        )
        if current_endpoint_updated:
            seed_p = seed_path[i]
            seed_o = seed_overlap[i]
            live_path_sum = 0.0 if not np.isfinite(seed_p) else float(seed_p)
            live_overlap_sum = 0.0 if not np.isfinite(seed_o) else float(seed_o)
            live_overlap_count = pivot_len
        else:
            live_path_sum += abs(price - close[i - 1]) if i > 0 else 0.0
            live_overlap_sum += float(bar_overlap[i]) if np.isfinite(bar_overlap[i]) else 0.0
            live_overlap_count += 1

        live_leg_eff = 0.0
        if not np.isnan(end_price) and live_path_sum > EPS:
            live_leg_eff = float(_clamp(_safe_div(abs(price - end_price), live_path_sum, 0.0), 0.0, 1.0))
        live_leg_overlap = (
            float(_clamp(live_overlap_sum / live_overlap_count, 0.0, 1.0))
            if live_overlap_count > 0
            else 0.5
        )
        live_leg_quality = float(
            _clamp(100.0 * (live_leg_eff * 0.55 + (1.0 - live_leg_overlap) * 0.45), 0.0, 100.0)
        )

        # --- 回撤比例与成熟度 -----------------------------------------------
        bullish_range = leg_span if latest_bull else math.nan
        bearish_range = leg_span if latest_bear else math.nan
        bull_retr = math.nan
        bear_retr = math.nan
        if latest_bull and bullish_range > EPS:
            bull_retr = (last_high - price) / bullish_range
        if latest_bear and bearish_range > EPS:
            bear_retr = (price - last_low) / bearish_range

        q_frac = ref_quality / 100.0
        if q_frac >= 0.67:
            preferred_center = float(_lerp(normal_center, shallow_center, (q_frac - 0.67) / 0.33))
        else:
            preferred_center = float(_lerp(weak_center, normal_center, q_frac / 0.67))

        active_range = bullish_range if latest_bull else bearish_range if latest_bear else math.nan
        fib_softness = 0.12
        if not np.isnan(active_range) and active_range > EPS:
            fib_softness = 0.075 + fib_tolerance_atr * atr_i / active_range
        fib_softness = float(_clamp(fib_softness, 0.05, 0.30))

        bull_zone = (
            float(_gaussian_like(bull_retr, preferred_center, fib_softness))
            if not np.isnan(bull_retr)
            else 0.0
        )
        bear_zone = (
            float(_gaussian_like(bear_retr, preferred_center, fib_softness))
            if not np.isnan(bear_retr)
            else 0.0
        )
        rev_zone_bull = (
            float(_gaussian_like(bull_retr, reversal_center, reversal_width)) if not np.isnan(bull_retr) else 0.0
        )
        rev_zone_bear = (
            float(_gaussian_like(bear_retr, reversal_center, reversal_width)) if not np.isnan(bear_retr) else 0.0
        )
        presence_bull = float(_clamp(bull_retr / correction_presence_min, 0.0, 1.0)) if not np.isnan(bull_retr) else 0.0
        presence_bear = float(_clamp(bear_retr / correction_presence_min, 0.0, 1.0)) if not np.isnan(bear_retr) else 0.0

        time_ratio = math.nan
        if (
            (latest_bull or latest_bear)
            and not np.isnan(ref_bars)
            and ref_bars > 0
            and live_leg_bars is not None
        ):
            time_ratio = live_leg_bars / ref_bars
        time_factor = (
            float(_gaussian_like(time_ratio, time_maturity_center, time_maturity_width))
            if not np.isnan(time_ratio)
            else 1.0
        )
        rev_time_factor = (
            float(_gaussian_like(time_ratio, reversal_time_maturity_center, reversal_time_maturity_width))
            if not np.isnan(time_ratio)
            else 1.0
        )

        cont_long = float(_clamp(100.0 * bull_zone * presence_bull * time_factor, 0.0, 100.0))
        cont_short = float(_clamp(100.0 * bear_zone * presence_bear * time_factor, 0.0, 100.0))
        structural_rev_short = float(_clamp(100.0 * rev_zone_bull * presence_bull * rev_time_factor, 0.0, 100.0))
        structural_rev_long = float(_clamp(100.0 * rev_zone_bear * presence_bear * rev_time_factor, 0.0, 100.0))

        leg_ext_atr = (
            active_range / max(atr_at_leg_end if np.isfinite(atr_at_leg_end) else atr_i, tick)
            if not np.isnan(active_range)
            else math.nan
        )
        endpoint_overextension = (
            float(_clamp((leg_ext_atr - overextension_atr) / overextension_atr, 0.0, 1.0))
            if not np.isnan(leg_ext_atr)
            else 0.0
        )
        endpoint_presence_bull = (
            float(_clamp(1.0 - bull_retr / endpoint_presence_max, 0.0, 1.0)) if not np.isnan(bull_retr) else 0.0
        )
        endpoint_presence_bear = (
            float(_clamp(1.0 - bear_retr / endpoint_presence_max, 0.0, 1.0)) if not np.isnan(bear_retr) else 0.0
        )
        endpoint_rev_short = float(_clamp(100.0 * endpoint_overextension * endpoint_presence_bull, 0.0, 100.0))
        endpoint_rev_long = float(_clamp(100.0 * endpoint_overextension * endpoint_presence_bear, 0.0, 100.0))
        rev_long = max(structural_rev_long, endpoint_rev_long)
        rev_short = max(structural_rev_short, endpoint_rev_short)

        # --- 背离 -----------------------------------------------------------
        if new_low:
            prev_low_osc, prev_low_price, prev_low_div_bar = last_low_osc, last_low_price, last_low_div_bar
            back = i - pivot_len
            last_low_osc = osc_smoothed[back] if back >= 0 else math.nan
            last_low_price = float(pivot_low[i])
            last_low_div_bar = float(back)
        if new_high:
            prev_high_osc, prev_high_price, prev_high_div_bar = last_high_osc, last_high_price, last_high_div_bar
            back = i - pivot_len
            last_high_osc = osc_smoothed[back] if back >= 0 else math.nan
            last_high_price = float(pivot_high[i])
            last_high_div_bar = float(back)

        valid_low_span = (
            not np.isnan(last_low_div_bar)
            and not np.isnan(prev_low_div_bar)
            and last_low_div_bar - prev_low_div_bar <= divergence_max_bars
        )
        valid_high_span = (
            not np.isnan(last_high_div_bar)
            and not np.isnan(prev_high_div_bar)
            and last_high_div_bar - prev_high_div_bar <= divergence_max_bars
        )
        regular_bull = bool(
            divergence_enabled
            and valid_low_span
            and last_low_price < prev_low_price - pivot_tolerance
            and last_low_osc > prev_low_osc
        )
        hidden_bull = bool(
            divergence_enabled
            and valid_low_span
            and last_low_price > prev_low_price + pivot_tolerance
            and last_low_osc < prev_low_osc
        )
        regular_bear = bool(
            divergence_enabled
            and valid_high_span
            and last_high_price > prev_high_price + pivot_tolerance
            and last_high_osc < prev_high_osc
        )
        hidden_bear = bool(
            divergence_enabled
            and valid_high_span
            and last_high_price < prev_high_price - pivot_tolerance
            and last_high_osc > prev_high_osc
        )
        if regular_bull or hidden_bull:
            bull_div_bar = float(i)
            bull_div_strength = 1.0 if regular_bull else 0.75
        if regular_bear or hidden_bear:
            bear_div_bar = float(i)
            bear_div_strength = 1.0 if regular_bear else 0.75

        decay_bars = pivot_len * 3
        bull_div_decay = (
            float(_clamp(1.0 - (i - bull_div_bar) / decay_bars, 0.0, 1.0))
            if not np.isnan(bull_div_bar)
            else 0.0
        )
        bear_div_decay = (
            float(_clamp(1.0 - (i - bear_div_bar) / decay_bars, 0.0, 1.0))
            if not np.isnan(bear_div_bar)
            else 0.0
        )
        bull_div_bonus = divergence_strength * bull_div_decay * bull_div_strength
        bear_div_bonus = divergence_strength * bear_div_decay * bear_div_strength

        # --- 衰竭 -----------------------------------------------------------
        accel = float(accel_norm[i]) if np.isfinite(accel_norm[i]) else 0.0
        recent_eff = float(_clamp(_safe_div(abs(recent_net[i]), recent_path[i], 0.0), 0.0, 1.0))
        exhaustion_raw = 0.0
        if live_leg_bars is not None and live_leg_bars >= exhaustion_recent_bars:
            exhaustion_raw = 100.0 * float(_clamp(live_leg_eff - recent_eff, 0.0, 1.0))
        bear_exhaustion = (
            float(_clamp(exhaustion_raw + 15.0 * max(accel, 0.0) + bull_div_bonus, 0.0, 100.0))
            if latest_bull
            else 0.0
        )
        bull_exhaustion = (
            float(_clamp(exhaustion_raw + 15.0 * max(-accel, 0.0) + bear_div_bonus, 0.0, 100.0))
            if latest_bear
            else 0.0
        )
        bull_impulse_exhaustion = (
            float(_clamp(100.0 * bear_div_decay * bear_div_strength, 0.0, 100.0)) if latest_bull else 0.0
        )
        bear_impulse_exhaustion = (
            float(_clamp(100.0 * bull_div_decay * bull_div_strength, 0.0, 100.0)) if latest_bear else 0.0
        )

        # --- 持仓健康 -------------------------------------------------------
        live_trend_long = live_trend_quality if swing_dir == 1 else 100.0 - live_trend_quality
        live_trend_short = live_trend_quality if swing_dir == -1 else 100.0 - live_trend_quality
        counter_long = 100.0 - live_leg_quality if swing_dir == 1 else live_leg_quality
        counter_short = 100.0 - live_leg_quality if swing_dir == -1 else live_leg_quality
        live_quality_long = float(_weighted_avg2(live_trend_long, 60.0, counter_long, 40.0))
        live_quality_short = float(_weighted_avg2(live_trend_short, 60.0, counter_short, 40.0))

        blend_long = float(_weighted_avg2(cont_long, 65.0, bear_exhaustion, 35.0))
        blend_short = float(_weighted_avg2(cont_short, 65.0, bull_exhaustion, 35.0))
        pullback_health_long = (
            float(_lerp(100.0, blend_long, presence_bull)) if swing_dir == 1 else rev_long
        )
        pullback_health_short = (
            float(_lerp(100.0, blend_short, presence_bear)) if swing_dir == -1 else rev_short
        )

        close_back = close[i - structural_progress_lookback] if i >= structural_progress_lookback else price
        long_net_atr = (price - close_back) / atr_i
        short_net_atr = (close_back - price) / atr_i
        trend_progress_long = float(_clamp(100.0 * long_net_atr / trend_progress_full_credit_atr, 0.0, 100.0))
        trend_progress_short = float(_clamp(100.0 * short_net_atr / trend_progress_full_credit_atr, 0.0, 100.0))

        scaled_ratio = range_ratio / trend_fatigue_growth_scale if range_ratio >= 0 else range_ratio
        trend_fatigue = float(_clamp(50.0 + 50.0 * float(_clamp(scaled_ratio, -1.0, 1.0)), 0.0, 100.0))

        have_leg_range = not np.isnan(last_high) and not np.isnan(last_low) and leg_span > EPS
        long_recovery = float(_clamp((price - last_low) / leg_span, 0.0, 1.0)) if have_leg_range else 0.0
        short_recovery = float(_clamp((last_high - price) / leg_span, 0.0, 1.0)) if have_leg_range else 0.0
        structure_long = (
            100.0 * (1.0 - float(_clamp((bull_retr if not np.isnan(bull_retr) else 0.0) / leg_invalidation_retracement, 0.0, 1.0)))
            if swing_dir == 1
            else 100.0 * float(_clamp(long_recovery / leg_invalidation_retracement, 0.0, 1.0))
        )
        structure_short = (
            100.0 * (1.0 - float(_clamp((bear_retr if not np.isnan(bear_retr) else 0.0) / leg_invalidation_retracement, 0.0, 1.0)))
            if swing_dir == -1
            else 100.0 * float(_clamp(short_recovery / leg_invalidation_retracement, 0.0, 1.0))
        )

        regime_eff = float(_safe_div(regime_net[i], regime_path[i], 0.5))
        regime_score = float(_clamp(100.0 * regime_eff, 0.0, 100.0))

        total_weight = max(
            w_live_quality + w_pullback_health + w_trend_progress + w_fatigue
            + w_structure_intact + w_regime,
            EPS,
        )
        health_long_raw = (
            w_live_quality * live_quality_long
            + w_pullback_health * pullback_health_long
            + w_trend_progress * trend_progress_long
            + w_fatigue * trend_fatigue
            + w_structure_intact * structure_long
            + w_regime * regime_score
        ) / total_weight
        health_short_raw = (
            w_live_quality * live_quality_short
            + w_pullback_health * pullback_health_short
            + w_trend_progress * trend_progress_short
            + w_fatigue * trend_fatigue
            + w_structure_intact * structure_short
            + w_regime * regime_score
        ) / total_weight

        # --- Trigger 状态机 -------------------------------------------------
        candle_range = max(high[i] - low[i], tick)
        long_rejection = (
            i > 0
            and low[i] < low[i - 1]
            and price > low[i - 1]
            and float(_safe_div(price - low[i], candle_range, 0.0)) >= rejection_close_loc
        )
        short_rejection = (
            i > 0
            and high[i] > high[i - 1]
            and price < high[i - 1]
            and float(_safe_div(high[i] - price, candle_range, 0.0)) >= rejection_close_loc
        )
        prev_micro_high = micro_last_high[i - 1] if i > 0 else math.nan
        prev_micro_low = micro_last_low[i - 1] if i > 0 else math.nan
        long_micro_break = (
            not np.isnan(micro_last_high[i])
            and not np.isnan(prev_micro_high)
            and price > micro_last_high[i]
            and close[i - 1] <= prev_micro_high
        )
        short_micro_break = (
            not np.isnan(micro_last_low[i])
            and not np.isnan(prev_micro_low)
            and price < micro_last_low[i]
            and close[i - 1] >= prev_micro_low
        )

        if not np.isnan(long_epoch) and long_epoch != reference_epoch:
            long_state, long_level, long_state_bar, long_epoch = 0, math.nan, -1, math.nan
        if long_state == 0:
            if long_rejection:
                long_state, long_level, long_state_bar, long_epoch = 1, high[i], i, float(reference_epoch)
            elif long_micro_break:
                long_state, long_level, long_state_bar, long_epoch = 2, float(micro_last_high[i]), i, float(reference_epoch)
        elif long_state == 1:
            if price > long_level:
                long_state, long_state_bar = 2, i
            elif price < long_level - atr_i * 0.3:
                long_state, long_level, long_epoch = 0, math.nan, math.nan
        elif long_state == 2:
            if price < long_level:
                long_state, long_level, long_epoch = 0, math.nan, math.nan
            elif i - long_state_bar >= acceptance_bars:
                long_state = 3
        else:
            if price < long_level:
                long_state, long_level, long_epoch = 0, math.nan, math.nan

        if not np.isnan(short_epoch) and short_epoch != reference_epoch:
            short_state, short_level, short_state_bar, short_epoch = 0, math.nan, -1, math.nan
        if short_state == 0:
            if short_rejection:
                short_state, short_level, short_state_bar, short_epoch = 1, low[i], i, float(reference_epoch)
            elif short_micro_break:
                short_state, short_level, short_state_bar, short_epoch = 2, float(micro_last_low[i]), i, float(reference_epoch)
        elif short_state == 1:
            if price < short_level:
                short_state, short_state_bar = 2, i
            elif price > short_level + atr_i * 0.3:
                short_state, short_level, short_epoch = 0, math.nan, math.nan
        elif short_state == 2:
            if price > short_level:
                short_state, short_level, short_epoch = 0, math.nan, math.nan
            elif i - short_state_bar >= acceptance_bars:
                short_state = 3
        else:
            if price > short_level:
                short_state, short_level, short_epoch = 0, math.nan, math.nan

        def trigger_score(state: int, state_bar: int) -> tuple[float, float]:
            if state == 1:
                return 30.0, 0.0
            if state == 2:
                frac = float(_clamp((i - state_bar) / acceptance_bars, 0.0, 1.0))
                return 50.0 + 25.0 * frac, 100.0 * frac
            if state == 3:
                return 100.0, 100.0
            return 0.0, 0.0

        long_trigger_score, long_acceptance = trigger_score(long_state, long_state_bar)
        short_trigger_score, short_acceptance = trigger_score(short_state, short_state_bar)

        # --- Permission -----------------------------------------------------
        ctx_gate = float(_gated_factor(ref_quality, context_floor))
        trend_gate = float(_gated_factor(live_trend_quality, live_trend_quality_floor))
        cont_long_perm = (
            100.0 * ctx_gate * trend_gate
            * float(_gated_factor(cont_long, maturity_floor))
            * float(_gated_factor(bear_exhaustion, exhaustion_floor))
        )
        rev_long_perm = (
            100.0 * ctx_gate * trend_gate
            * float(_gated_factor(rev_long, maturity_floor))
            * float(_gated_factor(bear_impulse_exhaustion, exhaustion_floor))
        )
        cont_short_perm = (
            100.0 * ctx_gate * trend_gate
            * float(_gated_factor(cont_short, maturity_floor))
            * float(_gated_factor(bull_exhaustion, exhaustion_floor))
        )
        rev_short_perm = (
            100.0 * ctx_gate * trend_gate
            * float(_gated_factor(rev_short, maturity_floor))
            * float(_gated_factor(bull_impulse_exhaustion, exhaustion_floor))
        )

        if htf_enabled and valid_htf:
            htf_trend = float(htf_trend_signed[i])
            cont_long_perm = float(_clamp(cont_long_perm + htf_influence * htf_trend, 0.0, 100.0))
            cont_short_perm = float(_clamp(cont_short_perm - htf_influence * htf_trend, 0.0, 100.0))
            long_factor = reversal_htf_aligned_factor if htf_trend >= 0 else reversal_htf_conflict_factor
            rev_long_perm = float(_clamp(rev_long_perm + htf_influence * long_factor * htf_trend, 0.0, 100.0))
            short_factor = reversal_htf_aligned_factor if htf_trend <= 0 else reversal_htf_conflict_factor
            rev_short_perm = float(_clamp(rev_short_perm - htf_influence * short_factor * htf_trend, 0.0, 100.0))

        if np.isnan(long_sticky_epoch) or long_sticky_epoch != reference_epoch:
            long_sticky = cont_long_perm >= rev_long_perm
            long_sticky_epoch = float(reference_epoch)
        elif cont_long_perm > rev_long_perm + scenario_hysteresis_margin:
            long_sticky = True
        elif rev_long_perm > cont_long_perm + scenario_hysteresis_margin:
            long_sticky = False

        if np.isnan(short_sticky_epoch) or short_sticky_epoch != reference_epoch:
            short_sticky = cont_short_perm >= rev_short_perm
            short_sticky_epoch = float(reference_epoch)
        elif cont_short_perm > rev_short_perm + scenario_hysteresis_margin:
            short_sticky = True
        elif rev_short_perm > cont_short_perm + scenario_hysteresis_margin:
            short_sticky = False

        long_permission = max(cont_long_perm, rev_long_perm)
        short_permission = max(cont_short_perm, rev_short_perm)

        # --- 目标 / 失效位 ---------------------------------------------------
        min_t1 = min_target_distance_atr * atr_i
        t1_long = _nearest_pivot(pivot_high_hist, price, min_t1, above=True)
        if t1_long is None:
            t1_long = price + min_target_distance_atr * 2.0 * atr_i
        t1_short = _nearest_pivot(pivot_low_hist, price, min_t1, above=False)
        if t1_short is None:
            t1_short = price - min_target_distance_atr * 2.0 * atr_i

        t2_long = inv_long = math.nan
        if long_sticky:
            if not np.isnan(low_since_last_high) and not np.isnan(last_high):
                t2_long = low_since_last_high + extension_mult * (last_high - low_since_last_high)
            if not np.isnan(low_since_last_high):
                inv_long = low_since_last_high - atr_i * invalidation_atr_buffer
        else:
            if not np.isnan(high_since_last_low) and not np.isnan(last_high) and not np.isnan(last_low):
                t2_long = last_high + extension_mult * (high_since_last_low - last_low)
            if not np.isnan(low_since_last_low):
                inv_long = low_since_last_low - atr_i * invalidation_atr_buffer

        t2_short = inv_short = math.nan
        if short_sticky:
            if not np.isnan(high_since_last_low) and not np.isnan(last_low):
                t2_short = high_since_last_low - extension_mult * (high_since_last_low - last_low)
            if not np.isnan(high_since_last_low):
                inv_short = high_since_last_low + atr_i * invalidation_atr_buffer
        else:
            if not np.isnan(low_since_last_high) and not np.isnan(last_low) and not np.isnan(last_high):
                t2_short = last_low - extension_mult * (last_high - low_since_last_high)
            if not np.isnan(high_since_last_high):
                inv_short = high_since_last_high + atr_i * invalidation_atr_buffer

        if np.isnan(t2_long) or (t2_long - t1_long) < min_t1:
            t2_long = math.nan
        if np.isnan(t2_short) or (t1_short - t2_short) < min_t1:
            t2_short = math.nan

        # --- Confirmation / setup / 交易生命周期 ----------------------------
        long_confirmation = float(_weighted_avg2(long_trigger_score, trigger_weight, long_acceptance, acceptance_weight))
        short_confirmation = float(_weighted_avg2(short_trigger_score, trigger_weight, short_acceptance, acceptance_weight))
        long_pre_crv = long_permission * (
            confirmation_floor + (1.0 - confirmation_floor) * long_confirmation / 100.0
        )
        short_pre_crv = short_permission * (
            confirmation_floor + (1.0 - confirmation_floor) * short_confirmation / 100.0
        )

        long_setup = long_permission >= permission_threshold and long_permission > short_permission
        short_setup = short_permission >= permission_threshold and short_permission > long_permission
        new_long_setup = long_setup and not long_setup_prev
        new_short_setup = short_setup and not short_setup_prev
        long_setup_prev, short_setup_prev = long_setup, short_setup

        long_cooldown_ok = (i - last_long_trigger_bar) > cooldown_bars
        short_cooldown_ok = (i - last_short_trigger_bar) > cooldown_bars

        if long_state == 0:
            long_consumed = False
        if short_state == 0:
            short_consumed = False

        if long_state >= 2 and not long_consumed and (i - long_state_bar) > trigger_validity_bars:
            long_state, long_level, long_epoch = 0, math.nan, math.nan
        if short_state >= 2 and not short_consumed and (i - short_state_bar) > trigger_validity_bars:
            short_state, short_level, short_epoch = 0, math.nan, math.nan

        long_armed = long_state >= 2 and (i - long_state_bar) <= trigger_validity_bars and not long_consumed
        short_armed = short_state >= 2 and (i - short_state_bar) <= trigger_validity_bars and not short_consumed

        has_long_t2_prior = use_t2 and not np.isnan(t2_long_f)
        has_short_t2_prior = use_t2 and not np.isnan(t2_short_f)
        long_open_prior = (
            not np.isnan(t1_long_f)
            and not long_inv_hit
            and not long_cancelled
            and (not long_t2_hit if has_long_t2_prior else not long_t1_hit)
        )
        short_open_prior = (
            not np.isnan(t1_short_f)
            and not short_inv_hit
            and not short_cancelled
            and (not short_t2_hit if has_short_t2_prior else not short_t1_hit)
        )

        long_candidate = long_armed and long_cooldown_ok and long_setup and not long_open_prior
        short_candidate = short_armed and short_cooldown_ok and short_setup and not short_open_prior
        simultaneous = long_candidate and short_candidate
        new_long_trigger = long_candidate and (not simultaneous or long_pre_crv > short_pre_crv)
        new_short_trigger = short_candidate and (not simultaneous or short_pre_crv > long_pre_crv)

        if new_long_trigger:
            last_long_trigger_bar = i
            long_consumed = True
            short_state, short_level, short_epoch = 0, math.nan, math.nan
        if new_short_trigger:
            last_short_trigger_bar = i
            short_consumed = True
            long_state, long_level, long_epoch = 0, math.nan, math.nan

        if new_long_trigger:
            t1_long_f, t2_long_f, inv_long_f = t1_long, t2_long, inv_long
            long_t1_hit = long_t2_hit = long_inv_hit = long_cancelled = False
        if new_short_trigger:
            t1_short_f, t2_short_f, inv_short_f = t1_short, t2_short, inv_short
            short_t1_hit = short_t2_hit = short_inv_hit = short_cancelled = False

        has_long_t2 = use_t2 and not np.isnan(t2_long_f)
        has_short_t2 = use_t2 and not np.isnan(t2_short_f)
        long_open = (
            not np.isnan(t1_long_f)
            and not long_inv_hit
            and not long_cancelled
            and (not long_t2_hit if has_long_t2 else not long_t1_hit)
        )
        short_open = (
            not np.isnan(t1_short_f)
            and not short_inv_hit
            and not short_cancelled
            and (not short_t2_hit if has_short_t2 else not short_t1_hit)
        )
        if new_short_trigger and long_open:
            long_cancelled = True
        if new_long_trigger and short_open:
            short_cancelled = True

        long_still_open = long_open and not long_cancelled
        short_still_open = short_open and not short_cancelled

        long_inv_now = long_still_open and not np.isnan(inv_long_f) and low[i] <= inv_long_f
        if long_inv_now:
            long_inv_hit = True
        long_t1_now = (
            long_still_open and not long_t1_hit and not long_inv_now
            and not np.isnan(t1_long_f) and high[i] >= t1_long_f
        )
        long_t2_now = (
            long_still_open and use_t2 and not long_t2_hit and not long_inv_now
            and not np.isnan(t2_long_f) and high[i] >= t2_long_f
        )
        if long_t1_now:
            long_t1_hit = True
        if long_t2_now:
            long_t2_hit = True

        short_inv_now = short_still_open and not np.isnan(inv_short_f) and high[i] >= inv_short_f
        if short_inv_now:
            short_inv_hit = True
        short_t1_now = (
            short_still_open and not short_t1_hit and not short_inv_now
            and not np.isnan(t1_short_f) and low[i] <= t1_short_f
        )
        short_t2_now = (
            short_still_open and use_t2 and not short_t2_hit and not short_inv_now
            and not np.isnan(t2_short_f) and low[i] <= t2_short_f
        )
        if short_t1_now:
            short_t1_hit = True
        if short_t2_now:
            short_t2_hit = True

        # 对侧状态可能已在上面被清零；按 bar 收盘时的最终状态重算触发分，
        # 让导出的 state 与 score 永远一致（Pine 在同 bar 上会留下一个
        # “state 已复位、score 仍是 100”的瞬时不一致，这里按收盘口径统一）。
        long_trigger_score, long_acceptance = trigger_score(long_state, long_state_bar)
        short_trigger_score, short_acceptance = trigger_score(short_state, short_state_bar)
        long_confirmation = float(_weighted_avg2(long_trigger_score, trigger_weight, long_acceptance, acceptance_weight))
        short_confirmation = float(_weighted_avg2(short_trigger_score, trigger_weight, short_acceptance, acceptance_weight))
        long_pre_crv = long_permission * (
            confirmation_floor + (1.0 - confirmation_floor) * long_confirmation / 100.0
        )
        short_pre_crv = short_permission * (
            confirmation_floor + (1.0 - confirmation_floor) * short_confirmation / 100.0
        )

        # --- CRV / 衰减 / 最终分 --------------------------------------------
        long_crv_factor = 1.0
        if long_still_open:
            crv_target = t2_long_f if (long_t1_hit and has_long_t2) else t1_long_f
            if not np.isnan(inv_long_f) and not np.isnan(crv_target):
                crv = float(_safe_div(crv_target - price, max(price - inv_long_f, atr_i * 0.1), 0.0))
                long_crv_factor = crv_floor + (1.0 - crv_floor) * float(_clamp(crv / min_crv, 0.0, 1.0))
        elif not np.isnan(inv_long):
            crv = float(_safe_div(t1_long - price, max(price - inv_long, atr_i * 0.1), 0.0))
            long_crv_factor = crv_floor + (1.0 - crv_floor) * float(_clamp(crv / min_crv, 0.0, 1.0))

        short_crv_factor = 1.0
        if short_still_open:
            crv_target = t2_short_f if (short_t1_hit and has_short_t2) else t1_short_f
            if not np.isnan(inv_short_f) and not np.isnan(crv_target):
                crv = float(_safe_div(price - crv_target, max(inv_short_f - price, atr_i * 0.1), 0.0))
                short_crv_factor = crv_floor + (1.0 - crv_floor) * float(_clamp(crv / min_crv, 0.0, 1.0))
        elif not np.isnan(inv_short):
            crv = float(_safe_div(price - t1_short, max(inv_short - price, atr_i * 0.1), 0.0))
            short_crv_factor = crv_floor + (1.0 - crv_floor) * float(_clamp(crv / min_crv, 0.0, 1.0))

        atr_back = atr_safe[i - decay_atr_lookback] if i >= decay_atr_lookback else math.nan
        long_progress = high[i] > high[i - 1] or accel > 0.05 or atr_i > (atr_back if np.isfinite(atr_back) else atr_i)
        short_progress = low[i] < low[i - 1] or accel < -0.05 or atr_i > (atr_back if np.isfinite(atr_back) else atr_i)
        stale_long = 0.0 if long_progress else stale_long + 1.0
        stale_short = 0.0 if short_progress else stale_short + 1.0
        long_decay = float(_clamp(1.0 - stale_long * decay_per_bar / 100.0, 0.0, 1.0))
        short_decay = float(_clamp(1.0 - stale_short * decay_per_bar / 100.0, 0.0, 1.0))

        score_raw_long[i] = long_pre_crv * long_crv_factor * long_decay
        score_raw_short[i] = short_pre_crv * short_crv_factor * short_decay
        long_signal = bool(
            new_long_trigger and long_permission > 0
            and score_raw_long[i] >= long_permission * signal_retention / 100.0
        )
        short_signal = bool(
            new_short_trigger and short_permission > 0
            and score_raw_short[i] >= short_permission * signal_retention / 100.0
        )

        # --- 落盘 -----------------------------------------------------------
        cols["swing_direction"][i] = swing_dir
        cols["last_high"][i] = last_high
        cols["last_low"][i] = last_low
        cols["leg_range_atr"][i] = leg_span / atr_i if not np.isnan(leg_span) else math.nan
        cols["reference_leg_quality"][i] = ref_quality
        cols["live_trend_quality"][i] = live_trend_quality
        cols["live_leg_quality"][i] = live_leg_quality
        cols["bullish_retracement"][i] = bull_retr
        cols["bearish_retracement"][i] = bear_retr
        cols["preferred_fib_center"][i] = preferred_center
        cols["continuation_maturity_long"][i] = cont_long
        cols["continuation_maturity_short"][i] = cont_short
        cols["reversal_maturity_long"][i] = rev_long
        cols["reversal_maturity_short"][i] = rev_short
        cols["exhaustion_bull"][i] = bull_exhaustion
        cols["exhaustion_bear"][i] = bear_exhaustion
        cols["impulse_exhaustion_bull"][i] = bull_impulse_exhaustion
        cols["impulse_exhaustion_bear"][i] = bear_impulse_exhaustion
        cols["divergence_bull_bonus"][i] = bull_div_bonus
        cols["divergence_bear_bonus"][i] = bear_div_bonus
        cols["cont_permission_long"][i] = cont_long_perm
        cols["cont_permission_short"][i] = cont_short_perm
        cols["rev_permission_long"][i] = rev_long_perm
        cols["rev_permission_short"][i] = rev_short_perm
        cols["permission_long"][i] = long_permission
        cols["permission_short"][i] = short_permission
        cols["t1_long"][i] = t1_long
        cols["t2_long"][i] = t2_long
        cols["inv_long"][i] = inv_long
        cols["t1_short"][i] = t1_short
        cols["t2_short"][i] = t2_short
        cols["inv_short"][i] = inv_short
        cols["trigger_score_long"][i] = long_trigger_score
        cols["trigger_score_short"][i] = short_trigger_score
        cols["acceptance_long"][i] = long_acceptance
        cols["acceptance_short"][i] = short_acceptance
        cols["confirmation_long"][i] = long_confirmation
        cols["confirmation_short"][i] = short_confirmation
        cols["crv_factor_long"][i] = long_crv_factor
        cols["crv_factor_short"][i] = short_crv_factor
        cols["decay_factor_long"][i] = long_decay
        cols["decay_factor_short"][i] = short_decay
        cols["stale_bars_long"][i] = stale_long
        cols["stale_bars_short"][i] = stale_short
        cols["score_raw_long"][i] = score_raw_long[i]
        cols["score_raw_short"][i] = score_raw_short[i]
        cols["live_quality_long"][i] = live_quality_long
        cols["live_quality_short"][i] = live_quality_short
        cols["pullback_health_long"][i] = pullback_health_long
        cols["pullback_health_short"][i] = pullback_health_short
        cols["trend_progress_long"][i] = trend_progress_long
        cols["trend_progress_short"][i] = trend_progress_short
        cols["trend_fatigue"][i] = trend_fatigue
        cols["structure_intact_long"][i] = structure_long
        cols["structure_intact_short"][i] = structure_short
        cols["regime_score"][i] = regime_score
        cols["health_long"][i] = health_long_raw
        cols["health_short"][i] = health_short_raw
        cols["atr"][i] = atr_i
        cols["oscillator"][i] = osc_smoothed[i]
        cols["acceleration_norm"][i] = accel
        cols["fib_softness"][i] = fib_softness
        cols["time_maturity_ratio"][i] = time_ratio
        cols["htf_trend_signed"][i] = htf_trend_signed[i]

        flags["new_leg_bull"][i] = int(new_reference_leg and swing_dir == 1)
        flags["new_leg_bear"][i] = int(new_reference_leg and swing_dir == -1)
        flags["regular_bull_div"][i] = int(regular_bull)
        flags["hidden_bull_div"][i] = int(hidden_bull)
        flags["regular_bear_div"][i] = int(regular_bear)
        flags["hidden_bear_div"][i] = int(hidden_bear)
        flags["trigger_state_long"][i] = long_state
        flags["trigger_state_short"][i] = short_state
        flags["long_is_continuation"][i] = int(long_sticky)
        flags["setup_long"][i] = int(long_setup)
        flags["setup_short"][i] = int(short_setup)
        flags["new_setup_long"][i] = int(new_long_setup)
        flags["new_setup_short"][i] = int(new_short_setup)
        flags["trigger_long"][i] = int(new_long_trigger)
        flags["trigger_short"][i] = int(new_short_trigger)
        flags["signal_long"][i] = int(long_signal)
        flags["signal_short"][i] = int(short_signal)
        flags["t1_hit_long"][i] = int(long_t1_now)
        flags["t2_hit_long"][i] = int(long_t2_now)
        flags["inv_hit_long"][i] = int(long_inv_now)
        flags["cancelled_long"][i] = int(new_short_trigger and long_open)
        flags["t1_hit_short"][i] = int(short_t1_now)
        flags["t2_hit_short"][i] = int(short_t2_now)
        flags["inv_hit_short"][i] = int(short_inv_now)
        flags["cancelled_short"][i] = int(new_long_trigger and short_open)

    # --- 平滑与健康状态码 ---------------------------------------------------
    score_long = ema(pd.Series(score_raw_long), smoothing).to_numpy(dtype=float)
    score_short = ema(pd.Series(score_raw_short), smoothing).to_numpy(dtype=float)
    health_long = ema(pd.Series(cols["health_long"]), smoothing).to_numpy(dtype=float)
    health_short = ema(pd.Series(cols["health_short"]), smoothing).to_numpy(dtype=float)

    def health_state(values: np.ndarray) -> np.ndarray:
        state = np.zeros(len(values), dtype=np.int64)
        state = np.where(np.nan_to_num(values, nan=50.0) >= health_reduce_threshold, 1, state)
        state = np.where(np.nan_to_num(values, nan=50.0) >= health_protect_threshold, 2, state)
        state = np.where(np.nan_to_num(values, nan=50.0) >= health_hold_threshold, 3, state)
        return state

    cols["health_long"] = np.nan_to_num(health_long, nan=50.0)
    cols["health_short"] = np.nan_to_num(health_short, nan=50.0)

    out: dict[str, object] = {}
    for name, values in cols.items():
        out["tpe_" + name] = values
    out["tpe_score_long"] = score_long
    out["tpe_score_short"] = score_short
    for name, values in flags.items():
        out["tpe_" + name] = pd.array(values, dtype="Int64")
    flags["health_state_long"] = health_state(cols["health_long"])
    flags["health_state_short"] = health_state(cols["health_short"])
    out["tpe_health_state_long"] = pd.array(flags["health_state_long"], dtype="Int64")
    out["tpe_health_state_short"] = pd.array(flags["health_state_short"], dtype="Int64")
    return pd.DataFrame(out, index=frame.index)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _bar_seconds(frame: pd.DataFrame) -> float:
    """图表周期（秒），用于 Auto 预设与 HTF 选择。"""
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        return 3600.0
    deltas = pd.Series(frame.index).diff().dt.total_seconds().dropna()
    median = float(deltas.median()) if len(deltas) else 3600.0
    return median if median > 0 else 3600.0


def _auto_htf_seconds(bar_seconds: float) -> float:
    """Pine ``f_autoHtf``：按图表周期选上一级周期。"""
    if bar_seconds <= 300:
        return 900.0
    if bar_seconds <= 900:
        return 3600.0
    if bar_seconds <= 3600:
        return 14400.0
    if bar_seconds <= 14400:
        return 86400.0
    if bar_seconds <= 86400:
        return 604800.0
    return 2592000.0


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """把稀疏的枢轴值向前填充成台阶序列（对应 Pine 的 ``var`` 保持）。"""
    out = np.full(len(values), np.nan)
    current = np.nan
    for i, value in enumerate(values):
        if not np.isnan(value):
            current = value
        out[i] = current
    return out


def _recency_weighted_avg(values: list[float]) -> float | None:
    """Pine ``f_recencyWeightedAvg``：权重 1..n 的加权平均。"""
    if not values:
        return None
    total_weight = 0.0
    weighted = 0.0
    for index, value in enumerate(values):
        weight = float(index + 1)
        weighted += value * weight
        total_weight += weight
    return weighted / total_weight if total_weight > EPS else None


def _recent_half_vs_older_half(values: list[float]) -> float:
    """Pine ``f_recentHalfVsOlderHalf``：近半均值 − 旧半均值（少于 4 个返回 0）。"""
    size = len(values)
    if size < 4:
        return 0.0
    half = size // 2
    older = sum(values[:half]) / half
    recent = sum(values[half:]) / (size - half)
    return recent - older


def _recent_vs_older_ratio(values: list[float]) -> float:
    """Pine ``f_recentVsOlderRatio``：近半均值/旧半均值 − 1（相对而非绝对）。"""
    size = len(values)
    if size < 4:
        return 0.0
    half = size // 2
    older = sum(values[:half]) / half
    recent = sum(values[half:]) / (size - half)
    return (recent - older) / max(older, EPS)


def _nearest_pivot(history: list[float], price: float, min_distance: float, *, above: bool) -> float | None:
    """Pine ``f_nearestPivotAbove/Below``：最近的、距离达标的确认枢轴。"""
    best: float | None = None
    for value in history:
        if above:
            if value - price >= min_distance and (best is None or value < best):
                best = value
        else:
            if price - value >= min_distance and (best is None or value > best):
                best = value
    return best
