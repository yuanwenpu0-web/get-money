"""Regime Detector Pro + MA Zones Overlay 移植。

Port of ``data/pine-collection/indicators/trend_strength/regime_detector/regime_detector.pine``
（清单里的 ``regime_detector_ma_zones_overlay``）。

Pine 原脚本是一个七段的复合引擎（大量 ``type`` 结构体承载状态）。本移植逐段保留：

1. **自适应核**（KAMA 混合）：``er = |src−src[n]| / (Σ|Δsrc| )``，``alpha`` 由 ER 经
   ``gamma`` 非线性映射到 [alphaMin, alphaMax]，自适应线是 ``prev + alpha·(src−prev)``
   的递归；``slope = Δline / ATR``。ATR 给出进场/出场带（进场带宽 > 出场带宽 = 滞回）。
2. **制度识别**：ER 进/出双阈值 + 动量门（``|slope| ≥ sensitivity``）+ 置信度门槛，
   三态**带滞回**状态机（0 无 / 1 多 / −1 空），置信度在 0 态取进场口径、
   在换向那一根取进场口径、维持原向时取出场口径。
3. **MTF 门**：按图表周期自动叠三级 HTF（15m→60/240/D、1h→240/D/W、4h→D/W/M、
   1D→W/M/3M、更高→M/3M/6M），每级方向 = ``close vs EMA50``，按权重算加权一致率、
   净偏置与背离强度。
4. **过滤器**：ATR/价格 波动门槛（Static / Adaptive / Dynamic 三模式）、量能门、
   市场状态（趋势/盘整）、合流折扣，综合成 0–1 的 ``signal_quality``。
5. **信号**：确认根数 + 冷却 + 质量/HTF 门 + Tier A/B 分级；出场支持
   Conservative（带外）/ Aggressive（制度失效）/ Hybrid（两者或）。
6. **绩效**：按仓位翻转记账，输出胜率、盈亏比、样本数。
7. **制度上下文（v1.1）**：波动率维度（ATR 与布林带宽的分位秩 → Quiet/Normal/Volatile）、
   制度成熟度（峰值强度衰减 → Aging）、状态切换事件、playbook（映射为码）。
8. **MA 区间覆盖**：三条 MA（SMA/EMA/DEMA/TEMA/JMA）+ ATR 间距强度门
   （Off / By MA Type / Auto 三种归一），由 RDP 制度或 MA 排列本身门控多/空区间，
   区间翻转时给出进场位与 ATR 止损位。

未移植：绘图/表格/告警、``log.*``、SSA（Smart Signals Assistant，是独立的另一套信号引擎）、
RS 相对基准（需要第二个标的的行情，单帧无法 ``request.security`` → 恒为中性）、
WIP 项（trailing stop / risk management / smart confirm / ML 分类，原脚本标注未接入引擎）。

``barstate.isconfirmed`` 在历史数据上恒为真，因此 ``bullFromMAs/bearFromMAs`` 不需要冻结逻辑。

输出 ``rdp_`` 前缀列。
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
    percentrank,
    rma,
    sma,
    stdev,
)

_PROFILES = ("Smart", "Futures", "Spot", "ETF", "Crypto", "Custom")
_REGIME_MODES = ("Trend Only", "Momentum Only", "Trend + Momentum", "Full Analysis")
_VOL_MODES = ("Static", "Adaptive", "Dynamic")
_HTF_WEIGHTS = ("Primary Focus", "Balanced", "Long-term Focus")
_EXIT_MODES = ("Conservative", "Balanced", "Hybrid", "Aggressive")
_MA_TYPES = ("SMA", "EMA", "DEMA", "TEMA", "JMA")
_STRENGTH_MODES = ("Off", "By MA Type", "Auto")

# playbook 文本 → 码（0 = 无）
PLAYBOOK_NONE = 0
PLAYBOOK_COMPRESSION = 1
PLAYBOOK_CHOPPY = 2
PLAYBOOK_RANGE = 3
PLAYBOOK_AGING = 4
PLAYBOOK_STRETCHED = 5
PLAYBOOK_HEALTHY = 6
PLAYBOOK_TREND = 7


def _clamp01(values):
    """Pine ``f_clamp01``。"""
    return np.maximum(0.0, np.minimum(1.0, values))


def _safe_div(numerator, denominator, fallback: float = 0.0):
    """除零守卫：分母为 0 或 na → ``fallback``（Pine 的 ``!= 0 ? a/b : fallback``）。"""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.isfinite(den) & (den != 0.0)
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _shift(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 0:
        return arr.copy()
    if len(arr) <= length:
        return np.full(len(arr), np.nan)
    return np.concatenate([np.full(length, np.nan), arr[: len(arr) - length]])


def _jma(src: np.ndarray, length: int, phase: float, power: float) -> np.ndarray:
    """Pine ``calculateJMA``（开源 Jurik 近似）。"""
    phase_clamped = max(-100.0, min(100.0, phase))
    phase_ratio = (phase_clamped + 100.0) / 200.0
    safe_length = max(1.0, float(length))
    beta = 0.45 * (safe_length - 1.0) / (0.45 * (safe_length - 1.0) + 2.0)
    alpha = beta ** power
    n = len(src)
    out = np.full(n, np.nan)
    e0 = e1 = e2 = value = math.nan
    for i in range(n):
        price = src[i]
        seed = price if not np.isfinite(value) else value
        e0 = (1.0 - alpha) * price + alpha * (price if not np.isfinite(e0) else e0)
        e1 = (price - e0) * (1.0 - beta) + beta * (0.0 if not np.isfinite(e1) else e1)
        e2 = (e0 + phase_ratio * e1 - seed) * (1.0 - alpha) ** 2 + alpha ** 2 * (
            0.0 if not np.isfinite(e2) else e2
        )
        value = e2 + seed
        out[i] = value
    return out


def _ma_of_type(kind: str, series: pd.Series, length: int) -> np.ndarray:
    """``f_ma_zone`` 的五种均线。"""
    if kind == "SMA":
        return sma(series, length).to_numpy(dtype=float)
    if kind == "EMA":
        return ema(series, length).to_numpy(dtype=float)
    if kind == "DEMA":
        first = ema(series, length)
        return (2.0 * first - ema(first, length)).to_numpy(dtype=float)
    if kind == "TEMA":
        first = ema(series, length)
        second = ema(first, length)
        return (3.0 * first - 3.0 * second + ema(second, length)).to_numpy(dtype=float)
    if kind == "JMA":
        return _jma(series.to_numpy(dtype=float), length, 0.0, 2.0)
    return ema(series, length).to_numpy(dtype=float)


def _strength_factor(kind: str) -> float:
    """Pine ``f_strengthFactor_zone``。"""
    return {"SMA": 1.00, "EMA": 1.00, "DEMA": 1.05, "TEMA": 0.80, "JMA": 0.85}.get(kind, 1.00)


def _bar_seconds(frame: pd.DataFrame) -> float:
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        return 3600.0
    deltas = pd.Series(frame.index).diff().dt.total_seconds().dropna()
    median = float(deltas.median()) if len(deltas) else 3600.0
    return median if median > 0 else 3600.0


def _htf_timeframes(chart_seconds: float) -> tuple[str, str, str]:
    """Pine 的 HTF 自动叠级。"""
    if chart_seconds <= 900:
        return ("60", "240", "D")
    if chart_seconds <= 3600:
        return ("240", "D", "W")
    if chart_seconds <= 14400:
        return ("D", "W", "M")
    if chart_seconds <= 86400:
        return ("W", "M", "3M")
    return ("M", "3M", "6M")


_TF_SECONDS = {
    "60": 3600.0, "240": 14400.0, "D": 86400.0, "W": 604800.0,
    "M": 2592000.0, "3M": 7776000.0, "6M": 15552000.0,
}


def _htf_direction(frame: pd.DataFrame, tf_label: str, ema_length: int = 50) -> np.ndarray:
    """HTF 方向：``close vs EMA50``，映射回图表 bar。

    对应 Pine 的 ``request.security(..., close > ta.ema(close, 50) ? 1 : ...,
    gaps_off, lookahead_off)``。``lookahead_off`` 拿到的是**正在形成**的那根 HTF bar
    的值，所以这里用「上一根已完成 HTF bar 的 EMA50」与当前图表收盘价比 —— 这正是
    单帧数据里能拿到的等价信息。
    """
    n = len(frame)
    target = _TF_SECONDS.get(tf_label, 0.0)
    chart = _bar_seconds(frame)
    if target <= 0 or target <= chart:
        return np.zeros(n, dtype=np.int64)
    idx = pd.DatetimeIndex(frame.index)
    if target >= 28 * 86400:
        key = pd.PeriodIndex(idx, freq="M")
    elif target >= 6 * 86400:
        key = pd.PeriodIndex(idx, freq="W")
    elif target >= 86400:
        key = pd.PeriodIndex(idx, freq="D")
    else:
        key = idx.floor(pd.Timedelta(seconds=target))
    grouped = frame.groupby(key).agg(close=("close", "last"))
    # shift(1)：取**上一根已完成** HTF bar 的 EMA，否则当期 bar 的收尾价会被用到（未来函数）
    htf_ema = ema(grouped["close"], ema_length).shift(1)
    bars = pd.Series(np.asarray(key))
    ema_a = bars.map(htf_ema).to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    direction = np.zeros(n, dtype=np.int64)
    valid = np.isfinite(ema_a)
    direction[valid & (close > ema_a)] = 1
    direction[valid & (close < ema_a)] = -1
    return direction


def regime_detector_ma_zones_overlay(
    frame: pd.DataFrame,
    *,
    profile_preset: str = "Smart",
    source: str = "close",
    enable_adaptive_engine: bool = True,
    er_length: int = 20,
    alpha_min: float = 0.03,
    alpha_max: float = 0.30,
    gamma: float = 1.25,
    atr_length: int = 14,
    entry_multiplier: float = 1.6,
    exit_multiplier: float = 1.2,
    enable_advanced_regime: bool = True,
    regime_mode: str = "Trend + Momentum",
    er_threshold: float = 0.2,
    er_threshold_exit: float = 0.15,
    regime_sensitivity: float = 0.6,
    regime_confidence_min: float = 0.65,
    regime_confidence_exit: float = 0.55,
    enable_vola_dim: bool = True,
    vola_lookback: int = 100,
    vola_quiet_max: int = 33,
    vola_vol_min: int = 66,
    enable_mtf_gate: bool = True,
    htf_weights: str = "Balanced",
    htf_min_agreement: float = 0.6,
    enable_htf_divergence: bool = True,
    enable_advanced_filters: bool = True,
    min_atr_percent: float = 0.0007,
    volatility_mode: str = "Adaptive",
    use_volume_filter: bool = False,
    volume_length: int = 50,
    volume_quantile: float = 0.3,
    enable_market_condition: bool = True,
    market_condition_period: int = 100,
    enable_confluence: bool = True,
    confluence_threshold: float = 0.6,
    confirm_bars: int = 2,
    enable_advanced_exits: bool = True,
    exit_mode: str = "Hybrid",
    cooldown_period: int = 5,
    tier_a_conf: float = 0.80,
    tier_a_agree: float = 0.65,
    tier_a_quality: float = 0.70,
    tier_b_conf: float = 0.70,
    tier_b_agree: float = 0.55,
    tier_b_quality: float = 0.60,
    enable_ma_zones: bool = False,
    use_rdp_regime: bool = True,
    ma_type: str = "DEMA",
    len1: int = 21,
    len2: int = 55,
    len3: int = 89,
    atr_len_zones: int = 20,
    mult_zones: float = 2.0,
    strength_mode_zones: str = "By MA Type",
    base_min_strength_zones: float = 0.6,
    auto_lookback_zones: int = 200,
    min_confidence_for_zones: float = 0.6,
    min_htf_agreement_for_zones: float = 0.5,
    warning_divergence_threshold: float = 0.65,
    warning_regime_strength_min: float = 0.30,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Regime Detector Pro（含 MA 区间覆盖）。

    Args:
        frame: OHLCV（``volume`` 可选），索引为时间。
        profile_preset: ``Smart`` / ``Futures`` / ``Spot`` / ``ETF`` / ``Crypto`` / ``Custom``。
            非 Custom 时覆盖 alpha/gamma 三个参数（Pine 的 profile 预设）。
        source: 自适应引擎的输入序列（列名或 ``hl2``/``hlc3``/``ohlc4``）。
        enable_adaptive_engine: 自适应引擎总开关。
        er_length: 效率比窗口（5..100）。
        alpha_min: 平滑系数下限。
        alpha_max: 平滑系数上限。
        gamma: alpha 对 ER 的非线性硬度（0.5..3.0）。
        atr_length: ATR 长度（2..50）。
        entry_multiplier: 进场带的 ATR 倍数。
        exit_multiplier: 出场带的 ATR 倍数（小于进场 = 滞回）。
        enable_advanced_regime: 是否启用高级制度识别（关闭则任何 ER 都放行）。
        regime_mode: ``Trend Only`` / ``Momentum Only`` / ``Trend + Momentum`` / ``Full Analysis``。
        er_threshold: 进入趋势制度的最小 ER。
        er_threshold_exit: 退出趋势制度的最小 ER（滞回）。
        regime_sensitivity: 动量门要求的 ``|slope|`` 下限（退出取 80%）。
        regime_confidence_min: 进入制度的最小置信度。
        regime_confidence_exit: 维持制度的最小置信度。
        enable_vola_dim: 是否计算波动率维度（Quiet/Normal/Volatile）。
        vola_lookback: 波动率分位秩的窗口（20..500）。
        vola_quiet_max: 低于该分位视为 Quiet（5..49）。
        vola_vol_min: 高于该分位视为 Volatile（51..95）。
        enable_mtf_gate: 是否启用多周期门。
        htf_weights: ``Primary Focus`` / ``Balanced`` / ``Long-term Focus``。
        htf_min_agreement: 最小加权一致率。
        enable_htf_divergence: 是否计算 HTF 背离强度。
        enable_advanced_filters: 是否启用过滤器系统。
        min_atr_percent: ATR/价格 的绝对下限。
        volatility_mode: ``Static`` / ``Adaptive`` / ``Dynamic``。
        use_volume_filter: 是否启用量能门。
        volume_length: 量能均线长度（5..200）。
        volume_quantile: 量能相对均线的下限倍数。
        enable_market_condition: 是否计算市场状态（1 趋势 / −1 盘整 / 0 中性）。
        market_condition_period: 市场状态的分析窗口（50..500）。
        enable_confluence: 是否对低质量信号打折。
        confluence_threshold: 合流门槛。
        confirm_bars: 信号确认根数（1..10）。
        enable_advanced_exits: 是否启用进阶出场。
        exit_mode: ``Conservative`` / ``Balanced`` / ``Hybrid`` / ``Aggressive``。
        cooldown_period: 同向信号冷却 bar 数（0..50）。
        tier_a_conf: Tier A 的最小置信度。
        tier_a_agree: Tier A 的最小 HTF 一致率。
        tier_a_quality: Tier A 的最小信号质量。
        tier_b_conf: Tier B 的最小置信度。
        tier_b_agree: Tier B 的最小 HTF 一致率。
        tier_b_quality: Tier B 的最小信号质量。
        enable_ma_zones: 是否启用 MA 区间覆盖。
        use_rdp_regime: 区间由 RDP 制度门控（否则由 MA 排列门控）。
        ma_type: ``SMA`` / ``EMA`` / ``DEMA`` / ``TEMA`` / ``JMA``。
        len1: 第一条 MA 长度。
        len2: 第二条 MA 长度。
        len3: 第三条 MA 长度。
        atr_len_zones: 区间用的 ATR 长度。
        mult_zones: 止损距离的 ATR 倍数。
        strength_mode_zones: ``Off`` / ``By MA Type`` / ``Auto``。
        base_min_strength_zones: 基础最小强度（MA 间距/ATR）。
        auto_lookback_zones: Auto 归一的回看窗口（≥50）。
        min_confidence_for_zones: 区间所需的最小制度置信度。
        min_htf_agreement_for_zones: 区间所需的最小 HTF 一致率。
        warning_divergence_threshold: 背离警告阈值。
        warning_regime_strength_min: 制度强度过低的警告阈值。
        volume_is_real: ``False`` 表示量能不可信（量能门退化为放行）。

    Returns:
        ``pd.DataFrame``，索引与 ``frame`` 一致，列为 ``rdp_*``。

    Raises:
        ValueError: 参数越界或取值不在允许集合内。
    """
    profile_preset = choice("profile_preset", profile_preset, _PROFILES)
    regime_mode = choice("regime_mode", regime_mode, _REGIME_MODES)
    volatility_mode = choice("volatility_mode", volatility_mode, _VOL_MODES)
    htf_weights = choice("htf_weights", htf_weights, _HTF_WEIGHTS)
    exit_mode = choice("exit_mode", exit_mode, _EXIT_MODES)
    ma_type = choice("ma_type", ma_type, _MA_TYPES)
    strength_mode_zones = choice("strength_mode_zones", strength_mode_zones, _STRENGTH_MODES)
    enable_adaptive_engine = boolean("enable_adaptive_engine", enable_adaptive_engine)
    er_length = integer("er_length", er_length, 5, 100)
    alpha_min = number("alpha_min", alpha_min, 0.001, 0.5)
    alpha_max = number("alpha_max", alpha_max, 0.01, 1.0)
    gamma = number("gamma", gamma, 0.5, 3.0)
    atr_length = integer("atr_length", atr_length, 2, 50)
    entry_multiplier = number("entry_multiplier", entry_multiplier, 0.5, 5.0)
    exit_multiplier = number("exit_multiplier", exit_multiplier, 0.3, 5.0)
    er_threshold = number("er_threshold", er_threshold, 0.0, 1.0)
    er_threshold_exit = number("er_threshold_exit", er_threshold_exit, 0.0, 1.0)
    regime_sensitivity = number("regime_sensitivity", regime_sensitivity, 0.1, 1.0)
    regime_confidence_min = number("regime_confidence_min", regime_confidence_min, 0.0, 1.0)
    regime_confidence_exit = number("regime_confidence_exit", regime_confidence_exit, 0.0, 1.0)
    vola_lookback = integer("vola_lookback", vola_lookback, 20, 500)
    vola_quiet_max = integer("vola_quiet_max", vola_quiet_max, 5, 49)
    vola_vol_min = integer("vola_vol_min", vola_vol_min, 51, 95)
    htf_min_agreement = number("htf_min_agreement", htf_min_agreement, 0.0, 1.0)
    min_atr_percent = number("min_atr_percent", min_atr_percent, 0.0, 1.0)
    use_volume_filter = boolean("use_volume_filter", use_volume_filter)
    volume_length = integer("volume_length", volume_length, 5, 200)
    volume_quantile = number("volume_quantile", volume_quantile, 0.0, 1.0)
    market_condition_period = integer("market_condition_period", market_condition_period, 50, 500)
    confluence_threshold = number("confluence_threshold", confluence_threshold, 0.0, 1.0)
    confirm_bars = integer("confirm_bars", confirm_bars, 1, 10)
    cooldown_period = integer("cooldown_period", cooldown_period, 0, 50)
    tier_a_conf = number("tier_a_conf", tier_a_conf, 0.0, 1.0)
    tier_a_agree = number("tier_a_agree", tier_a_agree, 0.0, 1.0)
    tier_a_quality = number("tier_a_quality", tier_a_quality, 0.0, 1.0)
    tier_b_conf = number("tier_b_conf", tier_b_conf, 0.0, 1.0)
    tier_b_agree = number("tier_b_agree", tier_b_agree, 0.0, 1.0)
    tier_b_quality = number("tier_b_quality", tier_b_quality, 0.0, 1.0)
    len1 = integer("len1", len1, 1, 5000)
    len2 = integer("len2", len2, 1, 5000)
    len3 = integer("len3", len3, 1, 5000)
    atr_len_zones = integer("atr_len_zones", atr_len_zones, 1, 500)
    mult_zones = number("mult_zones", mult_zones, 0.1, 100.0)
    base_min_strength_zones = number("base_min_strength_zones", base_min_strength_zones, 0.0, 100.0)
    auto_lookback_zones = integer("auto_lookback_zones", auto_lookback_zones, 50, 5000)
    min_confidence_for_zones = number("min_confidence_for_zones", min_confidence_for_zones, 0.0, 1.0)
    min_htf_agreement_for_zones = number(
        "min_htf_agreement_for_zones", min_htf_agreement_for_zones, 0.0, 1.0
    )
    warning_divergence_threshold = number(
        "warning_divergence_threshold", warning_divergence_threshold, 0.0, 1.0
    )
    warning_regime_strength_min = number(
        "warning_regime_strength_min", warning_regime_strength_min, 0.0, 1.0
    )
    enable_advanced_regime = boolean("enable_advanced_regime", enable_advanced_regime)
    enable_vola_dim = boolean("enable_vola_dim", enable_vola_dim)
    enable_mtf_gate = boolean("enable_mtf_gate", enable_mtf_gate)
    enable_htf_divergence = boolean("enable_htf_divergence", enable_htf_divergence)
    enable_advanced_filters = boolean("enable_advanced_filters", enable_advanced_filters)
    enable_market_condition = boolean("enable_market_condition", enable_market_condition)
    enable_confluence = boolean("enable_confluence", enable_confluence)
    enable_advanced_exits = boolean("enable_advanced_exits", enable_advanced_exits)
    enable_ma_zones = boolean("enable_ma_zones", enable_ma_zones)
    use_rdp_regime = boolean("use_rdp_regime", use_rdp_regime)

    high_s, low_s, close_s = ohlcv(frame, "high", "low", "close")
    close = close_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    n = len(frame)
    src = _resolve_source(frame, source, high, low, close)

    # --- profile 预设（非 Custom 覆盖 alpha/gamma） -------------------------
    if profile_preset != "Custom":
        presets = {
            "Smart": (0.025, 0.35, 1.15),
            "Futures": (0.02, 0.28, 1.35),
            "Spot": (0.03, 0.30, 1.25),
            "ETF": (0.015, 0.25, 1.45),
            "Crypto": (0.035, 0.40, 1.05),
        }
        if profile_preset in presets:
            alpha_min, alpha_max, gamma = presets[profile_preset]

    # --- 自适应核 -----------------------------------------------------------
    atr_a = atr(frame, atr_length).to_numpy(dtype=float)
    er_arr = _safe_div(
        np.abs(src - _shift(src, er_length)),
        sma(pd.Series(np.abs(np.diff(src, prepend=src[0]))), er_length).to_numpy(dtype=float) * er_length,
        0.0,
    )
    er_arr = np.where(np.isfinite(er_arr), er_arr, 0.0)
    alpha_arr = alpha_min + (alpha_max - alpha_min) * np.power(np.fmax(er_arr, 0.0), gamma)

    line = np.full(n, np.nan)
    prev_line = math.nan
    for i in range(n):
        base = src[i] if not np.isfinite(prev_line) else prev_line
        current = base + alpha_arr[i] * (src[i] - base)
        line[i] = current
        prev_line = current

    slope = np.where(
        np.isfinite(atr_a) & (atr_a > 0),
        _safe_div(line - _shift(line, 1), atr_a, 0.0),
        0.0,
    )

    if not enable_adaptive_engine:
        # Pine 的 processAdaptiveEngine 并未读取该开关（它只是 UI 占位）；
        # 这里保持一致：仍然计算，开关只作为元信息。
        pass

    upper_entry = line + entry_multiplier * atr_a
    lower_entry = line - entry_multiplier * atr_a
    upper_exit = line + exit_multiplier * atr_a
    lower_exit = line - exit_multiplier * atr_a

    # --- 制度识别 -----------------------------------------------------------
    strength = _clamp01(np.abs(slope))
    er_enter_ok = er_arr >= er_threshold
    er_exit_ok = er_arr >= er_threshold_exit
    base_conf_enter = np.where(
        er_enter_ok,
        _clamp01(_safe_div(er_arr - er_threshold, 1.0 - er_threshold, 0.0)),
        0.0,
    )
    base_conf_exit = np.where(
        er_exit_ok,
        _clamp01(_safe_div(er_arr - er_threshold_exit, 1.0 - er_threshold_exit, 0.0)),
        0.0,
    )
    conf_enter = base_conf_enter
    conf_exit = base_conf_exit

    mom_enter_ok = strength >= regime_sensitivity
    mom_exit_ok = strength >= regime_sensitivity * 0.8
    if not enable_advanced_regime:
        mode_enter_ok = np.ones(n, dtype=bool)
        mode_exit_ok = np.ones(n, dtype=bool)
    elif regime_mode == "Trend Only":
        mode_enter_ok, mode_exit_ok = er_enter_ok, er_exit_ok
    elif regime_mode == "Momentum Only":
        mode_enter_ok, mode_exit_ok = mom_enter_ok, mom_exit_ok
    else:  # Trend + Momentum / Full Analysis
        mode_enter_ok = er_enter_ok & mom_enter_ok
        mode_exit_ok = er_exit_ok & mom_exit_ok

    regime = np.zeros(n, dtype=np.int64)
    confidence = np.zeros(n)
    current = 0
    for i in range(n):
        prev = current
        if prev == 0:
            if er_enter_ok[i] and mode_enter_ok[i] and conf_enter[i] >= regime_confidence_min:
                current = 1 if slope[i] > 0 else (-1 if slope[i] < 0 else 0)
        elif prev == 1:
            if (
                er_exit_ok[i] and mode_exit_ok[i]
                and conf_exit[i] >= regime_confidence_exit and slope[i] > 0
            ):
                current = 1
            elif (
                er_enter_ok[i] and mode_enter_ok[i]
                and conf_enter[i] >= regime_confidence_min and slope[i] < 0
            ):
                current = -1
            else:
                current = 0
        else:
            if (
                er_exit_ok[i] and mode_exit_ok[i]
                and conf_exit[i] >= regime_confidence_exit and slope[i] < 0
            ):
                current = -1
            elif (
                er_enter_ok[i] and mode_enter_ok[i]
                and conf_enter[i] >= regime_confidence_min and slope[i] > 0
            ):
                current = 1
            else:
                current = 0
        regime[i] = current
        if current == 0:
            confidence[i] = conf_enter[i]
        elif current == prev:
            confidence[i] = conf_exit[i]
        else:
            confidence[i] = conf_enter[i]

    # --- MTF 门 -------------------------------------------------------------
    chart_seconds = _bar_seconds(frame)
    tf1, tf2, tf3 = _htf_timeframes(chart_seconds)
    if htf_weights == "Primary Focus":
        w1, w2, w3 = 0.6, 0.3, 0.1
    elif htf_weights == "Long-term Focus":
        w1, w2, w3 = 0.2, 0.3, 0.5
    else:
        w1, w2, w3 = 0.4, 0.35, 0.25

    htf_agreement = np.ones(n)
    htf_bias = np.zeros(n, dtype=np.int64)
    htf_divergence = np.zeros(n)
    if enable_mtf_gate:
        dirs = [
            (_htf_direction(frame, tf1), w1),
            (_htf_direction(frame, tf2), w2),
            (_htf_direction(frame, tf3), w3),
        ]
        total_w = w1 + w2 + w3
        bias_sum = np.zeros(n)
        agree_w = np.zeros(n)
        for direction, weight in dirs:
            bias_sum += direction * weight
            agree_w += np.where(
                (regime != 0) & (direction == regime), weight, 0.0
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            htf_agreement = np.where(
                regime == 0,
                np.abs(bias_sum) / total_w,
                agree_w / total_w,
            )
        htf_agreement = np.nan_to_num(htf_agreement, nan=0.0)
        htf_bias = np.where(bias_sum > 0, 1, np.where(bias_sum < 0, -1, 0))
        if enable_htf_divergence:
            htf_divergence = 1.0 - htf_agreement

    # --- 过滤器 -------------------------------------------------------------
    if "volume" in frame.columns and volume_is_real is not False:
        volume = np.nan_to_num(frame["volume"].to_numpy(dtype=float), nan=0.0)
        volume_real = bool(np.nansum(volume) > 0)
    else:
        volume = np.zeros(n, dtype=float)
        volume_real = False
    volume_sma = sma(pd.Series(volume), volume_length).to_numpy(dtype=float)

    atr_pct = np.where(close > 0, _safe_div(atr_a, close, 0.0), 0.0)
    atr_pct_avg = sma(pd.Series(atr_pct), market_condition_period).to_numpy(dtype=float)
    if volatility_mode == "Static":
        atr_threshold = np.full(n, min_atr_percent)
    elif volatility_mode == "Adaptive":
        atr_threshold = np.fmax(min_atr_percent, atr_pct_avg * 0.7)
    else:  # Dynamic
        atr_threshold = np.fmax(
            min_atr_percent, atr_pct_avg * np.where(regime != 0, 0.6, 0.85)
        )
    vol_ok = atr_pct >= atr_threshold
    vol_gate = np.ones(n, dtype=bool)
    if use_volume_filter and volume_real:
        vol_gate = (volume_sma > 0) & (volume >= volume_sma * volume_quantile)

    trend_score = sma(pd.Series(np.abs(slope)), market_condition_period).to_numpy(dtype=float)
    er_avg = sma(pd.Series(er_arr), market_condition_period).to_numpy(dtype=float)
    market_condition = np.zeros(n, dtype=np.int64)
    if enable_market_condition:
        market_condition = np.where(trend_score > 0.35, 1, np.where(er_avg < 0.15, -1, 0))

    if not enable_advanced_filters:
        signal_quality = np.ones(n)
    else:
        if use_volume_filter and not volume_real:
            quality = np.where(vol_ok, 0.5 / 0.7, 0.0) + np.where(trend_score > 0.2, 0.2 / 0.7, 0.0)
        else:
            quality = (
                np.where(vol_ok, 0.5, 0.0)
                + np.where(vol_gate, 0.3, 0.0)
                + np.where(trend_score > 0.2, 0.2, 0.0)
            )
        if enable_confluence:
            quality = np.where(quality < confluence_threshold, quality * 0.5, quality)
        signal_quality = _clamp01(quality)

    # --- 信号 ---------------------------------------------------------------
    bull = regime == 1
    bear = regime == -1
    if confirm_bars <= 1:
        bars_ok_bull = bull
        bars_ok_bear = bear
    else:
        bars_ok_bull = sma(pd.Series(bull.astype(float)), confirm_bars).to_numpy(dtype=float) == 1.0
        bars_ok_bear = sma(pd.Series(bear.astype(float)), confirm_bars).to_numpy(dtype=float) == 1.0

    htf_gate_ok = (not enable_mtf_gate) or (htf_agreement >= htf_min_agreement)
    filter_ok = (signal_quality >= 0.5) & htf_gate_ok
    htf_agree = htf_agreement if enable_mtf_gate else np.ones(n)

    qual_a = (
        (confidence >= tier_a_conf) & (htf_agree >= tier_a_agree) & (signal_quality >= tier_a_quality)
    )
    qual_b = (
        (~qual_a)
        & (confidence >= tier_b_conf)
        & (htf_agree >= tier_b_agree)
        & (signal_quality >= tier_b_quality)
    )
    allow_entry = qual_a | qual_b

    position = np.zeros(n, dtype=np.int64)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    exit_long = np.zeros(n, dtype=bool)
    exit_short = np.zeros(n, dtype=bool)
    low_quality_long = np.zeros(n, dtype=bool)
    low_quality_short = np.zeros(n, dtype=bool)
    last_signal_bar = -10**9
    current_position = 0

    for i in range(n):
        can_fire = (last_signal_bar < 0) or (i - last_signal_bar >= cooldown_period)
        long_candidate = (
            can_fire and filter_ok[i] and bars_ok_bull[i] and current_position != 1
        )
        short_candidate = (
            can_fire and filter_ok[i] and bars_ok_bear[i] and current_position != -1
        )
        entry_long = long_candidate and allow_entry[i]
        entry_short = short_candidate and allow_entry[i]
        low_quality_long[i] = long_candidate and not allow_entry[i]
        low_quality_short[i] = short_candidate and not allow_entry[i]

        exit_by_band_long = current_position == 1 and close[i] < lower_exit[i]
        exit_by_band_short = current_position == -1 and close[i] > upper_exit[i]
        exit_by_regime_long = current_position == 1 and not bull[i]
        exit_by_regime_short = current_position == -1 and not bear[i]

        if enable_advanced_exits:
            if exit_mode == "Conservative":
                out_long, out_short = exit_by_band_long, exit_by_band_short
            elif exit_mode == "Aggressive":
                out_long, out_short = exit_by_regime_long, exit_by_regime_short
            else:
                out_long = exit_by_band_long or exit_by_regime_long
                out_short = exit_by_band_short or exit_by_regime_short
        else:
            out_long, out_short = exit_by_regime_long, exit_by_regime_short

        exit_long[i] = out_long
        exit_short[i] = out_short
        long_entry[i] = entry_long
        short_entry[i] = entry_short

        if entry_long:
            current_position = 1
            last_signal_bar = i
        elif entry_short:
            current_position = -1
            last_signal_bar = i
        elif out_long or out_short:
            current_position = 0
        position[i] = current_position

    tier = np.where(qual_a, 1, np.where(qual_b, 2, 0)).astype(np.int64)

    # --- 绩效 ---------------------------------------------------------------
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    win_rate = 0.0
    profit_factor = 0.0
    sample_count = np.zeros(n, dtype=np.int64)
    win_rate_arr = np.zeros(n)
    profit_factor_arr = np.zeros(n)
    entry_price = math.nan
    last_position = 0
    for i in range(n):
        if position[i] != last_position:
            if last_position == 0 and position[i] != 0:
                entry_price = close[i]
            elif last_position != 0 and position[i] == 0 and np.isfinite(entry_price):
                pnl = (close[i] - entry_price) * (1 if last_position == 1 else -1)
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += abs(pnl)
                sample_count[i] = wins + losses
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    profit_factor = 99.0
                else:
                    profit_factor = 0.0
        last_position = position[i]
        sample_count[i] = wins + losses
        win_rate_arr[i] = win_rate
        profit_factor_arr[i] = profit_factor

    # --- 制度上下文 ---------------------------------------------------------
    ctx_atr = atr_a
    atr_rank = percentrank(pd.Series(ctx_atr), vola_lookback).to_numpy(dtype=float)
    bb_basis = sma(close_s, 20).to_numpy(dtype=float)
    bb_dev = stdev(close_s, 20).to_numpy(dtype=float)
    bb_width = _safe_div(bb_dev * 4.0, bb_basis, 0.0)
    bbw_rank = percentrank(pd.Series(bb_width), vola_lookback).to_numpy(dtype=float)
    vola_score = (np.nan_to_num(atr_rank, nan=50.0) + np.nan_to_num(bbw_rank, nan=50.0)) / 2.0
    vola_state = np.where(
        vola_score < vola_quiet_max, -1, np.where(vola_score > vola_vol_min, 1, 0)
    ).astype(np.int64)

    regime_changed = np.zeros(n, dtype=bool)
    if n:
        regime_changed[0] = True
        regime_changed[1:] = regime[1:] != regime[:-1]
    bars_in_regime = np.zeros(n, dtype=np.int64)
    counter = 0
    for i in range(n):
        counter = 0 if regime_changed[i] else counter + 1
        bars_in_regime[i] = counter

    peak_strength = np.zeros(n)
    running_peak = 0.0
    for i in range(n):
        if regime_changed[i]:
            running_peak = strength[i]
        else:
            running_peak = max(running_peak, strength[i])
        peak_strength[i] = running_peak
    fade_ratio = np.where(
        peak_strength > 0, _safe_div(strength, peak_strength, 1.0), 1.0
    )

    slope_against = ((regime == 1) & (slope < 0)) | ((regime == -1) & (slope > 0))
    regime_weakening = (regime != 0) & (
        (er_arr < er_threshold_exit) | (fade_ratio < 0.5) | slope_against
    )
    vola_compression = np.zeros(n, dtype=bool)
    vola_expansion = np.zeros(n, dtype=bool)
    if enable_vola_dim and n:
        vola_compression[1:] = (vola_state[1:] == -1) & (vola_state[:-1] != -1)
        vola_expansion[1:] = (vola_state[1:] == 1) & (vola_state[:-1] != 1)
    weakening_event = np.zeros(n, dtype=bool)
    if n:
        weakening_event[1:] = regime_weakening[1:] & (~regime_weakening[:-1])

    playbook = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if regime[i] == 0:
            if enable_vola_dim and vola_state[i] == -1:
                playbook[i] = PLAYBOOK_COMPRESSION
            elif enable_vola_dim and vola_state[i] == 1:
                playbook[i] = PLAYBOOK_CHOPPY
            else:
                playbook[i] = PLAYBOOK_RANGE
        elif regime_weakening[i]:
            playbook[i] = PLAYBOOK_AGING
        elif enable_vola_dim and vola_state[i] == 1:
            playbook[i] = PLAYBOOK_STRETCHED
        elif enable_vola_dim and vola_state[i] == -1:
            playbook[i] = PLAYBOOK_HEALTHY
        else:
            playbook[i] = PLAYBOOK_TREND

    # --- MA 区间覆盖 --------------------------------------------------------
    ma1 = _ma_of_type(ma_type, close_s, len1)
    ma2 = _ma_of_type(ma_type, close_s, len2)
    ma3 = _ma_of_type(ma_type, close_s, len3)
    ma_avg = (ma1 + ma2 + ma3) / 3.0
    atr_zones = atr(frame, atr_len_zones).to_numpy(dtype=float)
    sl_distance = atr_zones * mult_zones
    highest_ma = np.fmax(ma1, np.fmax(ma2, ma3))
    lowest_ma = np.fmin(ma1, np.fmin(ma2, ma3))
    spread_atr = np.where(atr_zones > 0, _safe_div(highest_ma - lowest_ma, atr_zones, 0.0), 0.0)

    min_strength_by_type = base_min_strength_zones * _strength_factor(ma_type)
    ama_step = np.where(
        atr_zones > 0,
        _safe_div(rma(pd.Series(np.abs(ma_avg - _shift(ma_avg, 1))), atr_len_zones), atr_zones, 0.0),
        0.0,
    )
    ama_step_ref = sma(pd.Series(ama_step), auto_lookback_zones).to_numpy(dtype=float)
    auto_factor = np.where(
        np.isfinite(ama_step_ref) & (ama_step_ref > 0),
        _safe_div(ama_step, ama_step_ref, 1.0),
        1.0,
    )
    min_strength_auto = base_min_strength_zones * auto_factor

    if strength_mode_zones == "Off":
        min_strength = np.full(n, base_min_strength_zones)
    elif strength_mode_zones == "By MA Type":
        min_strength = np.full(n, min_strength_by_type)
    else:  # Auto
        min_strength = min_strength_auto

    strong_trend = spread_atr >= min_strength
    bull_from_mas = (
        (close >= ma1) & (close >= ma2) & (close >= ma3) & strong_trend
    )
    bear_from_mas = (
        (close <= ma1) & (close <= ma2) & (close <= ma3) & strong_trend
    )
    htf_ok_zones = (not enable_mtf_gate) or (htf_agreement >= min_htf_agreement_for_zones)

    if use_rdp_regime:
        zone_bull = (regime == 1) & (confidence >= min_confidence_for_zones) & htf_ok_zones
        zone_bear = (regime == -1) & (confidence >= min_confidence_for_zones) & htf_ok_zones
    else:
        zone_bull = bull_from_mas
        zone_bear = bear_from_mas
    zone_neutral = (~zone_bull) & (~zone_bear)

    zone_bull_flip = np.zeros(n, dtype=bool)
    zone_bear_flip = np.zeros(n, dtype=bool)
    zone_exit_flip = np.zeros(n, dtype=bool)
    if n:
        zone_bull_flip[1:] = zone_bull[1:] & (~zone_bull[:-1])
        zone_bear_flip[1:] = zone_bear[1:] & (~zone_bear[:-1])
        zone_exit_flip[1:] = zone_neutral[1:] & (~zone_neutral[:-1])

    zone_long_entry = np.full(n, np.nan)
    zone_long_sl = np.full(n, np.nan)
    zone_short_entry = np.full(n, np.nan)
    zone_short_sl = np.full(n, np.nan)
    cur_long_entry = cur_long_sl = cur_short_entry = cur_short_sl = math.nan
    for i in range(n):
        if zone_bull_flip[i]:
            cur_long_entry = highest_ma[i]
            cur_long_sl = highest_ma[i] - sl_distance[i]
            cur_short_entry = cur_short_sl = math.nan
        if zone_bear_flip[i]:
            cur_short_entry = lowest_ma[i]
            cur_short_sl = lowest_ma[i] + sl_distance[i]
            cur_long_entry = cur_long_sl = math.nan
        if zone_exit_flip[i]:
            cur_long_entry = cur_long_sl = cur_short_entry = cur_short_sl = math.nan
        zone_long_entry[i] = cur_long_entry
        zone_long_sl[i] = cur_long_sl
        zone_short_entry[i] = cur_short_entry
        zone_short_sl[i] = cur_short_sl

    # --- 警告 ---------------------------------------------------------------
    warn_divergence = (htf_divergence >= warning_divergence_threshold) & (regime != 0)
    warn_low_strength = (regime != 0) & (strength < warning_regime_strength_min)

    # --- 组装 ---------------------------------------------------------------
    out: dict[str, object] = {
        "rdp_line": line,
        "rdp_atr": atr_a,
        "rdp_er": er_arr,
        "rdp_alpha": alpha_arr,
        "rdp_slope": slope,
        "rdp_strength": strength,
        "rdp_confidence": confidence,
        "rdp_upper_entry": upper_entry,
        "rdp_lower_entry": lower_entry,
        "rdp_upper_exit": upper_exit,
        "rdp_lower_exit": lower_exit,
        "rdp_htf_agreement": htf_agreement,
        "rdp_htf_divergence": htf_divergence,
        "rdp_signal_quality": signal_quality,
        "rdp_atr_percent": atr_pct,
        "rdp_win_rate": win_rate_arr,
        "rdp_profit_factor": profit_factor_arr,
        "rdp_vola_score": vola_score,
        "rdp_atr_rank": atr_rank,
        "rdp_bbw_rank": bbw_rank,
        "rdp_fade_ratio": fade_ratio,
        "rdp_ma1": ma1,
        "rdp_ma2": ma2,
        "rdp_ma3": ma3,
        "rdp_ma_avg": ma_avg,
        "rdp_highest_ma": highest_ma,
        "rdp_lowest_ma": lowest_ma,
        "rdp_spread_atr": spread_atr,
        "rdp_min_strength": min_strength,
        "rdp_zone_atr": atr_zones,
        "rdp_zone_sl_distance": sl_distance,
        "rdp_zone_long_entry": zone_long_entry,
        "rdp_zone_long_sl": zone_long_sl,
        "rdp_zone_short_entry": zone_short_entry,
        "rdp_zone_short_sl": zone_short_sl,
        "rdp_regime": pd.array(regime, dtype="Int64"),
        "rdp_htf_bias": pd.array(htf_bias, dtype="Int64"),
        "rdp_market_condition": pd.array(market_condition, dtype="Int64"),
        "rdp_position": pd.array(position, dtype="Int64"),
        "rdp_tier": pd.array(tier, dtype="Int64"),
        "rdp_vola_state": pd.array(vola_state, dtype="Int64"),
        "rdp_playbook": pd.array(playbook, dtype="Int64"),
        "rdp_bars_in_regime": pd.array(bars_in_regime, dtype="Int64"),
        "rdp_samples": pd.array(sample_count, dtype="Int64"),
        "rdp_long_entry": _flag(long_entry),
        "rdp_short_entry": _flag(short_entry),
        "rdp_exit_long": _flag(exit_long),
        "rdp_exit_short": _flag(exit_short),
        "rdp_low_quality_long": _flag(low_quality_long),
        "rdp_low_quality_short": _flag(low_quality_short),
        "rdp_vola_compression": _flag(vola_compression),
        "rdp_vola_expansion": _flag(vola_expansion),
        "rdp_regime_weakening": _flag(regime_weakening),
        "rdp_regime_weakening_event": _flag(weakening_event),
        "rdp_strong_trend": _flag(strong_trend),
        "rdp_zone_bull": _flag(zone_bull & np.full(n, enable_ma_zones)),
        "rdp_zone_bear": _flag(zone_bear & np.full(n, enable_ma_zones)),
        "rdp_zone_bull_flip": _flag(zone_bull_flip),
        "rdp_zone_bear_flip": _flag(zone_bear_flip),
        "rdp_zone_exit_flip": _flag(zone_exit_flip),
        "rdp_warn_divergence": _flag(warn_divergence),
        "rdp_warn_low_strength": _flag(warn_low_strength),
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
