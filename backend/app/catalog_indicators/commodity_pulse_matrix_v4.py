"""Commodity Pulse Matrix v4 [WavesUnchained] v4.2.1 的 Python 移植。

图表周期入场引擎 + 多周期合流上下文。三层结构：

**第一层：六个类别分数**（每个都在 [-1, +1]）
------------------------------------------------------------------
* **FLOW** —— MFI / VFI / OBV 斜率，各自 Z 标准化后加权（0.5 / 0.3 / 0.2）
* **MOM** —— RSI / ROC / StochRSI / CCI，Z 标准化后加权（0.275 / 0.225 / 0.375 / 0.125）
* **TREND** —— ADX 强度 × DMI 方向 / JMA 栈排列 / VWAP 位置（0.50 / 0.35 / 0.15）
* **VOL** —— ATR% 反向 / 布林-肯特纳挤压 / Williams VIX Fix（0.50 / 0.30 / 0.20）
* **STRUCT** —— 价格 vs JMA 三线位置 / 距离 / 唐奇安突破（0.40 / 0.40 / 0.20）
* **DIV** —— 价格与振荡器的常规背离（每个 ±0.5）

**第二层：Heat Score（0–1）**
------------------------------------------------------------------
五个组件：MTF 一致率(0.30) / 信号幅度(0.25) / setup 状态(0.20) /
波动相位(0.15) / 指标合流(0.10)，最后做 ``pow(x, 1.2)`` 非线性拉伸。

**第三层：入场引擎 V2**
------------------------------------------------------------------
三种入场（突破 / 回踩 / 延续）× 密度阈值 × 反向趋势惩罚 × S/R 放宽，
再过入场状态机（FIRST / REENTRY / UPGRADE + 方向性冷却），
最后过**分级证据门**：时序触发必须成立，再按 ``final_min_gate`` 要求 0–3 个上下文证据点。

关键设计点
----------
* **MTF 一致率只数「占多数方向」的周期**，不是「所有周期同向」—— 后者在真实市场里
  几乎不成立，会把 heat 永久压低。
* **反向趋势用 1.15 倍阈值**而不是 1.3（原脚本下调过），且只在「S/R 之外的上下文」
  才启用硬动量门 —— 在支撑/阻力位附近动量往往已经转向，硬门会挡掉最好的入场。
* **分级证据门的三个证据点都是「上下文」，不能是触发本身** —— 否则证据恒为满分，
  门形同虚设（原脚本的注释明确记录了这个历史 bug）。
* **方向性冷却**：同向信号才受冷却约束，反向可以直接翻（反转不该被等）。

移植说明
--------
MTF 用重采样近似 ``request.security``（三个更高周期各自跑同一套 6 类分数），
与图表周期一起构成 4 个周期的一致率。未移植：Matrix 表格 / 绘图 / 告警 /
Diamond Zone 箱体 / 确认摆动标记 / 仓位与风险模块（这些是展示与执行层）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    dmi,
    ema,
    highest,
    integer,
    lowest,
    mfi,
    number,
    obv,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
    sma,
    stdev,
    stoch,
    valuewhen,
)

__all__ = ["commodity_pulse_matrix_v4"]

_DENSITIES = ("Few (Top Signals)", "Moderate", "Many (Scalper)")

# 密度 → (bpc 阈值, final 阈值)
_DENSITY_THRESHOLDS = {
    "Few (Top Signals)": (0.50, 0.60),
    "Moderate": (0.35, 0.45),
    "Many (Scalper)": (0.20, 0.30),
}

# Heat 权重
_W_MTF, _W_SIGNAL, _W_SETUP, _W_VOL, _W_CONFLUENCE = 0.30, 0.25, 0.20, 0.15, 0.10

_SETUP_STRONG_REVERSAL, _SETUP_DIVERGENCE, _SETUP_BUILDING = 0, 1, 2
_SETUP_TREND_CONT, _SETUP_CONSOLIDATION, _SETUP_NONE = 3, 4, 5
_SETUP_SCORES = {
    _SETUP_STRONG_REVERSAL: 1.0,
    _SETUP_DIVERGENCE: 0.85,
    _SETUP_BUILDING: 0.6,
    _SETUP_TREND_CONT: 0.5,
    _SETUP_CONSOLIDATION: 0.3,
    _SETUP_NONE: 0.1,
}

_PHASE_ACCEL, _PHASE_EXPAND, _PHASE_EQUIL, _PHASE_COMPRESS = 0, 1, 2, 3
_PHASE_SCORES = {_PHASE_ACCEL: 1.0, _PHASE_EXPAND: 0.8, _PHASE_EQUIL: 0.5, _PHASE_COMPRESS: 0.3}

_ENTRY_TYPE_NONE, _ENTRY_TYPE_BREAKOUT, _ENTRY_TYPE_PULLBACK, _ENTRY_TYPE_CONT = 0, 1, 2, 3
_EVENT_NONE, _EVENT_FIRST, _EVENT_REENTRY, _EVENT_UPGRADE = 0, 1, 2, 3


def _jma(values: np.ndarray, length: int, phase: float = 0.0, power: float = 2.0) -> np.ndarray:
    """``calculateJMA``：JMA 闭式近似。"""
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


def _z_local(values: np.ndarray, length: int) -> np.ndarray:
    """``zLocal``：Z 分数；标准差为 0 时取 0（不是 nan）。"""
    series = pd.Series(values)
    mean = sma(series, length).to_numpy(dtype=float)
    sigma = stdev(series, length).to_numpy(dtype=float)
    return np.where(np.isfinite(sigma) & (sigma != 0.0),
                    (values - mean) / np.where(sigma != 0.0, sigma, 1.0), 0.0)


def _nzmap(values, low: float, high: float) -> np.ndarray:
    """``nzmap``：线性映射到 [0,1] 并夹逼。"""
    span = max(0.001, high - low)
    return np.clip((np.asarray(values, dtype=float) - low) / span, 0.0, 1.0)


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if 0 < steps < len(values):
        out[steps:] = values[:-steps]
    return out


def _period_key(index: pd.DatetimeIndex, seconds: int):
    if seconds >= 2592000:
        return pd.PeriodIndex(index, freq="M")
    if seconds >= 604800:
        return pd.PeriodIndex(index, freq="W")
    if seconds >= 86400:
        return pd.PeriodIndex(index, freq="D")
    return index.floor(pd.Timedelta(seconds=int(seconds)))


def _resample(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(frame.index)
    if seconds >= 2592000:
        key = pd.PeriodIndex(idx, freq="M")
    elif seconds >= 604800:
        key = pd.PeriodIndex(idx, freq="W")
    elif seconds >= 86400:
        key = pd.PeriodIndex(idx, freq="D")
    else:
        key = idx.floor(pd.Timedelta(seconds=int(seconds)))
    grouped = frame.groupby(key).agg(open=("open", "first"), high=("high", "max"),
                                     low=("low", "min"), close=("close", "last"),
                                     volume=("volume", "sum"))
    # 用「上一根已完成」的周期值，避免未来函数
    return grouped.shift(1).dropna(how="all")


def _six_scores(bar: pd.DataFrame, volume_is_real: bool, z_len: int) -> dict[str, np.ndarray]:
    """在一个周期上算六类分数的**逐 bar 序列**（长度与 ``bar`` 相同）。

    短帧（不足 60 根）返回全 0 —— 指标还没暖机，任何读数都是噪声。
    """
    n = len(bar)
    if n < 60:
        return {name: np.zeros(n) for name in ("flow", "mom", "trend", "vol", "struct", "div")}
    open_a = bar["open"].to_numpy(dtype=float)
    high = bar["high"].to_numpy(dtype=float)
    low = bar["low"].to_numpy(dtype=float)
    close = bar["close"].to_numpy(dtype=float)
    volume = bar["volume"].to_numpy(dtype=float)
    idx = bar.index
    hlc3_s = (bar["high"] + bar["low"] + bar["close"]) / 3.0

    # --- FLOW ---
    if volume_is_real:
        typical = hlc3_s.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inter = np.where((typical > 0) & (_shift(typical, 1) > 0),
                             np.log(np.where(typical > 0, typical, 1.0))
                             - np.log(np.where(_shift(typical, 1) > 0, _shift(typical, 1), 1.0)), np.nan)
        vinter = stdev(pd.Series(inter, index=idx), 30).to_numpy(dtype=float)
        cutoff = 0.2 * vinter * close
        vol_sma30 = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), 30).to_numpy(dtype=float)
        delta = close - _shift(close, 1)
        confirmed = np.where((np.abs(delta) > cutoff) & (vol_sma30 * cutoff != 0),
                             volume * delta / np.where(vol_sma30 * cutoff != 0, vol_sma30 * cutoff, 1.0), 0.0)
        vfi = _jma(confirmed, 30)
        vfi_z = _z_local(vfi, z_len)
        mfi_z = _z_local(mfi(bar, 14).to_numpy(dtype=float), z_len)
        obv_series = obv(bar).to_numpy(dtype=float)
        obv_slope = np.where(np.isfinite(obv_series),
                             (obv_series - _shift(obv_series, 5)) / 5.0, np.nan)
        obv_z_raw = _z_local(np.nan_to_num(obv_slope, nan=0.0), z_len)
        obv_z = np.where(np.isfinite(obv_slope), obv_z_raw, 0.0)
        raw_flow = np.where(np.isfinite(obv_slope), mfi_z * 0.5 + vfi_z * 0.3 + obv_z * 0.2,
                            mfi_z * 0.625 + vfi_z * 0.375)
        flow = np.clip(raw_flow, -3.0, 3.0) / 3.0
    else:
        flow = np.zeros(n)

    # --- MOM ---
    rsi_a = rsi(bar["close"], 14).to_numpy(dtype=float)
    roc_a = np.where(_shift(close, 9) != 0, (close - _shift(close, 9)) / np.where(_shift(close, 9) != 0, _shift(close, 9), 1.0), 0.0)
    cci_a = (hlc3_s - sma(hlc3_s, 20)) / (0.015 * stdev(hlc3_s, 20).replace(0, np.nan))
    cci_a = cci_a.to_numpy(dtype=float)
    rsi_s = pd.Series(rsi_a, index=idx)
    stoch_rsi = stoch(rsi_s, rsi_s, rsi_s, 14).to_numpy(dtype=float)
    stoch_norm = np.where(np.isfinite(stoch_rsi), (stoch_rsi - 50.0) / 50.0, 0.0)
    cci_norm = np.clip(np.nan_to_num(cci_a) / 100.0, -3.0, 3.0) / 3.0
    mom_raw = (_z_local(rsi_a, z_len) * 0.275 + _z_local(roc_a, z_len) * 0.225
               + stoch_norm * 0.375 + cci_norm * 0.125)
    mom = np.clip(mom_raw, -3.0, 3.0) / 3.0

    # --- TREND ---
    plus_di, minus_di, adx_s = dmi(bar, 14)
    adx_a = adx_s.to_numpy(dtype=float)
    plus_a = plus_di.to_numpy(dtype=float)
    minus_a = minus_di.to_numpy(dtype=float)
    adx_norm = _nzmap(adx_a, 15.0, 35.0)
    adx_dir = np.where(adx_a > 25.0, np.where(plus_a > minus_a, 1.0, -1.0), 0.0)
    jma20 = _jma(close, 20)
    jma50 = _jma(close, 50)
    jma200 = _jma(close, 200)
    stack_dir = np.where((jma20 > jma50) & (jma50 > jma200), 1.0,
                         np.where((jma20 < jma50) & (jma50 < jma200), -1.0, 0.0))
    # VWAP 用会话重置（这里简化为整段累计，无量能时为 na → 权重重归一）
    if volume_is_real:
        pv = np.nancumsum(close * np.nan_to_num(volume, nan=0.0))
        cv = np.nancumsum(np.nan_to_num(volume, nan=0.0))
        vwap = np.where(cv > 0, pv / np.where(cv > 0, cv, 1.0), np.nan)
        vwap_dist = np.where(np.isfinite(vwap) & (vwap != 0), (close - vwap) / np.where(vwap != 0, vwap, 1.0), 0.0)
        vwap_norm = np.clip(vwap_dist, -0.05, 0.05) / 0.05
        trend_raw = adx_norm * adx_dir * 0.50 + stack_dir * 0.35 + vwap_norm * 0.15
    else:
        trend_raw = adx_norm * adx_dir * 0.59 + stack_dir * 0.41
    trend = np.clip(trend_raw, -1.0, 1.0)

    # --- VOL ---
    atr_a = atr(bar, 14).to_numpy(dtype=float)
    atr_pct = np.where(close != 0, atr_a / np.where(close != 0, close, 1.0) * 100.0, 0.0)
    atr_score = 1.0 - _nzmap(atr_pct, 0.5, 3.0)
    basis = sma(bar["close"], 20).to_numpy(dtype=float)
    dev = stdev(bar["close"], 20).to_numpy(dtype=float)
    kelt_basis = _jma(close, 20)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - _shift(close, 1)),
                                                  np.abs(low - _shift(close, 1))))
    kelt_range = _jma(np.nan_to_num(true_range, nan=0.0), 20)
    squeeze = ((basis - 2.0 * dev) > (kelt_basis - 1.5 * kelt_range)) & \
              ((basis + 2.0 * dev) < (kelt_basis + 1.5 * kelt_range))
    highest_close = highest(bar["close"], 22).to_numpy(dtype=float)
    wvf = np.where(highest_close > 0, (highest_close - low) / np.where(highest_close > 0, highest_close, 1.0) * 100.0, 0.0)
    wvf_ma = sma(pd.Series(wvf, index=idx), 22).to_numpy(dtype=float)
    wvf_boost = (wvf > wvf_ma).astype(float)
    vol_score = (atr_score * 0.50 + squeeze.astype(float) * 0.30 + wvf_boost * 0.20)
    vol = np.clip(vol_score * 2.0 - 1.0, -1.0, 1.0)

    # --- STRUCT ---
    above20 = np.where(close > jma20, 1.0, -1.0)
    above50 = np.where(close > jma50, 1.0, -1.0)
    above200 = np.where(np.isfinite(jma200), np.where(close > jma200, 1.0, -1.0), 0.0)
    dist20 = np.where(jma20 != 0, (close - jma20) / np.where(jma20 != 0, jma20, 1.0), 0.0)
    dist50 = np.where(jma50 != 0, (close - jma50) / np.where(jma50 != 0, jma50, 1.0), 0.0)
    dist200 = np.where(np.isfinite(jma200) & (jma200 != 0), (close - jma200) / np.where(jma200 != 0, jma200, 1.0), 0.0)
    has200 = np.isfinite(jma200)
    dist_score = np.where(has200, dist20 * 0.5 + dist50 * 0.3 + dist200 * 0.2,
                          dist20 * 0.625 + dist50 * 0.375)
    dist_norm = np.clip(dist_score, -0.10, 0.10) / 0.10
    donch_high = highest(bar["high"], 20).to_numpy(dtype=float)
    donch_low = lowest(bar["low"], 20).to_numpy(dtype=float)
    donch = np.where(close >= donch_high, 1.0, np.where(close <= donch_low, -1.0, 0.0))
    ema_pos = np.where(has200, above20 * 0.4 + above50 * 0.3 + above200 * 0.3,
                       above20 * 0.571 + above50 * 0.429)
    struct_raw = ema_pos * 0.40 + dist_norm * 0.40 + donch * 0.20
    struct = np.clip(struct_raw, -1.0, 1.0)

    # --- DIV ---
    osc = pd.Series(rsi_a, index=idx)
    price_low = pivotlow(bar["low"], 5, 5)
    price_high = pivothigh(bar["high"], 5, 5)
    osc_low = pivotlow(osc, 5, 5)
    osc_high = pivothigh(osc, 5, 5)
    prev_price_low = valuewhen(price_low.notna(), price_low, 1).to_numpy(dtype=float)
    prev_osc_low = valuewhen(osc_low.notna(), osc_low, 1).to_numpy(dtype=float)
    prev_price_high = valuewhen(price_high.notna(), price_high, 1).to_numpy(dtype=float)
    prev_osc_high = valuewhen(osc_high.notna(), osc_high, 1).to_numpy(dtype=float)
    cur_price_low = price_low.to_numpy(dtype=float)
    cur_osc_low = osc_low.to_numpy(dtype=float)
    cur_price_high = price_high.to_numpy(dtype=float)
    cur_osc_high = osc_high.to_numpy(dtype=float)
    bull_div = (np.isfinite(cur_price_low) & np.isfinite(cur_osc_low) & np.isfinite(prev_price_low)
                & np.isfinite(prev_osc_low) & (cur_price_low < prev_price_low) & (cur_osc_low > prev_osc_low))
    bear_div = (np.isfinite(cur_price_high) & np.isfinite(cur_osc_high) & np.isfinite(prev_price_high)
                & np.isfinite(prev_osc_high) & (cur_price_high > prev_price_high) & (cur_osc_high < prev_osc_high))
    div = np.clip(bull_div.astype(float) * 0.5 - bear_div.astype(float) * 0.5, -1.0, 1.0)

    return {"flow": flow, "mom": mom, "trend": trend, "vol": vol, "struct": struct, "div": div}


def commodity_pulse_matrix_v4(
    frame: pd.DataFrame,
    *,
    entry_density: str = "Moderate",
    use_breakout_entries: bool = True,
    use_pullback_entries: bool = True,
    use_continuation_entries: bool = True,
    allow_counter_trend_entries: bool = False,
    use_trend_filter: bool = False,
    vol_stop_atr_mult: float = 2.0,
    entry_jma_fast: int = 50,
    entry_jma_slow: int = 100,
    use_rsi_boost: bool = True,
    rsi_boost_threshold: float = 25.0,
    rsi_boost_amount: float = 0.15,
    use_wt1_boost: bool = True,
    wt1_boost_threshold: float = 60.0,
    wt1_boost_amount: float = 0.10,
    strength_threshold: float = 0.55,
    entry_heat_min: float = 0.40,
    final_min_gate: int = 4,
    show_early_reversal: bool = True,
    early_require_price_reaction: bool = True,
    early_min_heat: float = 0.25,
    early_cooldown: int = 8,
    use_quality_gate: bool = False,
    quality_atr_contract_thresh: float = 0.8,
    quality_atr_expand_thresh: float = 1.2,
    quality_vol_low_thresh: float = 0.7,
    quality_vol_high_thresh: float = 1.2,
    quality_block_low_vol: bool = True,
    quality_block_range: bool = True,
    quality_allow_mean_reversion: bool = True,
    entry_cooldown: int = 5,
    upgrade_delta: float = 0.35,
    mtf_timeframes: tuple[str, ...] | None = None,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Commodity Pulse Matrix v4：六类分数 + Heat + 入场引擎 V2 + 分级证据门。"""
    entry_density = choice("entry_density", entry_density, _DENSITIES)
    final_min_gate = integer("final_min_gate", final_min_gate, 2, 5)
    entry_heat_min = number("entry_heat_min", entry_heat_min, 0.0, 1.0)
    strength_threshold = number("strength_threshold", strength_threshold, 0.20, 0.70)
    early_min_heat = number("early_min_heat", early_min_heat, 0.0, 1.0)
    early_cooldown = integer("early_cooldown", early_cooldown, 1, 50)
    entry_cooldown = integer("entry_cooldown", entry_cooldown, 1, 100)
    upgrade_delta = number("upgrade_delta", upgrade_delta, 0.0, 1.0)
    entry_jma_fast = integer("entry_jma_fast", entry_jma_fast, 10, 100)
    entry_jma_slow = integer("entry_jma_slow", entry_jma_slow, 50, 200)
    if entry_jma_fast >= entry_jma_slow:
        raise ValueError("entry_jma_fast 必须小于 entry_jma_slow")
    vol_stop_atr_mult = number("vol_stop_atr_mult", vol_stop_atr_mult, 1.0, 4.0)

    open_s, high_s, low_s, close_s, volume_s = ohlcv(frame, "open", "high", "low", "close", "volume")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    if volume_is_real is None:
        volume_is_real = bool(np.isfinite(volume).any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)

    # --- 图表周期的六类分数（逐 bar） --------------------------------------
    chart_scores = _six_scores(frame, volume_is_real, 50)
    chart_total = (chart_scores["flow"] * 0.40 + chart_scores["mom"] * 0.40
                   + chart_scores["trend"] * 0.10 + chart_scores["vol"] * 0.05
                   + chart_scores["struct"] * 0.05)

    # --- MTF：三个更高周期（重采样 + 上一根已完成） ------------------------
    if mtf_timeframes is not None and len(mtf_timeframes) != 3:
        raise ValueError("mtf_timeframes 必须提供 3 个周期")
    defaults = ("240", "D", "W")
    tokens = mtf_timeframes if mtf_timeframes is not None else defaults
    mtf_totals: list[np.ndarray] = []
    for token in tokens:
        seconds = {"D": 86400, "W": 604800, "M": 2592000}.get(str(token).upper())
        if seconds is None:
            seconds = max(1, int(float(token))) * 60
        key = _period_key(pd.DatetimeIndex(idx), seconds)
        resampled = _resample(frame, seconds)
        if len(resampled) < 60:
            mtf_totals.append(np.zeros(n))
            continue
        scores = _six_scores(resampled, volume_is_real, 50)
        period_total = (scores["flow"] * 0.40 + scores["mom"] * 0.40
                        + scores["trend"] * 0.10 + scores["vol"] * 0.05 + scores["struct"] * 0.05)
        series = pd.Series(period_total, index=resampled.index)
        mtf_totals.append(pd.Series(np.asarray(key)).map(series).to_numpy(dtype=float))

    # --- 指标 ---------------------------------------------------------------
    atr_a = atr(frame, 14).to_numpy(dtype=float)
    atr_sma = sma(pd.Series(atr_a, index=idx), 50).to_numpy(dtype=float)
    atr_norm = np.where(atr_sma > 0, atr_a / np.where(atr_sma > 0, atr_sma, 1.0), 1.0)
    is_ranging = atr_norm < 0.7

    jma_fast = _jma(close, entry_jma_fast, 50.0, 2.0)
    jma_slow = _jma(close, entry_jma_slow, 0.0, 2.0)
    trend_up = jma_fast > jma_slow
    trend_down = jma_fast < jma_slow

    # Volatility Stop（多空两条单向棘轮）
    hl2 = (high + low) / 2.0
    upper_band = hl2 + vol_stop_atr_mult * atr_a
    lower_band = hl2 - vol_stop_atr_mult * atr_a
    vol_stop_long = np.full(n, np.nan)
    vol_stop_short = np.full(n, np.nan)
    prev_close = _shift(close, 1)
    for i in range(n):
        if i == 0 or not np.isfinite(vol_stop_long[i - 1]):
            vol_stop_long[i] = lower_band[i]
            vol_stop_short[i] = upper_band[i]
            continue
        vol_stop_long[i] = (max(lower_band[i], vol_stop_long[i - 1])
                            if prev_close[i] > vol_stop_long[i - 1] else lower_band[i])
        vol_stop_short[i] = (min(upper_band[i], vol_stop_short[i - 1])
                             if prev_close[i] < vol_stop_short[i - 1] else upper_band[i])
    midpoint = (vol_stop_long + vol_stop_short) / 2.0
    long_bias = close > midpoint
    short_bias = close < midpoint

    # --- 支撑/阻力（20 根摆动 + 0.5 ATR 容差） ----------------------------
    swing_low20 = lowest(frame["low"], 20).to_numpy(dtype=float)
    swing_high20 = highest(frame["high"], 20).to_numpy(dtype=float)
    near_swing_low = np.abs(low - swing_low20) < atr_a * 0.5
    near_swing_high = np.abs(high - swing_high20) < atr_a * 0.5
    near_support = near_swing_low
    near_resistance = near_swing_high

    # --- 强度评分 -----------------------------------------------------------
    rsi_a = rsi(frame["close"], 14).to_numpy(dtype=float)
    momentum = (rsi_a - 50.0) / 50.0

    trend_align_long = np.where(trend_up & long_bias, 1.0,
                                np.where(trend_up | long_bias, 0.6, 0.3))
    trend_align_short = np.where(trend_down & short_bias, 1.0,
                                 np.where(trend_down | short_bias, 0.6, 0.3))
    zone_long = np.where(near_support, 1.0, np.where(near_resistance, 0.8, 0.5))
    zone_short = np.where(near_resistance, 1.0, np.where(near_support, 0.8, 0.5))

    mom_long = np.where(momentum > 0, momentum, 0.0)
    mom_short = np.where(momentum < 0, np.abs(momentum), 0.0)
    long_strength = mom_long * 0.40 + trend_align_long * 0.45 + zone_long * 0.15
    short_strength = mom_short * 0.40 + trend_align_short * 0.45 + zone_short * 0.15

    # --- RSI / WT1 boost ----------------------------------------------------
    hlc3_s = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    wt_esa = ema(hlc3_s, 12).to_numpy(dtype=float)
    wt_dev = ema((hlc3_s - pd.Series(wt_esa, index=idx)).abs(), 12).to_numpy(dtype=float)
    wt_ci = np.where(wt_dev != 0, (hlc3_s.to_numpy(dtype=float) - wt_esa) / (0.015 * np.where(wt_dev != 0, wt_dev, 1.0)), 0.0)
    wt1 = ema(ema(pd.Series(wt_ci, index=idx), 21), 4).to_numpy(dtype=float)

    rsi_oversold = rsi_a < rsi_boost_threshold
    rsi_overbought = rsi_a > (100.0 - rsi_boost_threshold)
    rsi_boost_long = np.where(use_rsi_boost & rsi_oversold, rsi_boost_amount, 0.0)
    rsi_boost_short = np.where(use_rsi_boost & rsi_overbought, rsi_boost_amount, 0.0)
    wt1_oversold = wt1 < -wt1_boost_threshold
    wt1_overbought = wt1 > wt1_boost_threshold
    wt1_boost_long = np.where(use_wt1_boost & wt1_oversold, wt1_boost_amount, 0.0)
    wt1_boost_short = np.where(use_wt1_boost & wt1_overbought, wt1_boost_amount, 0.0)
    boosted_long = long_strength + rsi_boost_long + wt1_boost_long
    boosted_short = short_strength + rsi_boost_short + wt1_boost_short

    # --- 入场类型 -----------------------------------------------------------
    lookback = 8
    recent_high = highest(pd.Series(_shift(high, 1), index=idx), lookback).to_numpy(dtype=float)
    recent_low = lowest(pd.Series(_shift(low, 1), index=idx), lookback).to_numpy(dtype=float)
    breakout_long = close > recent_high
    breakout_short = close < recent_low

    jma_entry_fast = _jma(close, entry_jma_fast)
    pullback_long = near_support & (low < jma_entry_fast) & (close > jma_entry_fast) & trend_up
    pullback_short = near_resistance & (high > jma_entry_fast) & (close < jma_entry_fast) & trend_down

    recent_high3 = highest(pd.Series(_shift(high, 1), index=idx), 3).to_numpy(dtype=float)
    recent_low3 = lowest(pd.Series(_shift(low, 1), index=idx), 3).to_numpy(dtype=float)
    cont_long = long_bias & (close > recent_high3)
    cont_short = short_bias & (close < recent_low3)

    # --- 密度阈值 + 入场判定 -----------------------------------------------
    bpc_thresh, final_thresh = _DENSITY_THRESHOLDS[entry_density]
    effective_final_thresh = strength_threshold if strength_threshold > 0 else final_thresh

    allow_breakout = use_breakout_entries & ~is_ranging
    allow_cont = use_continuation_entries & ~is_ranging
    raw_long = ((allow_breakout & breakout_long) | (use_pullback_entries & pullback_long)
                | (allow_cont & cont_long))
    raw_short = ((allow_breakout & breakout_short) | (use_pullback_entries & pullback_short)
                 | (allow_cont & cont_short))

    counter_long = raw_long & short_bias
    counter_short = raw_short & long_bias
    ct_mult = 1.15
    long_thresh_bpc = np.where(counter_long, bpc_thresh * ct_mult, bpc_thresh)
    long_thresh_final = np.where(counter_long, effective_final_thresh * ct_mult, effective_final_thresh)
    short_thresh_bpc = np.where(counter_short, bpc_thresh * ct_mult, bpc_thresh)
    short_thresh_final = np.where(counter_short, effective_final_thresh * ct_mult, effective_final_thresh)

    # S/R 附近放宽阈值
    long_thresh_bpc = np.where(near_support, np.maximum(0.0, long_thresh_bpc - 0.10), long_thresh_bpc)
    long_thresh_final = np.where(near_support, np.maximum(0.0, long_thresh_final - 0.15), long_thresh_final)
    short_thresh_bpc = np.where(near_resistance, np.maximum(0.0, short_thresh_bpc - 0.10), short_thresh_bpc)
    short_thresh_final = np.where(near_resistance, np.maximum(0.0, short_thresh_final - 0.15), short_thresh_final)

    sr_context = near_support | near_resistance
    mom_supports_long = sr_context | (momentum >= -0.05)
    mom_supports_short = sr_context | (momentum <= 0.05)

    bpc_long = raw_long & (boosted_long >= long_thresh_bpc) & (allow_counter_trend_entries | ~counter_long) & mom_supports_long
    bpc_short = raw_short & (boosted_short >= short_thresh_bpc) & (allow_counter_trend_entries | ~counter_short) & mom_supports_short
    engine_final_long = raw_long & (boosted_long >= long_thresh_final) & (allow_counter_trend_entries | ~counter_long) & mom_supports_long
    engine_final_short = raw_short & (boosted_short >= short_thresh_final) & (allow_counter_trend_entries | ~counter_short) & mom_supports_short

    entry_type = np.where(breakout_long | breakout_short, _ENTRY_TYPE_BREAKOUT,
                          np.where(pullback_long | pullback_short, _ENTRY_TYPE_PULLBACK,
                                   np.where(cont_long | cont_short, _ENTRY_TYPE_CONT, _ENTRY_TYPE_NONE)))

    # --- 入场状态机（FIRST / REENTRY / UPGRADE） --------------------------
    emit_entry = np.zeros(n, dtype=np.int64)
    event_type = np.zeros(n, dtype=np.int64)
    state_direction = np.zeros(n, dtype=np.int64)
    state_strength = np.full(n, np.nan)
    current_dir = 0
    current_strength = math.nan
    last_entry_bar = -1
    for i in range(n):
        new_dir = 1 if engine_final_long[i] else (-1 if engine_final_short[i] else 0)
        new_strength = (boosted_long[i] if new_dir == 1 else
                        boosted_short[i] if new_dir == -1 else 0.0)
        is_change = new_dir != current_dir and new_dir != 0
        is_same = new_dir == current_dir and new_dir != 0
        cooldown_passed = last_entry_bar < 0 or (i - last_entry_bar) >= entry_cooldown
        prev_long = engine_final_long[i - 1] if i > 0 else False
        prev_short = engine_final_short[i - 1] if i > 0 else False
        fresh_trigger = (engine_final_long[i] and not prev_long) or (engine_final_short[i] and not prev_short)
        is_reentry = is_same and cooldown_passed and fresh_trigger
        is_upgrade = (is_same and not cooldown_passed and np.isfinite(current_strength)
                      and (new_strength - current_strength) >= upgrade_delta)
        if is_change:
            code = _EVENT_FIRST
        elif is_reentry:
            code = _EVENT_REENTRY
        elif is_upgrade:
            code = _EVENT_UPGRADE
        else:
            code = _EVENT_NONE
        if code != _EVENT_NONE:
            emit_entry[i] = 1
            current_dir = new_dir
            current_strength = new_strength
            last_entry_bar = i
        event_type[i] = code
        state_direction[i] = current_dir
        state_strength[i] = current_strength

    # --- Heat Score（逐 bar） -----------------------------------------------
    # MTF 一致率：四个周期（3 MTF + 图表）里有多少个与「方向之和」同向。
    # 注意这是「占多数的方向」，不是「全部同向」—— 后者在真实市场几乎不成立。
    # MTF 暖机期是 NaN —— 先归零（与 Pine 的 nz(x, 0.0) 一致）再做方向判定
    directions = [np.sign(np.nan_to_num(t)).astype(np.int64) for t in mtf_totals]
    dir_sum = sum(directions) + np.sign(np.nan_to_num(chart_total)).astype(np.int64)
    dominant = np.sign(dir_sum).astype(np.int64)
    aligned = sum((d == dominant) & (dominant != 0) for d in directions)
    chart_dir = np.sign(np.nan_to_num(chart_total)).astype(np.int64)
    aligned = aligned + ((chart_dir == dominant) & (dominant != 0)).astype(np.int64)
    mtf_alignment = aligned.astype(float) / 4.0

    magnitudes = [np.abs(np.nan_to_num(t)) for t in mtf_totals] + [np.abs(np.nan_to_num(chart_total))]
    signal_strength = np.minimum(1.0, np.mean(magnitudes, axis=0))

    # setup 状态（逐 bar）
    div_series = chart_scores["div"]
    setup_status = np.where((div_series > 0.4) | (div_series < -0.4), _SETUP_DIVERGENCE,
                            np.where(is_ranging, _SETUP_CONSOLIDATION,
                                     np.where(trend_up | trend_down, _SETUP_TREND_CONT, _SETUP_BUILDING))).astype(np.int64)
    setup_score = np.select(
        [setup_status == _SETUP_STRONG_REVERSAL, setup_status == _SETUP_DIVERGENCE,
         setup_status == _SETUP_BUILDING, setup_status == _SETUP_TREND_CONT,
         setup_status == _SETUP_CONSOLIDATION],
        [1.0, 0.85, 0.6, 0.5, 0.3], default=0.1)

    # 波动相位（StochRSI + MACD 柱）
    rsi_s = pd.Series(rsi_a, index=idx)
    stoch_rsi = stoch(rsi_s, rsi_s, rsi_s, 14).to_numpy(dtype=float)
    stoch_norm = np.where(np.isfinite(stoch_rsi), (stoch_rsi - 50.0) / 50.0, 0.0)
    macd_line = ema(frame["close"], 12) - ema(frame["close"], 26)
    macd_hist = (macd_line - ema(macd_line, 9)).to_numpy(dtype=float)
    macd_hist = np.nan_to_num(macd_hist)
    momentum_strength = np.abs(stoch_norm) * np.where(stoch_norm * macd_hist > 0, 1.5, 0.5)
    phase = np.where(momentum_strength >= 0.75, _PHASE_ACCEL,
                     np.where(momentum_strength >= 0.50, _PHASE_EXPAND,
                              np.where(momentum_strength >= 0.25, _PHASE_EQUIL, _PHASE_COMPRESS))).astype(np.int64)
    phase_score = np.select([phase == _PHASE_ACCEL, phase == _PHASE_EXPAND,
                             phase == _PHASE_EQUIL, phase == _PHASE_COMPRESS],
                            [1.0, 0.8, 0.5, 0.3], default=0.5)
    atr_context = np.where(atr_norm > 1.2, 1.0, np.where(atr_norm < 0.8, 0.3, 0.6))
    volatility_component = phase_score * 0.7 + atr_context * 0.3

    # 指标合流（逐 bar）
    flow_s, mom_s, trend_s = chart_scores["flow"], chart_scores["mom"], chart_scores["trend"]
    vol_s, struct_s, div_s = chart_scores["vol"], chart_scores["struct"], chart_scores["div"]
    agreements = ((flow_s > 0) & (trend_s > 0) | (flow_s < 0) & (trend_s < 0)).astype(np.int64) \
        + ((mom_s > 0) & (div_s > 0) | (mom_s < 0) & (div_s < 0)).astype(np.int64) \
        + ((struct_s > 0) & (vol_s > 0) | (struct_s < 0) & (vol_s < 0)).astype(np.int64)
    all_pos = (flow_s > 0) & (mom_s > 0) & (trend_s > 0) & (vol_s > 0) & (struct_s > 0) & (div_s > 0)
    all_neg = (flow_s < 0) & (mom_s < 0) & (trend_s < 0) & (vol_s < 0) & (struct_s < 0) & (div_s < 0)
    confluence = np.where(all_pos | all_neg, 1.0, agreements / 3.0)

    heat_raw = (mtf_alignment * _W_MTF + signal_strength * _W_SIGNAL + setup_score * _W_SETUP
                + volatility_component * _W_VOL + confluence * _W_CONFLUENCE)
    heat_score = np.clip(np.maximum(0.0, heat_raw) ** 1.2, 0.0, 1.0)
    heat_level = np.where(heat_score >= 0.65, 3, np.where(heat_score >= 0.45, 2,
                                                          np.where(heat_score >= 0.30, 1, 0))).astype(np.int64)

    # --- 质量门（ATR + 量能） ----------------------------------------------
    volume_sma = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), 50).to_numpy(dtype=float)
    atr_ratio = np.where(atr_sma > 0, atr_a / np.where(atr_sma > 0, atr_sma, 1.0), 0.0)
    vol_ratio = np.where(volume_is_real & (volume_sma > 0),
                         volume / np.where(volume_sma > 0, volume_sma, 1.0), 0.0)
    atr_contracting = (atr_ratio > 0) & (atr_ratio < quality_atr_contract_thresh)
    atr_expanding = atr_ratio >= quality_atr_expand_thresh
    vol_low = np.asarray(volume_is_real) & (vol_ratio > 0) & (vol_ratio < quality_vol_low_thresh)
    vol_high = np.asarray(volume_is_real) & (vol_ratio >= quality_vol_high_thresh)
    allow_entries = ~vol_low
    allow_trend_signals = ~atr_contracting

    # --- 最终门：触发 + 三级证据 ------------------------------------------
    trigger_long = (emit_entry == 1) & engine_final_long
    trigger_short = (emit_entry == 1) & engine_final_short
    near_sr = near_support | near_resistance
    quality_context_ok = allow_entries & (allow_trend_signals | (quality_allow_mean_reversion & near_sr))
    trending_now = trend_up | trend_down
    trend_aligned_long = trending_now & (close > jma_slow)
    trend_aligned_short = trending_now & (close < jma_slow)
    heat_ok = heat_score >= entry_heat_min
    quality_veto = np.full(n, use_quality_gate) & (
        (quality_block_low_vol & ~allow_entries)
        | (quality_block_range & ~allow_trend_signals
           & ~(quality_allow_mean_reversion & near_sr)))
    trend_ok_long = (not use_trend_filter) or trend_aligned_long
    trend_ok_short = (not use_trend_filter) or trend_aligned_short
    evidence_threshold = max(0, min(3, final_min_gate - 2))
    long_evidence = quality_context_ok.astype(np.int64) + trend_aligned_long.astype(np.int64) + heat_ok.astype(np.int64)
    short_evidence = quality_context_ok.astype(np.int64) + trend_aligned_short.astype(np.int64) + heat_ok.astype(np.int64)

    final_long = trigger_long & ~quality_veto & trend_ok_long & (long_evidence >= evidence_threshold)
    final_short = trigger_short & ~quality_veto & trend_ok_short & (short_evidence >= evidence_threshold)

    # --- 早反转候选 ---------------------------------------------------------
    early_long_evidence = (div_s > 0.2) | rsi_oversold | wt1_oversold
    early_short_evidence = (div_s < -0.2) | rsi_overbought | wt1_overbought
    jma20 = _jma(close, 20)
    prev_close_arr = _shift(close, 1)
    prev_jma20 = _shift(jma20, 1)
    cross_above = (close > jma20) & (prev_close_arr <= prev_jma20)
    cross_below = (close < jma20) & (prev_close_arr >= prev_jma20)
    if early_require_price_reaction:
        early_long_reaction = cross_above | (close > jma20) & (close > prev_close_arr)
        early_short_reaction = cross_below | (close < jma20) & (close < prev_close_arr)
    else:
        early_long_reaction = np.ones(n, dtype=bool)
        early_short_reaction = np.ones(n, dtype=bool)
    early_long_candidate = show_early_reversal & ~final_long & early_long_evidence & early_long_reaction & (heat_score >= early_min_heat)
    early_short_candidate = show_early_reversal & ~final_short & early_short_evidence & early_short_reaction & (heat_score >= early_min_heat)

    # --- 止损 / 止盈 --------------------------------------------------------
    stop_long = close - atr_a * 1.5
    stop_short = close + atr_a * 1.5
    tp1_long = close + atr_a * 1.0
    tp2_long = close + atr_a * 1.5
    tp3_long = close + atr_a * 2.0
    tp1_short = close - atr_a * 1.0
    tp2_short = close - atr_a * 1.5
    tp3_short = close - atr_a * 2.0

    def as_int(values) -> pd.Series:
        return pd.Series(np.asarray(values).astype(np.int64), index=idx).astype("Int64")

    data = {
        "cpm_flow_score": flow_s,
        "cpm_mom_score": mom_s,
        "cpm_trend_score": trend_s,
        "cpm_vol_score": vol_s,
        "cpm_struct_score": struct_s,
        "cpm_div_score": div_s,
        "cpm_chart_total": chart_total,
        "cpm_mtf_total_1": mtf_totals[0],
        "cpm_mtf_total_2": mtf_totals[1],
        "cpm_mtf_total_3": mtf_totals[2],
        "cpm_heat_score": heat_score,
        "cpm_heat_mtf_alignment": mtf_alignment,
        "cpm_heat_signal_strength": signal_strength,
        "cpm_heat_setup": setup_score,
        "cpm_heat_volatility": volatility_component,
        "cpm_heat_confluence": confluence,
        "cpm_setup_status": pd.Series(setup_status, index=idx).astype("Int64"),
        "cpm_heat_level": pd.Series(heat_level, index=idx).astype("Int64"),
        "cpm_market_phase": pd.Series(phase, index=idx).astype("Int64"),
        "cpm_is_ranging": as_int(is_ranging),
        "cpm_trend_up": as_int(trend_up),
        "cpm_trend_down": as_int(trend_down),
        "cpm_long_bias": as_int(long_bias),
        "cpm_short_bias": as_int(short_bias),
        "cpm_near_support": as_int(near_support),
        "cpm_near_resistance": as_int(near_resistance),
        "cpm_jma_fast": jma_fast,
        "cpm_jma_slow": jma_slow,
        "cpm_vol_stop_long": vol_stop_long,
        "cpm_vol_stop_short": vol_stop_short,
        "cpm_atr": atr_a,
        "cpm_atr_norm": atr_norm,
        "cpm_momentum": momentum,
        "cpm_long_strength": long_strength,
        "cpm_short_strength": short_strength,
        "cpm_boosted_long": boosted_long,
        "cpm_boosted_short": boosted_short,
        "cpm_breakout_long": as_int(breakout_long),
        "cpm_breakout_short": as_int(breakout_short),
        "cpm_pullback_long": as_int(pullback_long),
        "cpm_pullback_short": as_int(pullback_short),
        "cpm_cont_long": as_int(cont_long),
        "cpm_cont_short": as_int(cont_short),
        "cpm_entry_type": as_int(entry_type),
        "cpm_bpc_long": as_int(bpc_long),
        "cpm_bpc_short": as_int(bpc_short),
        "cpm_engine_final_long": as_int(engine_final_long),
        "cpm_engine_final_short": as_int(engine_final_short),
        "cpm_emit_entry": as_int(emit_entry),
        "cpm_event_type": as_int(event_type),
        "cpm_state_direction": as_int(state_direction),
        "cpm_state_strength": state_strength,
        "cpm_trigger_long": as_int(trigger_long),
        "cpm_trigger_short": as_int(trigger_short),
        "cpm_evidence_long": pd.Series(long_evidence, index=idx).astype("Int64"),
        "cpm_evidence_short": pd.Series(short_evidence, index=idx).astype("Int64"),
        "cpm_evidence_threshold": pd.Series(np.full(n, evidence_threshold), index=idx).astype("Int64"),
        "cpm_final_long": as_int(final_long),
        "cpm_final_short": as_int(final_short),
        "cpm_early_long": as_int(early_long_candidate),
        "cpm_early_short": as_int(early_short_candidate),
        "cpm_allow_entries": as_int(allow_entries),
        "cpm_allow_trend_signals": as_int(allow_trend_signals),
        "cpm_quality_veto": as_int(quality_veto),
        "cpm_rsi": rsi_a,
        "cpm_wt1": wt1,
        "cpm_stop_long": stop_long,
        "cpm_stop_short": stop_short,
        "cpm_tp1_long": tp1_long,
        "cpm_tp2_long": tp2_long,
        "cpm_tp3_long": tp3_long,
        "cpm_tp1_short": tp1_short,
        "cpm_tp2_short": tp2_short,
        "cpm_tp3_short": tp3_short,
    }
    return pd.DataFrame(data, index=idx)
