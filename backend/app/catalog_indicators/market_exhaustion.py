"""Market Exhaustion [WavesUnchained] — MFI/CCI 衰竭振荡器 + StochRSI 择时 + 背离。

**振荡器**（`osc_mode`）：MFI / CCI / MFI+CCI（后者用 `engine` 决定参与背离计算的
引擎）。MFI 需要真实成交量，缺列或全零成交量时整段为 NaN（Pine 的
`volumeIsReal`）。CCI 通过 `0.5+0.5*tanh(2*cci/scale)` 映射到 0..100。

**平滑**：JMA 近似（`beta=0.45*(len-1)/(0.45*(len-1)+2)`，`alpha=beta^power`，
相位 `pr = phase/100+1.5`，三段 e0/e1/e2 递归 + 积分 j），可分别作用于 MFI 与
StochRSI 的 K。

**动态区**：`upper = lo + (hi-lo)*upper_frac`、`lower = lo + (hi-lo)*lower_frac`，
hi/lo 为 MFI 在 `zone_lookback` 内的最高/最低；关闭时用静态 80/20。

**StochRSI**：`100*(rsi - lowest(rsi)) / max(highest(rsi)-lowest(rsi), 1e-10)`，
再 `sma(k_smooth)`，可再叠 JMA。

**HTF 上下文**：自动选择高周期（基础 ≤5m→15m、≤15m→60m、≤60m→240m、否则日线），
取**上一已完成周期**的 MFI（无未来函数），可平滑；`bias >= bias_level` 为多。

**信号**：`long = use_stoch and allow_long and (区确认 or mfi 上穿 lower) and k 上穿 stoch_os`；
`short` 镜像。区确认要求当前及前 `need_confirm` 根都在区外（`need_confirm=0` 时只看当前）。

**背离**（价格枢轴锚定，振荡器在枢轴 bar 采样）：
- 看涨：价格 LL（`p2 <= p1*(1+eps_price)`）且振荡器 HL（`m2 >= m1-eps_osc`），
  `Δ = m2-m1 >= min_delta`，摆动 ≥ `min_atr×ATR`，且按 `osob_mode` 检查 OS 区
  （缓冲 `osob_buffer_pct×区间`）。
- 看跌镜像。
- 连续同类背离 `seq` 计数（+1 / 反向清零），分项归一：MFI 增量、摆动幅度、时间
  间隔（dt/40）、序号（seq/3），可选 HTF 同向加 0.10。
- 反应分延后结算：枢轴后 `react_bars` 根窗口内（相对枢轴收盘）的价格反应
  `react_frac×ATR` 归一后追加，权重 `W_REACT`。

丢弃：全部 plot/fill/bgcolor/label/line 绘图、K 线着色、告警配置、诊断日志。

Source: pine-collection/indicators/momentum/market_exhaustion/market_exhaustion.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    cci,
    choice,
    crossover,
    crossunder,
    flags,
    highest,
    integer,
    lowest,
    mfi,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
    sma,
)
from ._tf import base_step_minutes, lift_series

__all__ = ["market_exhaustion"]

# 各分项权重（原 fatigue 权重按比例摊到其余项，分母 0.82）。
_W_MFI_DELTA = 0.18 / 0.82
_W_ATR = 0.16 / 0.82
_W_TIME = 0.18 / 0.82
_W_REACT = 0.18 / 0.82
_W_SEQ = 0.12 / 0.82

_PRESETS = {
    # tf: (piv_l, piv_r, mfi_delta, buffer, atr, osc_eps, price_eps)
    "5m": (3, 3, 5, 0.10, 0.25, 0.8, 0.0016),
    "15m": (5, 5, 7, 0.25, 0.45, 1.0, 0.0015),
    "1H": (6, 6, 9, 0.12, 0.60, 1.2, 0.0012),
    "4H": (5, 5, 10, 0.30, 0.70, 1.5, 0.0010),
    "1D": (9, 9, 16, 0.16, 1.00, 2.0, 0.0008),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _jma(series: pd.Series, length: int, phase: int, power: float) -> pd.Series:
    """Approximate JMA (Jurik) recursion used by the source script."""
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = math.pow(beta, power)
    e0 = e1 = e2 = j = 0.0
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for position, value in enumerate(values):
        if not np.isfinite(value):
            out[position] = np.nan
            continue
        e0 = (1 - alpha) * value + alpha * e0
        e1 = (value - e0) * (1 - beta) + beta * e1
        e2 = (e0 + pr * e1 - j) * math.pow(1 - alpha, 2) + math.pow(alpha, 2) * e2
        j = e2 + j
        out[position] = j
    return pd.Series(out, index=series.index)


def _cci_mapped(frame: pd.DataFrame, length: int, scale: float, source: str) -> pd.Series:
    """CCI squashed into 0..100 by `0.5 + 0.5*tanh(2*cci/scale)`."""
    raw = cci(frame, length, source)
    # `tanh` directly instead of (exp-1)/(exp+1): identical value, no overflow to inf.
    return 100.0 * (0.5 + 0.5 * np.tanh(2.0 * (raw / scale)))


def _auto_htf_minutes(frame: pd.DataFrame) -> int:
    """Pine's getAutoHTF(): 15/60/240 minutes, or daily (0) for 1D+."""
    minutes = base_step_minutes(frame)
    if minutes <= 5:
        return 15
    if minutes <= 15:
        return 60
    if minutes <= 60:
        return 240
    return 0


def _in_zone_confirmed(condition: np.ndarray, bars: int) -> np.ndarray:
    """`inZoneConfirmed`: current bar and the `bars` prior bars all satisfy `condition`."""
    if bars <= 0:
        return condition.copy()
    out = np.ones(len(condition), dtype=bool)
    for offset in range(bars + 1):
        shifted = np.zeros(len(condition), dtype=bool)
        shifted[offset:] = condition[: len(condition) - offset]
        out &= shifted
    return out


def market_exhaustion(
    frame: pd.DataFrame,
    *,
    mfi_len: int = 14,
    osc_mode: str = "MFI",
    engine: str = "MFI",
    cci_len: int = 34,
    cci_map_scale: float = 150.0,
    use_dynamic_zones: bool = True,
    zone_lookback: int = 200,
    dynamic_upper: float = 0.85,
    dynamic_lower: float = 0.15,
    static_upper: int = 80,
    static_lower: int = 20,
    use_stoch: bool = True,
    rsi_len: int = 14,
    stoch_len: int = 14,
    k_smooth: int = 3,
    stoch_os: int = 20,
    stoch_ob: int = 80,
    show_signals: bool = True,
    need_confirm: int = 1,
    use_jma: bool = True,
    jma_len: int = 20,
    jma_phase: int = 0,
    jma_power: float = 2.0,
    smooth_mfi: bool = True,
    smooth_stoch: bool = True,
    use_htf: bool = True,
    auto_htf: bool = True,
    htf_minutes: int = 60,
    use_htf_filter: bool = False,
    htf_smooth: bool = True,
    htf_ema_len: int = 10,
    bias_level: int = 50,
    show_divergence: bool = True,
    require_osob: bool = True,
    osob_mode: str = "StartOrEnd",
    osob_buffer_pct: float = 0.10,
    use_atr_filter: bool = True,
    atr_len: int = 14,
    score_strong: int = 65,
    react_bars: int = 10,
    react_atr_frac: float = 0.35,
    use_bias_bonus: bool = False,
    div_profile: str = "More Signals",
    use_preset: bool = True,
    pivot_left: int = 7,
    pivot_right: int = 7,
    min_osc_delta: float = 15.0,
    min_atr_move: float = 0.75,
    price_eps: float = 0.0010,
    osc_eps: float = 1.0,
) -> pd.DataFrame:
    """返回 MFI/CCI 振荡器、动态区、StochRSI、HTF 偏向、信号与背离评分。"""
    mfi_len = integer("mfi_len", mfi_len, 1)
    osc_mode = choice("osc_mode", osc_mode, ("MFI", "CCI", "MFI+CCI"))
    engine = choice("engine", engine, ("MFI", "CCI"))
    cci_len = integer("cci_len", cci_len, 1)
    cci_map_scale = number("cci_map_scale", cci_map_scale, 50.0, 500.0)
    use_dynamic_zones = boolean("use_dynamic_zones", use_dynamic_zones)
    zone_lookback = integer("zone_lookback", zone_lookback, 50)
    dynamic_upper = number("dynamic_upper", dynamic_upper, 0.5, 1.0)
    dynamic_lower = number("dynamic_lower", dynamic_lower, 0.0, 0.5)
    static_upper = integer("static_upper", static_upper, 50, 100)
    static_lower = integer("static_lower", static_lower, 0, 50)
    use_stoch = boolean("use_stoch", use_stoch)
    rsi_len = integer("rsi_len", rsi_len, 1)
    stoch_len = integer("stoch_len", stoch_len, 1)
    k_smooth = integer("k_smooth", k_smooth, 1)
    stoch_os = integer("stoch_os", stoch_os, 1, 49)
    stoch_ob = integer("stoch_ob", stoch_ob, 51, 99)
    show_signals = boolean("show_signals", show_signals)
    need_confirm = integer("need_confirm", need_confirm, 0, 5)
    use_jma = boolean("use_jma", use_jma)
    jma_len = integer("jma_len", jma_len, 1)
    jma_phase = integer("jma_phase", jma_phase, -100, 100)
    jma_power = number("jma_power", jma_power, 1.0, 8.0)
    smooth_mfi = boolean("smooth_mfi", smooth_mfi)
    smooth_stoch = boolean("smooth_stoch", smooth_stoch)
    use_htf = boolean("use_htf", use_htf)
    auto_htf = boolean("auto_htf", auto_htf)
    htf_minutes = integer("htf_minutes", htf_minutes, 1)
    use_htf_filter = boolean("use_htf_filter", use_htf_filter)
    htf_smooth = boolean("htf_smooth", htf_smooth)
    htf_ema_len = integer("htf_ema_len", htf_ema_len, 1)
    bias_level = integer("bias_level", bias_level, 1, 99)
    show_divergence = boolean("show_divergence", show_divergence)
    require_osob = boolean("require_osob", require_osob)
    osob_mode = choice("osob_mode", osob_mode, ("End", "StartOrEnd", "Both"))
    osob_buffer_pct = number("osob_buffer_pct", osob_buffer_pct, 0.0, 0.5)
    use_atr_filter = boolean("use_atr_filter", use_atr_filter)
    atr_len = integer("atr_len", atr_len, 1)
    score_strong = integer("score_strong", score_strong, 0, 100)
    react_bars = integer("react_bars", react_bars, 2, 100)
    react_atr_frac = number("react_atr_frac", react_atr_frac, 0.05, 2.0)
    use_bias_bonus = boolean("use_bias_bonus", use_bias_bonus)
    div_profile = choice("div_profile", div_profile, ("More Signals", "Conservative"))
    use_preset = boolean("use_preset", use_preset)
    pivot_left = integer("pivot_left", pivot_left, 1, 20)
    pivot_right = integer("pivot_right", pivot_right, 1, 20)
    min_osc_delta = number("min_osc_delta", min_osc_delta, 0.0, 100.0)
    min_atr_move = number("min_atr_move", min_atr_move, 0.0, 10.0)
    price_eps = number("price_eps", price_eps, 0.0, 0.02)
    osc_eps = number("osc_eps", osc_eps, 0.0, 10.0)

    high, low, close, open_ = ohlcv(frame, "high", "low", "close", "open")
    typical = (high + low + close) / 3.0

    has_volume = "volume" in frame.columns
    if has_volume:
        volume = pd.to_numeric(frame["volume"], errors="coerce").astype(float)
        volume_real = bool(volume.fillna(0.0).cumsum().iloc[-1] > 0) if len(volume) else False
    else:
        volume_real = False

    raw_mfi = mfi(frame, mfi_len) if volume_real else pd.Series(np.nan, index=frame.index)
    mfi_value = _jma(raw_mfi, jma_len, jma_phase, jma_power) if (use_jma and smooth_mfi) else raw_mfi

    cci100 = _cci_mapped(frame, cci_len, cci_map_scale, "hlc3")
    if osc_mode == "MFI":
        oscillator = mfi_value
    elif osc_mode == "CCI":
        oscillator = cci100
    else:
        oscillator = cci100 if engine == "CCI" else mfi_value

    hi_zone = highest(mfi_value, zone_lookback)
    lo_zone = lowest(mfi_value, zone_lookback)
    if use_dynamic_zones:
        upper = lo_zone + (hi_zone - lo_zone) * _clamp01(dynamic_upper)
        lower = lo_zone + (hi_zone - lo_zone) * _clamp01(dynamic_lower)
    else:
        upper = pd.Series(float(static_upper), index=frame.index)
        lower = pd.Series(float(static_lower), index=frame.index)
    is_ob = flags(mfi_value > upper)
    is_os = flags(mfi_value < lower)

    rsi_values = rsi(close, rsi_len)
    rsi_low = lowest(rsi_values, stoch_len)
    rsi_high = highest(rsi_values, stoch_len)
    stoch_rsi = 100 * (rsi_values - rsi_low) / np.maximum(rsi_high - rsi_low, 1e-10)
    k_raw = sma(stoch_rsi, k_smooth)
    k_value = _jma(k_raw, jma_len, jma_phase, jma_power) if (use_jma and smooth_stoch) else k_raw

    htf_values = pd.Series(np.nan, index=frame.index)
    if use_htf or use_htf_filter:
        minutes = _auto_htf_minutes(frame) if auto_htf else htf_minutes
        if minutes and volume_real:
            base = base_step_minutes(frame)
            if minutes % base == 0 and minutes >= base:
                htf_raw = lift_series(frame, raw_mfi, minutes)
                if htf_smooth:
                    htf_raw = (_jma(htf_raw, jma_len, jma_phase, jma_power) if use_jma
                               else htf_raw.ewm(span=htf_ema_len, adjust=False).mean())
                htf_values = htf_raw
    if use_htf:
        htf_bull = flags(htf_values.notna() & (htf_values >= bias_level))
        htf_bear = flags(htf_values.notna() & (htf_values < bias_level))
    else:
        htf_bull = pd.Series(False, index=frame.index)
        htf_bear = pd.Series(False, index=frame.index)
    if use_htf_filter:
        allow_long = htf_bull
        allow_short = htf_bear
    else:
        allow_long = pd.Series(True, index=frame.index)
        allow_short = pd.Series(True, index=frame.index)

    os_confirm = _in_zone_confirmed(flags(mfi_value < lower).to_numpy(), need_confirm)
    ob_confirm = _in_zone_confirmed(flags(mfi_value > upper).to_numpy(), need_confirm)
    long_ok = (os_confirm | crossover(mfi_value, lower).to_numpy()) \
        & crossover(k_value, stoch_os).to_numpy()
    short_ok = (ob_confirm | crossunder(mfi_value, upper).to_numpy()) \
        & crossunder(k_value, stoch_ob).to_numpy()
    gate = bool(show_signals and use_stoch)
    long_signal = pd.Series(gate & allow_long.to_numpy() & long_ok, index=frame.index, dtype=bool)
    short_signal = pd.Series(gate & allow_short.to_numpy() & short_ok, index=frame.index, dtype=bool)

    # --- divergence parameters ------------------------------------------------
    tf_minutes = base_step_minutes(frame)
    preset = ("5m" if tf_minutes <= 5 else "15m" if tf_minutes <= 15
              else "1H" if tf_minutes <= 60 else "4H" if tf_minutes <= 240 else "1D")
    p_left, p_right, delta_base, buffer_base, atr_base, osc_base, price_base = _PRESETS[preset]
    conservative = div_profile == "Conservative"
    if use_preset:
        effective_left = p_left
        effective_right = p_right
        effective_delta = delta_base + 4 if conservative else max(1, delta_base - 2)
        effective_buffer = max(0.05, buffer_base - 0.07) if conservative else min(0.35, buffer_base + 0.05)
        effective_atr = atr_base + 0.30 if conservative else max(0.15, atr_base - 0.15)
        effective_osc_eps = max(0.2, osc_base - 0.3) if conservative else osc_base + 0.5
        effective_price_eps = max(0.0001, price_base - 0.0004) if conservative else price_base + 0.0005
    else:
        effective_left = pivot_left
        effective_right = pivot_right
        effective_delta = min_osc_delta
        effective_buffer = osob_buffer_pct
        effective_atr = min_atr_move
        effective_osc_eps = osc_eps
        effective_price_eps = price_eps

    osc_hi = highest(oscillator, zone_lookback)
    osc_lo = lowest(oscillator, zone_lookback)
    if use_dynamic_zones:
        div_upper = osc_lo + (osc_hi - osc_lo) * _clamp01(dynamic_upper)
        div_lower = osc_lo + (osc_hi - osc_lo) * _clamp01(dynamic_lower)
    else:
        div_upper = pd.Series(float(static_upper), index=frame.index)
        div_lower = pd.Series(float(static_lower), index=frame.index)
    zones_ready = div_upper.notna() & div_lower.notna()
    band = (div_upper - div_lower).clip(lower=1.0)
    atr_div = atr(frame, atr_len)

    def _in_os(values: pd.Series) -> np.ndarray:
        dynamic = (values <= div_lower + band * effective_buffer).to_numpy()
        fallback = (values <= 20.0 + 5.0).to_numpy()
        return np.where(zones_ready.to_numpy(), dynamic, fallback)

    def _in_ob(values: pd.Series) -> np.ndarray:
        dynamic = (values >= div_upper - band * effective_buffer).to_numpy()
        fallback = (values >= 80.0 - 5.0).to_numpy()
        return np.where(zones_ready.to_numpy(), dynamic, fallback)

    price_low_pivot = pivotlow(low, effective_left, effective_right)
    price_high_pivot = pivothigh(high, effective_left, effective_right)
    osc_series = oscillator
    osc_at_low = price_low_pivot.copy()
    osc_at_high = price_high_pivot.copy()
    for position in np.flatnonzero(price_low_pivot.notna().to_numpy()):
        source_bar = position - effective_right
        if 0 <= source_bar < len(frame):
            osc_at_low.iloc[position] = osc_series.iloc[source_bar]
    for position in np.flatnonzero(price_high_pivot.notna().to_numpy()):
        source_bar = position - effective_right
        if 0 <= source_bar < len(frame):
            osc_at_high.iloc[position] = osc_series.iloc[source_bar]

    react_lag = max(effective_right - react_bars, 0)
    react_high_window = highest(high, react_bars)
    react_low_window = lowest(low, react_bars)
    react_pivot_close = close.shift(react_bars + react_lag)
    react_up = (react_high_window.shift(react_lag) - react_pivot_close).to_numpy()
    react_down = (react_pivot_close - react_low_window.shift(react_lag)).to_numpy()

    size = len(frame)
    out = {name: np.zeros(size, dtype=bool) for name in
           ("bull_div", "bear_div", "div_finalized")}
    out["div_score"] = np.full(size, np.nan)
    out["div_final_score"] = np.full(size, np.nan)
    out["div_delta"] = np.full(size, np.nan)
    out["div_bars"] = np.full(size, np.nan)
    out["div_seq"] = np.zeros(size, dtype=int)
    out["div_react"] = np.full(size, np.nan)

    pending: list[dict] = []
    last_low_price = last_low_osc = math.nan
    last_low_bar = -1
    last_high_price = last_high_osc = math.nan
    last_high_bar = -1
    bull_seq = bear_seq = 0
    os_flags = _in_os(oscillator)
    ob_flags = _in_ob(oscillator)
    prices_low = price_low_pivot.to_numpy()
    prices_high = price_high_pivot.to_numpy()
    osc_low = osc_at_low.to_numpy()
    osc_high = osc_at_high.to_numpy()
    atr_values = atr_div.to_numpy()

    for bar in range(size):
        # --- bullish divergence ---------------------------------------------
        if show_divergence and not np.isnan(prices_low[bar]):
            pivot_bar = bar - effective_right
            p2, m2 = prices_low[bar], osc_low[bar]
            is_bull = False
            if not (np.isnan(last_low_price) or np.isnan(last_low_osc)):
                p1, m1, x1 = last_low_price, last_low_osc, last_low_bar
                if osob_mode == "End":
                    area_ok = os_flags[pivot_bar]
                elif osob_mode == "StartOrEnd":
                    area_ok = os_flags[x1] or os_flags[pivot_bar]
                else:
                    area_ok = os_flags[x1] and os_flags[pivot_bar]
                if not require_osob:
                    area_ok = True
                delta = m2 - m1
                swing_ok = (not use_atr_filter) or (abs(p1 - p2) >= atr_values[bar] * effective_atr)
                price_ok = p2 <= p1 * (1 + effective_price_eps)
                osc_ok = m2 >= m1 - effective_osc_eps
                is_bull = bool(price_ok and osc_ok and area_ok and delta >= effective_delta and swing_ok)
                if is_bull:
                    bull_seq += 1
                    bear_seq = 0
                    bars_between = max(1, pivot_bar - x1)
                    time01 = _clamp01(bars_between / 40.0)
                    delta01 = 1.0 if effective_delta <= 0 else _clamp01(delta / (effective_delta * 2.0))
                    atr01 = _clamp01((abs(p1 - p2) / max(atr_values[bar] * effective_atr, 1e-10)) / 2.0)
                    seq01 = _clamp01(bull_seq / 3.0)
                    bonus = 0.10 if (use_bias_bonus and htf_bull.iloc[bar]) else 0.0
                    base01 = (_W_MFI_DELTA * delta01 + _W_ATR * atr01 + _W_TIME * time01
                              + _W_SEQ * seq01 + bonus)
                    out["bull_div"][bar] = True
                    out["div_score"][bar] = round(_clamp01(base01) * 100)
                    out["div_delta"][bar] = delta
                    out["div_bars"][bar] = bars_between
                    out["div_seq"][bar] = bull_seq
                    pending.append({"bar": pivot_bar, "bull": True, "base": base01,
                                    "seq": bull_seq, "delta": delta, "bars": bars_between})
            if not is_bull:
                bull_seq = 0
            last_low_price, last_low_osc, last_low_bar = p2, m2, pivot_bar

        # --- bearish divergence ---------------------------------------------
        if show_divergence and not np.isnan(prices_high[bar]):
            pivot_bar = bar - effective_right
            p2, m2 = prices_high[bar], osc_high[bar]
            is_bear = False
            if not (np.isnan(last_high_price) or np.isnan(last_high_osc)):
                p1, m1, x1 = last_high_price, last_high_osc, last_high_bar
                if osob_mode == "End":
                    area_ok = ob_flags[pivot_bar]
                elif osob_mode == "StartOrEnd":
                    area_ok = ob_flags[x1] or ob_flags[pivot_bar]
                else:
                    area_ok = ob_flags[x1] and ob_flags[pivot_bar]
                if not require_osob:
                    area_ok = True
                delta = m1 - m2
                swing_ok = (not use_atr_filter) or (abs(p2 - p1) >= atr_values[bar] * effective_atr)
                price_ok = p2 >= p1 * (1 - effective_price_eps)
                osc_ok = m2 <= m1 + effective_osc_eps
                is_bear = bool(price_ok and osc_ok and area_ok and delta >= effective_delta and swing_ok)
                if is_bear:
                    bear_seq += 1
                    bull_seq = 0
                    bars_between = max(1, pivot_bar - x1)
                    time01 = _clamp01(bars_between / 40.0)
                    delta01 = 1.0 if effective_delta <= 0 else _clamp01(delta / (effective_delta * 2.0))
                    atr01 = _clamp01((abs(p2 - p1) / max(atr_values[bar] * effective_atr, 1e-10)) / 2.0)
                    seq01 = _clamp01(bear_seq / 3.0)
                    bonus = 0.10 if (use_bias_bonus and htf_bear.iloc[bar]) else 0.0
                    base01 = (_W_MFI_DELTA * delta01 + _W_ATR * atr01 + _W_TIME * time01
                              + _W_SEQ * seq01 + bonus)
                    out["bear_div"][bar] = True
                    out["div_score"][bar] = round(_clamp01(base01) * 100)
                    out["div_delta"][bar] = delta
                    out["div_bars"][bar] = bars_between
                    out["div_seq"][bar] = -bear_seq
                    pending.append({"bar": pivot_bar, "bull": False, "base": base01,
                                    "seq": bear_seq, "delta": delta, "bars": bars_between})
            if not is_bear:
                bear_seq = 0
            last_high_price, last_high_osc, last_high_bar = p2, m2, pivot_bar

        # --- finalize pending divergences (reaction window closed) -----------
        while pending and bar - pending[0]["bar"] >= react_bars + react_lag:
            item = pending.pop(0)
            move = react_up[bar] if item["bull"] else react_down[bar]
            denom = max(atr_values[bar] * react_atr_frac, 1e-10)
            react01 = _clamp01((move / denom) if np.isfinite(move) else 0.0)
            score = round(_clamp01(item["base"] + _W_REACT * react01) * 100)
            out["div_finalized"][bar] = True
            out["div_final_score"][bar] = score
            out["div_react"][bar] = react01

    index = frame.index
    final_score = np.where(np.isnan(out["div_final_score"]), out["div_score"],
                           out["div_final_score"])
    data: dict[str, pd.Series] = {
        "mfi": mfi_value,
        "mfi_raw": raw_mfi,
        "cci_100": cci100,
        "oscillator": oscillator,
        "zone_upper": upper,
        "zone_lower": lower,
        "is_ob": is_ob,
        "is_os": is_os,
        "stoch_k": k_value,
        "stoch_k_raw": k_raw,
        "htf_mfi": htf_values,
        "htf_bull": htf_bull,
        "htf_bear": htf_bear,
        "volume_real": pd.Series(bool(volume_real), index=index),
        "long_signal": long_signal,
        "short_signal": short_signal,
        "bull_div": flags(pd.Series(out["bull_div"], index=index)),
        "bear_div": flags(pd.Series(out["bear_div"], index=index)),
        "div_score": pd.Series(out["div_score"], index=index).astype("Int64"),
        "div_delta": pd.Series(out["div_delta"], index=index),
        "div_bars": pd.Series(out["div_bars"], index=index).astype("Int64"),
        "div_seq": pd.Series(out["div_seq"], index=index).astype("int64"),
        "div_finalized": flags(pd.Series(out["div_finalized"], index=index)),
        "div_final_score": pd.Series(out["div_final_score"], index=index).astype("Int64"),
        "div_react": pd.Series(out["div_react"], index=index),
        "div_strong": pd.Series(np.nan_to_num(final_score, nan=-1.0) >= score_strong,
                                index=index),
    }
    return pd.DataFrame(data, index=index)
