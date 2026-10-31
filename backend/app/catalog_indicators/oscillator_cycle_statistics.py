"""Oscillator Cycle Statistics [WavesUnchained] v1.5.6 的 Python 移植。

追踪「从一个极值到相反极值」的完整振荡器周期，把每一次已完成的尝试存进样本库，
再回答「历史上到达过同样进度桶、且存活到同样年龄的周期，有多大比例真正抵达
对侧极值」——并且按市场状态（牛/熊/震荡）分组。

核心流水线
----------
1. 振荡器：RSI / StochRSI / Stochastic / 归一化 Williams %R / MFI / CCI /
   WaveTrend / DeMarker / RVI / EOM / Klinger，全部重标定到 0..100。
   三个量能振荡器（MFI/EOM/Klinger）在无量能数据时钉在 50（中性、不产生周期）。
2. 市场状态：EMA(200) 与 ATR 归一的 EMA 斜率双门 → +1 / 0 / -1。
3. 周期状态机：进场（OS→OB 或 OB→OS）→ 武装（离开起点区）→ 成功 / 失败
   （回到起点区）/ 超时，四类结果都落样本。
4. 经验统计：按方向 + 回溯窗 + 进度桶 + 年龄门过滤样本，给出完成率、
   时长/MFE/MAE 中位数、Wilson 区间。
5. 校准：周期跑到 ``calib_checkpoint`` 进度时快照当时的条件预测（只用已解析样本，
   无前视），周期结束时与真实结果配对，算 Brier 与 Brier Skill Score。

移植说明
--------
* ``confirmed_only`` 保留为参数：批量模式下每根 bar 都是已确认 bar，故无影响。
* Pine 的 ``barstate.islast`` 仪表盘/标签块在这里退化为「逐 bar 计算」，输出的是
  **每根 bar 收盘时**的读数，便于当作特征列使用。
* 绘图/表格/折线/标记全部不移植（本目录只产出数值）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._pine import atr, cci, choice, clamp, ema, integer, lowest, mfi, boolean, number, ohlcv, rsi, sma, stoch

__all__ = ["oscillator_cycle_statistics"]

_BUCKETS = 10  # 进度按 10 个百分点分桶，保证「按年龄可比」是时间一致的

_OSCILLATORS = (
    "RSI",
    "Stochastic RSI",
    "Stochastic",
    "Williams %R (normalized = Fast Stoch %K)",
    "MFI (volume)",
    "CCI",
    "WaveTrend",
    "DeMarker",
    "RVI",
    "EOM (volume)",
    "Klinger (volume)",
)

_VOLUME_OSCILLATORS = ("MFI (volume)", "EOM (volume)", "Klinger (volume)")
_EIGHTY_TWENTY = (
    "Stochastic RSI",
    "Stochastic",
    "Williams %R (normalized = Fast Stoch %K)",
    "MFI (volume)",
)

# 周期阶段码（纯描述，不是交易阈值）
_STAGE_WAITING, _STAGE_ORIGIN, _STAGE_EARLY = 0, 1, 2
_STAGE_DEVELOPING, _STAGE_MATURE, _STAGE_LATE = 3, 4, 5

# 标签状态码
_LABEL_WAIT, _LABEL_LOW_DATA, _LABEL_STALLED = 0, 1, 2
_LABEL_UNCERTAIN, _LABEL_LATE, _LABEL_OPPORTUNITY, _LABEL_WATCH = 3, 4, 5, 6

_STAGE_EARLY_END = 0.30
_STAGE_DEVELOPING_END = 0.60
_STAGE_MATURE_END = 0.80
_COMPLETION_MAJORITY = 50.0


def _ema_skipna(series: pd.Series, length: int) -> pd.Series:
    """``ta.ema`` 的 na 容忍版本：跳过开头暖机期的 na，从第一个有效值起步。"""
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


def _safe_div(numerator: np.ndarray, denominator: np.ndarray, fallback: float) -> np.ndarray:
    """分母为 0 时取 ``fallback``；分母为 na 时保持 na。"""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    usable = np.isfinite(den) & (den != 0.0)
    return np.where(usable, num / np.where(usable, den, 1.0), fallback)


def _wilson(success: int, total: int) -> tuple[float, float]:
    """Wilson 得分区间（百分比）。样本为 0 时返回 (nan, nan)。"""
    if total <= 0:
        return math.nan, math.nan
    z = 1.96
    n = float(total)
    p = success / n
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    margin = z / (1.0 + z2 / n) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


@dataclass
class _SampleStore:
    """已完成的周期样本（定长上限，超限丢最旧）。"""

    capacity: int
    rows: list[tuple] = field(default_factory=list)
    _cache: tuple | None = None

    def push(self, row: tuple) -> None:
        if len(self.rows) >= self.capacity:
            self.rows.pop(0)
        self.rows.append(row)
        self._cache = None

    def arrays(self) -> tuple:
        """把行式存储展开成列式 numpy 数组（按需构建并缓存）。"""
        if self._cache is None:
            if not self.rows:
                empty_f = np.empty(0, dtype=float)
                empty_i = np.empty(0, dtype=np.int64)
                self._cache = (empty_i, empty_i, empty_i, empty_i, empty_i, empty_i,
                               empty_f, empty_f, empty_f, np.empty((0, _BUCKETS), dtype=np.int64))
            else:
                cols = list(zip(*self.rows))
                self._cache = (
                    np.asarray(cols[0], dtype=np.int64),   # direction
                    np.asarray(cols[1], dtype=np.int64),   # success
                    np.asarray(cols[2], dtype=np.int64),   # reason
                    np.asarray(cols[3], dtype=np.int64),   # regime
                    np.asarray(cols[4], dtype=np.int64),   # duration
                    np.asarray(cols[5], dtype=np.int64),   # end bar
                    np.asarray(cols[6], dtype=float),      # max progress
                    np.asarray(cols[7], dtype=float),      # mfe (ATR)
                    np.asarray(cols[8], dtype=float),      # mae (ATR)
                    np.stack([np.asarray(c, dtype=np.int64) for c in cols[9]]),
                )
        return self._cache


@dataclass
class _CalibStore:
    """走前校准记录：周期中途快照的预测 vs 该周期最终结果。"""

    capacity: int
    rows: list[tuple] = field(default_factory=list)
    _cache: tuple | None = None

    def push(self, row: tuple) -> None:
        self.rows.append(row)
        if len(self.rows) > self.capacity:
            self.rows.pop(0)
        self._cache = None

    def arrays(self) -> tuple:
        if self._cache is None:
            if not self.rows:
                e = np.empty(0, dtype=float)
                i = np.empty(0, dtype=np.int64)
                self._cache = (e, e, e, i, i, i)
            else:
                cols = list(zip(*self.rows))
                self._cache = (
                    np.asarray(cols[0], dtype=float),
                    np.asarray(cols[1], dtype=float),
                    np.asarray(cols[2], dtype=float),
                    np.asarray(cols[3], dtype=np.int64),
                    np.asarray(cols[4], dtype=np.int64),
                    np.asarray(cols[5], dtype=np.int64),
                )
        return self._cache


def _statistics(
    store: _SampleStore,
    bar_index: int,
    direction: int,
    lookback: int,
    filter_regime: bool,
    wanted_regime: int,
    progress_threshold: float,
    age_threshold: int,
    collect: bool,
) -> dict[str, float]:
    """按方向 / 回溯 / 进度桶 / 年龄门过滤样本并汇总。

    ``age_threshold < 0`` 表示不按年龄过滤（只看最终进度）。
    """
    directions, successes, _reasons, regimes, durations, end_bars, max_prog, mfe, mae, first_reach = store.arrays()
    result = {
        "success": 0, "total": 0, "prob": math.nan,
        "cond_success": 0, "cond_total": 0, "cond_prob": math.nan,
        "median_duration": math.nan, "median_mfe": math.nan, "median_mae": math.nan,
        "median_failed_mfe": math.nan, "median_failed_mae": math.nan,
    }
    if directions.size == 0:
        return result

    mask = (bar_index - end_bars <= lookback) & (directions == direction)
    if filter_regime:
        mask &= regimes == wanted_regime
    total = int(mask.sum())
    result["total"] = total
    result["success"] = int(successes[mask].sum()) if total else 0
    if total:
        result["prob"] = 100.0 * result["success"] / total
    if total == 0:
        return result

    bucket = int(math.floor(min(max(progress_threshold, 0.0), 1.0) * _BUCKETS + 0.000001))
    bucket_progress = bucket * 1.0 / _BUCKETS
    cond = mask & (max_prog >= bucket_progress)
    if age_threshold >= 0:
        cond &= durations > age_threshold
        if bucket > 0:
            reach = first_reach[:, bucket - 1]
            cond &= (reach >= 0) & (reach <= age_threshold)
    cond_total = int(cond.sum())
    result["cond_total"] = cond_total
    if cond_total:
        cond_success = int(successes[cond].sum())
        result["cond_success"] = cond_success
        result["cond_prob"] = 100.0 * cond_success / cond_total
        if collect:
            won = cond & (successes == 1)
            lost = cond & (successes == 0)
            if won.any():
                result["median_duration"] = float(np.median(durations[won]))
                result["median_mfe"] = float(np.median(mfe[won]))
                result["median_mae"] = float(np.median(mae[won]))
            if lost.any():
                result["median_failed_mfe"] = float(np.median(mfe[lost]))
                result["median_failed_mae"] = float(np.median(mae[lost]))
    return result


def _direction_base_rates(store: _SampleStore, bar_index: int, lookback: int) -> tuple[float, int, float, int]:
    """每个方向的朴素完成率（不带进度/年龄条件）。"""
    directions, successes, _r, _g, _d, end_bars, *_rest = store.arrays()
    if directions.size == 0:
        return math.nan, 0, math.nan, 0
    mask = bar_index - end_bars <= lookback
    down = mask & (directions == -1)
    up = mask & (directions == 1)
    down_n, up_n = int(down.sum()), int(up.sum())
    down_rate = 100.0 * float(successes[down].sum()) / down_n if down_n else math.nan
    up_rate = 100.0 * float(successes[up].sum()) / up_n if up_n else math.nan
    return down_rate, down_n, up_rate, up_n


def oscillator_cycle_statistics(
    frame: pd.DataFrame,
    *,
    oscillator: str = "RSI",
    primary_length: int = 14,
    smooth_k: int = 3,
    wt_channel_length: int = 10,
    wt_average_length: int = 21,
    cci_reference: float = 100.0,
    wt_reference: float = 60.0,
    rvi_reference: float = 0.3,
    eom_reference: float = 1.0,
    klinger_fast_length: int = 34,
    klinger_slow_length: int = 55,
    klinger_reference: float = 100000.0,
    use_oscillator_default_zones: bool = True,
    custom_overbought: float = 70.0,
    custom_oversold: float = 30.0,
    max_cycle_bars: int = 150,
    confirmed_only: bool = True,
    recent_lookback: int = 300,
    medium_lookback: int = 1000,
    long_lookback: int = 3000,
    max_stored_samples: int = 1500,
    min_conditional_samples: int = 20,
    use_regime_stats: bool = True,
    regime_ema_length: int = 200,
    regime_slope_bars: int = 10,
    regime_atr_length: int = 14,
    regime_slope_threshold: float = 0.20,
    calib_checkpoint: float = 50.0,
    opportunity_edge_threshold: float = 15.0,
    opportunity_zone_min: int = 30,
    opportunity_zone_max: int = 65,
    opportunity_max_ci_width: float = 40.0,
    volume_is_real: bool | None = None,
) -> pd.DataFrame:
    """Oscillator Cycle Statistics：逐 bar 的振荡器周期统计面。

    参数镜像 Pine 的 input 默认值（keyword-only）。返回 ``ocs_`` 前缀的 DataFrame，
    索引与 ``frame`` 一致。
    """
    oscillator = choice("oscillator", oscillator, _OSCILLATORS)
    primary_length = integer("primary_length", primary_length, 2, 200)
    smooth_k = integer("smooth_k", smooth_k, 1, 20)
    wt_channel_length = integer("wt_channel_length", wt_channel_length, 2, 100)
    wt_average_length = integer("wt_average_length", wt_average_length, 2, 100)
    cci_reference = number("cci_reference", cci_reference, 20.0)
    wt_reference = number("wt_reference", wt_reference, 10.0)
    rvi_reference = number("rvi_reference", rvi_reference, 0.05)
    eom_reference = number("eom_reference", eom_reference, 0.05)
    klinger_fast_length = integer("klinger_fast_length", klinger_fast_length, 2, 200)
    klinger_slow_length = integer("klinger_slow_length", klinger_slow_length, 2, 300)
    klinger_reference = number("klinger_reference", klinger_reference, 100.0)
    use_oscillator_default_zones = boolean("use_oscillator_default_zones", use_oscillator_default_zones)
    custom_overbought = number("custom_overbought", custom_overbought, 51.0, 99.0)
    custom_oversold = number("custom_oversold", custom_oversold, 1.0, 49.0)
    max_cycle_bars = integer("max_cycle_bars", max_cycle_bars, 5, 2000)
    confirmed_only = boolean("confirmed_only", confirmed_only)
    recent_lookback = integer("recent_lookback", recent_lookback, 50, 10000)
    medium_lookback = integer("medium_lookback", medium_lookback, 100, 20000)
    long_lookback = integer("long_lookback", long_lookback, 200, 50000)
    max_stored_samples = integer("max_stored_samples", max_stored_samples, 100, 10000)
    min_conditional_samples = integer("min_conditional_samples", min_conditional_samples, 5, 500)
    use_regime_stats = boolean("use_regime_stats", use_regime_stats)
    regime_ema_length = integer("regime_ema_length", regime_ema_length, 20, 500)
    regime_slope_bars = integer("regime_slope_bars", regime_slope_bars, 1, 100)
    regime_atr_length = integer("regime_atr_length", regime_atr_length, 2, 100)
    regime_slope_threshold = number("regime_slope_threshold", regime_slope_threshold, 0.0, 5.0)
    calib_checkpoint = number("calib_checkpoint", calib_checkpoint, 10.0, 90.0)
    opportunity_edge_threshold = number("opportunity_edge_threshold", opportunity_edge_threshold, 5.0, 50.0)
    opportunity_zone_min = integer("opportunity_zone_min", opportunity_zone_min, 0, 90)
    opportunity_zone_max = integer("opportunity_zone_max", opportunity_zone_max, 10, 90)
    opportunity_max_ci_width = number("opportunity_max_ci_width", opportunity_max_ci_width, 10.0, 100.0)
    if klinger_fast_length >= klinger_slow_length:
        raise ValueError("klinger_fast_length must be smaller than klinger_slow_length")
    if opportunity_zone_min > opportunity_zone_max:
        raise ValueError("opportunity_zone_min must not exceed opportunity_zone_max")

    open_s, high_s, low_s, close_s, volume_s = ohlcv(frame, "open", "high", "low", "close", "volume")
    open_a = open_s.to_numpy(dtype=float)
    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)
    volume = volume_s.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    # --- 区间 ---------------------------------------------------------------
    if use_oscillator_default_zones:
        overbought = 80.0 if oscillator in _EIGHTY_TWENTY else 70.0
        oversold = 20.0 if oscillator in _EIGHTY_TWENTY else 30.0
    else:
        overbought, oversold = custom_overbought, custom_oversold
    extreme_overbought = max(90.0, overbought)
    extreme_oversold = min(10.0, oversold)

    # --- 振荡器 -------------------------------------------------------------
    hlc3 = (high + low + close) / 3.0
    hlc3_s = pd.Series(hlc3, index=idx)

    rsi_a = rsi(close_s, primary_length).to_numpy(dtype=float)
    rsi_series = pd.Series(rsi_a, index=idx)
    stoch_rsi = sma(stoch(rsi_series, rsi_series, rsi_series, primary_length), smooth_k).to_numpy(dtype=float)
    stochastic_raw = stoch(close_s, high_s, low_s, primary_length).to_numpy(dtype=float)
    stochastic = sma(pd.Series(stochastic_raw, index=idx), smooth_k).to_numpy(dtype=float)

    mfi_raw = mfi(frame, primary_length).to_numpy(dtype=float)
    mfi_vol_avg = sma(pd.Series(np.nan_to_num(volume, nan=0.0), index=idx), primary_length).to_numpy(dtype=float)
    mfi_a = np.where((mfi_vol_avg > 0) & np.isfinite(mfi_raw), mfi_raw, 50.0)

    cci_raw = cci(frame, primary_length, "hlc3").to_numpy(dtype=float)
    cci_norm = np.clip(50.0 + cci_raw * (20.0 / cci_reference), 0.0, 100.0)

    wt_ap = hlc3
    wt_esa = ema(hlc3_s, wt_channel_length)
    wt_dev = _ema_skipna((hlc3_s - wt_esa).abs(), wt_channel_length).to_numpy(dtype=float)
    wt_diff = (hlc3_s - wt_esa).to_numpy(dtype=float)
    wt_ci = np.where(np.isfinite(wt_dev) & (wt_dev != 0.0), wt_diff / (0.015 * np.where(wt_dev == 0.0, 1.0, wt_dev)), 0.0)
    wt_ci = np.where(np.isfinite(wt_ci), wt_ci, 0.0)
    wt1 = ema(pd.Series(wt_ci, index=idx), wt_average_length).to_numpy(dtype=float)
    wt_norm = np.clip(50.0 + wt1 * (20.0 / wt_reference), 0.0, 100.0)

    prev_high = np.r_[np.nan, high[:-1]]
    prev_low = np.r_[np.nan, low[:-1]]
    de_max = np.where(high > prev_high, high - prev_high, 0.0)
    de_min = np.where(low < prev_low, prev_low - low, 0.0)
    de_max_avg = sma(pd.Series(de_max, index=idx), primary_length).to_numpy(dtype=float)
    de_min_avg = sma(pd.Series(de_min, index=idx), primary_length).to_numpy(dtype=float)
    de_denom = de_max_avg + de_min_avg
    de_marker = np.where(de_denom > 0, 100.0 * de_max_avg / np.where(de_denom > 0, de_denom, 1.0), 50.0)

    def _lag(values: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(values, np.nan, dtype=float)
        if k < len(values):
            out[k:] = values[:-k]
        return out

    rvi_num4 = ((close - open_a) + 2.0 * (_lag(close, 1) - _lag(open_a, 1))
                + 2.0 * (_lag(close, 2) - _lag(open_a, 2)) + (_lag(close, 3) - _lag(open_a, 3)))
    rvi_den4 = ((high - low) + 2.0 * (_lag(high, 1) - _lag(low, 1))
                + 2.0 * (_lag(high, 2) - _lag(low, 2)) + (_lag(high, 3) - _lag(low, 3)))
    rvi_num_s = sma(pd.Series(rvi_num4, index=idx), primary_length).to_numpy(dtype=float)
    rvi_den_s = sma(pd.Series(rvi_den4, index=idx), primary_length).to_numpy(dtype=float)
    rvi_raw = _safe_div(rvi_num_s, rvi_den_s, 0.0)
    rvi_norm = np.clip(50.0 + rvi_raw * (20.0 / rvi_reference), 0.0, 100.0)

    mid = (high + low) / 2.0
    eom_distance = mid - _lag(mid, 1)
    eom_range = high - low
    eom_box = np.where(eom_range != 0.0,
                       (np.nan_to_num(volume, nan=0.0) / 100000000.0) / np.where(eom_range == 0.0, 1.0, eom_range),
                       0.0)
    eom_raw = _safe_div(eom_distance, eom_box, 0.0)
    eom_a = sma(pd.Series(eom_raw, index=idx), primary_length).to_numpy(dtype=float)
    eom_norm = np.clip(50.0 + eom_a * (20.0 / eom_reference), 0.0, 100.0)

    # Klinger：趋势方向持续期间累积 dm，翻向时用前一根 dm 重置，再取快慢 EMA 差
    klinger_cm = 0.0
    klinger_trend = -1
    klinger_dm_series = high - low
    klinger_vf = np.zeros(n, dtype=float)
    hlc3_sum = high + low + close
    for i in range(n):
        prev_sum = hlc3_sum[i - 1] if i > 0 and np.isfinite(hlc3_sum[i - 1]) else hlc3_sum[i]
        new_trend = 1 if hlc3_sum[i] > prev_sum else -1
        dm = klinger_dm_series[i]
        if new_trend == klinger_trend:
            klinger_cm += dm
        else:
            prev_dm = klinger_dm_series[i - 1] if i > 0 and np.isfinite(klinger_dm_series[i - 1]) else dm
            klinger_cm = prev_dm + dm
            klinger_trend = new_trend
        klinger_vf[i] = (np.nan_to_num(volume[i], nan=0.0) * 2.0
                         * (dm / klinger_cm - 1.0) * klinger_trend * 100.0) if klinger_cm != 0 else 0.0
    klinger_fast = ema(pd.Series(klinger_vf, index=idx), klinger_fast_length).to_numpy(dtype=float)
    klinger_slow = ema(pd.Series(klinger_vf, index=idx), klinger_slow_length).to_numpy(dtype=float)
    klinger_raw = klinger_fast - klinger_slow
    klinger_norm = np.clip(50.0 + klinger_raw * (20.0 / klinger_reference), 0.0, 100.0)

    # 量能有效性：只有真实成交量为真，tick 计数不算（这里无法识别 tick，按有无量能给判断）
    if volume_is_real is None:
        finite_volume = np.isfinite(volume)
        volume_is_real = bool(finite_volume.any() and np.nansum(volume) > 0)
    else:
        volume_is_real = boolean("volume_is_real", volume_is_real)
    volume_blocked = oscillator in _VOLUME_OSCILLATORS and not volume_is_real

    table = {
        "RSI": rsi_a,
        "Stochastic RSI": stoch_rsi,
        "Stochastic": stochastic,
        "Williams %R (normalized = Fast Stoch %K)": stochastic_raw,
        "MFI (volume)": mfi_a,
        "CCI": cci_norm,
        "DeMarker": de_marker,
        "RVI": rvi_norm,
        "EOM (volume)": eom_norm,
        "Klinger (volume)": klinger_norm,
        "WaveTrend": wt_norm,
    }
    osc = np.clip(table[oscillator], 0.0, 100.0)
    if volume_blocked:
        osc = np.full(n, 50.0)

    # --- 市场状态 -----------------------------------------------------------
    regime_ema = ema(close_s, regime_ema_length).to_numpy(dtype=float)
    regime_atr = atr(frame, regime_atr_length).to_numpy(dtype=float)
    ema_slope = np.zeros(n, dtype=float)
    if regime_slope_bars < n:
        lagged = _lag(regime_ema, regime_slope_bars)
        ok = np.isfinite(lagged) & np.isfinite(regime_ema) & np.isfinite(regime_atr) & (regime_atr > 0)
        ema_slope = np.where(ok, (regime_ema - lagged) / np.where(ok, regime_atr, 1.0), 0.0)
    current_regime = np.zeros(n, dtype=np.int64)
    current_regime = np.where((close > regime_ema) & (ema_slope > regime_slope_threshold), 1, current_regime)
    current_regime = np.where((close < regime_ema) & (ema_slope < -regime_slope_threshold), -1, current_regime)

    # --- 区间状态 -----------------------------------------------------------
    zone_state = np.full(n, 4, dtype=np.int64)
    zone_state = np.where(osc < 45.0, 2, zone_state)
    zone_state = np.where((osc >= 45.0) & (osc <= 55.0), 3, zone_state)
    zone_state = np.where(osc >= overbought, 5, zone_state)
    zone_state = np.where(osc >= extreme_overbought, 6, zone_state)
    zone_state = np.where(osc <= oversold, 1, zone_state)
    zone_state = np.where(osc <= extreme_oversold, 0, zone_state)

    prev_osc = np.r_[np.nan, osc[:-1]]
    entered_ob = (osc >= overbought) & (np.where(np.isfinite(prev_osc), prev_osc, overbought - 1.0) < overbought)
    entered_os = (osc <= oversold) & (np.where(np.isfinite(prev_osc), prev_osc, oversold + 1.0) > oversold)

    # --- 输出数组 -----------------------------------------------------------
    cols: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in (
        "progress", "max_progress", "mfe_atr", "mae_atr", "edge",
        "recent_prob", "medium_prob", "cond_prob", "long_prob", "regime_cond_prob",
        "cond_ci_lower", "cond_ci_upper", "cond_ci_width",
        "median_duration", "median_mfe", "median_mae",
        "median_failed_mfe", "median_failed_mae", "estimated_remaining",
        "base_rate_down", "base_rate_up", "captured_prediction",
        "base_brier", "regime_brier", "naive_brier", "brier_skill",
    )}
    ints: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.int64) for name in (
        "active_direction", "cycle_age", "armed", "new_cycle", "success", "failure",
        "timeout", "opportunity", "stage", "label_state",
        "recent_n", "medium_n", "cond_n", "long_n", "regime_cond_n",
        "base_n_down", "base_n_up", "captured_n", "sample_count", "calib_count",
    )}

    samples = _SampleStore(capacity=max_stored_samples)
    calibrations = _CalibStore(capacity=max_stored_samples)
    width_zone_min = opportunity_zone_min / 100.0
    width_zone_max = opportunity_zone_max / 100.0
    calib_target = calib_checkpoint / 100.0

    # 活动周期状态
    direction = 0
    start_bar = -1
    start_regime = 0
    start_price = math.nan
    start_atr = math.nan
    armed = False
    max_progress = 0.0
    mfe_atr = 0.0
    mae_atr = 0.0
    first_reach = np.full(_BUCKETS, -1, dtype=np.int64)
    prediction_captured = False
    captured_prediction = math.nan
    captured_n = 0
    captured_regime_prediction = math.nan
    captured_regime_n = 0
    captured_base_rate = math.nan
    opportunity_marked = False

    def _start_cycle(new_direction: int, bar: int) -> None:
        nonlocal direction, start_bar, start_regime, start_price, start_atr
        nonlocal armed, max_progress, mfe_atr, mae_atr, first_reach
        nonlocal prediction_captured, captured_prediction, captured_n
        nonlocal captured_regime_prediction, captured_regime_n, captured_base_rate
        nonlocal opportunity_marked
        direction = new_direction
        start_bar = bar
        start_regime = int(current_regime[bar])
        start_price = close[bar]
        start_atr = regime_atr[bar]
        armed = False
        max_progress = 0.0
        mfe_atr = 0.0
        mae_atr = 0.0
        first_reach = np.full(_BUCKETS, -1, dtype=np.int64)
        prediction_captured = False
        opportunity_marked = False
        captured_prediction = math.nan
        captured_n = 0
        captured_regime_prediction = math.nan
        captured_regime_n = 0
        captured_base_rate = math.nan

    def _store_sample(bar: int, result: int, reason: int, duration: int) -> None:
        samples.push((direction, result, reason, start_regime, duration, bar,
                      max_progress, mfe_atr, mae_atr, first_reach.copy()))

    def _store_calibration(bar: int, result: int) -> None:
        if not prediction_captured or not np.isfinite(captured_prediction):
            return
        if captured_n < min_conditional_samples:
            return
        calibrations.push((
            captured_prediction,
            captured_regime_prediction if captured_regime_n >= min_conditional_samples else math.nan,
            captured_base_rate,
            result, bar, direction,
        ))

    span = overbought - oversold

    for i in range(n):
        oi = osc[i]
        dir_before = direction

        # 本 bar 的进度（状态迁移前）
        if dir_before == -1:
            cur_progress = min(max((overbought - oi) / span, 0.0), 1.0) if np.isfinite(oi) else 0.0
        elif dir_before == 1:
            cur_progress = min(max((oi - oversold) / span, 0.0), 1.0) if np.isfinite(oi) else 0.0
        else:
            cur_progress = 0.0

        # 价格偏移（ATR 归一）
        if direction != 0 and np.isfinite(start_atr) and start_atr > 0 and np.isfinite(oi):
            if direction == -1:
                favorable = max(0.0, (start_price - low[i]) / start_atr)
                adverse = max(0.0, (high[i] - start_price) / start_atr)
            else:
                favorable = max(0.0, (high[i] - start_price) / start_atr)
                adverse = max(0.0, (start_price - low[i]) / start_atr)
            mfe_atr = max(mfe_atr, favorable)
            mae_atr = max(mae_atr, adverse)

        active_age = (i - start_bar) if direction != 0 else 0
        target_now = (direction == -1 and oi <= oversold) or (direction == 1 and oi >= overbought)
        timeout_now = direction != 0 and active_age >= max_cycle_bars
        new_cycle = success = failure = timeout = opportunity = False

        # 进度桶推进 + 机会点（只在跨桶瞬间评估）
        if direction != 0:
            if cur_progress > max_progress:
                max_progress = cur_progress
            progress_age = i - start_bar
            for bucket in range(_BUCKETS):
                threshold = (bucket + 1.0) / _BUCKETS
                if cur_progress >= threshold and first_reach[bucket] < 0:
                    first_reach[bucket] = progress_age
                    if (not opportunity_marked and not target_now and not timeout_now
                            and width_zone_min <= threshold <= width_zone_max):
                        stats = _statistics(samples, i, direction, medium_lookback, False, 0,
                                            threshold, progress_age, False)
                        down_rate, _dn, up_rate, _un = _direction_base_rates(samples, i, medium_lookback)
                        base_rate = down_rate if direction == -1 else up_rate
                        edge = (stats["cond_prob"] - base_rate
                                if np.isfinite(stats["cond_prob"]) and np.isfinite(base_rate) else math.nan)
                        lower, upper = _wilson(stats["cond_success"], stats["cond_total"])
                        ci_width = (upper - lower) if np.isfinite(lower) else math.nan
                        if (stats["cond_total"] >= min_conditional_samples
                                and np.isfinite(edge) and edge >= opportunity_edge_threshold
                                and np.isfinite(ci_width) and ci_width <= opportunity_max_ci_width):
                            opportunity = True
                            opportunity_marked = True

        # 校准快照：周期首次跨过检查点时记下当时的条件预测
        if (direction != 0 and not target_now and not timeout_now
                and not prediction_captured and cur_progress >= calib_target):
            stats = _statistics(samples, i, direction, medium_lookback, False, 0,
                                calib_target, active_age, False)
            rstats = _statistics(samples, i, direction, medium_lookback, True, start_regime,
                                 calib_target, active_age, False)
            down_rate, _dn, up_rate, _un = _direction_base_rates(samples, i, medium_lookback)
            captured_prediction = stats["cond_prob"]
            captured_n = stats["cond_total"]
            captured_regime_prediction = rstats["cond_prob"] if use_regime_stats else math.nan
            captured_regime_n = rstats["cond_total"] if use_regime_stats else 0
            captured_base_rate = down_rate if direction == -1 else up_rate
            prediction_captured = True

        # --- 状态迁移 -------------------------------------------------------
        if dir_before == 0:
            if entered_ob[i]:
                _start_cycle(-1, i)
                new_cycle = True
            elif entered_os[i]:
                _start_cycle(1, i)
                new_cycle = True
        elif dir_before == -1:
            age = i - start_bar
            if not armed and oi < overbought:
                armed = True
            if oi <= oversold:
                _store_sample(i, 1, 0, age)
                _store_calibration(i, 1)
                success = True
                _start_cycle(1, i)
                new_cycle = True
            elif armed and oi >= overbought:
                _store_sample(i, 0, 1, age)
                _store_calibration(i, 0)
                failure = True
                _start_cycle(-1, i)
                new_cycle = True
            elif age >= max_cycle_bars:
                _store_sample(i, 0, 2, age)
                _store_calibration(i, 0)
                failure = True
                timeout = True
                direction = 0
                start_bar = -1
                start_price = math.nan
                start_atr = math.nan
                armed = False
                max_progress = 0.0
                mfe_atr = 0.0
                mae_atr = 0.0
                first_reach = np.full(_BUCKETS, -1, dtype=np.int64)
                prediction_captured = False
                opportunity_marked = False
                captured_prediction = math.nan
                captured_n = 0
                captured_regime_prediction = math.nan
                captured_regime_n = 0
                captured_base_rate = math.nan
        elif dir_before == 1:
            age = i - start_bar
            if not armed and oi > oversold:
                armed = True
            if oi >= overbought:
                _store_sample(i, 1, 0, age)
                _store_calibration(i, 1)
                success = True
                _start_cycle(-1, i)
                new_cycle = True
            elif armed and oi <= oversold:
                _store_sample(i, 0, 1, age)
                _store_calibration(i, 0)
                failure = True
                _start_cycle(1, i)
                new_cycle = True
            elif age >= max_cycle_bars:
                _store_sample(i, 0, 2, age)
                _store_calibration(i, 0)
                failure = True
                timeout = True
                direction = 0
                start_bar = -1
                start_price = math.nan
                start_atr = math.nan
                armed = False
                max_progress = 0.0
                mfe_atr = 0.0
                mae_atr = 0.0
                first_reach = np.full(_BUCKETS, -1, dtype=np.int64)
                prediction_captured = False
                opportunity_marked = False
                captured_prediction = math.nan
                captured_n = 0
                captured_regime_prediction = math.nan
                captured_regime_n = 0
                captured_base_rate = math.nan

        # 迁移后的进度
        if direction == -1:
            post_progress = min(max((overbought - oi) / span, 0.0), 1.0) if np.isfinite(oi) else 0.0
        elif direction == 1:
            post_progress = min(max((oi - oversold) / span, 0.0), 1.0) if np.isfinite(oi) else 0.0
        else:
            post_progress = 0.0

        ints["active_direction"][i] = direction
        ints["cycle_age"][i] = (i - start_bar) if direction != 0 else 0
        ints["armed"][i] = 1 if armed else 0
        ints["new_cycle"][i] = 1 if new_cycle else 0
        ints["success"][i] = 1 if success else 0
        ints["failure"][i] = 1 if failure else 0
        ints["timeout"][i] = 1 if timeout else 0
        ints["opportunity"][i] = 1 if opportunity else 0
        cols["progress"][i] = post_progress
        cols["max_progress"][i] = max_progress
        cols["mfe_atr"][i] = mfe_atr
        cols["mae_atr"][i] = mae_atr

        # --- 逐 bar 统计读数 -------------------------------------------------
        active = direction != 0
        stat_direction = direction if active else (-1 if (np.isfinite(oi) and oi >= 50.0) else 1)
        progress_for_stats = max_progress if active else 0.0
        cycle_age = ints["cycle_age"][i]

        rec = _statistics(samples, i, stat_direction, recent_lookback, False, 0, progress_for_stats, -1, False)
        med = _statistics(samples, i, stat_direction, medium_lookback, False, 0, progress_for_stats, cycle_age, True)
        lng = _statistics(samples, i, stat_direction, long_lookback, False, 0, progress_for_stats, -1, False)
        regime_for_stats = start_regime if active else int(current_regime[i])
        reg = _statistics(samples, i, stat_direction, medium_lookback, use_regime_stats, regime_for_stats,
                          progress_for_stats, cycle_age, False)
        down_rate, down_n, up_rate, up_n = _direction_base_rates(samples, i, medium_lookback)

        cols["recent_prob"][i] = rec["prob"]
        ints["recent_n"][i] = rec["total"]
        cols["medium_prob"][i] = med["prob"]
        ints["medium_n"][i] = med["total"]
        cols["cond_prob"][i] = med["cond_prob"]
        ints["cond_n"][i] = med["cond_total"]
        cols["long_prob"][i] = lng["prob"]
        ints["long_n"][i] = lng["total"]
        cols["regime_cond_prob"][i] = reg["cond_prob"]
        ints["regime_cond_n"][i] = reg["cond_total"]
        cols["base_rate_down"][i] = down_rate
        ints["base_n_down"][i] = down_n
        cols["base_rate_up"][i] = up_rate
        ints["base_n_up"][i] = up_n
        for name in ("median_duration", "median_mfe", "median_mae",
                     "median_failed_mfe", "median_failed_mae"):
            cols[name][i] = med[name]
        cols["estimated_remaining"][i] = (max(0.0, med["median_duration"] - cycle_age)
                                          if np.isfinite(med["median_duration"]) else math.nan)

        lower, upper = _wilson(med["cond_success"], med["cond_total"])
        cols["cond_ci_lower"][i] = lower
        cols["cond_ci_upper"][i] = upper
        cols["cond_ci_width"][i] = (upper - lower) if np.isfinite(lower) else math.nan

        # --- 阶段 / 标签状态 -------------------------------------------------
        if active:
            if not armed:
                stage = _STAGE_ORIGIN
            elif post_progress < _STAGE_EARLY_END:
                stage = _STAGE_EARLY
            elif post_progress < _STAGE_DEVELOPING_END:
                stage = _STAGE_DEVELOPING
            elif post_progress < _STAGE_MATURE_END:
                stage = _STAGE_MATURE
            else:
                stage = _STAGE_LATE
        else:
            stage = _STAGE_WAITING
        ints["stage"][i] = stage

        label_state = _LABEL_WAIT
        if active:
            base_rate = down_rate if direction == -1 else up_rate
            edge = (med["cond_prob"] - base_rate
                    if np.isfinite(med["cond_prob"]) and np.isfinite(base_rate) else math.nan)
            cols["edge"][i] = edge
            low_data = med["cond_total"] < min_conditional_samples or not np.isfinite(edge)
            stalled = np.isfinite(med["median_duration"]) and cycle_age > med["median_duration"]
            uncertain = (not low_data) and (not np.isfinite(cols["cond_ci_width"][i])
                                            or cols["cond_ci_width"][i] > opportunity_max_ci_width)
            late = post_progress > width_zone_max
            in_zone = width_zone_min <= post_progress <= width_zone_max
            if low_data:
                label_state = _LABEL_LOW_DATA
            elif stalled:
                label_state = _LABEL_STALLED
            elif uncertain:
                label_state = _LABEL_UNCERTAIN
            elif late:
                label_state = _LABEL_LATE
            elif edge >= opportunity_edge_threshold and in_zone:
                label_state = _LABEL_OPPORTUNITY
            elif edge >= opportunity_edge_threshold / 2.0:
                label_state = _LABEL_WATCH
        ints["label_state"][i] = label_state
        cols["captured_prediction"][i] = captured_prediction
        ints["captured_n"][i] = captured_n
        ints["sample_count"][i] = len(samples.rows)
        ints["calib_count"][i] = len(calibrations.rows)

        # --- 校准 / Brier ----------------------------------------------------
        predicted, regime_pred, naive_pred, outcome, calib_end, _cdir = calibrations.arrays()
        if predicted.size:
            in_window = i - calib_end <= medium_lookback
            if in_window.any():
                p = predicted[in_window]
                o = outcome[in_window]
                base_err = p / 100.0 - o
                cols["base_brier"][i] = float(np.mean(base_err ** 2))
                r = regime_pred[in_window]
                paired = np.isfinite(r)
                if paired.any():
                    cols["regime_brier"][i] = float(np.mean((r[paired] / 100.0 - o[paired]) ** 2))
                nv = naive_pred[in_window]
                naive_ok = np.isfinite(nv)
                naive_brier = float(np.mean((nv[naive_ok] / 100.0 - o[naive_ok]) ** 2)) if naive_ok.any() else math.nan
                cols["naive_brier"][i] = naive_brier
                base_brier = cols["base_brier"][i]
                if (naive_ok.sum() >= min_conditional_samples and in_window.sum() >= min_conditional_samples
                        and np.isfinite(naive_brier) and naive_brier > 0):
                    cols["brier_skill"][i] = 100.0 * (1.0 - base_brier / naive_brier)

    data = {
        "ocs_osc": osc,
        "ocs_zone_state": zone_state,
        "ocs_overbought": np.full(n, overbought),
        "ocs_oversold": np.full(n, oversold),
        "ocs_extreme_overbought": np.full(n, extreme_overbought),
        "ocs_extreme_oversold": np.full(n, extreme_oversold),
        "ocs_volume_blocked": np.full(n, 1 if volume_blocked else 0, dtype=np.int64),
        "ocs_regime": current_regime,
        "ocs_regime_ema": regime_ema,
        "ocs_regime_atr": regime_atr,
        "ocs_ema_slope_atr": ema_slope,
    }
    for name, values in cols.items():
        data["ocs_" + name] = np.asarray(values, dtype=float)
    for name, values in ints.items():
        data["ocs_" + name] = pd.Series(values, index=idx).astype("Int64")

    result = pd.DataFrame(data, index=idx)
    for name in ("ocs_zone_state", "ocs_regime", "ocs_volume_blocked"):
        result[name] = pd.Series(result[name].to_numpy(), index=idx).astype("Int64")
    return result
