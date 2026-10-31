"""Vein: Accumulation Phase 4H [WavesUnchained] — 吸筹/派发阶段识别。

双向 Wyckoff 风格阶段评分（各 0..10 分，五传感器各封顶 2）：

- **趋势衰竭**：EMA20 斜率（lookback 前下行 + 现在贴平）等三项。
- **区间环境**：效率比 < 0.3（+0.25 加深）+ 区间宽度 2..8×ATR。
- **波动压缩**：BBW 比 < 0.7 + ATR 比 < 0.8 + K 线均幅比 < 0.7。
- **流动性**：sweep low/high 与 spring/upthrust 计数（每 lookback/2 根衰减 1）。
- **结构确认**：HL/LH + BOS + RSI 区间回升/回落。

accumScore = 趋势衰竭 + (衰竭>0 才计入的区间/压缩) + 流动性 + 结构。
≥3 激活 → transition（未满 min_phase_bars）→ early(≥3)/developing(≥5)/
mature(≥6 且流动性≥1 且区间≥1)/breakout_ready(≥8 且结构≥1.5)。
派发端镜像（distState）。

丢弃：背景着色/表格/标签绘图与 display 模式。

Source: pine-collection/indicators/trend_direction/vein/vein_accumulation_phase.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    ema,
    flags,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
    sma,
    stdev,
    sum_,
)

__all__ = ["vein_accumulation_phase_4h"]


def vein_accumulation_phase_4h(
    frame: pd.DataFrame,
    *,
    atr_len: int = 14,
    lookback: int = 20,
    min_phase_bars: int = 10,
    pivot_left: int = 3,
    pivot_right: int = 3,
    er_range_thresh: float = 0.3,
    bbw_compress_thresh: float = 0.7,
    atr_compress_thresh: float = 0.8,
    display_mode: str = "Light",
) -> pd.DataFrame:
    """计算吸筹/派发阶段评分与状态。"""
    atr_len = integer("atr_len", atr_len, 1)
    lookback = integer("lookback", lookback, 10)
    min_phase_bars = integer("min_phase_bars", min_phase_bars, 3)
    pivot_left = integer("pivot_left", pivot_left, 1)
    pivot_right = integer("pivot_right", pivot_right, 1)
    er_range_thresh = number("er_range_thresh", er_range_thresh, 0.05)
    bbw_compress_thresh = number("bbw_compress_thresh", bbw_compress_thresh, 0.3)
    atr_compress_thresh = number("atr_compress_thresh", atr_compress_thresh, 0.3)
    choice("display_mode", display_mode, ("Light", "Debug", "Research"))  # 展示参数

    high, low, close = ohlcv(frame, "high", "low", "close")
    closes = close.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(frame)

    atr14 = atr(frame, atr_len)
    atr_arr = atr14.to_numpy(dtype=float)
    atr_avg = sma(atr14, lookback).to_numpy(dtype=float)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema20_slope = (ema20.diff(3) / 3.0).to_numpy(dtype=float)
    ema20_arr = ema20.to_numpy(dtype=float)
    ema50_arr = ema50.to_numpy(dtype=float)

    bb_basis = sma(close, 20).to_numpy(dtype=float)
    bb_dev = stdev(close, 20).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = np.where(bb_basis > 0, (2 * bb_dev * 2.0) / bb_basis, 0.0)
    bbw_avg = sma(pd.Series(bb_width, index=frame.index), lookback).to_numpy(dtype=float)
    bbw_ratio = np.where(bbw_avg > 0, bb_width / np.where(bbw_avg > 0, bbw_avg, 1.0), 1.0)
    atr_ratio = np.where(atr_avg > 0, atr_arr / np.where(atr_avg > 0, atr_avg, 1.0), 1.0)

    close_shift = pd.Series(closes, index=frame.index).shift(lookback)
    net_move = np.abs(closes - close_shift.to_numpy(dtype=float))
    diff_abs = pd.Series(np.abs(np.diff(closes, prepend=closes[0])), index=frame.index)
    total_path = sum_(diff_abs, lookback).to_numpy(dtype=float)
    eff_ratio = np.where(total_path > 0, net_move / np.where(total_path > 0, total_path, 1.0), 1.0)

    candle_range = pd.Series(highs - lows, index=frame.index)
    avg_candle = sma(candle_range, lookback).to_numpy(dtype=float)
    candle_ratio = np.where(avg_candle > 0, (highs - lows) / np.where(avg_candle > 0, avg_candle, 1.0), 1.0)

    range_high = highest(high, lookback).to_numpy(dtype=float)
    range_low = lowest(low, lookback).to_numpy(dtype=float)
    range_width = np.where(atr_arr > 0, (range_high - range_low) / np.where(atr_arr > 0, atr_arr, 1.0), 0.0)

    rsi14 = rsi(close, 14).to_numpy(dtype=float)
    rsi_shift = rsi(close, 14).shift(lookback).to_numpy(dtype=float)

    ph_arr = pivothigh(high, pivot_left, pivot_right).to_numpy(dtype=float)
    pl_arr = pivotlow(low, pivot_left, pivot_right).to_numpy(dtype=float)

    out = {
        "accum_score": np.full(n, np.nan),
        "accum_state": np.full(n, "none", dtype=object),
        "accum_duration": np.zeros(n, dtype=int),
        "dist_score": np.full(n, np.nan),
        "dist_state": np.full(n, "none", dtype=object),
        "dist_duration": np.zeros(n, dtype=int),
        "efficiency_ratio": np.full(n, np.nan),
        "bbw_ratio": np.full(n, np.nan),
        "atr_ratio": np.full(n, np.nan),
        "bull_bos": np.zeros(n, dtype=bool),
        "bear_bos": np.zeros(n, dtype=bool),
        "spring": np.zeros(n, dtype=bool),
        "upthrust": np.zeros(n, dtype=bool),
        "last_swing_high": np.full(n, np.nan),
        "last_swing_low": np.full(n, np.nan),
    }

    last_swing_high = prev_swing_high = np.nan
    last_swing_low = prev_swing_low = np.nan
    sweep_count = spring_count = sweep_high_count = upthrust_count = 0
    sweep_low_prev = spring_prev = sweep_high_prev = upthrust_prev = False
    accum_start: int | None = None
    dist_start: int | None = None
    decay_every = max(1, round(lookback / 2))

    for bar in range(n):
        if not np.isnan(ph_arr[bar]):
            prev_swing_high = last_swing_high
            last_swing_high = ph_arr[bar]
        if not np.isnan(pl_arr[bar]):
            prev_swing_low = last_swing_low
            last_swing_low = pl_arr[bar]
        out["last_swing_high"][bar] = last_swing_high
        out["last_swing_low"][bar] = last_swing_low

        prev_close = closes[bar - 1] if bar > 0 else closes[bar]
        bull_bos = (not np.isnan(last_swing_high) and closes[bar] > last_swing_high
                    and prev_close <= last_swing_high)
        bear_bos = (not np.isnan(last_swing_low) and closes[bar] < last_swing_low
                    and prev_close >= last_swing_low)
        spring = (not np.isnan(last_swing_low) and lows[bar] < last_swing_low
                  and closes[bar] > last_swing_low)
        sweep_low = not np.isnan(last_swing_low) and lows[bar] < last_swing_low
        upthrust = (not np.isnan(last_swing_high) and highs[bar] > last_swing_high
                    and closes[bar] < last_swing_high)
        sweep_high = not np.isnan(last_swing_high) and highs[bar] > last_swing_high
        has_hl = (not np.isnan(last_swing_low) and not np.isnan(prev_swing_low)
                  and last_swing_low > prev_swing_low)
        has_ll = (not np.isnan(last_swing_low) and not np.isnan(prev_swing_low)
                  and last_swing_low < prev_swing_low)
        has_lh = (not np.isnan(last_swing_high) and not np.isnan(prev_swing_high)
                  and last_swing_high < prev_swing_high)
        has_hh = (not np.isnan(last_swing_high) and not np.isnan(prev_swing_high)
                  and last_swing_high > prev_swing_high)
        out["bull_bos"][bar] = bull_bos
        out["bear_bos"][bar] = bear_bos
        out["spring"][bar] = spring
        out["upthrust"][bar] = upthrust

        # 计数与衰减。
        if sweep_low and not sweep_low_prev:
            sweep_count += 1
        if spring and not spring_prev:
            spring_count += 1
        if sweep_high and not sweep_high_prev:
            sweep_high_count += 1
        if upthrust and not upthrust_prev:
            upthrust_count += 1
        if bar % decay_every == 0:
            sweep_count = max(0, sweep_count - 1)
            spring_count = max(0, spring_count - 1)
            sweep_high_count = max(0, sweep_high_count - 1)
            upthrust_count = max(0, upthrust_count - 1)
        sweep_low_prev, spring_prev = sweep_low, spring
        sweep_high_prev, upthrust_prev = sweep_high, upthrust

        if np.isnan(atr_arr[bar]) or np.isnan(eff_ratio[bar]):
            continue

        # 吸筹端传感器。
        ema20_lag = ema20_arr[bar - lookback] if bar >= lookback else np.nan
        ema50_lag = ema50_arr[bar - lookback] if bar >= lookback else np.nan
        slope_flattening = (-0.0001 * closes[bar] < ema20_slope[bar]
                            < 0.0001 * closes[bar])
        slope_was_down = (not np.isnan(ema20_lag) and not np.isnan(ema50_lag)
                          and ema20_lag < ema50_lag)
        slope_now_flat = (abs(ema20_arr[bar] - ema50_arr[bar]) < atr_arr[bar] * 0.5)
        trend_loss = 0.0
        if slope_was_down and slope_now_flat:
            trend_loss += 1.0
        if slope_was_down and not bull_bos and not has_ll:
            trend_loss += 0.5
        if slope_flattening and slope_was_down:
            trend_loss += 0.5
        trend_loss = min(trend_loss, 2.0)

        range_score = 0.0
        if eff_ratio[bar] < er_range_thresh:
            range_score += 1.0
        if eff_ratio[bar] < er_range_thresh * 0.5:
            range_score += 0.5
        if 2.0 < range_width[bar] < 8.0:
            range_score += 0.5
        range_score = min(range_score, 2.0)

        vol_comp = 0.0
        if bbw_ratio[bar] < bbw_compress_thresh:
            vol_comp += 1.0
        if atr_ratio[bar] < atr_compress_thresh:
            vol_comp += 0.5
        if candle_ratio[bar] < 0.7:
            vol_comp += 0.5
        vol_comp = min(vol_comp, 2.0)

        liq = min(0.5 * (sweep_count >= 1) + 0.5 * (sweep_count >= 2)
                  + 0.5 * (spring_count >= 1) + 0.5 * (spring_count >= 2), 2.0)
        bull_struct = 0.0
        if has_hl:
            bull_struct += 1.0
        if bull_bos:
            bull_struct += 1.0
        if not np.isnan(rsi14[bar]) and not np.isnan(rsi_shift[bar]) \
                and rsi14[bar] > 40 and rsi_shift[bar] < 35:
            bull_struct += 0.5
        bull_struct = min(bull_struct, 2.0)

        accum_score = (trend_loss + (range_score if trend_loss > 0 else 0.0)
                       + (vol_comp if trend_loss > 0 else 0.0) + liq + bull_struct)
        accum_active = accum_score >= 3.0
        if accum_active and accum_start is None:
            accum_start = bar
        if not accum_active:
            accum_start = None
        accum_duration = bar - accum_start if accum_start is not None else 0
        meets_min = accum_duration >= min_phase_bars
        if not accum_active:
            accum_state = "none"
        elif not meets_min:
            accum_state = "transition"
        elif accum_score >= 8.0 and bull_struct >= 1.5:
            accum_state = "breakout_ready"
        elif accum_score >= 6.0 and liq >= 1.0 and (trend_loss > 0 and range_score >= 1.0):
            accum_state = "mature"
        elif accum_score >= 5.0:
            accum_state = "developing"
        else:
            accum_state = "early"
        out["accum_score"][bar] = accum_score
        out["accum_state"][bar] = accum_state
        out["accum_duration"][bar] = accum_duration

        # 派发端传感器（镜像）。
        dist_slope_was_up = (not np.isnan(ema20_lag) and not np.isnan(ema50_lag)
                             and ema20_lag > ema50_lag)
        dist_trend_loss = 0.0
        if dist_slope_was_up and slope_now_flat:
            dist_trend_loss += 1.0
        if dist_slope_was_up and not bull_bos and not has_hh:
            dist_trend_loss += 0.5
        if slope_flattening and dist_slope_was_up:
            dist_trend_loss += 0.5
        dist_trend_loss = min(dist_trend_loss, 2.0)
        dist_liq = min(0.5 * (sweep_high_count >= 1) + 0.5 * (sweep_high_count >= 2)
                       + 0.5 * (upthrust_count >= 1) + 0.5 * (upthrust_count >= 2), 2.0)
        bear_struct = 0.0
        if has_lh:
            bear_struct += 1.0
        if bear_bos:
            bear_struct += 1.0
        if not np.isnan(rsi14[bar]) and not np.isnan(rsi_shift[bar]) \
                and rsi14[bar] < 60 and rsi_shift[bar] > 65:
            bear_struct += 0.5
        bear_struct = min(bear_struct, 2.0)
        dist_score = (dist_trend_loss + (range_score if dist_trend_loss > 0 else 0.0)
                      + (vol_comp if dist_trend_loss > 0 else 0.0) + dist_liq
                      + bear_struct)
        dist_active = dist_score >= 3.0
        if dist_active and dist_start is None:
            dist_start = bar
        if not dist_active:
            dist_start = None
        dist_duration = bar - dist_start if dist_start is not None else 0
        dist_meets = dist_duration >= min_phase_bars
        if not dist_active:
            dist_state = "none"
        elif not dist_meets:
            dist_state = "transition"
        elif dist_score >= 8.0 and bear_struct >= 1.5:
            dist_state = "breakdown_ready"
        elif dist_score >= 6.0 and dist_liq >= 1.0 and (dist_trend_loss > 0 and range_score >= 1.0):
            dist_state = "mature"
        elif dist_score >= 5.0:
            dist_state = "developing"
        else:
            dist_state = "early"
        out["dist_score"][bar] = dist_score
        out["dist_state"][bar] = dist_state
        out["dist_duration"][bar] = dist_duration

        out["efficiency_ratio"][bar] = eff_ratio[bar]
        out["bbw_ratio"][bar] = bbw_ratio[bar]
        out["atr_ratio"][bar] = atr_ratio[bar]

    index = frame.index
    data: dict[str, pd.Series] = {
        "accum_score": pd.Series(out["accum_score"], index=index),
        "accum_state": pd.Series(out["accum_state"], index=index, dtype=object),
        "accum_duration": pd.Series(out["accum_duration"], index=index, dtype="int64"),
        "dist_score": pd.Series(out["dist_score"], index=index),
        "dist_state": pd.Series(out["dist_state"], index=index, dtype=object),
        "dist_duration": pd.Series(out["dist_duration"], index=index, dtype="int64"),
        "efficiency_ratio": pd.Series(out["efficiency_ratio"], index=index),
        "bbw_ratio": pd.Series(out["bbw_ratio"], index=index),
        "atr_ratio": pd.Series(out["atr_ratio"], index=index),
        "bull_bos": flags(pd.Series(out["bull_bos"], index=index)),
        "bear_bos": flags(pd.Series(out["bear_bos"], index=index)),
        "spring": flags(pd.Series(out["spring"], index=index)),
        "upthrust": flags(pd.Series(out["upthrust"], index=index)),
        "last_swing_high": pd.Series(out["last_swing_high"], index=index),
        "last_swing_low": pd.Series(out["last_swing_low"], index=index),
    }
    return pd.DataFrame(data, index=index)
