"""Crypto AMT Toolkit v7.3 — 拍卖市场（AMT）+ 成交量分布 + VWAP 信号引擎。

**成交量分布（滚动 VP）**：最近 `vp_lookback` 根的最高/最低划成 `vp_rows` 行，
每根 bar 的成交量按「与行的重叠比例」分摊（`volume*overlap/bar_range`；十字星
按收盘落点整根计入）。POC = 成交量最大的行中心；价值区从 POC 向两侧贪心扩展，
每次取成交量更大的一侧，直到累计 ≥ `value_area_pct%`；VAH = 高界、VAL = 低界。

**VWAP**：按 `vwap_anchor`（日/周/月）锚定重算，`vwap = Σ(hlc3*vol)/Σvol`；
带宽用 `stdev(hlc3 - vwap, band_len)` × `band_mult`。

**趋势**：快/慢 EMA + close 位置定义趋势；HTF 用**上一已完成周期**的
`close` 与 EMA 比较（无未来函数）。

**订单流代理**：本库无法取低周期 intrabar，按 Pine 的 fallback 分支
`close>open → +volume`、`close<open → −volume`；`delta_abs_avg = ema(|delta|)`，
`rel_vol = volume / sma(volume)`；强多流 = delta > delta_abs_avg×delta_mult 且
volume ≥ avg×vol_mult。CVD 按 `cvd_reset` 在周期切换时归零，否则累加；
CVD 多头 = cvd > ema(cvd) 且 cvd 上行。

**价格行为**：swing pivot 追踪最近高低点 → BOS；看涨吞没/看跌吞没；
下影 ≥ `wick_atr×ATR` 且收阳 → 看涨拒斥（镜像）；实体 ≥ `body_atr×ATR`
且量能达标 → 冲量。

**评分**：VWAP 位置 ±1、价值区位置 ±2、趋势 ±1、HTF ±1、订单流 ±1、CVD ±1、
价格行为 ±1、冲量 ±1，合计 −9..9。`setup_score = max(2, required−1或−2)`，
`required = max(min_score, 模式分 4/5/6)`。

**触发**：突破（价值区接受 + VWAP 同向 + 流）、回收（POC/VWAP 交叉）、
回测（关键位附近 + 反转 K 线）、VAL/VAH 拒斥、BOS。

**免交易区**：中间价值区、低相对量能、震荡（无趋势且中间价值）、远离关键位
（仅 Adaptive 且要求触位），冲量例外可豁免。冷却按 `cooldown_bars`（Scalp 折半、
Swing 1.5 倍）计。

**风险**：`Structure` 用 VAL/最近 swing 低点 − 容差（空头镜像）；`Tight` 用最近
下方关键位；`Hybrid` 取两者更紧者。目标 = 入场 ± 风险×`target_1r/2r`。

丢弃：全部 plot/box/line/label/table/barcolor 绘图、dashboard、告警配置、
Spec Watch 实时态（历史回放里等价于 pending）、新手标签。

Source: pine-crypto-collection/latifangren_crypto_amt/indicator/crypto_amt_toolkit_v7_3_auto_performance.pine
License: latifangren 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    crossover,
    crossunder,
    ema,
    flags,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    sma,
    stdev,
)
from ._tf import base_step_minutes, lift_series

__all__ = ["crypto_amt_toolkit"]

_PRESETS = {"Scalp": (4, 2, 0.5, 1.15), "Intraday": (5, 1, 1.0, None), "Swing": (6, 1, 1.5, None)}


def _resolve_perf(mode: str, minutes: int) -> str:
    if mode != "Auto":
        return mode
    return "Fast" if minutes >= 60 else "Balanced"


def _anchored_vwap(frame: pd.DataFrame, anchor: str) -> pd.Series:
    """Session-anchored VWAP: the cumulative sum restarts at every anchor boundary."""
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3.0
    utc = frame.index.tz_convert("UTC").tz_localize(None)
    period = utc.to_period({"Daily": "D", "Weekly": "W", "Monthly": "M"}[anchor])
    flow = typical * volume
    cumulative_flow = flow.groupby(period.asi8).cumsum()
    cumulative_volume = volume.groupby(period.asi8).cumsum()
    return cumulative_flow / cumulative_volume.replace(0, np.nan)


def _volume_profile(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    volumes: np.ndarray, bar: int, lookback: int, rows: int,
                    value_area_pct: float) -> tuple[float, float, float, np.ndarray, float, float]:
    """Rolling volume profile → POC / VAH / VAL plus the raw row volumes."""
    start = max(0, bar - lookback + 1)
    window_high = highs[start: bar + 1]
    window_low = lows[start: bar + 1]
    if not len(window_high):
        return (math.nan, math.nan, math.nan, np.zeros(rows), math.nan, math.nan)
    top, bottom = window_high.max(), window_low.min()
    span = max(top - bottom, 1e-12)
    row_height = span / rows
    row_low = bottom + row_height * np.arange(rows)
    row_high = row_low + row_height
    volumes_row = np.zeros(rows)
    for offset in range(len(window_high)):
        index = bar - offset
        bar_high, bar_low = highs[index], lows[index]
        bar_range = max(bar_high - bar_low, 1e-12)
        if abs(bar_high - bar_low) <= 1e-12:
            position = int(np.floor((closes[index] - bottom) / row_height)) if row_height else 0
            position = min(max(position, 0), rows - 1)
            volumes_row[position] += volumes[index]
            continue
        overlap = np.maximum(0.0, np.minimum(bar_high, row_high) - np.maximum(bar_low, row_low))
        volumes_row += volumes[index] * overlap / bar_range
    total = volumes_row.sum()
    poc_index = int(np.argmax(volumes_row)) if total > 0 else 0
    low_index = high_index = poc_index
    target = total * value_area_pct / 100.0
    accumulated = volumes_row[poc_index]
    while accumulated < target and (low_index > 0 or high_index < rows - 1):
        next_low = volumes_row[low_index - 1] if low_index > 0 else -1.0
        next_high = volumes_row[high_index + 1] if high_index < rows - 1 else -1.0
        if next_high >= next_low:
            high_index += 1
            accumulated += max(next_high, 0.0)
        else:
            low_index -= 1
            accumulated += max(next_low, 0.0)
    poc = bottom + row_height * (poc_index + 0.5)
    val = bottom + row_height * low_index
    vah = bottom + row_height * (high_index + 1.0)
    return poc, vah, val, volumes_row, bottom, row_height


def crypto_amt_toolkit(
    frame: pd.DataFrame,
    *,
    vp_lookback: int = 160,
    vp_rows: int = 36,
    value_area_pct: float = 70.0,
    accept_bars: int = 2,
    level_tolerance_atr: float = 0.35,
    perf_mode: str = "Auto",
    vwap_anchor: str = "Daily",
    band_len: int = 50,
    band_mult: float = 1.0,
    ema_fast_len: int = 50,
    ema_slow_len: int = 200,
    use_htf_filter: bool = True,
    htf_minutes: int = 240,
    htf_ema_len: int = 50,
    delta_len: int = 20,
    delta_mult: float = 1.35,
    vol_len: int = 20,
    vol_mult: float = 1.10,
    cvd_reset: str = "Daily",
    swing_len: int = 3,
    atr_len: int = 14,
    wick_atr_mult: float = 0.40,
    body_atr_mult: float = 0.25,
    enable_signals: bool = True,
    trade_mode: str = "Intraday",
    min_score: int = 5,
    cooldown_bars: int = 8,
    confirm_on_close: bool = True,
    no_trade_filter: str = "Adaptive",
    block_middle_value: bool = True,
    require_level_touch: bool = True,
    target_1r: float = 1.0,
    target_2r: float = 2.0,
    risk_mode: str = "Structure",
) -> pd.DataFrame:
    """返回成交量分布、VWAP、趋势/订单流/价格行为、信号评分与风险计划。"""
    vp_lookback = integer("vp_lookback", vp_lookback, 30, 500)
    vp_rows = integer("vp_rows", vp_rows, 10, 70)
    value_area_pct = number("value_area_pct", value_area_pct, 50.0, 90.0)
    accept_bars = integer("accept_bars", accept_bars, 1, 10)
    level_tolerance_atr = number("level_tolerance_atr", level_tolerance_atr, 0.05, 2.0)
    perf_mode = choice("perf_mode", perf_mode, ("Auto", "Balanced", "Fast", "Full"))
    vwap_anchor = choice("vwap_anchor", vwap_anchor, ("Daily", "Weekly", "Monthly"))
    band_len = integer("band_len", band_len, 10, 300)
    band_mult = number("band_mult", band_mult, 0.25, 5.0)
    ema_fast_len = integer("ema_fast_len", ema_fast_len, 5, 300)
    ema_slow_len = integer("ema_slow_len", ema_slow_len, 20, 500)
    use_htf_filter = boolean("use_htf_filter", use_htf_filter)
    htf_minutes = integer("htf_minutes", htf_minutes, 1)
    htf_ema_len = integer("htf_ema_len", htf_ema_len, 10, 300)
    delta_len = integer("delta_len", delta_len, 5, 200)
    delta_mult = number("delta_mult", delta_mult, 0.5, 5.0)
    vol_len = integer("vol_len", vol_len, 5, 200)
    vol_mult = number("vol_mult", vol_mult, 0.5, 5.0)
    cvd_reset = choice("cvd_reset", cvd_reset, ("None", "Daily", "Weekly"))
    swing_len = integer("swing_len", swing_len, 1, 20)
    atr_len = integer("atr_len", atr_len, 5, 100)
    wick_atr_mult = number("wick_atr_mult", wick_atr_mult, 0.05, 2.0)
    body_atr_mult = number("body_atr_mult", body_atr_mult, 0.05, 2.0)
    enable_signals = boolean("enable_signals", enable_signals)
    trade_mode = choice("trade_mode", trade_mode, ("Scalp", "Intraday", "Swing"))
    min_score = integer("min_score", min_score, 3, 9)
    cooldown_bars = integer("cooldown_bars", cooldown_bars, 0, 100)
    confirm_on_close = boolean("confirm_on_close", confirm_on_close)
    no_trade_filter = choice("no_trade_filter", no_trade_filter, ("Adaptive", "Middle Value", "Off"))
    block_middle_value = boolean("block_middle_value", block_middle_value)
    require_level_touch = boolean("require_level_touch", require_level_touch)
    target_1r = number("target_1r", target_1r, 0.25, 5.0)
    target_2r = number("target_2r", target_2r, 0.5, 8.0)
    risk_mode = choice("risk_mode", risk_mode, ("Structure", "Hybrid", "Tight"))

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    if "volume" not in frame.columns:
        raise ValueError("缺少行情列: volume")
    volume = pd.to_numeric(frame["volume"], errors="raise").astype(float)

    highs, lows = high.to_numpy(), low.to_numpy()
    opens, closes, volumes = open_.to_numpy(), close.to_numpy(), volume.to_numpy()
    size = len(frame)

    minutes = base_step_minutes(frame)
    resolved = _resolve_perf(perf_mode, minutes)
    effective_lookback = vp_lookback if resolved == "Full" else (
        min(vp_lookback, 100) if resolved == "Fast" else min(vp_lookback, 200))
    effective_rows = vp_rows if resolved == "Full" else (
        min(vp_rows, 24) if resolved == "Fast" else min(vp_rows, 40))

    typical = (high + low + close) / 3.0
    vwap_value = _anchored_vwap(frame, vwap_anchor)
    vwap_dev = stdev(typical - vwap_value, band_len)
    vwap_upper = vwap_value + vwap_dev * band_mult
    vwap_lower = vwap_value - vwap_dev * band_mult

    ema_fast = ema(close, ema_fast_len)
    ema_slow = ema(close, ema_slow_len)
    trend_up = flags((ema_fast > ema_slow) & (close > ema_fast))
    trend_down = flags((ema_fast < ema_slow) & (close < ema_fast))

    if use_htf_filter:
        if htf_minutes % minutes == 0 and htf_minutes >= minutes:
            htf_close = lift_series(frame, close, htf_minutes)
            htf_ema_raw = lift_series(frame, ema(close, htf_ema_len), htf_minutes)
        else:
            htf_close = pd.Series(np.nan, index=frame.index)
            htf_ema_raw = pd.Series(np.nan, index=frame.index)
        htf_bull = flags(htf_close > htf_ema_raw)
        htf_bear = flags(htf_close < htf_ema_raw)
    else:
        htf_bull = pd.Series(False, index=frame.index)
        htf_bear = pd.Series(False, index=frame.index)

    atr_values = atr(frame, atr_len)
    atr_baseline_len = max(atr_len * 5, 20)
    atr_baseline = sma(atr_values, atr_baseline_len)
    atr_ratio = atr_values / atr_baseline.replace(0, np.nan)

    # --- orderflow proxy ----------------------------------------------------
    bar_delta = np.where(closes > opens, volumes, np.where(closes < opens, -volumes, 0.0))
    delta_abs_avg = ema(pd.Series(np.abs(bar_delta), index=frame.index), delta_len)
    avg_volume = sma(volume, vol_len)
    rel_vol = (volume / avg_volume.replace(0, np.nan)).fillna(0.0)
    bull_flow = flags((pd.Series(bar_delta, index=frame.index) > delta_abs_avg * delta_mult)
                      & (volume >= avg_volume * vol_mult))
    bear_flow = flags((pd.Series(bar_delta, index=frame.index) < -delta_abs_avg * delta_mult)
                      & (volume >= avg_volume * vol_mult))

    utc = frame.index.tz_convert("UTC").tz_localize(None)
    reset_rule = {"Daily": "D", "Weekly": "W", "None": None}[cvd_reset]
    cvd = np.zeros(size)
    running = 0.0
    previous_period = None
    for bar in range(size):
        if reset_rule is not None:
            period = utc[bar].to_period(reset_rule)
            if period != previous_period:
                running = 0.0
                previous_period = period
        running += bar_delta[bar]
        cvd[bar] = running
    cvd_series = pd.Series(cvd, index=frame.index)
    cvd_ema = ema(cvd_series, delta_len)
    cvd_bull = flags((cvd_series > cvd_ema) & (cvd_series > cvd_series.shift(1)))
    cvd_bear = flags((cvd_series < cvd_ema) & (cvd_series < cvd_series.shift(1)))

    # --- price action -------------------------------------------------------
    pivot_high = pivothigh(high, swing_len, swing_len).to_numpy()
    pivot_low = pivotlow(low, swing_len, swing_len).to_numpy()
    last_high = np.full(size, np.nan)
    last_low = np.full(size, np.nan)
    current_high = current_low = math.nan
    for bar in range(size):
        if not np.isnan(pivot_high[bar]):
            current_high = pivot_high[bar]
        if not np.isnan(pivot_low[bar]):
            current_low = pivot_low[bar]
        last_high[bar] = current_high
        last_low[bar] = current_low

    previous_close = close.shift(1)
    previous_open = open_.shift(1)
    bos_up = flags((close > pd.Series(last_high, index=frame.index))
                   & (previous_close <= pd.Series(last_high, index=frame.index)))
    bos_down = flags((close < pd.Series(last_low, index=frame.index))
                     & (previous_close >= pd.Series(last_low, index=frame.index)))
    bull_engulfing = flags((close > open_) & (previous_close < previous_open)
                           & (close >= previous_open) & (open_ <= previous_close))
    bear_engulfing = flags((close < open_) & (previous_close > previous_open)
                           & (close <= previous_open) & (open_ >= previous_close))

    lower_wick = np.minimum(opens, closes) - lows
    upper_wick = highs - np.maximum(opens, closes)
    body = np.abs(closes - opens)
    bull_rejection = flags((pd.Series(lower_wick, index=frame.index) >= atr_values * wick_atr_mult)
                           & (close > open_))
    bear_rejection = flags((pd.Series(upper_wick, index=frame.index) >= atr_values * wick_atr_mult)
                           & (close < open_))
    bull_impulse = flags((close > open_) & (pd.Series(body, index=frame.index) >= atr_values * body_atr_mult)
                         & (volume >= avg_volume * vol_mult))
    bear_impulse = flags((close < open_) & (pd.Series(body, index=frame.index) >= atr_values * body_atr_mult)
                         & (volume >= avg_volume * vol_mult))
    bull_pa = bull_engulfing | bull_rejection | bull_impulse | bos_up
    bear_pa = bear_engulfing | bear_rejection | bear_impulse | bos_down

    # --- per-bar state machine ---------------------------------------------
    mode_score, setup_penalty, cooldown_factor, mode_vol = _PRESETS[trade_mode]
    required_score = max(min_score, mode_score)
    setup_score = max(2, required_score - (2 if trade_mode == "Scalp" else 1))
    effective_cooldown = (max(0, int(round(cooldown_bars * 0.5))) if trade_mode == "Scalp"
                          else max(cooldown_bars, int(round(cooldown_bars * 1.5))
                                   if trade_mode == "Swing" else cooldown_bars))
    required_rel_vol = mode_vol if mode_vol is not None else (
        vol_mult * 1.20 if trade_mode == "Swing" else vol_mult)
    low_rel_vol_threshold = 0.85 if trade_mode in ("Scalp", "Swing") else 0.70

    out = {name: np.zeros(size, dtype=bool) for name in
           ("above_value", "below_value", "in_value", "middle_value", "accept_above",
            "accept_below", "touched_vah", "touched_val", "touched_poc", "touched_vwap",
            "val_rejection", "vah_rejection", "poc_cross_up", "poc_cross_down",
            "vwap_cross_up", "vwap_cross_down", "bull_flow", "bear_flow", "cvd_bull",
            "cvd_bear", "bos_up", "bos_down", "bull_engulfing", "bear_engulfing",
            "bull_rejection", "bear_rejection", "bull_impulse", "bear_impulse",
            "trigger_long", "trigger_short", "no_trade_zone", "setup_long", "setup_short",
            "confirmed_long", "confirmed_short", "pending_long", "pending_short",
            "invalidated", "early_warning")}
    for name in ("poc", "vah", "val", "vwap", "vwap_upper", "vwap_lower", "value_position",
                 "atr_ratio", "rel_vol", "bar_delta", "cvd", "score",
                 "active_entry", "active_invalidation", "active_target_1",
                 "active_target_2", "proposed_long_inv", "proposed_short_inv"):
        out[name] = np.full(size, np.nan)
    out["score"] = np.zeros(size, dtype=int)
    out["active_dir"] = np.zeros(size, dtype=int)
    out["signal_grade"] = np.full(size, "", dtype=object)
    out["trigger_text"] = np.full(size, "", dtype=object)
    out["decision"] = np.full(size, "", dtype=object)
    out["block_reason"] = np.full(size, "", dtype=object)
    out["volatility_regime"] = np.full(size, "", dtype=object)
    out["perf_mode"] = pd.Series(resolved, index=frame.index, dtype=object)
    out["ltf_source"] = pd.Series("FALLBACK", index=frame.index, dtype=object)

    active_dir = 0
    active_entry = active_invalidation = active_target_1 = active_target_2 = math.nan
    last_signal_bar = None
    last_warning_bar = None
    previous_warning_candidate = False

    for bar in range(size):
        poc, vah, val, _, _, _ = _volume_profile(highs, lows, closes, volumes, bar,
                                                 effective_lookback, effective_rows,
                                                 value_area_pct)
        out["poc"][bar] = poc
        out["vah"][bar] = vah
        out["val"][bar] = val

        price = closes[bar]
        atr_now = atr_values.iloc[bar]
        tolerance = max(atr_now * level_tolerance_atr if np.isfinite(atr_now) else 0.0, 1e-12)
        value_width = max(vah - val, 1e-12) if np.isfinite(vah) else math.nan
        value_position = (price - val) / value_width if np.isfinite(value_width) else math.nan

        in_value = bool(np.isfinite(vah) and val <= price <= vah)
        above_value = bool(np.isfinite(vah) and price > vah)
        below_value = bool(np.isfinite(val) and price < val)
        middle_value = bool(in_value and np.isfinite(value_position)
                            and 0.35 < value_position < 0.65)
        out["in_value"][bar] = in_value
        out["above_value"][bar] = above_value
        out["below_value"][bar] = below_value
        out["middle_value"][bar] = middle_value
        out["value_position"][bar] = value_position

        window = max(0, bar - accept_bars + 1)
        accept_above = bool(np.isfinite(vah) and np.all(closes[window: bar + 1] - vah > 0))
        accept_below = bool(np.isfinite(val) and np.all(closes[window: bar + 1] - val < 0))
        out["accept_above"][bar] = accept_above
        out["accept_below"][bar] = accept_below

        touched_vah = bool(np.isfinite(vah) and highs[bar] >= vah - tolerance
                           and lows[bar] <= vah + tolerance)
        touched_val = bool(np.isfinite(val) and highs[bar] >= val - tolerance
                           and lows[bar] <= val + tolerance)
        touched_poc = bool(np.isfinite(poc) and highs[bar] >= poc - tolerance
                           and lows[bar] <= poc + tolerance)
        touched_vwap = bool(np.isfinite(vwap_value.iloc[bar])
                            and highs[bar] >= vwap_value.iloc[bar] - tolerance
                            and lows[bar] <= vwap_value.iloc[bar] + tolerance)
        out["touched_vah"][bar] = touched_vah
        out["touched_val"][bar] = touched_val
        out["touched_poc"][bar] = touched_poc
        out["touched_vwap"][bar] = touched_vwap

        near_vah = touched_vah or bool(np.isfinite(vah) and abs(price - vah) <= tolerance)
        near_val = touched_val or bool(np.isfinite(val) and abs(price - val) <= tolerance)
        near_poc = touched_poc or bool(np.isfinite(poc) and abs(price - poc) <= tolerance)
        near_vwap = touched_vwap or bool(np.isfinite(vwap_value.iloc[bar])
                                         and abs(price - vwap_value.iloc[bar]) <= tolerance)
        near_any = near_vah or near_val or near_poc or near_vwap

        val_rejection = bool(np.isfinite(val) and lows[bar] <= val + tolerance
                             and price > val and closes[bar] > opens[bar])
        vah_rejection = bool(np.isfinite(vah) and highs[bar] >= vah - tolerance
                             and price < vah and closes[bar] < opens[bar])
        out["val_rejection"][bar] = val_rejection
        out["vah_rejection"][bar] = vah_rejection

        poc_cross_up = bool(np.isfinite(poc) and bar > 0 and np.isfinite(out["poc"][bar - 1])
                            and price > poc and closes[bar - 1] <= out["poc"][bar - 1])
        poc_cross_down = bool(np.isfinite(poc) and bar > 0 and np.isfinite(out["poc"][bar - 1])
                              and price < poc and closes[bar - 1] >= out["poc"][bar - 1])
        vwap_now = vwap_value.iloc[bar]
        vwap_prev = vwap_value.iloc[bar - 1] if bar else math.nan
        vwap_cross_up = bool(np.isfinite(vwap_now) and np.isfinite(vwap_prev)
                             and price > vwap_now and closes[bar - 1] <= vwap_prev)
        vwap_cross_down = bool(np.isfinite(vwap_now) and np.isfinite(vwap_prev)
                               and price < vwap_now and closes[bar - 1] >= vwap_prev)
        out["poc_cross_up"][bar] = poc_cross_up
        out["poc_cross_down"][bar] = poc_cross_down
        out["vwap_cross_up"][bar] = vwap_cross_up
        out["vwap_cross_down"][bar] = vwap_cross_down
        out["vwap"][bar] = vwap_now
        out["vwap_upper"][bar] = vwap_upper.iloc[bar]
        out["vwap_lower"][bar] = vwap_lower.iloc[bar]

        for name, series in (("bull_flow", bull_flow), ("bear_flow", bear_flow),
                             ("cvd_bull", cvd_bull), ("cvd_bear", cvd_bear),
                             ("bos_up", bos_up), ("bos_down", bos_down),
                             ("bull_engulfing", bull_engulfing), ("bear_engulfing", bear_engulfing),
                             ("bull_rejection", bull_rejection), ("bear_rejection", bear_rejection),
                             ("bull_impulse", bull_impulse), ("bear_impulse", bear_impulse)):
            out[name][bar] = bool(series.iloc[bar])

        out["rel_vol"][bar] = rel_vol.iloc[bar]
        out["bar_delta"][bar] = bar_delta[bar]
        out["cvd"][bar] = cvd[bar]
        out["atr_ratio"][bar] = atr_ratio.iloc[bar]
        ratio = atr_ratio.iloc[bar]
        out["volatility_regime"][bar] = ("NA" if not np.isfinite(ratio)
                                        else "LOW" if ratio < 0.75
                                        else "NORMAL" if ratio < 1.25
                                        else "HIGH" if ratio < 1.75 else "EXTREME")

        # --- scoring ------------------------------------------------------
        score_vwap = 1 if price > vwap_now else -1 if price < vwap_now else 0
        score_value = 2 if above_value else -2 if below_value else 0
        score_trend = 1 if trend_up.iloc[bar] else -1 if trend_down.iloc[bar] else 0
        score_htf = (1 if htf_bull.iloc[bar] else -1 if htf_bear.iloc[bar] else 0) \
            if use_htf_filter else 0
        score_flow = 1 if bull_flow.iloc[bar] else -1 if bear_flow.iloc[bar] else 0
        score_cvd = 1 if cvd_bull.iloc[bar] else -1 if cvd_bear.iloc[bar] else 0
        score_pa = 1 if bull_pa.iloc[bar] else -1 if bear_pa.iloc[bar] else 0
        score_impulse = 1 if bull_impulse.iloc[bar] else -1 if bear_impulse.iloc[bar] else 0
        score = score_vwap + score_value + score_trend + score_htf + score_flow \
            + score_cvd + score_pa + score_impulse
        out["score"][bar] = score

        # --- triggers -----------------------------------------------------
        long_breakout = bool(accept_above and price > vwap_now
                             and (bull_flow.iloc[bar] or cvd_bull.iloc[bar] or bull_impulse.iloc[bar]))
        short_breakdown = bool(accept_below and price < vwap_now
                               and (bear_flow.iloc[bar] or cvd_bear.iloc[bar] or bear_impulse.iloc[bar]))
        long_reclaim = bool((poc_cross_up or vwap_cross_up) and not below_value
                            and (bull_pa.iloc[bar] or bull_flow.iloc[bar] or cvd_bull.iloc[bar]))
        short_reclaim = bool((poc_cross_down or vwap_cross_down) and not above_value
                             and (bear_pa.iloc[bar] or bear_flow.iloc[bar] or cvd_bear.iloc[bar]))
        long_retest = bool((near_val or near_poc or near_vwap) and closes[bar] > opens[bar]
                           and price > closes[bar - 1]
                           and (bull_rejection.iloc[bar] or bull_engulfing.iloc[bar]
                                or bull_flow.iloc[bar] or cvd_bull.iloc[bar]))
        short_retest = bool((near_vah or near_poc or near_vwap) and closes[bar] < opens[bar]
                            and price < closes[bar - 1]
                            and (bear_rejection.iloc[bar] or bear_engulfing.iloc[bar]
                                 or bear_flow.iloc[bar] or cvd_bear.iloc[bar]))
        trigger_long = bool(long_breakout or val_rejection or long_reclaim or long_retest
                            or bos_up.iloc[bar])
        trigger_short = bool(short_breakdown or vah_rejection or short_reclaim or short_retest
                             or bos_down.iloc[bar])
        out["trigger_long"][bar] = trigger_long
        out["trigger_short"][bar] = trigger_short
        out["trigger_text"][bar] = (
            "LONG BREAKOUT" if long_breakout else "VAL REJECT" if val_rejection
            else "LONG RECLAIM" if long_reclaim else "LONG RETEST" if long_retest
            else "BOS UP") if trigger_long else (
            "SHORT BREAKDOWN" if short_breakdown else "VAH REJECT" if vah_rejection
            else "SHORT RECLAIM" if short_reclaim else "SHORT RETEST" if short_retest
            else "BOS DOWN") if trigger_short else "NONE"

        # --- no-trade zone -------------------------------------------------
        middle_block = bool(no_trade_filter != "Off" and block_middle_value and middle_value
                            and not near_poc and not near_vwap)
        rel_vol_now = rel_vol.iloc[bar]
        low_rel_vol_block = bool(no_trade_filter == "Adaptive"
                                 and rel_vol_now < low_rel_vol_threshold and not near_any)
        chop_block = bool(no_trade_filter == "Adaptive" and not trend_up.iloc[bar]
                          and not trend_down.iloc[bar] and middle_value)
        htf_ok_long = bool(not use_htf_filter or htf_bull.iloc[bar]
                           or (trade_mode == "Scalp" and not htf_bear.iloc[bar])
                           or long_breakout or val_rejection)
        htf_ok_short = bool(not use_htf_filter or htf_bear.iloc[bar]
                            or (trade_mode == "Scalp" and not htf_bull.iloc[bar])
                            or short_breakdown or vah_rejection)
        impulse_exception_long = bool(confirm_on_close and trigger_long and not trigger_short
                                      and bull_impulse.iloc[bar] and rel_vol_now >= required_rel_vol
                                      and score >= required_score + 1 and htf_ok_long
                                      and not middle_block and not low_rel_vol_block
                                      and not chop_block)
        impulse_exception_short = bool(confirm_on_close and trigger_short and not trigger_long
                                       and bear_impulse.iloc[bar] and rel_vol_now >= required_rel_vol
                                       and score <= -required_score - 1 and htf_ok_short
                                       and not middle_block and not low_rel_vol_block
                                       and not chop_block)
        away_from_level_block = bool(no_trade_filter == "Adaptive" and require_level_touch
                                     and not near_any and in_value
                                     and not (impulse_exception_long or impulse_exception_short))
        no_trade_zone = bool(no_trade_filter != "Off"
                             and (middle_block or low_rel_vol_block or chop_block
                                  or away_from_level_block))
        out["no_trade_zone"][bar] = no_trade_zone
        out["block_reason"][bar] = (
            "OFF" if no_trade_filter == "Off" else "MIDDLE VALUE" if middle_block
            else "LOW RELVOL" if low_rel_vol_block else "CHOP" if chop_block
            else "AWAY FROM LEVEL" if away_from_level_block else "OPEN")

        # --- setup / confirmation ------------------------------------------
        setup_long = bool(enable_signals and score >= setup_score
                          and (price > vwap_now or above_value or near_val or near_poc or near_vwap)
                          and htf_ok_long and not no_trade_zone)
        setup_short = bool(enable_signals and score <= -setup_score
                           and (price < vwap_now or below_value or near_vah or near_poc or near_vwap)
                           and htf_ok_short and not no_trade_zone)
        out["setup_long"][bar] = setup_long
        out["setup_short"][bar] = setup_short

        level_ok_long = bool(not require_level_touch or near_any or accept_above
                             or poc_cross_up or vwap_cross_up or val_rejection
                             or impulse_exception_long)
        level_ok_short = bool(not require_level_touch or near_any or accept_below
                              or poc_cross_down or vwap_cross_down or vah_rejection
                              or impulse_exception_short)
        if trade_mode == "Scalp":
            quality_long = bool(bull_pa.iloc[bar] or bull_flow.iloc[bar] or cvd_bull.iloc[bar]
                                or bull_impulse.iloc[bar] or long_breakout)
            quality_short = bool(bear_pa.iloc[bar] or bear_flow.iloc[bar] or cvd_bear.iloc[bar]
                                 or bear_impulse.iloc[bar] or short_breakdown)
        elif trade_mode == "Swing":
            quality_long = bool((bull_flow.iloc[bar] or cvd_bull.iloc[bar]) and bull_pa.iloc[bar]
                                and rel_vol_now >= required_rel_vol
                                and (trend_up.iloc[bar] or htf_bull.iloc[bar]))
            quality_short = bool((bear_flow.iloc[bar] or cvd_bear.iloc[bar]) and bear_pa.iloc[bar]
                                 and rel_vol_now >= required_rel_vol
                                 and (trend_down.iloc[bar] or htf_bear.iloc[bar]))
        else:
            quality_long = bool((bull_flow.iloc[bar] or cvd_bull.iloc[bar] or bull_impulse.iloc[bar])
                                and (bull_pa.iloc[bar] or long_breakout or val_rejection
                                     or long_reclaim))
            quality_short = bool((bear_flow.iloc[bar] or cvd_bear.iloc[bar] or bear_impulse.iloc[bar])
                                 and (bear_pa.iloc[bar] or short_breakdown or vah_rejection
                                      or short_reclaim))

        scalp_rel_vol_ok = bool(trade_mode != "Scalp" or rel_vol_now >= required_rel_vol)
        raw_long = bool(setup_long and score >= required_score and trigger_long
                        and level_ok_long and quality_long and scalp_rel_vol_ok)
        raw_short = bool(setup_short and score <= -required_score and trigger_short
                         and level_ok_short and quality_short and scalp_rel_vol_ok)
        cooldown_ok = bool(last_signal_bar is None
                           or bar - last_signal_bar > effective_cooldown)
        raw_confirmed_long = bool(raw_long and cooldown_ok and not raw_short)
        raw_confirmed_short = bool(raw_short and cooldown_ok and not raw_long)
        # Every bar in a historical replay is already closed, so Pine's
        # `barstate.isconfirmed` is always true here: nothing can be merely
        # pending and the realtime-only Spec Watch state never appears.
        confirmed_long = raw_confirmed_long
        confirmed_short = raw_confirmed_short
        pending_long = False
        pending_short = False
        out["confirmed_long"][bar] = confirmed_long
        out["confirmed_short"][bar] = confirmed_short
        out["pending_long"][bar] = pending_long
        out["pending_short"][bar] = pending_short

        # --- early warning -------------------------------------------------
        warning_candidate = bool(enable_signals and cooldown_ok and not no_trade_zone
                                 and not raw_confirmed_long and not raw_confirmed_short
                                 and ((setup_long and (trigger_long or score >= required_score - 1))
                                      or (setup_short and (trigger_short
                                                           or score <= -required_score + 1)))
                                 )
        warning_cooldown = max(1, int(round(max(effective_cooldown, 1) * 0.5)))
        warning_cooldown_ok = bool(last_warning_bar is None
                                   or bar - last_warning_bar > warning_cooldown)
        early_warning = bool(warning_candidate and not previous_warning_candidate
                             and warning_cooldown_ok)
        out["early_warning"][bar] = early_warning
        if early_warning:
            last_warning_bar = bar
        previous_warning_candidate = warning_candidate

        # --- risk plan -----------------------------------------------------
        last_low_now = last_low[bar]
        last_high_now = last_high[bar]
        structure_long_inv = (min(val, last_low_now if np.isfinite(last_low_now) else val)
                              - tolerance)
        structure_short_inv = (max(vah, last_high_now if np.isfinite(last_high_now) else vah)
                               + tolerance)
        tight_long_level = math.nan
        for level in (val, poc, vwap_now, vah):
            if np.isfinite(level) and level < price and (not np.isfinite(tight_long_level)
                                                         or level > tight_long_level):
                tight_long_level = level
        tight_short_level = math.nan
        for level in (vah, poc, vwap_now, val):
            if np.isfinite(level) and level > price and (not np.isfinite(tight_short_level)
                                                         or level < tight_short_level):
                tight_short_level = level
        tight_long_inv = (structure_long_inv if not np.isfinite(tight_long_level)
                          else min(price - 1e-12, tight_long_level - tolerance))
        tight_short_inv = (structure_short_inv if not np.isfinite(tight_short_level)
                           else max(price + 1e-12, tight_short_level + tolerance))
        hybrid_long_inv = max(structure_long_inv, tight_long_inv)
        hybrid_short_inv = min(structure_short_inv, tight_short_inv)
        proposed_long_inv = (tight_long_inv if risk_mode == "Tight"
                             else hybrid_long_inv if risk_mode == "Hybrid"
                             else structure_long_inv)
        proposed_short_inv = (tight_short_inv if risk_mode == "Tight"
                              else hybrid_short_inv if risk_mode == "Hybrid"
                              else structure_short_inv)
        out["proposed_long_inv"][bar] = proposed_long_inv
        out["proposed_short_inv"][bar] = proposed_short_inv

        stop_invalidated = bool((active_dir == 1 and np.isfinite(active_invalidation)
                                 and price < active_invalidation)
                                or (active_dir == -1 and np.isfinite(active_invalidation)
                                    and price > active_invalidation))
        opposite_invalidated = bool((active_dir == 1 and confirmed_short)
                                    or (active_dir == -1 and confirmed_long))
        invalidated = bool(enable_signals and active_dir != 0
                           and (stop_invalidated or opposite_invalidated))
        out["invalidated"][bar] = invalidated
        if invalidated and not confirmed_long and not confirmed_short:
            active_dir = 0
            active_entry = active_invalidation = active_target_1 = active_target_2 = math.nan

        if confirmed_long:
            last_signal_bar = bar
            active_dir = 1
            active_entry = price
            active_invalidation = proposed_long_inv
            risk = max(price - proposed_long_inv, 1e-12)
            active_target_1 = price + risk * target_1r
            active_target_2 = price + risk * target_2r
        elif confirmed_short:
            last_signal_bar = bar
            active_dir = -1
            active_entry = price
            active_invalidation = proposed_short_inv
            risk = max(proposed_short_inv - price, 1e-12)
            active_target_1 = price - risk * target_1r
            active_target_2 = price - risk * target_2r

        out["active_dir"][bar] = active_dir
        out["active_entry"][bar] = active_entry
        out["active_invalidation"][bar] = active_invalidation
        out["active_target_1"][bar] = active_target_1
        out["active_target_2"][bar] = active_target_2

        # --- decision / grade ---------------------------------------------
        grade_long_base = bool(enable_signals and not no_trade_zone and score >= setup_score
                               and level_ok_long and rel_vol_now >= low_rel_vol_threshold
                               and (trigger_long or setup_long))
        grade_short_base = bool(enable_signals and not no_trade_zone and score <= -setup_score
                                and level_ok_short and rel_vol_now >= low_rel_vol_threshold
                                and (trigger_short or setup_short))
        a_plus_long = bool(grade_long_base and score >= required_score + 2 and quality_long
                           and rel_vol_now >= required_rel_vol)
        a_plus_short = bool(grade_short_base and score <= -required_score - 2 and quality_short
                            and rel_vol_now >= required_rel_vol)
        a_long = bool(grade_long_base and score >= required_score and quality_long)
        a_short = bool(grade_short_base and score <= -required_score and quality_short)
        b_long = bool(grade_long_base and score >= setup_score)
        b_short = bool(grade_short_base and score <= -setup_score)
        out["signal_grade"][bar] = (
            "BLOCKED" if (not enable_signals or no_trade_zone) else
            "A+ LONG" if a_plus_long else "A+ SHORT" if a_plus_short else
            "A LONG" if a_long else "A SHORT" if a_short else
            "B LONG" if b_long else "B SHORT" if b_short else "WAIT")
        out["decision"][bar] = (
            "CONFIRMED LONG" if confirmed_long else "CONFIRMED SHORT" if confirmed_short
            else "INVALIDATED" if invalidated else "PENDING LONG" if pending_long
            else "PENDING SHORT" if pending_short else "EARLY WARNING" if early_warning
            else "SETUP LONG" if setup_long else "SETUP SHORT" if setup_short
            else "WAIT / BLOCKED" if no_trade_zone else "WAIT")

    index = frame.index
    data: dict[str, pd.Series] = {
        "vwap": pd.Series(out["vwap"], index=index),
        "vwap_upper": pd.Series(out["vwap_upper"], index=index),
        "vwap_lower": pd.Series(out["vwap_lower"], index=index),
        "poc": pd.Series(out["poc"], index=index),
        "vah": pd.Series(out["vah"], index=index),
        "val": pd.Series(out["val"], index=index),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "htf_bull": htf_bull,
        "htf_bear": htf_bear,
        "atr_ratio": pd.Series(out["atr_ratio"], index=index),
        "volatility_regime": pd.Series(out["volatility_regime"], index=index, dtype=object),
        "value_position": pd.Series(out["value_position"], index=index),
        "in_value": pd.Series(out["in_value"], index=index, dtype=bool),
        "above_value": pd.Series(out["above_value"], index=index, dtype=bool),
        "below_value": pd.Series(out["below_value"], index=index, dtype=bool),
        "middle_value": pd.Series(out["middle_value"], index=index, dtype=bool),
        "accept_above": pd.Series(out["accept_above"], index=index, dtype=bool),
        "accept_below": pd.Series(out["accept_below"], index=index, dtype=bool),
        "touched_vah": pd.Series(out["touched_vah"], index=index, dtype=bool),
        "touched_val": pd.Series(out["touched_val"], index=index, dtype=bool),
        "touched_poc": pd.Series(out["touched_poc"], index=index, dtype=bool),
        "touched_vwap": pd.Series(out["touched_vwap"], index=index, dtype=bool),
        "val_rejection": pd.Series(out["val_rejection"], index=index, dtype=bool),
        "vah_rejection": pd.Series(out["vah_rejection"], index=index, dtype=bool),
        "poc_cross_up": pd.Series(out["poc_cross_up"], index=index, dtype=bool),
        "poc_cross_down": pd.Series(out["poc_cross_down"], index=index, dtype=bool),
        "vwap_cross_up": pd.Series(out["vwap_cross_up"], index=index, dtype=bool),
        "vwap_cross_down": pd.Series(out["vwap_cross_down"], index=index, dtype=bool),
        "bar_delta": pd.Series(out["bar_delta"], index=index),
        "cvd": pd.Series(out["cvd"], index=index),
        "rel_vol": pd.Series(out["rel_vol"], index=index),
        "bull_flow": pd.Series(out["bull_flow"], index=index, dtype=bool),
        "bear_flow": pd.Series(out["bear_flow"], index=index, dtype=bool),
        "cvd_bull": pd.Series(out["cvd_bull"], index=index, dtype=bool),
        "cvd_bear": pd.Series(out["cvd_bear"], index=index, dtype=bool),
        "bos_up": pd.Series(out["bos_up"], index=index, dtype=bool),
        "bos_down": pd.Series(out["bos_down"], index=index, dtype=bool),
        "bull_engulfing": pd.Series(out["bull_engulfing"], index=index, dtype=bool),
        "bear_engulfing": pd.Series(out["bear_engulfing"], index=index, dtype=bool),
        "bull_rejection": pd.Series(out["bull_rejection"], index=index, dtype=bool),
        "bear_rejection": pd.Series(out["bear_rejection"], index=index, dtype=bool),
        "bull_impulse": pd.Series(out["bull_impulse"], index=index, dtype=bool),
        "bear_impulse": pd.Series(out["bear_impulse"], index=index, dtype=bool),
        "trigger_long": pd.Series(out["trigger_long"], index=index, dtype=bool),
        "trigger_short": pd.Series(out["trigger_short"], index=index, dtype=bool),
        "trigger_text": pd.Series(out["trigger_text"], index=index, dtype=object),
        "no_trade_zone": pd.Series(out["no_trade_zone"], index=index, dtype=bool),
        "block_reason": pd.Series(out["block_reason"], index=index, dtype=object),
        "setup_long": pd.Series(out["setup_long"], index=index, dtype=bool),
        "setup_short": pd.Series(out["setup_short"], index=index, dtype=bool),
        "confirmed_long": pd.Series(out["confirmed_long"], index=index, dtype=bool),
        "confirmed_short": pd.Series(out["confirmed_short"], index=index, dtype=bool),
        "pending_long": pd.Series(out["pending_long"], index=index, dtype=bool),
        "pending_short": pd.Series(out["pending_short"], index=index, dtype=bool),
        "early_warning": pd.Series(out["early_warning"], index=index, dtype=bool),
        "invalidated": pd.Series(out["invalidated"], index=index, dtype=bool),
        "score": pd.Series(out["score"], index=index).astype("int64"),
        "required_score": pd.Series(required_score, index=index).astype("int64"),
        "setup_score": pd.Series(setup_score, index=index).astype("int64"),
        "signal_grade": pd.Series(out["signal_grade"], index=index, dtype=object),
        "decision": pd.Series(out["decision"], index=index, dtype=object),
        "active_dir": pd.Series(out["active_dir"], index=index).astype("int64"),
        "active_entry": pd.Series(out["active_entry"], index=index),
        "active_invalidation": pd.Series(out["active_invalidation"], index=index),
        "active_target_1": pd.Series(out["active_target_1"], index=index),
        "active_target_2": pd.Series(out["active_target_2"], index=index),
        "proposed_long_inv": pd.Series(out["proposed_long_inv"], index=index),
        "proposed_short_inv": pd.Series(out["proposed_short_inv"], index=index),
        "perf_mode": pd.Series(out["perf_mode"], index=index, dtype=object),
        "ltf_source": pd.Series(out["ltf_source"], index=index, dtype=object),
    }
    return pd.DataFrame(data, index=index)
