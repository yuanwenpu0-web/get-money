"""Exhaustion Scanner [WavesUnchained] — 顶部/底部衰竭评分与信号。

三角色评分（预设权重：Stocks 35/35/30、Commodities 40/35/25、Forex 45/30/25
且无流量、Crypto 35/40/25、Custom 自定）：

- **Stretch**：0.45×max(|close−EMA50|/ATR, |BB z|/2.5) + 0.35×近 80 根高/低
  + 0.20×出带。
- **Exhaustion**：0.55×动量核（RSI/StochRSI/WaveTrend 越界加权）+ 0.25×流量
  （MFI + 量能尖峰；无量能或预设关流量时 0.70 动量 + 0.30 背离）+ 0.20×
  WT pivot 背离激活（12 根内且未破极值价）。
- **Reaction**：0.45×reclaim-fail（刺出带/前高后收回）+ 0.30×WT 极值交叉
  + 0.25×弱收盘长影线。

总分 = clamp((stretch·wS + exhaust·wE + react·wR)/Σw × 波动因子, 0, 100)，
波动因子 = 0.85 + 0.30×volScore（ATR 扩张 + K 线幅度）。
阈值 = base(Early 30/Normal 40/Conservative 50) + 逆势惩罚 12 − 背离减让 8，
钳 20..80。信号：Normal 模式要求 score≥阈值 且当根有反应；同侧信号间隔
≥ `min_signal_gap`。regime（ADX≥25 + EMA 堆叠）决定标签类型。

丢弃：背景热图/区域框/标签绘图。

Source: pine-collection/indicators/momentum/exhaustion_scanner/exhaustion_scanner.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    crossover,
    crossunder,
    dmi,
    ema,
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
    stoch,
    stdev,
)

__all__ = ["exhaustion_scanner"]

_PRESETS = {"Stocks": (35, 35, 30, True),
            "Commodities": (40, 35, 25, True),
            "Forex/CFD": (45, 30, 25, False),
            "Crypto": (35, 40, 25, True)}


def exhaustion_scanner(
    frame: pd.DataFrame,
    *,
    preset: str = "Commodities",
    signal_mode: str = "Normal",
    confirm_mode: str = "Candle Reaction",
    min_signal_gap: int = 8,
    use_divergence_assist: bool = False,
    regime_fast: int = 20,
    regime_slow: int = 100,
    adx_len: int = 14,
    adx_trend: float = 25.0,
    regime_penalty: float = 12.0,
    ema_len: int = 50,
    atr_len: int = 14,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    rsi_len: int = 14,
    stoch_len: int = 14,
    wt_n1: int = 10,
    wt_n2: int = 21,
    pivot_len: int = 5,
    div_memory: int = 12,
    div_assist: float = 8.0,
    structure_len: int = 80,
    near_atr_mult: float = 0.75,
    custom_stretch: float = 40.0,
    custom_exhaust: float = 35.0,
    custom_react: float = 25.0,
    custom_use_flow: bool = True,
) -> pd.DataFrame:
    """计算衰竭评分、阈值与信号。"""
    preset = choice("preset", preset,
                    ("Stocks", "Commodities", "Forex/CFD", "Crypto", "Custom"))
    signal_mode = choice("signal_mode", signal_mode,
                         ("Early", "Normal", "Conservative"))
    confirm_mode = choice("confirm_mode", confirm_mode,
                          ("Score Only", "Candle Reaction"))
    min_signal_gap = integer("min_signal_gap", min_signal_gap, 0)
    use_divergence_assist = boolean("use_divergence_assist", use_divergence_assist)
    regime_fast = integer("regime_fast", regime_fast, 5)
    regime_slow = integer("regime_slow", regime_slow, 20)
    adx_len = integer("adx_len", adx_len, 5)
    adx_trend = number("adx_trend", adx_trend, 1.0)
    regime_penalty = number("regime_penalty", regime_penalty, 0.0)
    ema_len = integer("ema_len", ema_len, 10)
    atr_len = integer("atr_len", atr_len, 5)
    bb_len = integer("bb_len", bb_len, 10)
    bb_mult = number("bb_mult", bb_mult, 0.5)
    rsi_len = integer("rsi_len", rsi_len, 5)
    stoch_len = integer("stoch_len", stoch_len, 5)
    wt_n1 = integer("wt_n1", wt_n1, 3)
    wt_n2 = integer("wt_n2", wt_n2, 5)
    pivot_len = integer("pivot_len", pivot_len, 2)
    div_memory = integer("div_memory", div_memory, 1)
    div_assist = number("div_assist", div_assist, 0.0)
    structure_len = integer("structure_len", structure_len, 20)
    near_atr_mult = number("near_atr_mult", near_atr_mult, 0.05)
    custom_stretch = number("custom_stretch", custom_stretch, 0.0)
    custom_exhaust = number("custom_exhaust", custom_exhaust, 0.0)
    custom_react = number("custom_react", custom_react, 0.0)
    custom_use_flow = boolean("custom_use_flow", custom_use_flow)

    if preset == "Custom":
        w_stretch, w_exhaust, w_react, use_flow = (custom_stretch, custom_exhaust,
                                                   custom_react, custom_use_flow)
    else:
        w_stretch, w_exhaust, w_react, use_flow = _PRESETS[preset]
    role_sum = max(w_stretch + w_exhaust + w_react, 1.0)

    high, low, open_, close, volume = ohlcv(frame, "high", "low", "open", "close",
                                            "volume")
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    vols = volume.to_numpy(dtype=float)
    n = len(frame)
    index = frame.index

    atr_series = atr(frame, atr_len)
    atr_arr = atr_series.to_numpy(dtype=float)
    ema_arr = ema(close, ema_len).to_numpy(dtype=float)
    basis = sma(close, bb_len).to_numpy(dtype=float)
    dev = stdev(close, bb_len).to_numpy(dtype=float)
    upper = basis + dev * bb_mult
    lower = basis - dev * bb_mult
    rsi_arr = rsi(close, rsi_len).to_numpy(dtype=float)
    stoch_raw = stoch(pd.Series(rsi_arr, index=index), pd.Series(rsi_arr, index=index),
                      pd.Series(rsi_arr, index=index), stoch_len)
    stoch_arr = sma(stoch_raw, 3).to_numpy(dtype=float)

    ap = (high + low + close) / 3.0
    esa = ema(ap, wt_n1)
    de = ema((ap - esa).abs(), wt_n1)
    ci = (ap - esa) / (0.015 * de.where(de > 0))
    wt1 = ema(ci, wt_n2)
    wt2 = sma(wt1, 4)
    wt1_arr = wt1.to_numpy(dtype=float)
    wt2_arr = wt2.to_numpy(dtype=float)

    vol_ma = sma(volume, 30).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_spike = np.where(vol_ma > 0, vols / np.where(vol_ma > 0, vol_ma, 1.0), 0.0)
    has_real_volume = np.nancumsum(np.nan_to_num(vols)) > 0
    mfi_arr = mfi(frame, 14).fillna(50.0).to_numpy(dtype=float)
    atr_ma = sma(atr_series, 50).to_numpy(dtype=float)
    atr_expansion = np.where(atr_ma > 0, atr_arr / np.where(atr_ma > 0, atr_ma, 1.0), 0.0)
    range_atr = np.where(atr_arr > 0, (highs - lows) / np.where(atr_arr > 0, atr_arr, 1.0), 0.0)

    recent_high = highest(high, structure_len).to_numpy(dtype=float)
    recent_low = lowest(low, structure_len).to_numpy(dtype=float)
    ema_fast = ema(close, regime_fast).to_numpy(dtype=float)
    ema_slow = ema(close, regime_slow).to_numpy(dtype=float)
    _, _, adx_arr = dmi(frame, adx_len)
    adx_arr = adx_arr.to_numpy(dtype=float)

    ph_arr = pivothigh(high, pivot_len, pivot_len).to_numpy(dtype=float)
    pl_arr = pivotlow(low, pivot_len, pivot_len).to_numpy(dtype=float)
    bar_range = highs - lows
    with np.errstate(divide="ignore", invalid="ignore"):
        upper_wick = np.where(bar_range > 0, (highs - np.maximum(opens, closes)) / np.where(bar_range > 0, bar_range, 1.0), 0.0)
        lower_wick = np.where(bar_range > 0, (np.minimum(opens, closes) - lows) / np.where(bar_range > 0, bar_range, 1.0), 0.0)

    out = {
        "top_score": np.full(n, np.nan),
        "bot_score": np.full(n, np.nan),
        "top_threshold": np.full(n, np.nan),
        "bot_threshold": np.full(n, np.nan),
        "top_extreme": np.zeros(n, dtype=bool),
        "bot_extreme": np.zeros(n, dtype=bool),
        "top_setup": np.zeros(n, dtype=bool),
        "bot_setup": np.zeros(n, dtype=bool),
        "top_signal": np.zeros(n, dtype=bool),
        "bot_signal": np.zeros(n, dtype=bool),
        "top_reason": np.full(n, "", dtype=object),
        "bot_reason": np.full(n, "", dtype=object),
        "top_kind": np.full(n, "", dtype=object),
        "bot_kind": np.full(n, "", dtype=object),
        "regime": np.full(n, "Range", dtype=object),
        "stretch_top": np.full(n, np.nan),
        "stretch_bot": np.full(n, np.nan),
        "exhaust_top": np.full(n, np.nan),
        "exhaust_bot": np.full(n, np.nan),
        "react_top": np.full(n, np.nan),
        "react_bot": np.full(n, np.nan),
        "bear_div_active": np.zeros(n, dtype=bool),
        "bull_div_active": np.zeros(n, dtype=bool),
    }

    prev_ph = prev_pl = np.nan
    prev_wt_at_ph = prev_wt_at_pl = np.nan
    bear_div_pivot = bull_div_pivot = np.nan
    bars_since_bear_div = bars_since_bull_div = 10 ** 9
    last_top_signal = last_bot_signal = -10 ** 9
    top_extreme_prev = bot_extreme_prev = top_setup_prev = bot_setup_prev = False

    def score01(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, 1.0) * 100.0

    def safe_div(a: np.ndarray, b: float) -> np.ndarray:
        return np.where(b == 0.0, 0.0, a / b) if np.isscalar(b) else np.divide(
            a, b, out=np.zeros_like(a), where=b != 0)

    for bar in range(n):
        if np.isnan(atr_arr[bar]) or np.isnan(dev[bar]) or dev[bar] == 0:
            continue
        dist_atr = (closes[bar] - ema_arr[bar]) / atr_arr[bar]
        bb_z = (closes[bar] - basis[bar]) / dev[bar]
        near_high = recent_high[bar] - closes[bar] <= atr_arr[bar] * near_atr_mult
        near_low = closes[bar] - recent_low[bar] <= atr_arr[bar] * near_atr_mult

        stretch_top = score01(np.array([
            0.45 * max(dist_atr / 3.0, bb_z / 2.5)
            + 0.35 * (1.0 if near_high else 0.0)
            + 0.20 * (1.0 if closes[bar] > upper[bar] else 0.0)]))[0]
        stretch_bot = score01(np.array([
            0.45 * max(-dist_atr / 3.0, -bb_z / 2.5)
            + 0.35 * (1.0 if near_low else 0.0)
            + 0.20 * (1.0 if closes[bar] < lower[bar] else 0.0)]))[0]

        mom_core_top = score01(np.array([
            0.40 * (rsi_arr[bar] - 60) / 20.0
            + 0.30 * (stoch_arr[bar] - 75) / 25.0
            + 0.30 * (wt1_arr[bar] - 45) / 25.0]))[0]
        mom_core_bot = score01(np.array([
            0.40 * (40 - rsi_arr[bar]) / 20.0
            + 0.30 * (25 - stoch_arr[bar]) / 25.0
            + 0.30 * (-45 - wt1_arr[bar]) / 25.0]))[0]

        # WT pivot 背离。
        bear_div = bull_div = False
        if not np.isnan(ph_arr[bar]):
            wt_at_pivot = wt1_arr[bar - pivot_len] if bar >= pivot_len else np.nan
            if not np.isnan(prev_ph) and not np.isnan(prev_wt_at_ph) \
                    and not np.isnan(wt_at_pivot) \
                    and ph_arr[bar] > prev_ph and wt_at_pivot < prev_wt_at_ph:
                bear_div = True
                bear_div_pivot = highs[bar - pivot_len]
            prev_ph = ph_arr[bar]
            prev_wt_at_ph = wt_at_pivot
        if not np.isnan(pl_arr[bar]):
            wt_at_pivot = wt1_arr[bar - pivot_len] if bar >= pivot_len else np.nan
            if not np.isnan(prev_pl) and not np.isnan(prev_wt_at_pl) \
                    and not np.isnan(wt_at_pivot) \
                    and pl_arr[bar] < prev_pl and wt_at_pivot > prev_wt_at_pl:
                bull_div = True
                bull_div_pivot = lows[bar - pivot_len]
            prev_pl = pl_arr[bar]
            prev_wt_at_pl = wt_at_pivot
        bars_since_bear_div = 0 if bear_div else bars_since_bear_div + 1
        bars_since_bull_div = 0 if bull_div else bars_since_bull_div + 1
        bear_div_active = (bars_since_bear_div <= div_memory
                           and not np.isnan(bear_div_pivot)
                           and highs[bar] <= bear_div_pivot)
        bull_div_active = (bars_since_bull_div <= div_memory
                           and not np.isnan(bull_div_pivot)
                           and lows[bar] >= bull_div_pivot)

        use_flow_final = use_flow and has_real_volume[bar] and vol_ma[bar] > 0
        if use_flow_final:
            mfi_v = mfi_arr[bar]
            flow_top = score01(np.array([
                0.55 * (mfi_v - 75) / 20.0
                + 0.45 * (vol_spike[bar] - 1.5) / 1.5]))[0]
            flow_bot = score01(np.array([
                0.55 * (25 - mfi_v) / 20.0
                + 0.45 * (vol_spike[bar] - 1.5) / 1.5]))[0]
            exhaust_top = 0.55 * mom_core_top + 0.25 * flow_top + 0.20 * (100.0 if bear_div_active else 0.0)
            exhaust_bot = 0.55 * mom_core_bot + 0.25 * flow_bot + 0.20 * (100.0 if bull_div_active else 0.0)
        else:
            exhaust_top = 0.70 * mom_core_top + 0.30 * (100.0 if bear_div_active else 0.0)
            exhaust_bot = 0.70 * mom_core_bot + 0.30 * (100.0 if bull_div_active else 0.0)

        prior_high = recent_high[bar - 1] if bar > 0 else recent_high[bar]
        prior_low = recent_low[bar - 1] if bar > 0 else recent_low[bar]
        top_reclaim_fail = ((highs[bar] > upper[bar] and closes[bar] < upper[bar])
                            or (highs[bar] > prior_high and closes[bar] < prior_high))
        bot_reclaim_fail = ((lows[bar] < lower[bar] and closes[bar] > lower[bar])
                            or (lows[bar] < prior_low and closes[bar] > prior_low))
        wt_cross_up = (wt1_arr[bar] > wt2_arr[bar]
                       and (bar == 0 or wt1_arr[bar - 1] <= wt2_arr[bar - 1]))
        wt_cross_dn = (wt1_arr[bar] < wt2_arr[bar]
                       and (bar == 0 or wt1_arr[bar - 1] >= wt2_arr[bar - 1]))
        wt_was_high = (np.nanmax(wt1_arr[max(0, bar - 2):bar + 1]) >= 45)
        wt_was_low = (np.nanmin(wt1_arr[max(0, bar - 2):bar + 1]) <= -45)
        top_wt_trigger = wt_cross_dn and wt_was_high
        bot_wt_trigger = wt_cross_up and wt_was_low
        hl2 = (highs[bar] + lows[bar]) / 2.0
        top_close_weak = closes[bar] < opens[bar] and closes[bar] < hl2
        bot_close_weak = closes[bar] > opens[bar] and closes[bar] > hl2
        top_wick_rej = top_close_weak and upper_wick[bar] >= 0.35
        bot_wick_rej = bot_close_weak and lower_wick[bar] >= 0.35
        react_top = score01(np.array([
            0.45 * (1.0 if top_reclaim_fail else 0.0)
            + 0.30 * (1.0 if top_wt_trigger else 0.0)
            + 0.25 * ((upper_wick[bar] - 0.35) / 0.35 if top_close_weak else 0.0)]))[0]
        react_bot = score01(np.array([
            0.45 * (1.0 if bot_reclaim_fail else 0.0)
            + 0.30 * (1.0 if bot_wt_trigger else 0.0)
            + 0.25 * ((lower_wick[bar] - 0.35) / 0.35 if bot_close_weak else 0.0)]))[0]
        top_reacted = top_reclaim_fail or top_wt_trigger or top_wick_rej
        bot_reacted = bot_reclaim_fail or bot_wt_trigger or bot_wick_rej

        vol_score = score01(np.array([
            0.55 * (atr_expansion[bar] - 1.1) / 0.7
            + 0.45 * (range_atr[bar] - 1.2) / 1.2]))[0]
        vol_factor = 0.85 + vol_score / 100.0 * 0.30
        top_score = min(max((stretch_top * w_stretch + exhaust_top * w_exhaust
                             + react_top * w_react) / role_sum * vol_factor, 0.0), 100.0)
        bot_score = min(max((stretch_bot * w_stretch + exhaust_bot * w_exhaust
                             + react_bot * w_react) / role_sum * vol_factor, 0.0), 100.0)

        trend_up = ema_fast[bar] > ema_slow[bar] and closes[bar] > ema_slow[bar]
        trend_dn = ema_fast[bar] < ema_slow[bar] and closes[bar] < ema_slow[bar]
        strong_trend = not np.isnan(adx_arr[bar]) and adx_arr[bar] >= adx_trend
        regime_up = strong_trend and trend_up
        regime_dn = strong_trend and trend_dn
        regime = "Trend ↑" if regime_up else "Trend ↓" if regime_dn else "Range"

        base_thresh = {"Early": 30, "Normal": 40, "Conservative": 50}[signal_mode]
        top_thresh = min(max(base_thresh + (regime_penalty if regime_up else 0.0)
                             - (div_assist if use_divergence_assist and bear_div_active else 0.0),
                             20.0), 80.0)
        bot_thresh = min(max(base_thresh + (regime_penalty if regime_dn else 0.0)
                             - (div_assist if use_divergence_assist and bull_div_active else 0.0),
                             20.0), 80.0)
        top_extreme = top_score >= top_thresh
        bot_extreme = bot_score >= bot_thresh
        top_setup = top_score >= top_thresh - 10
        bot_setup = bot_score >= bot_thresh - 10

        if confirm_mode == "Score Only":
            top_raw = top_extreme and not top_extreme_prev
            bot_raw = bot_extreme and not bot_extreme_prev
        else:
            top_raw = top_extreme and top_reacted
            bot_raw = bot_extreme and bot_reacted
        top_gap_ok = (min_signal_gap == 0 or bar - last_top_signal > min_signal_gap)
        bot_gap_ok = (min_signal_gap == 0 or bar - last_bot_signal > min_signal_gap)
        top_signal = top_raw and top_gap_ok
        bot_signal = bot_raw and bot_gap_ok
        if top_signal:
            last_top_signal = bar
        if bot_signal:
            last_bot_signal = bar

        def reason(stretch_s, exhaust_s, react_s, div_on):
            s_c = stretch_s * w_stretch
            e_c = exhaust_s * w_exhaust
            r_c = react_s * w_react
            if div_on:
                return "Divergence"
            if r_c >= e_c and r_c >= s_c:
                return "Rejection"
            return "Stretch" if s_c >= e_c else "Momentum"

        top_kind = ("CONTINUATION RISK" if regime_up else
                    "TREND EXHAUSTION" if regime_dn else "RANGE FADE")
        bot_kind = ("CONTINUATION RISK" if regime_dn else
                    "TREND EXHAUSTION" if regime_up else "RANGE FADE")

        out["top_score"][bar] = top_score
        out["bot_score"][bar] = bot_score
        out["top_threshold"][bar] = top_thresh
        out["bot_threshold"][bar] = bot_thresh
        out["top_extreme"][bar] = top_extreme
        out["bot_extreme"][bar] = bot_extreme
        out["top_setup"][bar] = top_setup
        out["bot_setup"][bar] = bot_setup
        out["top_signal"][bar] = top_signal
        out["bot_signal"][bar] = bot_signal
        out["top_reason"][bar] = reason(stretch_top, exhaust_top, react_top, bear_div_active)
        out["bot_reason"][bar] = reason(stretch_bot, exhaust_bot, react_bot, bull_div_active)
        out["top_kind"][bar] = top_kind
        out["bot_kind"][bar] = bot_kind
        out["regime"][bar] = regime
        out["stretch_top"][bar] = stretch_top
        out["stretch_bot"][bar] = stretch_bot
        out["exhaust_top"][bar] = exhaust_top
        out["exhaust_bot"][bar] = exhaust_bot
        out["react_top"][bar] = react_top
        out["react_bot"][bar] = react_bot
        out["bear_div_active"][bar] = bear_div_active
        out["bull_div_active"][bar] = bull_div_active
        top_extreme_prev, bot_extreme_prev = top_extreme, bot_extreme
        top_setup_prev, bot_setup_prev = top_setup, bot_setup

    data: dict[str, pd.Series] = {
        "top_score": pd.Series(out["top_score"], index=index),
        "bot_score": pd.Series(out["bot_score"], index=index),
        "top_threshold": pd.Series(out["top_threshold"], index=index),
        "bot_threshold": pd.Series(out["bot_threshold"], index=index),
        "top_reason": pd.Series(out["top_reason"], index=index, dtype=object),
        "bot_reason": pd.Series(out["bot_reason"], index=index, dtype=object),
        "top_kind": pd.Series(out["top_kind"], index=index, dtype=object),
        "bot_kind": pd.Series(out["bot_kind"], index=index, dtype=object),
        "regime": pd.Series(out["regime"], index=index, dtype=object),
        "stretch_top": pd.Series(out["stretch_top"], index=index),
        "stretch_bot": pd.Series(out["stretch_bot"], index=index),
        "exhaust_top": pd.Series(out["exhaust_top"], index=index),
        "exhaust_bot": pd.Series(out["exhaust_bot"], index=index),
        "react_top": pd.Series(out["react_top"], index=index),
        "react_bot": pd.Series(out["react_bot"], index=index),
        "bear_div_active": flags(pd.Series(out["bear_div_active"], index=index)),
        "bull_div_active": flags(pd.Series(out["bull_div_active"], index=index)),
    }
    for column in ("top_extreme", "bot_extreme", "top_setup", "bot_setup",
                   "top_signal", "bot_signal"):
        data[column] = flags(pd.Series(out[column], index=index))
    return pd.DataFrame(data, index=index)
