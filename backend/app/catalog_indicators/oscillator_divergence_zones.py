"""Oscillator Divergence Zones [WavesUnchained] — 九种振荡器的背离区。

**振荡器**（`osc_type`）：RSI / CCI / MFI / Fisher / TSI / STC / DPO / Roofing /
Cyber。自动源：RSI/STC/DPO/TSI 用 close，Fisher/Roofing/Cyber 用 hl2，
CCI/MFI 用 hlc3。

**背离检测**：pivot(pivLeft, pivRight) 在价格上找 swing；对每个新 pivot 取
最近 3 个历史同侧 pivot，逐一比较：

- 常规看涨：当前 low 更低 且 振荡器极值更高（`osc_low_bar` 必须新于前值）。
- 隐藏看涨：当前 low 更高 且 振荡器极值更低。
- 看跌镜像。间距需在 [min_range, max_range]。

**过滤**：
- 水平过滤（Off / Fixed 按 OB-OS 区间百分比 / Dynamic Zones 用分位阈值）。
- StdDev 过滤（常规背离要求振荡器极值在 mean±N×std 之外）。
- 隐藏背离可选趋势过滤（bull 需在 EMA 之上、bear 之下）。

**质量分 0..100**：间距分 + 幅度分（背离幅度/σ）+ 极值分（距 OB/OS 边界）
+ 趋势分（价格距 EMA 的 ATR 倍数），四项均值。

**区**：以 pivot 价为中心、半宽 `atr×zone_factor/2`；被穿透（可选 close）即
标记移除；价格再次进入 touches+1；超 `max_zones` 淘汰最旧。

丢弃：全部 line/box/label 绘图、dashboard、debug pivot 标记。

Source: pine-collection/indicators/momentum/oscillator_divergence_zones/oscillator_divergence_zones.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    cci,
    choice,
    ema,
    flags,
    highest,
    integer,
    lowest,
    mfi,
    number,
    ohlcv,
    percentile_nearest_rank,
    pivothigh,
    pivotlow,
    rsi,
    sma,
    stdev,
)

__all__ = ["oscillator_divergence_zones"]

_OSC_TYPES = ("RSI", "CCI", "MFI", "Fisher", "TSI", "STC", "DPO", "Roofing", "Cyber")
# 自动 OB/OS 水平（有界振荡器）。
_AUTO_LEVELS = {"RSI": (70.0, 30.0), "CCI": (100.0, -100.0), "MFI": (80.0, 20.0),
                "Fisher": (1.5, -1.5), "TSI": (25.0, -25.0), "STC": (75.0, 25.0)}


def _fisher(src: pd.Series, length: int) -> pd.Series:
    high = src.rolling(length).max()
    low = src.rolling(length).min()
    rng = high - low
    norm = ((src - low) / rng.where(rng > 0) - 0.5).fillna(0.0)
    values = np.zeros(len(src))
    fish = np.zeros(len(src))
    for i in range(len(src)):
        n_i = norm.iloc[i]
        prev_v = values[i - 1] if i > 0 else 0.0
        v = 0.66 * n_i + 0.67 * prev_v
        v = min(max(v, -0.999), 0.999)
        values[i] = v
        prev_f = fish[i - 1] if i > 0 else 0.0
        fish[i] = 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * prev_f
    return pd.Series(fish, index=src.index)


def _tsi(close: pd.Series, long_len: int, short_len: int) -> pd.Series:
    change = close.diff()
    smooth = ema(ema(change, long_len), short_len)
    smooth_abs = ema(ema(change.abs(), long_len), short_len)
    return (100.0 * smooth / smooth_abs.where(smooth_abs != 0)).fillna(0.0)


def _stc(close: pd.Series, fast: int, slow: int, cycle: int) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    ll1 = macd.rolling(cycle).min()
    hh1 = macd.rolling(cycle).max() - ll1
    f1 = ((macd - ll1) / hh1.where(hh1 > 0) * 100.0).fillna(0.0)
    d1 = f1.ewm(alpha=0.5, adjust=False).mean()
    ll2 = d1.rolling(cycle).min()
    hh2 = d1.rolling(cycle).max() - ll2
    f2 = ((d1 - ll2) / hh2.where(hh2 > 0) * 100.0).fillna(0.0)
    return f2.ewm(alpha=0.5, adjust=False).mean()


def _roofing(src: pd.Series, hp_len: int, ss_len: int) -> pd.Series:
    pi = 2.0 * np.arcsin(1.0)
    a1 = ((np.cos(0.707 * 2.0 * pi / hp_len)
           + np.sin(0.707 * 2.0 * pi / hp_len) - 1.0)
          / np.cos(0.707 * 2.0 * pi / hp_len))
    s = src.to_numpy(dtype=float)
    n = len(s)
    hp = np.zeros(n)
    for i in range(n):
        s1 = s[i - 1] if i >= 1 else 0.0
        s2 = s[i - 2] if i >= 2 else 0.0
        hp1 = hp[i - 1] if i >= 1 else 0.0
        hp2 = hp[i - 2] if i >= 2 else 0.0
        hp[i] = ((1.0 - a1 / 2.0) ** 2 * (s[i] - 2.0 * s1 + s2)
                 + 2.0 * (1.0 - a1) * hp1 - (1.0 - a1) ** 2 * hp2)
    a2 = np.exp(-1.414 * pi / ss_len)
    b2 = 2.0 * a2 * np.cos(1.414 * pi / ss_len)
    c2, c3 = b2, -a2 * a2
    c1 = 1.0 - c2 - c3
    out = np.zeros(n)
    for i in range(n):
        hp1 = hp[i - 1] if i >= 1 else 0.0
        o1 = out[i - 1] if i >= 1 else 0.0
        o2 = out[i - 2] if i >= 2 else 0.0
        out[i] = c1 * (hp[i] + hp1) / 2.0 + c2 * o1 + c3 * o2
    return pd.Series(out, index=src.index)


def _cyber(src: pd.Series, alpha: float) -> pd.Series:
    s = src.to_numpy(dtype=float)
    n = len(s)
    smooth = np.zeros(n)
    for i in range(n):
        s1 = s[i - 1] if i >= 1 else 0.0
        s2 = s[i - 2] if i >= 2 else 0.0
        s3 = s[i - 3] if i >= 3 else 0.0
        smooth[i] = (s[i] + 2.0 * s1 + 2.0 * s2 + s3) / 6.0
    out = np.zeros(n)
    for i in range(n):
        if i < 7:
            s1 = s[i - 1] if i >= 1 else 0.0
            s2 = s[i - 2] if i >= 2 else 0.0
            out[i] = (s[i] - 2.0 * s1 + s2) / 4.0
        else:
            sm1 = smooth[i - 1]
            sm2 = smooth[i - 2]
            o1 = out[i - 1]
            o2 = out[i - 2]
            out[i] = ((1.0 - 0.5 * alpha) ** 2 * (smooth[i] - 2.0 * sm1 + sm2)
                      + 2.0 * (1.0 - alpha) * o1
                      - (1.0 - alpha) ** 2 * o2)
    return pd.Series(out, index=src.index)


def oscillator_divergence_zones(
    frame: pd.DataFrame,
    *,
    osc_type: str = "RSI",
    auto_source: bool = True,
    osc_len: int = 14,
    tsi_short_len: int = 13,
    stc_fast: int = 23,
    stc_slow: int = 50,
    hp_len: int = 48,
    ss_len: int = 10,
    cyber_alpha: float = 0.07,
    auto_levels: bool = True,
    ob_level: float = 70.0,
    os_level: float = 30.0,
    piv_left: int = 7,
    piv_right: int = 7,
    min_range: int = 10,
    max_range: int = 100,
    show_hidden: bool = False,
    filter_mode: str = "Fixed",
    bull_filter_pct: float = 75.0,
    bear_filter_pct: float = 25.0,
    dyn_sample_len: int = 100,
    bull_dyn_pcntile: float = 10.0,
    bear_dyn_pcntile: float = 90.0,
    use_std_filter: bool = False,
    std_period: int = 100,
    std_mult: float = 1.5,
    trend_len: int = 200,
    use_hidden_trend_filter: bool = False,
    use_quality_score: bool = True,
    max_zones: int = 10,
    zone_extend: int = 15,
    atr_len: int = 14,
    zone_factor: float = 0.25,
    break_on_close: bool = False,
) -> pd.DataFrame:
    """计算振荡器背离、质量分与背离区状态。"""
    osc_type = choice("osc_type", osc_type, _OSC_TYPES)
    auto_source = boolean("auto_source", auto_source)
    osc_len = integer("osc_len", osc_len, 2)
    tsi_short_len = integer("tsi_short_len", tsi_short_len, 1)
    stc_fast = integer("stc_fast", stc_fast, 1)
    stc_slow = integer("stc_slow", stc_slow, 2)
    hp_len = integer("hp_len", hp_len, 2)
    ss_len = integer("ss_len", ss_len, 2)
    cyber_alpha = number("cyber_alpha", cyber_alpha, 0.01, 0.5)
    auto_levels = boolean("auto_levels", auto_levels)
    number("ob_level", ob_level, -500.0, 500.0)
    number("os_level", os_level, -500.0, 500.0)
    piv_left = integer("piv_left", piv_left, 1)
    piv_right = integer("piv_right", piv_right, 1)
    min_range = integer("min_range", min_range, 1)
    max_range = integer("max_range", max_range, 5)
    show_hidden = boolean("show_hidden", show_hidden)
    filter_mode = choice("filter_mode", filter_mode, ("Off", "Fixed", "Dynamic Zones"))
    bull_filter_pct = number("bull_filter_pct", bull_filter_pct, 0.0, 100.0)
    bear_filter_pct = number("bear_filter_pct", bear_filter_pct, 0.0, 100.0)
    dyn_sample_len = integer("dyn_sample_len", dyn_sample_len, 10)
    bull_dyn_pcntile = number("bull_dyn_pcntile", bull_dyn_pcntile, 1.0, 50.0)
    bear_dyn_pcntile = number("bear_dyn_pcntile", bear_dyn_pcntile, 50.0, 99.0)
    use_std_filter = boolean("use_std_filter", use_std_filter)
    std_period = integer("std_period", std_period, 10)
    std_mult = number("std_mult", std_mult, 0.5)
    trend_len = integer("trend_len", trend_len, 2)
    use_hidden_trend_filter = boolean("use_hidden_trend_filter",
                                      use_hidden_trend_filter)
    use_quality_score = boolean("use_quality_score", use_quality_score)
    max_zones = integer("max_zones", max_zones, 1, 50)
    zone_extend = integer("zone_extend", zone_extend, 1)
    atr_len = integer("atr_len", atr_len, 1)
    zone_factor = number("zone_factor", zone_factor, 0.05, 3.0)
    break_on_close = boolean("break_on_close", break_on_close)

    high, low, close, open_ = ohlcv(frame, "high", "low", "close", "open")
    hl2 = (high + low) / 2.0
    hlc3 = (high + low + close) / 3.0
    n = len(frame)
    index = frame.index
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)

    # 振荡器序列。
    if osc_type == "RSI":
        osc_series = rsi(close, osc_len)
        src_used = close
    elif osc_type == "CCI":
        osc_series = cci(frame, osc_len)
        src_used = hlc3
    elif osc_type == "MFI":
        osc_series = mfi(frame, osc_len)
        src_used = hlc3
    elif osc_type == "Fisher":
        src_used = hl2
        osc_series = _fisher(hl2, osc_len)
    elif osc_type == "TSI":
        src_used = close
        osc_series = _tsi(close, osc_len, tsi_short_len)
    elif osc_type == "STC":
        src_used = close
        osc_series = _stc(close, stc_fast, stc_slow, osc_len)
    elif osc_type == "DPO":
        src_used = close
        shift = int(osc_len / 2) + 1
        osc_series = close - sma(close, osc_len).shift(shift)
    elif osc_type == "Roofing":
        src_used = hl2
        osc_series = _roofing(hl2, hp_len, ss_len)
    else:
        src_used = hl2
        osc_series = _cyber(hl2, cyber_alpha)
    if not auto_source:
        # 手动源：用 close 重算有界型振荡器。
        src_used = close
    osc_arr = osc_series.to_numpy(dtype=float)

    unbounded = osc_type in ("DPO", "Roofing", "Cyber")
    osc_std = stdev(osc_series, std_period).to_numpy(dtype=float)
    lvl_band = osc_std * 1.5
    if auto_levels:
        if unbounded:
            eff_ob = lvl_band
            eff_os = -lvl_band
        else:
            ob, os_ = _AUTO_LEVELS.get(osc_type, (70.0, 30.0))
            eff_ob = np.full(n, ob)
            eff_os = np.full(n, os_)
    else:
        eff_ob = np.full(n, ob_level)
        eff_os = np.full(n, os_level)

    atr_arr = atr(frame, atr_len).to_numpy(dtype=float)
    trend_ema = ema(close, trend_len).to_numpy(dtype=float)

    ph_arr = pivothigh(high, piv_left, piv_right).to_numpy(dtype=float)
    pl_arr = pivotlow(low, piv_left, piv_right).to_numpy(dtype=float)

    win_len = piv_left + piv_right + 1
    osc_low_series = lowest(osc_series, win_len)
    osc_high_series = highest(osc_series, win_len)
    osc_low_arr = osc_low_series.to_numpy(dtype=float)
    osc_high_arr = osc_high_series.to_numpy(dtype=float)
    # 极值 bar 偏移（距当根，负值）。
    osc_low_bar_arr = np.zeros(n, dtype=int)
    osc_high_bar_arr = np.zeros(n, dtype=int)
    for bar in range(n):
        start = max(0, bar - win_len + 1)
        window = osc_arr[start:bar + 1]
        if len(window) and not np.all(np.isnan(window)):
            # Pine: bar_index + lowestbars(...) -> absolute bar index.
            osc_low_bar_arr[bar] = start + int(np.nanargmin(window))
            osc_high_bar_arr[bar] = start + int(np.nanargmax(window))

    osc_mean = sma(osc_series, std_period).to_numpy(dtype=float)
    upper_band = osc_mean + std_mult * osc_std
    lower_band = osc_mean - std_mult * osc_std
    dzone_above = percentile_nearest_rank(osc_series, dyn_sample_len,
                                          bear_dyn_pcntile).to_numpy(dtype=float)
    dzone_below = percentile_nearest_rank(osc_series, dyn_sample_len,
                                          bull_dyn_pcntile).to_numpy(dtype=float)

    out = {
        "osc_value": np.full(n, np.nan),
        "ob_level": np.full(n, np.nan),
        "os_level": np.full(n, np.nan),
        "bull_divergence": np.zeros(n, dtype=bool),
        "bear_divergence": np.zeros(n, dtype=bool),
        "bull_hidden": np.zeros(n, dtype=bool),
        "bear_hidden": np.zeros(n, dtype=bool),
        "bull_score": np.full(n, np.nan),
        "bear_score": np.full(n, np.nan),
        "bull_zone_count": np.zeros(n, dtype=int),
        "bear_zone_count": np.zeros(n, dtype=int),
        "zone_touches_max": np.zeros(n, dtype=int),
        "zone_created": np.zeros(n, dtype=bool),
    }

    # 历史 pivot 记录：[(pivot_bar, price, osc_extreme, osc_extreme_bar)]
    hist_pl: list[tuple[int, float, float, int]] = []
    hist_ph: list[tuple[int, float, float, int]] = []

    bull_zones: list[dict] = []  # {level, half_w, touches, was_inside, bull}
    bear_zones: list[dict] = []

    def find_divergence(history: list, pivot_bar: int, price: float,
                        osc_extreme: float, osc_extreme_bar: int,
                        is_bull: bool, hidden: bool) -> tuple[bool, float, int]:
        for depth in (1, 2, 3):
            if len(history) < depth:
                continue
            prev_bar, prev_price, prev_osc, prev_osc_bar = history[-depth]
            spacing = pivot_bar - prev_bar
            if not (min_range <= spacing <= max_range):
                continue
            if not osc_extreme_bar > prev_osc_bar:
                continue
            if is_bull:
                ok = ((price > prev_price and osc_extreme < prev_osc) if hidden
                      else (price < prev_price and osc_extreme > prev_osc))
            else:
                ok = ((price < prev_price and osc_extreme > prev_osc) if hidden
                      else (price > prev_price and osc_extreme < prev_osc))
            if ok:
                return True, prev_osc, spacing
        return False, np.nan, 0

    def quality_score(spacing: int, pivot_osc: float, prev_osc: float,
                      is_bull: bool, hidden: bool, eff_ob_bar: float,
                      eff_os_bar: float, osc_std_bar: float) -> int:
        span = max(1, max_range - min_range)
        spacing_score = min(100.0, max(0.0, (spacing - min_range) / span * 100.0))
        magnitude = abs(pivot_osc - prev_osc)
        mag_score = (min(100.0, magnitude / osc_std_bar * 50.0)
                     if osc_std_bar > 0 else 50.0)
        bull_lvl = eff_os_bar + bull_filter_pct / 100.0 * (eff_ob_bar - eff_os_bar)
        bear_lvl = eff_os_bar + bear_filter_pct / 100.0 * (eff_ob_bar - eff_os_bar)
        if is_bull:
            ext_score = min(100.0, max(0.0, (bull_lvl - pivot_osc)
                                       / max(1e-10, bull_lvl - eff_os_bar) * 100.0))
        else:
            ext_score = min(100.0, max(0.0, (pivot_osc - bear_lvl)
                                       / max(1e-10, eff_ob_bar - bear_lvl) * 100.0))
        want_above = is_bull if hidden else (not is_bull)
        dist_atr = 0.0
        if not np.isnan(trend_ema[bar_now]):
            dist_atr = ((closes[bar_now] - trend_ema[bar_now]) / atr_arr[bar_now]
                        if want_above
                        else (trend_ema[bar_now] - closes[bar_now]) / atr_arr[bar_now])
        trend_score = min(100.0, max(0.0, dist_atr / 2.0 * 100.0))
        return int(round((spacing_score + mag_score + ext_score + trend_score) / 4.0))

    for bar in range(n):
        bar_now = bar  # 供 quality_score 闭包引用
        out["osc_value"][bar] = osc_arr[bar]
        out["ob_level"][bar] = eff_ob[bar] if not np.isnan(eff_ob[bar]) else np.nan
        out["os_level"][bar] = eff_os[bar] if not np.isnan(eff_os[bar]) else np.nan

        # 区维护。
        for zones, is_bull in ((bull_zones, True), (bear_zones, False)):
            for i in range(len(zones) - 1, -1, -1):
                zone = zones[i]
                lower = zone["level"] - zone["half_w"]
                upper = zone["level"] + zone["half_w"]
                broken = ((closes[bar] < lower) if break_on_close
                          else (lows[bar] < lower)) if is_bull else \
                    ((closes[bar] > upper) if break_on_close
                     else (highs[bar] > upper))
                inside = (not broken) and highs[bar] >= lower and lows[bar] <= upper
                if broken:
                    zones.pop(i)
                else:
                    if inside and not zone["was_inside"]:
                        zone["touches"] += 1
                    zone["was_inside"] = inside
        out["bull_zone_count"][bar] = len(bull_zones)
        out["bear_zone_count"][bar] = len(bear_zones)
        out["zone_touches_max"][bar] = max(
            [z["touches"] for z in bull_zones + bear_zones], default=0)

        # pivot 与背离。
        found_pl = not np.isnan(pl_arr[bar])
        found_ph = not np.isnan(ph_arr[bar])
        if not (found_pl or found_ph):
            continue
        if np.isnan(eff_ob[bar]) or np.isnan(eff_os[bar]):
            continue
        pivot_bar = bar - piv_right
        bull_filter_lvl = eff_os[bar] + bull_filter_pct / 100.0 * (eff_ob[bar] - eff_os[bar])
        bear_filter_lvl = eff_os[bar] + bear_filter_pct / 100.0 * (eff_ob[bar] - eff_os[bar])
        osc_low = osc_low_arr[bar]
        osc_high = osc_high_arr[bar]
        if filter_mode == "Fixed":
            bull_ok = osc_low < bull_filter_lvl
            bear_ok = osc_high > bear_filter_lvl
        elif filter_mode == "Dynamic Zones":
            bull_ok = osc_low < (dzone_below[bar - piv_right]
                                 if bar >= piv_right else np.nan)
            bear_ok = osc_high > (dzone_above[bar - piv_right]
                                  if bar >= piv_right else np.nan)
        else:
            bull_ok = bear_ok = True
        std_bull_ok = (not use_std_filter
                       or (bar >= piv_right and not np.isnan(lower_band[bar - piv_right])
                           and osc_low < lower_band[bar - piv_right]))
        std_bear_ok = (not use_std_filter
                       or (bar >= piv_right and not np.isnan(upper_band[bar - piv_right])
                           and osc_high > upper_band[bar - piv_right]))

        if found_pl:
            pivot_price = lows[pivot_bar] if pivot_bar >= 0 else lows[bar]
            hit, prev_osc, spacing = find_divergence(
                hist_pl, pivot_bar, pivot_price, osc_low,
                osc_low_bar_arr[bar], True, False)
            bull_div = bool(found_pl and bull_ok and std_bull_ok and hit)
            hit_h, prev_osc_h, spacing_h = find_divergence(
                hist_pl, pivot_bar, pivot_price, osc_low,
                osc_low_bar_arr[bar], True, True)
            trend_ok = (not use_hidden_trend_filter
                        or (not np.isnan(trend_ema[bar]) and closes[bar] > trend_ema[bar]))
            bull_hid = bool(found_pl and bull_ok and hit_h and trend_ok)
            if bull_div:
                out["bull_divergence"][bar] = True
                if use_quality_score:
                    out["bull_score"][bar] = quality_score(
                        spacing, osc_low, prev_osc, True, False,
                        eff_ob[bar], eff_os[bar], osc_std[bar])
                half_w = (atr_arr[bar] if not np.isnan(atr_arr[bar]) else 0.0) \
                    * zone_factor / 2.0
                bull_zones.append({"level": pivot_price, "half_w": half_w,
                                   "touches": 0, "was_inside": False,
                                   "bull": True})
                out["zone_created"][bar] = True
            if show_hidden and bull_hid:
                out["bull_hidden"][bar] = True
            hist_pl.append((pivot_bar, pivot_price, osc_low, osc_low_bar_arr[bar]))

        if found_ph:
            pivot_price = highs[pivot_bar] if pivot_bar >= 0 else highs[bar]
            hit, prev_osc, spacing = find_divergence(
                hist_ph, pivot_bar, pivot_price, osc_high,
                osc_high_bar_arr[bar], False, False)
            bear_div = bool(found_ph and bear_ok and std_bear_ok and hit)
            hit_h, prev_osc_h, spacing_h = find_divergence(
                hist_ph, pivot_bar, pivot_price, osc_high,
                osc_high_bar_arr[bar], False, True)
            trend_ok = (not use_hidden_trend_filter
                        or (not np.isnan(trend_ema[bar]) and closes[bar] < trend_ema[bar]))
            bear_hid = bool(found_ph and bear_ok and hit_h and trend_ok)
            if bear_div:
                out["bear_divergence"][bar] = True
                if use_quality_score:
                    out["bear_score"][bar] = quality_score(
                        spacing, osc_high, prev_osc, False, False,
                        eff_ob[bar], eff_os[bar], osc_std[bar])
                half_w = (atr_arr[bar] if not np.isnan(atr_arr[bar]) else 0.0) \
                    * zone_factor / 2.0
                bear_zones.append({"level": pivot_price, "half_w": half_w,
                                   "touches": 0, "was_inside": False,
                                   "bull": False})
                out["zone_created"][bar] = True
            if show_hidden and bear_hid:
                out["bear_hidden"][bar] = True
            hist_ph.append((pivot_bar, pivot_price, osc_high, osc_high_bar_arr[bar]))

        while len(bull_zones) > max_zones:
            bull_zones.pop(0)
        while len(bear_zones) > max_zones:
            bear_zones.pop(0)

    data: dict[str, pd.Series] = {
        "osc_value": pd.Series(out["osc_value"], index=index),
        "ob_level": pd.Series(out["ob_level"], index=index),
        "os_level": pd.Series(out["os_level"], index=index),
        "bull_score": pd.Series(out["bull_score"], index=index, dtype="Int64"),
        "bear_score": pd.Series(out["bear_score"], index=index, dtype="Int64"),
        "bull_zone_count": pd.Series(out["bull_zone_count"], index=index, dtype="int64"),
        "bear_zone_count": pd.Series(out["bear_zone_count"], index=index, dtype="int64"),
        "zone_touches_max": pd.Series(out["zone_touches_max"], index=index, dtype="int64"),
    }
    for column in ("bull_divergence", "bear_divergence", "bull_hidden", "bear_hidden",
                   "zone_created"):
        data[column] = flags(pd.Series(out[column], index=index))
    return pd.DataFrame(data, index=index)
