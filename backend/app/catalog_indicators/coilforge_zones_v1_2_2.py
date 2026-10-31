"""CoilForge Zones v1.2.2 [WavesUnchained] — 压缩区间（coil）探测与突破监视。

评分制区间检测器：

- **候选评分（0..80）**：ATR 压缩 25（range/ATR 越小越高）+ ADX 15（越低越高）
  + 量能干涸 15 + 冲量在-context 15 + 历史 S/R 复用 10；模块关闭时给满分。
- **结构证据**：range 压缩 + 触边结构（上/下沿各有 ≥1 次非极值 K 的重测，
  合计 ≥ min_touches）两项需满足 `min_struct_evidence` 项。
- **区间状态机**：candidate 且空闲 → 开区间（start 回溯 lookback-1 根）；
  candidate 且活跃 → 扩边界；失效 → 结束。区间存续 ≥ `min_zone_bars` 才武装
  （否则直接丢弃）；结束时把 zoneHigh/zoneLow 记入历史位。
- **区间解读**：upper/lowerTouches 压力差 + EMA(lookback/2) vs EMA(lookback)
  趋势 → Bullish/Bearish pressure；否则按冲量方向给 continuation。
- **突破**：武装后 `breakout_watch_bars` 内收盘越过 zone 边界 ± ATR×buffer
  （Early 0/Confirmed 0.05/Strict 0.12），实体 ≥ ATR×(0/0.12/0.25)、收盘位置
  ≥ 0/0.60/0.70，可选量能放大与解读方向一致；触发即解除武装。

丢弃：box/label/table 绘图、oldZone 生命周期管理（仅保留 armed 语义）。

Source: pine-collection/indicators/market_structure/coilforge_zones/coilforge_zones_v1.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    dmi,
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
)

__all__ = ["coilforge_zones_v1_2_2"]

# sensitivity -> (rangeAtrMult, adxMax, minScore, minTouches)
_PRESETS = {"Strict": (2.4, 20.0, 50, 3),
            "Balanced": (4.2, 27.0, 35, 2),
            "Aggressive": (5.0, 32.0, 25, 2)}


def coilforge_zones_v1_2_2(
    frame: pd.DataFrame,
    *,
    sensitivity: str = "Balanced",
    lookback: int = 16,
    min_zone_bars: int = 5,
    custom_range_atr_mult: float = 4.2,
    custom_adx_max: float = 27.0,
    custom_min_score: int = 35,
    custom_min_touches: int = 2,
    min_struct_evidence: int = 2,
    use_atr_compression: bool = True,
    use_adx_filter: bool = True,
    use_touches: bool = True,
    use_volume: bool = True,
    use_impulse: bool = True,
    use_historical: bool = True,
    atr_len: int = 14,
    adx_len: int = 14,
    adx_smooth: int = 14,
    touch_tol_atr: float = 0.25,
    vol_len: int = 20,
    impulse_lookback: int = 12,
    impulse_atr_mult: float = 1.8,
    hist_lookback: int = 250,
    hist_tol_atr: float = 0.5,
    pivot_len: int = 5,
    breakout_watch_bars: int = 20,
    breakout_quality: str = "Confirmed",
    avoid_contrary_read: bool = True,
) -> pd.DataFrame:
    """计算压缩区间候选、区间状态机与突破信号。"""
    sensitivity = choice("sensitivity", sensitivity,
                         ("Strict", "Balanced", "Aggressive", "Custom"))
    lookback = integer("lookback", lookback, 8, 100)
    min_zone_bars = integer("min_zone_bars", min_zone_bars, 3, 50)
    custom_range_atr_mult = number("custom_range_atr_mult", custom_range_atr_mult, 0.1)
    custom_adx_max = number("custom_adx_max", custom_adx_max, 1.0)
    custom_min_score = integer("custom_min_score", custom_min_score, 0, 100)
    custom_min_touches = integer("custom_min_touches", custom_min_touches, 2, 20)
    min_struct_evidence = integer("min_struct_evidence", min_struct_evidence, 1, 2)
    use_atr_compression = boolean("use_atr_compression", use_atr_compression)
    use_adx_filter = boolean("use_adx_filter", use_adx_filter)
    use_touches = boolean("use_touches", use_touches)
    use_volume = boolean("use_volume", use_volume)
    use_impulse = boolean("use_impulse", use_impulse)
    use_historical = boolean("use_historical", use_historical)
    atr_len = integer("atr_len", atr_len, 2)
    adx_len = integer("adx_len", adx_len, 2)
    adx_smooth = integer("adx_smooth", adx_smooth, 2)
    touch_tol_atr = number("touch_tol_atr", touch_tol_atr, 0.01)
    vol_len = integer("vol_len", vol_len, 2)
    impulse_lookback = integer("impulse_lookback", impulse_lookback, 3, 100)
    impulse_atr_mult = number("impulse_atr_mult", impulse_atr_mult, 0.1)
    hist_lookback = integer("hist_lookback", hist_lookback, 50, 2000)
    hist_tol_atr = number("hist_tol_atr", hist_tol_atr, 0.01)
    pivot_len = integer("pivot_len", pivot_len, 2, 20)
    breakout_watch_bars = integer("breakout_watch_bars", breakout_watch_bars, 1, 200)
    breakout_quality = choice("breakout_quality", breakout_quality,
                              ("Early", "Confirmed", "Strict"))
    avoid_contrary_read = boolean("avoid_contrary_read", avoid_contrary_read)

    if sensitivity == "Custom":
        range_atr_mult = custom_range_atr_mult
        adx_max = custom_adx_max
        min_score = custom_min_score
        min_touches = custom_min_touches
    else:
        range_atr_mult, adx_max, min_score, min_touches = _PRESETS[sensitivity]

    high, low, close, volume, open_ = ohlcv(frame, "high", "low", "close",
                                            "volume", "open")
    closes = close.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    vols = volume.to_numpy(dtype=float)
    n = len(frame)

    atr_value = atr(frame, atr_len)
    atr_arr = atr_value.to_numpy(dtype=float)
    di_plus, di_minus, adx_series = dmi(frame, adx_len)
    adx_arr = adx_series.to_numpy(dtype=float)
    range_high_arr = highest(high, lookback).to_numpy(dtype=float)
    range_low_arr = lowest(low, lookback).to_numpy(dtype=float)
    vol_ma = sma(volume, vol_len).to_numpy(dtype=float)
    vol_dry_ma = sma(volume, max(3, vol_len // 2)).to_numpy(dtype=float)
    atr_ma = sma(atr_value, lookback).to_numpy(dtype=float)

    fast_len = max(2, round(lookback / 2))
    ema_fast = ema(close, fast_len).to_numpy(dtype=float)
    ema_slow = ema(close, lookback).to_numpy(dtype=float)

    ph_arr = pivothigh(high, pivot_len, pivot_len).to_numpy(dtype=float)
    pl_arr = pivotlow(low, pivot_len, pivot_len).to_numpy(dtype=float)

    out = {
        "zone_score": np.full(n, np.nan),
        "in_zone": np.zeros(n, dtype=bool),
        "zone_high": np.full(n, np.nan),
        "zone_low": np.full(n, np.nan),
        "zone_read": np.full(n, "No active zone", dtype=object),
        "zone_bias": np.full(n, "", dtype=object),
        "upper_touches": np.zeros(n, dtype=int),
        "lower_touches": np.zeros(n, dtype=int),
        "hist_near_top": np.zeros(n, dtype=bool),
        "hist_near_bottom": np.zeros(n, dtype=bool),
        "impulse_dir": np.zeros(n, dtype=int),
        "zone_ended": np.zeros(n, dtype=bool),
        "zone_armed": np.zeros(n, dtype=bool),
        "armed_read": np.full(n, "No active zone", dtype=object),
        "breakout_up": np.zeros(n, dtype=bool),
        "breakout_down": np.zeros(n, dtype=bool),
    }

    hist_levels: list[tuple[int, float]] = []
    in_zone = False
    zone_start = 0
    zone_high = zone_low = np.nan
    upper_touches = lower_touches = 0
    armed = False
    armed_until = -1
    armed_high = armed_low = np.nan
    armed_read = "No active zone"

    buffer_mult = {"Early": 0.0, "Confirmed": 0.05, "Strict": 0.12}[breakout_quality]
    min_body = {"Early": 0.0, "Confirmed": 0.12, "Strict": 0.25}[breakout_quality]
    min_close_pos = {"Early": 0.0, "Confirmed": 0.60, "Strict": 0.70}[breakout_quality]

    for bar in range(n):
        if np.isnan(atr_arr[bar]) or np.isnan(range_high_arr[bar]):
            continue
        range_size = range_high_arr[bar] - range_low_arr[bar]
        range_ratio = range_size / atr_arr[bar] if atr_arr[bar] > 0 else np.inf
        range_ok = range_ratio <= range_atr_mult
        adx = adx_arr[bar]
        adx_ok = adx <= adx_max if not np.isnan(adx) else False

        touch_tol = atr_arr[bar] * touch_tol_atr
        # 极值 bar 偏移（Pine highestbars/lowestbars：距当根的偏移）。
        window_high = highs[max(0, bar - lookback + 1):bar + 1]
        window_low = lows[max(0, bar - lookback + 1):bar + 1]
        high_offset = int(np.argmax(window_high[::-1]))  # 距当根
        low_offset = int(np.argmin(window_low[::-1]))
        upper_touches_scan = lower_touches_scan = 0
        for i in range(min(lookback, bar + 1)):
            if i != high_offset and highs[bar - i] >= range_high_arr[bar] - touch_tol:
                upper_touches_scan += 1
            if i != low_offset and lows[bar - i] <= range_low_arr[bar] + touch_tol:
                lower_touches_scan += 1
        touch_ok = (upper_touches_scan > 0 and lower_touches_scan > 0
                    and upper_touches_scan + lower_touches_scan >= min_touches)

        # 冲量。
        prior_move = (closes[bar] - closes[bar - impulse_lookback]
                      if bar >= impulse_lookback else 0.0)
        impulse_dir = (1 if prior_move > atr_arr[bar] * impulse_atr_mult
                       else -1 if prior_move < -atr_arr[bar] * impulse_atr_mult else 0)
        out["impulse_dir"][bar] = impulse_dir

        # 历史 S/R。
        hist_near_top = hist_near_bottom = False
        if use_historical:
            hist_tol = atr_arr[bar] * hist_tol_atr
            cutoff = bar - lookback + 1
            for level_bar, level in hist_levels:
                if level_bar >= cutoff:
                    continue
                if abs(level - range_high_arr[bar]) <= hist_tol:
                    hist_near_top = True
                if abs(level - range_low_arr[bar]) <= hist_tol:
                    hist_near_bottom = True
        hist_reuse = hist_near_top or hist_near_bottom
        out["hist_near_top"][bar] = hist_near_top
        out["hist_near_bottom"][bar] = hist_near_bottom

        # 评分。
        score = 0.0
        if use_atr_compression:
            score += (25.0 * (1.0 - min(range_ratio / range_atr_mult, 1.0))
                      if range_ok else 0.0)
        else:
            score += 25.0
        if use_adx_filter:
            score += (15.0 * (1.0 - min(adx / adx_max, 1.0))
                      if adx_ok and not np.isnan(adx) else 0.0)
        else:
            score += 15.0
        if use_volume:
            volume_available = (vols[bar] > 0 and not np.isnan(vol_ma[bar])
                                and not np.isnan(vols[bar]))
            vol_drying = (volume_available and vols[bar] < vol_ma[bar]
                          and vol_dry_ma[bar] < vol_ma[bar])
            score += 0.0 if not volume_available else (15.0 if vol_drying else 0.0)
        else:
            score += 15.0
        if use_impulse:
            score += 15.0 if impulse_dir != 0 else 0.0
        else:
            score += 15.0
        if use_historical:
            score += 10.0 if hist_reuse else 0.0
        else:
            score += 10.0
        out["zone_score"][bar] = score

        # 结构证据。
        struct_evidence = 0
        if use_atr_compression and range_ok:
            struct_evidence += 1
        if use_touches and touch_ok:
            struct_evidence += 1
        active_modules = (1 if use_atr_compression else 0) + (1 if use_touches else 0)
        struct_required = min(min_struct_evidence, max(active_modules, 1))
        structure_ok = active_modules == 0 or struct_evidence >= struct_required
        candidate = structure_ok and score >= min_score

        # 突破（先于区间状态更新检查武装）。
        vol_ma_bar = vol_ma[bar]
        breakout_volume_ok = (not use_volume or np.isnan(vol_ma_bar)
                              or vols[bar] > vol_ma_bar)
        breakout_buffer = atr_arr[bar] * buffer_mult
        bar_range = highs[bar] - lows[bar]
        body = abs(closes[bar] - opens[bar])
        close_pos = (closes[bar] - lows[bar]) / bar_range if bar_range > 0 else 0.5
        body_ok = body >= atr_arr[bar] * min_body if atr_arr[bar] > 0 else True
        read_up_ok = (not avoid_contrary_read
                      or armed_read not in ("Bearish pressure",
                                            "Bearish continuation"))
        read_down_ok = (not avoid_contrary_read
                        or armed_read not in ("Bullish pressure",
                                              "Bullish continuation"))
        breakout_up = (armed and bar <= armed_until
                       and closes[bar] > armed_high + breakout_buffer
                       and breakout_volume_ok and body_ok
                       and close_pos >= min_close_pos and read_up_ok)
        breakout_down = (armed and bar <= armed_until
                         and closes[bar] < armed_low - breakout_buffer
                         and breakout_volume_ok and body_ok
                         and close_pos <= 1.0 - min_close_pos and read_down_ok)
        out["breakout_up"][bar] = breakout_up
        out["breakout_down"][bar] = breakout_down
        if breakout_up or breakout_down:
            armed = False
            armed_high = armed_low = np.nan
            armed_read = "No active zone"

        # 区间状态机。
        start_zone = candidate and not in_zone
        continue_zone = candidate and in_zone
        end_zone = not candidate and in_zone
        if start_zone:
            in_zone = True
            zone_start = bar - lookback + 1
            zone_high = range_high_arr[bar]
            zone_low = range_low_arr[bar]
            upper_touches = lower_touches = 0
        if continue_zone:
            zone_high = max(zone_high, range_high_arr[bar])
            zone_low = min(zone_low, range_low_arr[bar])
        out["in_zone"][bar] = in_zone
        out["zone_high"][bar] = zone_high if in_zone else np.nan
        out["zone_low"][bar] = zone_low if in_zone else np.nan
        out["upper_touches"][bar] = upper_touches
        out["lower_touches"][bar] = lower_touches
        if in_zone:
            if highs[bar] >= zone_high - touch_tol:
                upper_touches += 1
            if lows[bar] <= zone_low + touch_tol:
                lower_touches += 1
            out["upper_touches"][bar] = upper_touches
            out["lower_touches"][bar] = lower_touches
            pressure_raw = upper_touches - lower_touches
            pressure_needed = 1 if (not np.isnan(adx) and adx >= 20.0) else 2
            trend_up = (not np.isnan(ema_fast[bar]) and not np.isnan(ema_slow[bar])
                        and ema_fast[bar] > ema_slow[bar])
            trend_down = (not np.isnan(ema_fast[bar]) and not np.isnan(ema_slow[bar])
                          and ema_fast[bar] < ema_slow[bar])
            bias = "Neutral"
            read = "Balance / compression"
            if pressure_raw >= 3 or (pressure_raw >= pressure_needed and trend_up):
                bias, read = "Up", "Bullish pressure"
            elif pressure_raw <= -3 or (pressure_raw <= -pressure_needed and trend_down):
                bias, read = "Down", "Bearish pressure"
            elif impulse_dir == 1:
                bias, read = "Up / Continuation", "Bullish continuation"
            elif impulse_dir == -1:
                bias, read = "Down / Continuation", "Bearish continuation"
            out["zone_read"][bar] = read
            out["zone_bias"][bar] = bias
        if end_zone:
            out["zone_ended"][bar] = True
            zone_age = bar - zone_start + 1
            if zone_age >= min_zone_bars:
                armed = True
                armed_until = bar + breakout_watch_bars
                armed_high = zone_high
                armed_low = zone_low
                armed_read = out["zone_read"][bar]
                out["armed_read"][bar] = armed_read
                out["zone_armed"][bar] = True
                if use_historical:
                    hist_levels.append((bar, zone_high))
                    hist_levels.append((bar, zone_low))
            in_zone = False
            upper_touches = lower_touches = 0
            zone_high = zone_low = np.nan
            out["zone_read"][bar] = "No active zone"
            out["zone_bias"][bar] = ""
        if armed and bar > armed_until:
            armed = False
            armed_high = armed_low = np.nan
            armed_read = "No active zone"
        if use_historical and not np.isnan(ph_arr[bar]):
            hist_levels.append((bar - pivot_len, ph_arr[bar]))
        if use_historical and not np.isnan(pl_arr[bar]):
            hist_levels.append((bar - pivot_len, pl_arr[bar]))
        hist_levels = [x for x in hist_levels if bar - x[0] <= hist_lookback]

    index = frame.index
    data: dict[str, pd.Series] = {
        "armed_read": pd.Series(out["armed_read"], index=index, dtype=object),
        "zone_score": pd.Series(out["zone_score"], index=index),
        "in_zone": flags(pd.Series(out["in_zone"], index=index)),
        "zone_high": pd.Series(out["zone_high"], index=index),
        "zone_low": pd.Series(out["zone_low"], index=index),
        "zone_read": pd.Series(out["zone_read"], index=index, dtype=object),
        "zone_bias": pd.Series(out["zone_bias"], index=index, dtype=object),
        "upper_touches": pd.Series(out["upper_touches"], index=index, dtype="int64"),
        "lower_touches": pd.Series(out["lower_touches"], index=index, dtype="int64"),
        "hist_near_top": flags(pd.Series(out["hist_near_top"], index=index)),
        "hist_near_bottom": flags(pd.Series(out["hist_near_bottom"], index=index)),
        "impulse_dir": pd.Series(out["impulse_dir"], index=index, dtype="int64"),
        "zone_ended": flags(pd.Series(out["zone_ended"], index=index)),
        "zone_armed": flags(pd.Series(out["zone_armed"], index=index)),
        "breakout_up": flags(pd.Series(out["breakout_up"], index=index)),
        "breakout_down": flags(pd.Series(out["breakout_down"], index=index)),
    }
    return pd.DataFrame(data, index=index)
