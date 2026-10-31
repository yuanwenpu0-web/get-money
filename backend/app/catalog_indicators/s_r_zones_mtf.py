"""S/R Zones MTF [WavesUnchained] — 多周期 swing 支撑/阻力区跟踪。

**Swing 引擎**（ATR 反转阈值）：dir=0 时收盘越过 seed±ATR×mult 确立方向；
上涨中 extreme 取最大 high，收盘回落 extreme−thr → 输出 swing high 并转跌；
下跌镜像。图表 TF + HTF1（上一已完成周期投影）+ 可选 HTF2 三源供区。

**Zone 管理**：
- 新区 [price, price±width]（阻力 price 在顶部、支撑在底部）；
  与同侧既有区中点距离 ≤ ATR×`merge_atr_mult` → 合并（并集边界，+0.5 分 +
  TF 加成×0.25）；距离 ≤ `too_close_atr`×ATR → 丢弃；超 `max_zones` 时按
  filter 开关删最低分（否则删最旧）。
- **评分**：每根 −age；test（进入区且收盘留在区内）+wTouch + wCloseLoc×
  收盘位置质量 + wWick×反向影线 + wImpulse×反向实体；sweep（影线越界收盘
  回内）+0.7×wTouch + wWick；retest（突破后 ≥min_bars 回踩确认）+1.2×wTouch；
  break（Close/Wick 无效化 + 突破缓冲 + 可选量能放大）翻面并加分，`break_fail_bars`
  内收回旧侧则扣回。

丢弃：box/line/label 绘图与快照淡出。

Source: pine-collection/indicators/market_structure/sr_zones_mtf_v2/sr_zones_mtf_v2.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    flags,
    integer,
    number,
    ohlcv,
    sma,
)
from ._tf import htf_frame

__all__ = ["s_r_zones_mtf"]


def _struct_swings(frame: pd.DataFrame, rev_atr_mult: float, atr_len: int):
    """ATR 反转 swing：返回 (swing_high, swing_low, atr)。"""
    high, low, close = ohlcv(frame, "high", "low", "close")
    a = atr(frame, atr_len)
    thr_series = a * rev_atr_mult
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    thr_arr = thr_series.to_numpy(dtype=float)
    n = len(frame)
    new_high = np.full(n, np.nan)
    new_low = np.full(n, np.nan)
    direction = 0
    pivot = closes[0] if n else np.nan
    extreme = pivot
    for bar in range(n):
        thr = thr_arr[bar]
        if np.isnan(thr):
            continue
        if direction == 0:
            if closes[bar] > pivot + thr:
                direction = 1
                extreme = highs[bar]
            elif closes[bar] < pivot - thr:
                direction = -1
                extreme = lows[bar]
        elif direction == 1:
            extreme = max(extreme, highs[bar])
            if closes[bar] < extreme - thr:
                new_high[bar] = extreme
                direction = -1
                extreme = lows[bar]
        else:
            extreme = min(extreme, lows[bar])
            if closes[bar] > extreme + thr:
                new_low[bar] = extreme
                direction = 1
                extreme = highs[bar]
    return pd.Series(new_high, index=frame.index), \
        pd.Series(new_low, index=frame.index), a


def s_r_zones_mtf(
    frame: pd.DataFrame,
    *,
    htf1_minutes: int = 15,
    use_htf2: bool = False,
    htf2_minutes: int = 60,
    use_chart_swings: bool = True,
    max_zones: int = 10,
    atr_len: int = 14,
    zone_atr_mult_htf: float = 1.0,
    zone_atr_mult_ch: float = 0.8,
    break_atr_mult: float = 0.25,
    merge_on: bool = True,
    merge_atr_mult: float = 0.6,
    too_close_atr: float = 0.125,
    rev_atr_mult: float = 1.0,
    invalidation: str = "Close",
    avoid_false_breaks: bool = False,
    break_vol_mult: float = 1.2,
    break_fail_bars: int = 5,
    use_filter: bool = True,
    min_score: float = 0.0,
    w_touch: float = 1.0,
    w_wick: float = 0.8,
    w_close_loc: float = 0.7,
    w_impulse: float = 0.6,
    w_age: float = 0.001,
    retest_min_bars: int = 3,
    retest_cooldown: int = 3,
) -> pd.DataFrame:
    """计算多周期 S/R 区与事件。"""
    htf1_minutes = integer("htf1_minutes", htf1_minutes, 1)
    use_htf2 = boolean("use_htf2", use_htf2)
    htf2_minutes = integer("htf2_minutes", htf2_minutes, 1)
    use_chart_swings = boolean("use_chart_swings", use_chart_swings)
    max_zones = integer("max_zones", max_zones, 1, 50)
    atr_len = integer("atr_len", atr_len, 1)
    zone_atr_mult_htf = number("zone_atr_mult_htf", zone_atr_mult_htf, 0.1)
    zone_atr_mult_ch = number("zone_atr_mult_ch", zone_atr_mult_ch, 0.1)
    break_atr_mult = number("break_atr_mult", break_atr_mult, 0.0)
    merge_on = boolean("merge_on", merge_on)
    merge_atr_mult = number("merge_atr_mult", merge_atr_mult, 0.1)
    too_close_atr = number("too_close_atr", too_close_atr, 0.0)
    rev_atr_mult = number("rev_atr_mult", rev_atr_mult, 0.1)
    invalidation = choice("invalidation", invalidation, ("Wick", "Close"))
    avoid_false_breaks = boolean("avoid_false_breaks", avoid_false_breaks)
    break_vol_mult = number("break_vol_mult", break_vol_mult, 0.1, 5.0)
    break_fail_bars = integer("break_fail_bars", break_fail_bars, 1)
    use_filter = boolean("use_filter", use_filter)
    min_score = number("min_score", min_score, 0.0)
    w_touch = number("w_touch", w_touch, 0.0)
    w_wick = number("w_wick", w_wick, 0.0)
    w_close_loc = number("w_close_loc", w_close_loc, 0.0)
    w_impulse = number("w_impulse", w_impulse, 0.0)
    w_age = number("w_age", w_age, 0.0)
    retest_min_bars = integer("retest_min_bars", retest_min_bars, 1)
    retest_cooldown = integer("retest_cooldown", retest_cooldown, 0)

    high, low, open_, close, volume = ohlcv(frame, "high", "low", "open", "close",
                                            "volume")
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    vols = volume.to_numpy(dtype=float)
    n = len(frame)
    index = frame.index

    # 三源 swing。
    sw_h_ch, sw_l_ch, atr_ch_series = _struct_swings(frame, rev_atr_mult, atr_len)
    sw_h1, sw_l1, _ = _struct_swings(
        htf_frame(frame, htf1_minutes).reset_index(names="__ts"), rev_atr_mult, atr_len)
    sw_h1_lift = sw_h1.reindex(pd.DatetimeIndex(
        frame.index.floor(f"{htf1_minutes}min") - pd.Timedelta(minutes=htf1_minutes)
    )).to_numpy(dtype=float) if len(sw_h1) else np.full(n, np.nan)
    sw_l1_lift = sw_l1.reindex(pd.DatetimeIndex(
        frame.index.floor(f"{htf1_minutes}min") - pd.Timedelta(minutes=htf1_minutes)
    )).to_numpy(dtype=float) if len(sw_l1) else np.full(n, np.nan)
    if use_htf2:
        sw_h2, sw_l2, _ = _struct_swings(
            htf_frame(frame, htf2_minutes).reset_index(names="__ts"),
            rev_atr_mult, atr_len)
        key2 = pd.DatetimeIndex(
            frame.index.floor(f"{htf2_minutes}min") - pd.Timedelta(minutes=htf2_minutes))
        sw_h2_lift = sw_h2.reindex(key2).to_numpy(dtype=float)
        sw_l2_lift = sw_l2.reindex(key2).to_numpy(dtype=float)
    else:
        sw_h2_lift = sw_l2_lift = np.full(n, np.nan)

    atr_ch = atr_ch_series.to_numpy(dtype=float)
    atr_htf1 = atr(
        htf_frame(frame, htf1_minutes).reset_index(names="__ts"), atr_len)
    atr1_by_key = atr_htf1.reindex(pd.DatetimeIndex(
        frame.index.floor(f"{htf1_minutes}min") - pd.Timedelta(minutes=htf1_minutes)
    )).to_numpy(dtype=float)
    avg_vol = sma(volume, atr_len).to_numpy(dtype=float)
    volume_is_real = bool(np.nansum(vols) > 0)

    out = {
        "active_zones": np.zeros(n, dtype=int),
        "resistance_count": np.zeros(n, dtype=int),
        "support_count": np.zeros(n, dtype=bool) ,  # placeholder replaced below
        "break_up": np.zeros(n, dtype=bool),
        "break_down": np.zeros(n, dtype=bool),
        "retest": np.zeros(n, dtype=bool),
        "sweep": np.zeros(n, dtype=bool),
        "test": np.zeros(n, dtype=bool),
        "nearest_resistance": np.full(n, np.nan),
        "nearest_support": np.full(n, np.nan),
    }
    out["support_count"] = np.zeros(n, dtype=int)

    # zone 存储。
    tops: list[float] = []
    bots: list[float] = []
    res_flags: list[bool] = []
    scores: list[float] = []
    touches: list[int] = []
    brk_dirs: list[int] = []
    brk_bars: list[int | None] = []
    last_retest: list[int | None] = []
    brk_bonus: list[float] = []

    def find_merge_idx(mid: float, is_res: bool, dist: float) -> int:
        best, best_d = -1, 1e10
        for i, (top, bot, flag) in enumerate(zip(tops, bots, res_flags)):
            if flag == is_res:
                d = abs(mid - (top + bot) / 2.0)
                if d <= dist and d < best_d:
                    best_d, best = d, i
        return best

    def is_too_close(mid: float, is_res: bool, dist: float) -> bool:
        for top, bot, flag in zip(tops, bots, res_flags):
            if flag == is_res and abs(mid - (top + bot) / 2.0) <= dist:
                return True
        return False

    def add_or_merge(is_res: bool, price: float, width: float,
                     src_atr: float, tf_boost: float) -> None:
        top = price if is_res else price + width
        bot = price - width if is_res else price
        mid = (top + bot) / 2.0
        if too_close_atr > 0 and is_too_close(mid, is_res, src_atr * too_close_atr):
            return
        merge_idx = find_merge_idx(mid, is_res, src_atr * merge_atr_mult) \
            if merge_on and tops else -1
        if merge_idx != -1:
            tops[merge_idx] = max(tops[merge_idx], top)
            bots[merge_idx] = min(bots[merge_idx], bot)
            scores[merge_idx] += 0.5 + tf_boost * 0.25
            return
        tops.insert(0, top)
        bots.insert(0, bot)
        res_flags.insert(0, is_res)
        scores.insert(0, 0.0 + tf_boost)
        touches.insert(0, 0)
        brk_dirs.insert(0, 0)
        brk_bars.insert(0, None)
        last_retest.insert(0, None)
        brk_bonus.insert(0, 0.0)
        if len(tops) > max_zones:
            if use_filter:
                worst = int(np.argmin(scores))
            else:
                worst = len(tops) - 1
            for store in (tops, bots, res_flags, scores, touches, brk_dirs,
                          brk_bars, last_retest, brk_bonus):
                store.pop(worst)

    for bar in range(n):
        src_atr_ch = atr_ch[bar] if not np.isnan(atr_ch[bar]) else np.nan
        if np.isnan(src_atr_ch):
            continue
        # 新 swing → add/merge。
        if use_chart_swings and not np.isnan(sw_h_ch.iloc[bar]):
            add_or_merge(True, sw_h_ch.iloc[bar], src_atr_ch * zone_atr_mult_ch,
                         src_atr_ch, 0.5)
        if use_chart_swings and not np.isnan(sw_l_ch.iloc[bar]):
            add_or_merge(False, sw_l_ch.iloc[bar], src_atr_ch * zone_atr_mult_ch,
                         src_atr_ch, 0.5)
        if not np.isnan(sw_h1_lift[bar]):
            h1_atr = atr1_by_key[bar] if not np.isnan(atr1_by_key[bar]) else src_atr_ch
            add_or_merge(True, sw_h1_lift[bar], h1_atr * zone_atr_mult_htf,
                         h1_atr, 1.0)
        if not np.isnan(sw_l1_lift[bar]):
            h1_atr = atr1_by_key[bar] if not np.isnan(atr1_by_key[bar]) else src_atr_ch
            add_or_merge(False, sw_l1_lift[bar], h1_atr * zone_atr_mult_htf,
                         h1_atr, 1.0)
        if use_htf2 and not np.isnan(sw_h2_lift[bar]):
            add_or_merge(True, sw_h2_lift[bar], src_atr_ch * zone_atr_mult_htf,
                         src_atr_ch, 2.0)
        if use_htf2 and not np.isnan(sw_l2_lift[bar]):
            add_or_merge(False, sw_l2_lift[bar], src_atr_ch * zone_atr_mult_htf,
                         src_atr_ch, 2.0)

        # 事件扫描。
        inv_high = closes[bar] if invalidation == "Close" else highs[bar]
        inv_low = closes[bar] if invalidation == "Close" else lows[bar]
        vol_ok = (not avoid_false_breaks or not volume_is_real
                  or vols[bar] >= avg_vol[bar] * break_vol_mult)
        evt_break_up = evt_break_dn = evt_retest = evt_sweep = evt_test = False
        nearest_res = nearest_sup = np.nan
        for i in range(len(tops)):
            top, bot, is_res = tops[i], bots[i], res_flags[i]
            scores[i] -= w_age
            rng = max(top - bot, 1e-6)
            in_zone = lows[bar] <= top and highs[bar] >= bot
            sweep_r = is_res and highs[bar] > top and closes[bar] < top and closes[bar] > bot
            sweep_s = (not is_res) and lows[bar] < bot and closes[bar] > bot and closes[bar] < top
            test_r = is_res and in_zone and closes[bar] < top and closes[bar] > bot and not sweep_r
            test_s = (not is_res) and in_zone and closes[bar] > bot and closes[bar] < top and not sweep_s
            break_up = inv_high > top + src_atr_ch * break_atr_mult and vol_ok
            break_dn = inv_low < bot - src_atr_ch * break_atr_mult and vol_ok
            close_loc = min(max((closes[bar] - bot) / rng, 0.0), 1.0)
            q_close = (1.0 - close_loc) if is_res else close_loc
            upper_w = (highs[bar] - max(opens[bar], closes[bar])) / max(src_atr_ch, 1e-6)
            lower_w = (min(opens[bar], closes[bar]) - lows[bar]) / max(src_atr_ch, 1e-6)
            q_wick = upper_w if is_res else lower_w
            body = (closes[bar] - opens[bar]) / max(src_atr_ch, 1e-6)
            q_imp = -body if is_res else body
            if test_r or test_s:
                touches[i] += 1
                scores[i] += w_touch + w_close_loc * q_close + w_wick * q_wick \
                    + w_impulse * q_imp
                evt_test = True
            if sweep_r or sweep_s:
                scores[i] += w_touch * 0.7 + w_wick * q_wick
                evt_sweep = True
            # 挂起 break bonus 的回收。
            if brk_bonus[i] != 0.0 and brk_bars[i] is not None:
                since = bar - brk_bars[i]
                failed = since > 0 and ((brk_dirs[i] == 1 and closes[bar] < top)
                                        or (brk_dirs[i] == -1 and closes[bar] > bot))
                if failed:
                    scores[i] -= brk_bonus[i]
                    brk_bonus[i] = 0.0
                elif since >= break_fail_bars:
                    brk_bonus[i] = 0.0
            if is_res and break_up:
                res_flags[i] = False
                brk_dirs[i] = 1
                brk_bars[i] = bar
                bonus = w_impulse * 1.5 + w_close_loc * 0.5
                scores[i] += bonus
                brk_bonus[i] = bonus
                evt_break_up = True
            if (not is_res) and break_dn:
                res_flags[i] = True
                brk_dirs[i] = -1
                brk_bars[i] = bar
                bonus = w_impulse * 1.5 + w_close_loc * 0.5
                scores[i] += bonus
                brk_bonus[i] = bonus
                evt_break_dn = True
            can_retest = brk_bars[i] is not None and bar - brk_bars[i] >= retest_min_bars
            cooldown_ok = last_retest[i] is None or bar - last_retest[i] >= retest_cooldown
            retest_bull = (can_retest and cooldown_ok and brk_dirs[i] == 1
                           and in_zone and lows[bar] <= top and closes[bar] > top
                           and closes[bar] > opens[bar])
            retest_bear = (can_retest and cooldown_ok and brk_dirs[i] == -1
                           and in_zone and highs[bar] >= bot and closes[bar] < bot
                           and closes[bar] < opens[bar])
            if retest_bull or retest_bear:
                scores[i] += w_touch * 1.2 + w_wick * q_wick + w_impulse * abs(q_imp)
                last_retest[i] = bar
                evt_retest = True
            # 最近区（closes 在下方找阻力 / 上方找支撑）。
            if is_res and closes[bar] < bot:
                nearest_res = bot if np.isnan(nearest_res) else min(nearest_res, bot)
            if (not is_res) and closes[bar] > top:
                nearest_sup = top if np.isnan(nearest_sup) else max(nearest_sup, top)
        out["break_up"][bar] = evt_break_up
        out["break_down"][bar] = evt_break_dn
        out["retest"][bar] = evt_retest
        out["sweep"][bar] = evt_sweep
        out["test"][bar] = evt_test
        out["nearest_resistance"][bar] = nearest_res
        out["nearest_support"][bar] = nearest_sup
        out["active_zones"][bar] = len(tops)
        out["resistance_count"][bar] = sum(1 for f in res_flags if f)
        out["support_count"][bar] = len(res_flags) - sum(1 for f in res_flags if f)

    data: dict[str, pd.Series] = {
        "active_zones": pd.Series(out["active_zones"], index=index, dtype="int64"),
        "resistance_count": pd.Series(out["resistance_count"], index=index, dtype="int64"),
        "support_count": pd.Series(out["support_count"], index=index, dtype="int64"),
        "break_up": flags(pd.Series(out["break_up"], index=index)),
        "break_down": flags(pd.Series(out["break_down"], index=index)),
        "retest": flags(pd.Series(out["retest"], index=index)),
        "sweep": flags(pd.Series(out["sweep"], index=index)),
        "test": flags(pd.Series(out["test"], index=index)),
        "nearest_resistance": pd.Series(out["nearest_resistance"], index=index),
        "nearest_support": pd.Series(out["nearest_support"], index=index),
    }
    return pd.DataFrame(data, index=index)
