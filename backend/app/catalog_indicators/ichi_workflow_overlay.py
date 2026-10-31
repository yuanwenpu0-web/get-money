"""MMT Ichi Workflow Overlay — 一目均衡表工作流信号层。

四状态记分（各 -1/0/+1，`ctf_score` 合计 -4..+4）：

- **price**：收盘在云上（+1）/ 云下（-1）/ 云内（0），云为位移后（当前可见）云。
- **future**：当前 SpanA vs SpanB（云的前瞻方向）。
- **tk**：Tenkan vs Kijun。
- **chikou**：当前收盘 vs `displacement` 前的实体高低 + `2×displacement` 前的云
  （迟行线确认）。

**HTF** 四个状态取上一已完成周期（`lookahead_on` + `[1]` 语义）。

四类 setup（各有方向过滤 + HTF 一致性 + ctf 门槛）：

1. **Kumo Breakout**：前一根收盘在云内 → 当根收盘穿出云外。
2. **Full Checklist**：四状态全同向（ctf=±4）。
3. **Kijun Bounce**：ctf≥±3 且 price/future 同向，价格触及 Kijun 后收盘回
   到 Kijun 正确一侧且当根实体顺势、前一根收盘也在 Kijun 另一侧。
4. **Edge-to-Edge**：从云外进入云内（前一根在云下/云上 → 当根进云）。

**平目标**：扫描最近 250 根内 SpanA/SpanB 连续三根几乎相等（≤0.25 tick）的
平台价，取当前价上方最近两级 / 下方最近两级。

**活跃 setup 状态机**：新 setup 覆盖旧的；普通 setup 失效 = 反向 TK 交叉 或
收盘破 Kijun 或收盘回到云内；Edge 型失效 = 收盘越过云的对侧。

丢弃：dashboard/table、markers、label、regime 背景着色。

Source: pine-crypto-collection/memeticmoney_ichimoku/pine/Ichi_Workflow_Overlay.pine
License: memeticmoney 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    boolean,
    choice,
    flags,
    highest,
    integer,
    lowest,
    ohlcv,
)
from ._tf import _period_starts, calendar_frame

__all__ = ["ichi_workflow_overlay"]

_PRESETS = {"Crypto Doubled": (20, 61, 120, 30),
            "Traditional Standard": (9, 26, 52, 26),
            "Traditional Doubled": (18, 52, 104, 52)}


def ichi_workflow_overlay(
    frame: pd.DataFrame,
    *,
    preset: str = "Crypto Doubled",
    custom_conversion: int = 20,
    custom_base: int = 61,
    custom_span_b: int = 120,
    custom_displacement: int = 30,
    htf_rule: str = "D",
    enable_breakout: bool = True,
    enable_checklist: bool = True,
    enable_kijun: bool = True,
    enable_edge: bool = True,
    direction_filter: str = "Long Only",
    flat_scan_bars: int = 250,
) -> pd.DataFrame:
    """计算一目四状态、四类 setup 与活跃 setup 状态机。"""
    preset = choice("preset", preset,
                    ("Crypto Doubled", "Traditional Standard",
                     "Traditional Doubled", "Custom"))
    custom_conversion = integer("custom_conversion", custom_conversion, 1)
    custom_base = integer("custom_base", custom_base, 2)
    custom_span_b = integer("custom_span_b", custom_span_b, 2)
    custom_displacement = integer("custom_displacement", custom_displacement, 1)
    enable_breakout = boolean("enable_breakout", enable_breakout)
    enable_checklist = boolean("enable_checklist", enable_checklist)
    enable_kijun = boolean("enable_kijun", enable_kijun)
    enable_edge = boolean("enable_edge", enable_edge)
    direction_filter = choice("direction_filter", direction_filter,
                              ("Both", "Long Only", "Short Only"))
    flat_scan_bars = integer("flat_scan_bars", flat_scan_bars, 0, 250)

    if preset == "Custom":
        conversion, base, span_b_len, displacement = (custom_conversion, custom_base,
                                                      custom_span_b,
                                                      custom_displacement)
    else:
        conversion, base, span_b_len, displacement = _PRESETS[preset]

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    n = len(frame)
    index = frame.index

    tenkan = (highest(high, conversion) + lowest(low, conversion)) / 2.0
    kijun = (highest(high, base) + lowest(low, base)) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b = (highest(high, span_b_len) + lowest(low, span_b_len)) / 2.0

    # 当前可见云 = 位移后的 Span（Pine `spanA[displacement]`）。
    cloud_top_now = pd.concat([span_a.shift(displacement), span_b.shift(displacement)],
                              axis=1).max(axis=1)
    cloud_bottom_now = pd.concat([span_a.shift(displacement), span_b.shift(displacement)],
                                 axis=1).min(axis=1)
    cloud_top_arr = cloud_top_now.to_numpy(dtype=float)
    cloud_bottom_arr = cloud_bottom_now.to_numpy(dtype=float)
    tenkan_arr = tenkan.to_numpy(dtype=float)
    kijun_arr = kijun.to_numpy(dtype=float)
    span_a_arr = span_a.to_numpy(dtype=float)
    span_b_arr = span_b.to_numpy(dtype=float)

    def price_state(bar: int) -> int:
        top, bot = cloud_top_arr[bar], cloud_bottom_arr[bar]
        if np.isnan(top) or np.isnan(bot):
            return 0
        return 1 if closes[bar] > top else -1 if closes[bar] < bot else 0

    def future_state(bar: int) -> int:
        a, b = span_a_arr[bar], span_b_arr[bar]
        if np.isnan(a) or np.isnan(b):
            return 0
        return 1 if a > b else -1 if a < b else 0

    def tk_state(bar: int) -> int:
        t, k = tenkan_arr[bar], kijun_arr[bar]
        if np.isnan(t) or np.isnan(k):
            return 0
        return 1 if t > k else -1 if t < k else 0

    def chikou_state(bar: int) -> int:
        past = bar - displacement
        far = bar - 2 * displacement
        if past < 0 or far < 0:
            return 0
        body_hi = max(opens[past], closes[past])
        body_lo = min(opens[past], closes[past])
        cloud_hi = max(span_a_arr[far], span_b_arr[far])
        cloud_lo = min(span_a_arr[far], span_b_arr[far])
        if np.isnan(body_hi) or np.isnan(cloud_hi):
            return 0
        if closes[bar] > body_hi and closes[bar] > cloud_hi:
            return 1
        if closes[bar] < body_lo and closes[bar] < cloud_lo:
            return -1
        return 0

    price_arr = np.array([price_state(b) for b in range(n)], dtype=int)
    future_arr = np.array([future_state(b) for b in range(n)], dtype=int)
    tk_arr = np.array([tk_state(b) for b in range(n)], dtype=int)
    chikou_arr = np.array([chikou_state(b) for b in range(n)], dtype=int)
    ctf_arr = price_arr + future_arr + tk_arr + chikou_arr

    # HTF 四状态：上一已完成周期。
    daily = calendar_frame(frame[["high", "low", "open", "close"]], htf_rule)
    d_high = daily["high"].to_numpy(dtype=float)
    d_low = daily["low"].to_numpy(dtype=float)
    d_open = daily["open"].to_numpy(dtype=float)
    d_close = daily["close"].to_numpy(dtype=float)
    d_n = len(daily)
    d_tenkan = (highest(daily["high"], conversion) + lowest(daily["low"], conversion)) / 2.0
    d_kijun = (highest(daily["high"], base) + lowest(daily["low"], base)) / 2.0
    d_span_a = ((d_tenkan + d_kijun) / 2.0).to_numpy(dtype=float)
    d_span_b = (highest(daily["high"], span_b_len)
                + lowest(daily["low"], span_b_len)) / 2.0
    d_span_b_arr = d_span_b.to_numpy(dtype=float)
    d_tenkan_arr = d_tenkan.to_numpy(dtype=float)
    d_kijun_arr = d_kijun.to_numpy(dtype=float)
    d_cloud_top = np.full(d_n, np.nan)
    d_cloud_bottom = np.full(d_n, np.nan)
    for b in range(d_n):
        if b >= displacement:
            d_cloud_top[b] = max(d_span_a[b - displacement],
                                 d_span_b_arr[b - displacement])
            d_cloud_bottom[b] = min(d_span_a[b - displacement],
                                    d_span_b_arr[b - displacement])
    d_price = np.zeros(d_n, dtype=int)
    d_future = np.zeros(d_n, dtype=int)
    d_tk = np.zeros(d_n, dtype=int)
    d_chikou = np.zeros(d_n, dtype=int)
    for b in range(d_n):
        if not (np.isnan(d_cloud_top[b]) or np.isnan(d_cloud_bottom[b])):
            d_price[b] = (1 if d_close[b] > d_cloud_top[b]
                          else -1 if d_close[b] < d_cloud_bottom[b] else 0)
        if not (np.isnan(d_span_a[b]) or np.isnan(d_span_b_arr[b])):
            d_future[b] = (1 if d_span_a[b] > d_span_b_arr[b]
                           else -1 if d_span_a[b] < d_span_b_arr[b] else 0)
        if not (np.isnan(d_tenkan_arr[b]) or np.isnan(d_kijun_arr[b])):
            d_tk[b] = (1 if d_tenkan_arr[b] > d_kijun_arr[b]
                       else -1 if d_tenkan_arr[b] < d_kijun_arr[b] else 0)
        past = b - displacement
        far = b - 2 * displacement
        if past >= 0 and far >= 0:
            body_hi = max(d_open[past], d_close[past])
            body_lo = min(d_open[past], d_close[past])
            cloud_hi = max(d_span_a[far], d_span_b_arr[far])
            cloud_lo = min(d_span_a[far], d_span_b_arr[far])
            if not (np.isnan(body_hi) or np.isnan(cloud_hi)):
                if d_close[b] > body_hi and d_close[b] > cloud_hi:
                    d_chikou[b] = 1
                elif d_close[b] < body_lo and d_close[b] < cloud_lo:
                    d_chikou[b] = -1
    d_ctf = d_price + d_future + d_tk + d_chikou

    # 上一已完成 HTF 周期 → 投影到基础 bar。
    prev_states = {
        "price": np.roll(d_price, 1), "future": np.roll(d_future, 1),
        "tk": np.roll(d_tk, 1), "chikou": np.roll(d_chikou, 1),
    }
    prev_ctf = np.roll(d_ctf, 1)
    if len(prev_ctf):  # 数据不足一个完整周期时 HTF 序列为空
        prev_states["price"][0] = 0
        prev_states["future"][0] = 0
        prev_states["tk"][0] = 0
        prev_states["chikou"][0] = 0
        prev_ctf[0] = 0
    # 上一已完成日历周期 -> 基础 bar（calendar_frame 的索引是 period start，
    # 所以必须按 period-1 的 start_time 取，不能用 floor("W")）。
    previous_starts = _period_starts(
        (index.tz_localize(None).to_period(htf_rule) - 1))
    lifted = {}
    for name, values in prev_states.items():
        lifted[name] = pd.Series(values, index=daily.index).reindex(
            previous_starts).to_numpy(dtype=float)
    htf_ctf_lift = pd.Series(prev_ctf, index=daily.index).reindex(
        previous_starts).to_numpy(dtype=float)

    allow_long = direction_filter != "Short Only"
    allow_short = direction_filter != "Long Only"

    out = {
        "tenkan": tenkan_arr, "kijun": kijun_arr,
        "span_a": span_a_arr, "span_b": span_b_arr,
        "cloud_top": cloud_top_arr, "cloud_bottom": cloud_bottom_arr,
        "price_state": price_arr, "future_state": future_arr,
        "tk_state": tk_arr, "chikou_state": chikou_arr, "ctf_score": ctf_arr,
        "htf_ctf_score": np.full(n, np.nan),
        "htf_price_state": np.full(n, np.nan),
        "long_breakout": np.zeros(n, dtype=bool),
        "short_breakout": np.zeros(n, dtype=bool),
        "long_checklist": np.zeros(n, dtype=bool),
        "short_checklist": np.zeros(n, dtype=bool),
        "long_kijun": np.zeros(n, dtype=bool),
        "short_kijun": np.zeros(n, dtype=bool),
        "long_edge": np.zeros(n, dtype=bool),
        "short_edge": np.zeros(n, dtype=bool),
        "target_above_1": np.full(n, np.nan),
        "target_above_2": np.full(n, np.nan),
        "target_below_1": np.full(n, np.nan),
        "target_below_2": np.full(n, np.nan),
        "active_setup": np.full(n, "None", dtype=object),
        "active_dir": np.zeros(n, dtype=int),
        "active_entry": np.full(n, np.nan),
        "active_target": np.full(n, np.nan),
        "active_invalidation": np.full(n, np.nan),
    }

    prev_long_breakout = prev_short_breakout = False
    prev_long_checklist = prev_short_checklist = False
    prev_long_kijun = prev_short_kijun = False
    prev_long_edge = prev_short_edge = False
    active_setup = "None"
    active_dir = 0
    active_entry = np.nan
    active_target = np.nan
    active_invalidation = np.nan
    active_is_edge = False
    prev_tk = 0

    def nearly_equal(a: float, b: float) -> bool:
        return (not np.isnan(a)) and (not np.isnan(b)) and abs(a - b) <= 1e-12

    for bar in range(n):
        top, bot = cloud_top_arr[bar], cloud_bottom_arr[bar]
        prev_top = cloud_top_arr[bar - 1] if bar > 0 else np.nan
        prev_bot = cloud_bottom_arr[bar - 1] if bar > 0 else np.nan
        htf_score = htf_ctf_lift[bar]
        htf_bull_ok = bool(not np.isnan(htf_score) and htf_score >= 3)
        htf_bear_ok = bool(not np.isnan(htf_score) and htf_score <= -3)
        out["htf_ctf_score"][bar] = htf_score
        out["htf_price_state"][bar] = lifted["price"][bar]

        now_in_cloud = (not np.isnan(top) and not np.isnan(bot)
                        and bot <= closes[bar] <= top)
        prior_in_cloud = (bar > 0 and not np.isnan(prev_top) and not np.isnan(prev_bot)
                          and prev_bot <= closes[bar - 1] <= prev_top)
        entered_from_below = (bar > 0 and not np.isnan(prev_bot)
                              and closes[bar - 1] < prev_bot and now_in_cloud)
        entered_from_above = (bar > 0 and not np.isnan(prev_top)
                              and closes[bar - 1] > prev_top and now_in_cloud)
        bullish_tk_cross = tk_arr[bar] == 1 and prev_tk != 1
        bearish_tk_cross = tk_arr[bar] == -1 and prev_tk != -1

        # 四类 setup 的"状态"（持续条件）。
        long_breakout_state = (enable_breakout and allow_long and htf_bull_ok
                               and ctf_arr[bar] >= 2 and prior_in_cloud
                               and not np.isnan(top) and closes[bar] > top)
        short_breakout_state = (enable_breakout and allow_short and htf_bear_ok
                                and ctf_arr[bar] <= -2 and prior_in_cloud
                                and not np.isnan(bot) and closes[bar] < bot)
        long_checklist_state = (enable_checklist and allow_long and htf_bull_ok
                                and price_arr[bar] == 1 and future_arr[bar] == 1
                                and tk_arr[bar] == 1 and chikou_arr[bar] == 1
                                and ctf_arr[bar] == 4)
        short_checklist_state = (enable_checklist and allow_short and htf_bear_ok
                                 and price_arr[bar] == -1 and future_arr[bar] == -1
                                 and tk_arr[bar] == -1 and chikou_arr[bar] == -1
                                 and ctf_arr[bar] == -4)
        prev_kijun = kijun_arr[bar - 1] if bar > 0 else np.nan
        long_kijun_state = (enable_kijun and allow_long and htf_bull_ok
                            and ctf_arr[bar] >= 3 and price_arr[bar] == 1
                            and future_arr[bar] == 1
                            and not np.isnan(kijun_arr[bar])
                            and lows[bar] <= kijun_arr[bar]
                            and closes[bar] > kijun_arr[bar]
                            and closes[bar] > opens[bar]
                            and not np.isnan(prev_kijun)
                            and closes[bar - 1] > prev_kijun)
        short_kijun_state = (enable_kijun and allow_short and htf_bear_ok
                             and ctf_arr[bar] <= -3 and price_arr[bar] == -1
                             and future_arr[bar] == -1
                             and not np.isnan(kijun_arr[bar])
                             and highs[bar] >= kijun_arr[bar]
                             and closes[bar] < kijun_arr[bar]
                             and closes[bar] < opens[bar]
                             and not np.isnan(prev_kijun)
                             and closes[bar - 1] < prev_kijun)
        long_edge_state = enable_edge and allow_long and htf_bull_ok and entered_from_below
        short_edge_state = enable_edge and allow_short and htf_bear_ok and entered_from_above

        long_breakout = long_breakout_state and not prev_long_breakout
        short_breakout = short_breakout_state and not prev_short_breakout
        long_checklist = long_checklist_state and not prev_long_checklist
        short_checklist = short_checklist_state and not prev_short_checklist
        long_kijun = long_kijun_state and not prev_long_kijun
        short_kijun = short_kijun_state and not prev_short_kijun
        long_edge = long_edge_state and not prev_long_edge
        short_edge = short_edge_state and not prev_short_edge
        out["long_breakout"][bar] = long_breakout
        out["short_breakout"][bar] = short_breakout
        out["long_checklist"][bar] = long_checklist
        out["short_checklist"][bar] = short_checklist
        out["long_kijun"][bar] = long_kijun
        out["short_kijun"][bar] = short_kijun
        out["long_edge"][bar] = long_edge
        out["short_edge"][bar] = short_edge

        # 平目标扫描（SpanA/SpanB 三连平台）。
        max_scan = max(0, min(flat_scan_bars, bar - 2))
        above1 = above2 = below1 = below2 = np.nan
        above1_d = above2_d = below1_d = below2_d = np.nan
        for i in range(max_scan + 1):
            for series, arr in ((span_a_arr, None), (span_b_arr, None)):
                if i + 2 >= len(series):
                    continue
                flat = (nearly_equal(series[bar - i], series[bar - i - 1])
                        and nearly_equal(series[bar - i - 1], series[bar - i - 2]))
                if not flat:
                    continue
                level = series[bar - i]
                if np.isnan(level):
                    continue
                unique_up1 = np.isnan(above1) or abs(level - above1) > 1e-12
                unique_up2 = np.isnan(above2) or abs(level - above2) > 1e-12
                unique_dn1 = np.isnan(below1) or abs(level - below1) > 1e-12
                unique_dn2 = np.isnan(below2) or abs(level - below2) > 1e-12
                if level > closes[bar]:
                    dist = level - closes[bar]
                    if (np.isnan(above1_d) or dist < above1_d) and unique_up1:
                        above2, above2_d = above1, above1_d
                        above1, above1_d = level, dist
                    elif unique_up1 and unique_up2 and (np.isnan(above2_d) or dist < above2_d):
                        above2, above2_d = level, dist
                elif level < closes[bar]:
                    dist = closes[bar] - level
                    if (np.isnan(below1_d) or dist < below1_d) and unique_dn1:
                        below2, below2_d = below1, below1_d
                        below1, below1_d = level, dist
                    elif unique_dn1 and unique_dn2 and (np.isnan(below2_d) or dist < below2_d):
                        below2, below2_d = level, dist
        out["target_above_1"][bar] = above1
        out["target_above_2"][bar] = above2
        out["target_below_1"][bar] = below1
        out["target_below_2"][bar] = below2

        # 活跃 setup 状态机。
        if active_dir == 1:
            active_invalidation = (bot if active_is_edge
                                   else min(kijun_arr[bar] if not np.isnan(kijun_arr[bar]) else np.inf,
                                            bot if not np.isnan(bot) else np.inf))
            if active_is_edge and not np.isnan(top):
                active_target = top
            invalidated = ((not np.isnan(bot) and closes[bar] < bot) if active_is_edge
                           else (bearish_tk_cross
                                 or (not np.isnan(kijun_arr[bar])
                                     and closes[bar] < kijun_arr[bar])
                                 or (not np.isnan(top) and closes[bar] <= top)))
        elif active_dir == -1:
            active_invalidation = (top if active_is_edge
                                   else max(kijun_arr[bar] if not np.isnan(kijun_arr[bar]) else -np.inf,
                                            top if not np.isnan(top) else -np.inf))
            if active_is_edge and not np.isnan(bot):
                active_target = bot
            invalidated = ((not np.isnan(top) and closes[bar] > top) if active_is_edge
                           else (bullish_tk_cross
                                 or (not np.isnan(kijun_arr[bar])
                                     and closes[bar] > kijun_arr[bar])
                                 or (not np.isnan(bot) and closes[bar] >= bot)))
        else:
            invalidated = False

        if invalidated:
            active_setup = "None"
            active_dir = 0
            active_entry = active_target = active_invalidation = np.nan
            active_is_edge = False

        new_setup = None
        if long_checklist:
            new_setup = ("Checklist Long", 1, above1, min(kijun_arr[bar], bot), False)
        elif short_checklist:
            new_setup = ("Checklist Short", -1, below1, max(kijun_arr[bar], top), False)
        elif long_kijun:
            new_setup = ("Kijun Bounce Long", 1, above1, min(kijun_arr[bar], bot), False)
        elif short_kijun:
            new_setup = ("Kijun Bounce Short", -1, below1, max(kijun_arr[bar], top), False)
        elif long_breakout:
            new_setup = ("Breakout Long", 1, above1, min(kijun_arr[bar], bot), False)
        elif short_breakout:
            new_setup = ("Breakout Short", -1, below1, max(kijun_arr[bar], top), False)
        elif long_edge:
            new_setup = ("Edge-to-Edge Long", 1, top, bot, True)
        elif short_edge:
            new_setup = ("Edge-to-Edge Short", -1, bot, top, True)
        if new_setup is not None:
            active_setup, active_dir, active_target, active_invalidation, active_is_edge = new_setup
            active_entry = closes[bar]

        out["active_setup"][bar] = active_setup
        out["active_dir"][bar] = active_dir
        out["active_entry"][bar] = active_entry
        out["active_target"][bar] = active_target
        out["active_invalidation"][bar] = active_invalidation

        prev_long_breakout, prev_short_breakout = long_breakout_state, short_breakout_state
        prev_long_checklist, prev_short_checklist = long_checklist_state, short_checklist_state
        prev_long_kijun, prev_short_kijun = long_kijun_state, short_kijun_state
        prev_long_edge, prev_short_edge = long_edge_state, short_edge_state
        prev_tk = tk_arr[bar]

    data: dict[str, pd.Series] = {
        "tenkan": pd.Series(out["tenkan"], index=index),
        "kijun": pd.Series(out["kijun"], index=index),
        "span_a": pd.Series(out["span_a"], index=index),
        "span_b": pd.Series(out["span_b"], index=index),
        "cloud_top": pd.Series(out["cloud_top"], index=index),
        "cloud_bottom": pd.Series(out["cloud_bottom"], index=index),
        "price_state": pd.Series(out["price_state"], index=index, dtype="int64"),
        "future_state": pd.Series(out["future_state"], index=index, dtype="int64"),
        "tk_state": pd.Series(out["tk_state"], index=index, dtype="int64"),
        "chikou_state": pd.Series(out["chikou_state"], index=index, dtype="int64"),
        "ctf_score": pd.Series(out["ctf_score"], index=index, dtype="int64"),
        "htf_ctf_score": pd.Series(out["htf_ctf_score"], index=index),
        "htf_price_state": pd.Series(out["htf_price_state"], index=index),
        "target_above_1": pd.Series(out["target_above_1"], index=index),
        "target_above_2": pd.Series(out["target_above_2"], index=index),
        "target_below_1": pd.Series(out["target_below_1"], index=index),
        "target_below_2": pd.Series(out["target_below_2"], index=index),
        "active_setup": pd.Series(out["active_setup"], index=index, dtype=object),
        "active_dir": pd.Series(out["active_dir"], index=index, dtype="int64"),
        "active_entry": pd.Series(out["active_entry"], index=index),
        "active_target": pd.Series(out["active_target"], index=index),
        "active_invalidation": pd.Series(out["active_invalidation"], index=index),
    }
    for column in ("long_breakout", "short_breakout", "long_checklist",
                   "short_checklist", "long_kijun", "short_kijun", "long_edge",
                   "short_edge"):
        data[column] = flags(pd.Series(out[column], index=index))
    return pd.DataFrame(data, index=index)
