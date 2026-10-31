"""I2.0 SM Radar — Investisseur 2.0 聪明钱雷达（SMC 全套）。

**Order Blocks**：以 `ta.pivothigh/pivotlow(ob_pivot_len)` 追踪最近的 swing high/low，
当 `close` 上穿 swing high（且前一根未上穿）→ bullish BOS；此时在「BOS 前最近
`min(50, bar - swingIdx)` 根里找第一根阴线」的位置形成 bullish OB（top=该根 high，
bot=该根 low）。bearish 镜像（找第一根阳线）。OB 被价格穿越区间即 `mitigated`；
超 `ob_max_age` 根或被 mitigated 且 `ob_keep_mitigated=False` 时移除。

**Fair Value Gaps**：`low[1] > high[3]` → bullish FVG（top=low[1], bot=high[3]），
`high[1] < low[3]` → bearish FVG；`(top-bot)/close*100 >= fvg_min_size_pct` 才保留。
完全穿越（bull 需 low<=bot）即 filled 移除；部分触及则截断右端。超 `fvg_max_age` 移除。

**Equal Highs / Lows**：每出现新 OB 枢轴，回看前 5 个同侧枢轴（`eq_lookback` 根内），
`|cur-prev|/cur*100 <= eq_tol_pct` 则视为等高等低（流动性池）。被
`high > level + level*tol%`（或镜像）扫掉即 swept，5 根后移除。

**前日/周/月高低温**：上一已完成周期的 H/L（无未来函数）。

**Session**：Asia 00:00–08:00 / London 07:00–16:00 / New York 12:00–21:00 UTC；
每段会话记录高低温，会话结束输出 `session_high/low` 与「本根是否为会话末根」。
仅当基础周期 < 1D 时启用（与 Pine 的 `timeframe.in_seconds() < 86400` 一致）。

**市场结构 BOS/CHoCH**：`ta.pivothigh/pivotlow(ms_pivot_len)`，close 上穿最近 swing
high 时，若当前趋势为 -1 则判 CHoCH 否则 BOS，趋势置 +1（镜像对称）。

丢弃：全部 box/line/label/table 绘图、水印、告警配置。

Source: pine-crypto-collection/cedinvest/indicator.pine
License: Investisseur 2.0 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    boolean,
    flags,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
)
from ._tf import base_step_minutes, calendar_frame

__all__ = ["sm_radar"]

_SESSIONS = {
    "asia": (0, 8),
    "london": (7, 16),
    "ny": (12, 21),
}


class _Zone:
    """One Order Block or FVG zone tracked across bars."""

    __slots__ = ("bar", "top", "bot", "bull", "mitigated")

    def __init__(self, bar: int, top: float, bot: float, bull: bool) -> None:
        self.bar = bar
        self.top = top
        self.bot = bot
        self.bull = bull
        self.mitigated = False


def _previous_period_hl(frame: pd.DataFrame, rule: str) -> tuple[pd.Series, pd.Series]:
    """Previous completed calendar period high/low, projected without look-ahead."""
    columns = ["open", "high", "low", "close"]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    aggregated = calendar_frame(frame[columns], rule)
    previous = aggregated.index.tz_localize(None).to_period(rule) - 1
    keys = frame.index.tz_localize(None).to_period(rule)
    high_map = pd.Series(aggregated["high"].to_numpy(), index=previous)
    low_map = pd.Series(aggregated["low"].to_numpy(), index=previous)
    return (pd.Series(high_map.reindex(keys).to_numpy(dtype=float), index=frame.index),
            pd.Series(low_map.reindex(keys).to_numpy(dtype=float), index=frame.index))


def _session_windows(frame: pd.DataFrame, hours: tuple[int, int]) -> np.ndarray:
    """True when the bar's open hour falls inside the UTC [start, end) window."""
    utc = frame.index.tz_convert("UTC")
    start, end = hours
    return ((utc.hour.to_numpy() >= start) & (utc.hour.to_numpy() < end))


def sm_radar(
    frame: pd.DataFrame,
    *,
    show_ob: bool = True,
    ob_pivot_len: int = 5,
    ob_max_age: int = 80,
    ob_keep_mitigated: bool = False,
    show_fvg: bool = True,
    fvg_min_size_pct: float = 0.0,
    fvg_max_age: int = 60,
    show_equal_levels: bool = True,
    eq_tol_pct: float = 0.07,
    eq_lookback: int = 60,
    show_prev_hl: bool = True,
    show_sessions: bool = True,
    show_asia: bool = True,
    show_london: bool = True,
    show_ny: bool = True,
    show_structure: bool = True,
    ms_pivot_len: int = 7,
) -> pd.DataFrame:
    """返回 Order Blocks、FVG、等高等低、前周期 H/L、会话与 BOS/CHoCH 状态。"""
    show_ob = boolean("show_ob", show_ob)
    ob_pivot_len = integer("ob_pivot_len", ob_pivot_len, 2, 30)
    ob_max_age = integer("ob_max_age", ob_max_age, 10, 500)
    ob_keep_mitigated = boolean("ob_keep_mitigated", ob_keep_mitigated)
    show_fvg = boolean("show_fvg", show_fvg)
    fvg_min_size_pct = number("fvg_min_size_pct", fvg_min_size_pct, 0.0)
    fvg_max_age = integer("fvg_max_age", fvg_max_age, 5, 300)
    show_equal_levels = boolean("show_equal_levels", show_equal_levels)
    eq_tol_pct = number("eq_tol_pct", eq_tol_pct, 0.01)
    eq_lookback = integer("eq_lookback", eq_lookback, 10, 300)
    show_prev_hl = boolean("show_prev_hl", show_prev_hl)
    show_sessions = boolean("show_sessions", show_sessions)
    show_asia = boolean("show_asia", show_asia)
    show_london = boolean("show_london", show_london)
    show_ny = boolean("show_ny", show_ny)
    show_structure = boolean("show_structure", show_structure)
    ms_pivot_len = integer("ms_pivot_len", ms_pivot_len, 3, 30)

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    highs = high.to_numpy()
    lows = low.to_numpy()
    opens = open_.to_numpy()
    closes = close.to_numpy()
    size = len(frame)

    ph_ob = pivothigh(high, ob_pivot_len, ob_pivot_len).to_numpy()
    pl_ob = pivotlow(low, ob_pivot_len, ob_pivot_len).to_numpy()
    ph_ms = pivothigh(high, ms_pivot_len, ms_pivot_len).to_numpy()
    pl_ms = pivotlow(low, ms_pivot_len, ms_pivot_len).to_numpy()

    prev_day_high, prev_day_low = _previous_period_hl(frame, "D")
    prev_week_high, prev_week_low = _previous_period_hl(frame, "W")
    prev_month_high, prev_month_low = _previous_period_hl(frame, "M")

    intraday = base_step_minutes(frame) < 1440
    session_flags = {name: _session_windows(frame, window) for name, window in _SESSIONS.items()}

    out = {name: np.zeros(size, dtype=bool) for name in
           ("ob_bull", "ob_bear", "ob_mitigated", "fvg_bull", "fvg_bear", "fvg_filled",
            "eqh", "eql", "eq_swept", "bos_bull", "bos_bear", "choch_bull", "choch_bear",
            "asia_start", "asia_end", "london_start", "london_end", "ny_start", "ny_end")}
    out["ob_count"] = np.zeros(size, dtype=int)
    out["ob_top"] = np.full(size, np.nan)
    out["ob_bot"] = np.full(size, np.nan)
    out["fvg_count"] = np.zeros(size, dtype=int)
    out["fvg_top"] = np.full(size, np.nan)
    out["fvg_bot"] = np.full(size, np.nan)
    out["fvg_age"] = np.full(size, np.nan)
    out["eq_level"] = np.full(size, np.nan)
    out["eq_side"] = np.full(size, "", dtype=object)
    out["trend"] = np.zeros(size, dtype=int)
    out["pdh"] = prev_day_high.to_numpy() if show_prev_hl else np.full(size, np.nan)
    out["pdl"] = prev_day_low.to_numpy() if show_prev_hl else np.full(size, np.nan)
    out["pwh"] = prev_week_high.to_numpy() if show_prev_hl else np.full(size, np.nan)
    out["pwl"] = prev_week_low.to_numpy() if show_prev_hl else np.full(size, np.nan)
    out["pmh"] = prev_month_high.to_numpy() if show_prev_hl else np.full(size, np.nan)
    out["pml"] = prev_month_low.to_numpy() if show_prev_hl else np.full(size, np.nan)
    for name in _SESSIONS:
        out[f"{name}_high"] = np.full(size, np.nan)
        out[f"{name}_low"] = np.full(size, np.nan)
        out[f"{name}_active"] = np.zeros(size, dtype=bool)

    blocks: list[_Zone] = []
    gaps: list[_Zone] = []
    past_ph: list[tuple[float, int]] = []
    past_pl: list[tuple[float, int]] = []
    equal_levels: list[dict] = []

    last_swing_high = math.nan
    last_swing_high_idx = -1
    last_swing_low = math.nan
    last_swing_low_idx = -1
    ms_high = math.nan
    ms_high_idx = -1
    ms_low = math.nan
    ms_low_idx = -1
    trend = 0

    session_state = {name: {"active": False, "high": math.nan, "low": math.nan}
                     for name in _SESSIONS}

    for bar in range(size):
        # --- pivots (OB engine) -------------------------------------------
        pivot_bar = bar - ob_pivot_len
        if not np.isnan(ph_ob[bar]):
            last_swing_high = ph_ob[bar]
            last_swing_high_idx = pivot_bar
            past_ph.append((float(ph_ob[bar]), pivot_bar))
            if len(past_ph) > 30:
                past_ph.pop(0)
        if not np.isnan(pl_ob[bar]):
            last_swing_low = pl_ob[bar]
            last_swing_low_idx = pivot_bar
            past_pl.append((float(pl_ob[bar]), pivot_bar))
            if len(past_pl) > 30:
                past_pl.pop(0)

        price = closes[bar]
        previous_price = closes[bar - 1] if bar else math.nan

        bos_bull = (not math.isnan(last_swing_high) and price > last_swing_high
                    and (math.isnan(previous_price) or previous_price <= last_swing_high))
        bos_bear = (not math.isnan(last_swing_low) and price < last_swing_low
                    and (math.isnan(previous_price) or previous_price >= last_swing_low))
        out["bos_bull"][bar] = bos_bull
        out["bos_bear"][bar] = bos_bear

        if show_ob and bos_bull and last_swing_high_idx >= 0:
            limit = min(50, bar - last_swing_high_idx)
            for offset in range(1, limit + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] < opens[idx]:
                    blocks.append(_Zone(idx, highs[idx], lows[idx], True))
                    out["ob_bull"][bar] = True
                    break
        if show_ob and bos_bear and last_swing_low_idx >= 0:
            limit = min(50, bar - last_swing_low_idx)
            for offset in range(1, limit + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] > opens[idx]:
                    blocks.append(_Zone(idx, highs[idx], lows[idx], False))
                    out["ob_bear"][bar] = True
                    break

        # --- order block lifecycle ---------------------------------------
        keep: list[_Zone] = []
        for zone in blocks:
            if bar - zone.bar > ob_max_age:
                continue
            if not zone.mitigated and bar > zone.bar + 1:
                touched = ((lows[bar] <= zone.top and highs[bar] >= zone.bot) if zone.bull
                           else (highs[bar] >= zone.bot and lows[bar] <= zone.top))
                if touched:
                    zone.mitigated = True
                    out["ob_mitigated"][bar] = True
                    if not ob_keep_mitigated:
                        continue
            keep.append(zone)
        blocks = keep

        # --- fair value gaps ---------------------------------------------
        if show_fvg and bar > 3:
            if lows[bar - 1] > highs[bar - 3]:
                top, bot = lows[bar - 1], highs[bar - 3]
                if (top - bot) / price * 100 >= fvg_min_size_pct:
                    gaps.append(_Zone(bar - 2, top, bot, True))
                    out["fvg_bull"][bar] = True
            if highs[bar - 1] < lows[bar - 3]:
                top, bot = lows[bar - 3], highs[bar - 1]
                if (top - bot) / price * 100 >= fvg_min_size_pct:
                    gaps.append(_Zone(bar - 2, top, bot, False))
                    out["fvg_bear"][bar] = True

        keep_gaps: list[_Zone] = []
        for gap in gaps:
            age = bar - gap.bar
            if age > fvg_max_age:
                continue
            if not gap.mitigated and bar > gap.bar + 1:
                full_cross = lows[bar] <= gap.bot if gap.bull else highs[bar] >= gap.top
                if full_cross:
                    gap.mitigated = True
                    out["fvg_filled"][bar] = True
                    continue
            keep_gaps.append(gap)
        gaps = keep_gaps

        # --- equal highs / lows -------------------------------------------
        if show_equal_levels and not np.isnan(ph_ob[bar]) and len(past_ph) >= 2:
            cur_idx = pivot_bar
            for prev_price, prev_idx in reversed(past_ph[:-1][-5:]):
                if cur_idx - prev_idx > eq_lookback:
                    continue
                if abs(ph_ob[bar] - prev_price) / ph_ob[bar] * 100 <= eq_tol_pct:
                    equal_levels.append({"level": float(ph_ob[bar]), "side": "high",
                                         "last": bar})
                    out["eqh"][bar] = True
                    out["eq_level"][bar] = ph_ob[bar]
                    out["eq_side"][bar] = "EQH"
                    break
        if show_equal_levels and not np.isnan(pl_ob[bar]) and len(past_pl) >= 2:
            cur_idx = pivot_bar
            for prev_price, prev_idx in reversed(past_pl[:-1][-5:]):
                if cur_idx - prev_idx > eq_lookback:
                    continue
                if abs(pl_ob[bar] - prev_price) / pl_ob[bar] * 100 <= eq_tol_pct:
                    equal_levels.append({"level": float(pl_ob[bar]), "side": "low",
                                         "last": bar})
                    out["eql"][bar] = True
                    out["eq_level"][bar] = pl_ob[bar]
                    out["eq_side"][bar] = "EQL"
                    break

        live_levels: list[dict] = []
        for level in equal_levels:
            tol_abs = level["level"] * eq_tol_pct / 100
            if not level.get("swept"):
                swept = (highs[bar] > level["level"] + tol_abs if level["side"] == "high"
                         else lows[bar] < level["level"] - tol_abs)
                if swept:
                    level["swept"] = True
                    level["last"] = bar
                    out["eq_swept"][bar] = True
                live_levels.append(level)
            elif bar - level["last"] <= 5:
                live_levels.append(level)
        equal_levels = live_levels

        # --- market structure ---------------------------------------------
        ms_pivot_bar = bar - ms_pivot_len
        if not np.isnan(ph_ms[bar]):
            ms_high = ph_ms[bar]
            ms_high_idx = ms_pivot_bar
        if not np.isnan(pl_ms[bar]):
            ms_low = pl_ms[bar]
            ms_low_idx = ms_pivot_bar
        ms_bull = (not math.isnan(ms_high) and price > ms_high
                   and (math.isnan(previous_price) or previous_price <= ms_high))
        ms_bear = (not math.isnan(ms_low) and price < ms_low
                   and (math.isnan(previous_price) or previous_price >= ms_low))
        if show_structure and ms_bull and ms_high_idx >= 0:
            out["choch_bull"][bar] = trend == -1
            out["bos_bull"][bar] = out["bos_bull"][bar] or trend != -1
            trend = 1
        if show_structure and ms_bear and ms_low_idx >= 0:
            out["choch_bear"][bar] = trend == 1
            out["bos_bear"][bar] = out["bos_bear"][bar] or trend != 1
            trend = -1
        out["trend"][bar] = trend

        # --- sessions ------------------------------------------------------
        for name, flags_now in session_flags.items():
            if not intraday or not show_sessions:
                continue
            enabled = {"asia": show_asia, "london": show_london, "ny": show_ny}[name]
            if not enabled:
                continue
            state = session_state[name]
            inside = bool(flags_now[bar])
            if inside and not state["active"]:
                state["active"] = True
                state["high"] = highs[bar]
                state["low"] = lows[bar]
                out[f"{name}_start"][bar] = True
            elif inside and state["active"]:
                state["high"] = max(state["high"], highs[bar])
                state["low"] = min(state["low"], lows[bar])
            elif not inside and state["active"]:
                state["active"] = False
                out[f"{name}_end"][bar] = True
            out[f"{name}_active"][bar] = state["active"]
            out[f"{name}_high"][bar] = state["high"]
            out[f"{name}_low"][bar] = state["low"]

        # --- snapshots -----------------------------------------------------
        out["ob_count"][bar] = len(blocks)
        if blocks:
            newest = blocks[-1]
            out["ob_top"][bar] = newest.top
            out["ob_bot"][bar] = newest.bot
        out["fvg_count"][bar] = len(gaps)
        if gaps:
            newest_gap = gaps[-1]
            out["fvg_top"][bar] = newest_gap.top
            out["fvg_bot"][bar] = newest_gap.bot
            out["fvg_age"][bar] = bar - newest_gap.bar

    index = frame.index
    data: dict[str, pd.Series] = {
        "ob_bull": flags(pd.Series(out["ob_bull"], index=index)),
        "ob_bear": flags(pd.Series(out["ob_bear"], index=index)),
        "ob_mitigated": flags(pd.Series(out["ob_mitigated"], index=index)),
        "ob_count": pd.Series(out["ob_count"], index=index).astype("int64"),
        "ob_top": pd.Series(out["ob_top"], index=index),
        "ob_bot": pd.Series(out["ob_bot"], index=index),
        "fvg_bull": flags(pd.Series(out["fvg_bull"], index=index)),
        "fvg_bear": flags(pd.Series(out["fvg_bear"], index=index)),
        "fvg_filled": flags(pd.Series(out["fvg_filled"], index=index)),
        "fvg_count": pd.Series(out["fvg_count"], index=index).astype("int64"),
        "fvg_top": pd.Series(out["fvg_top"], index=index),
        "fvg_bot": pd.Series(out["fvg_bot"], index=index),
        "fvg_age": pd.Series(out["fvg_age"], index=index).astype("Int64"),
        "eqh": flags(pd.Series(out["eqh"], index=index)),
        "eql": flags(pd.Series(out["eql"], index=index)),
        "eq_swept": flags(pd.Series(out["eq_swept"], index=index)),
        "eq_level": pd.Series(out["eq_level"], index=index),
        "eq_side": pd.Series(out["eq_side"], index=index, dtype=object),
        "bos_bull": flags(pd.Series(out["bos_bull"], index=index)),
        "bos_bear": flags(pd.Series(out["bos_bear"], index=index)),
        "choch_bull": flags(pd.Series(out["choch_bull"], index=index)),
        "choch_bear": flags(pd.Series(out["choch_bear"], index=index)),
        "trend": pd.Series(out["trend"], index=index).astype("int64"),
        "prev_day_high": pd.Series(out["pdh"], index=index),
        "prev_day_low": pd.Series(out["pdl"], index=index),
        "prev_week_high": pd.Series(out["pwh"], index=index),
        "prev_week_low": pd.Series(out["pwl"], index=index),
        "prev_month_high": pd.Series(out["pmh"], index=index),
        "prev_month_low": pd.Series(out["pml"], index=index),
        "intraday": pd.Series(bool(intraday), index=index),
    }
    for name in _SESSIONS:
        data[f"{name}_active"] = flags(pd.Series(out[f"{name}_active"], index=index))
        data[f"{name}_high"] = pd.Series(out[f"{name}_high"], index=index)
        data[f"{name}_low"] = pd.Series(out[f"{name}_low"], index=index)
        data[f"{name}_start"] = flags(pd.Series(out[f"{name}_start"], index=index))
        data[f"{name}_end"] = flags(pd.Series(out[f"{name}_end"], index=index))
    return pd.DataFrame(data, index=index)
