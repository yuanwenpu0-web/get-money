"""Market Structure Pivot Map [WavesUnchained] — 多周期枢轴位图。

从上一根已完成周期（`request.security(..., lookahead_on)` 语义 → 上一周期收盘
数据）推导经典/斐波那契枢轴位集合：

- **CPR**：P=(H+L+C)/3，TC=max(2P−BC_raw, BC_raw)、BC=min(...)。
- **S/R**：Classic 公式 R1=2P−L…R4=R3+rng、Fibonacci 用 0.382/0.618/1.0/1.618×rng。
- **前周期高/低**：H/L 的上一周期值。
- **投影**：基准（日开盘/昨收/P）± {0.25,0.5,1.0}×范围（ADR/昨日振幅/日线 ATR）。
- **CPR 宽度分类**：|TC−BC|/日线 ATR：<0.20 Narrow、>0.55 Wide、否则 Normal。
- **偏向**：close 与日开盘同时 >TC → Bullish、同时 <BC → Bearish。
- **最近上方/下方位**：全部位按价与 close 比较取邻，距离用 ATR(14) 归一。
- **Confluence 聚类**：所有位排序后，间隔 ≤ ATR×mult 的归为同一簇，
  同簇不同 key（TC/BC/P 合并为 CPR key）≥ min_count 时输出簇中心。

丢弃：全部绘图（线/标签/box/table）、2nd/3rd 周期叠加、告警配置。

Source: pine-collection/indicators/market_structure/market_structure_pivot_map/market_structure_pivot_map.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    choice,
    integer,
    number,
    ohlcv,
    sma,
)
from ._tf import calendar_frame

__all__ = ["market_structure_pivot_map"]


def _pivot_set(high: float, low: float, close: float, kind: str) -> dict[str, float]:
    pivot = (high + low + close) / 3.0
    rng = high - low
    bc_raw = (high + low) / 2.0
    tc_raw = pivot + (pivot - bc_raw)
    tc = max(tc_raw, bc_raw)
    bc = min(tc_raw, bc_raw)
    classic = kind == "Classic"
    return {
        "p": pivot, "tc": tc, "bc": bc,
        "r1": 2.0 * pivot - low if classic else pivot + rng * 0.382,
        "s1": 2.0 * pivot - high if classic else pivot - rng * 0.382,
        "r2": pivot + rng if classic else pivot + rng * 0.618,
        "s2": pivot - rng if classic else pivot - rng * 0.618,
        "r3": high + 2.0 * (pivot - low) if classic else pivot + rng,
        "s3": low - 2.0 * (high - pivot) if classic else pivot - rng,
        "r4": (high + 2.0 * (pivot - low)) + rng if classic else pivot + rng * 1.618,
        "s4": (low - 2.0 * (high - pivot)) - rng if classic else pivot - rng * 1.618,
    }


def _daily_series(frame: pd.DataFrame, column: str, rule: str = "D") -> pd.Series:
    """上一已完成日的 `column` 值，投影到日内 bar（无未来函数）。"""
    daily = calendar_frame(frame[[column]], rule)[column]
    previous = daily.groupby(daily.index.floor("D")).last()
    shifted = previous.groupby(level=0).last().shift(1)
    keys = frame.index.floor("D")
    return pd.Series(shifted.reindex(keys).to_numpy(), index=frame.index)


def market_structure_pivot_map(
    frame: pd.DataFrame,
    *,
    pivot_type: str = "Classic",
    adr_len: int = 14,
    projection_base: str = "Day Open",
    projection_range: str = "ADR",
    show_cpr: bool = True,
    show_pivots: bool = True,
    show_prev_hl: bool = True,
    show_open: bool = True,
    show_proj_025: bool = True,
    show_proj_050: bool = True,
    show_proj_100: bool = True,
    show_confluence: bool = True,
    confl_atr_mult: float = 0.15,
    confl_min_count: int = 2,
) -> pd.DataFrame:
    """计算枢轴位集合、投影、CPR 分类、偏向、最近位与 confluence 聚类。"""
    pivot_type = choice("pivot_type", pivot_type, ("Classic", "Fibonacci"))
    adr_len = integer("adr_len", adr_len, 1)
    projection_base = choice("projection_base", projection_base,
                             ("Day Open", "Previous Close", "Pivot"))
    projection_range = choice("projection_range", projection_range,
                              ("ADR", "Previous Day Range", "ATR"))
    show_cpr = boolean("show_cpr", show_cpr)
    show_pivots = boolean("show_pivots", show_pivots)
    show_prev_hl = boolean("show_prev_hl", show_prev_hl)
    show_open = boolean("show_open", show_open)
    show_proj_025 = boolean("show_proj_025", show_proj_025)
    show_proj_050 = boolean("show_proj_050", show_proj_050)
    show_proj_100 = boolean("show_proj_100", show_proj_100)
    show_confluence = boolean("show_confluence", show_confluence)
    confl_atr_mult = number("confl_atr_mult", confl_atr_mult, 0.01)
    confl_min_count = integer("confl_min_count", confl_min_count, 2, 6)

    high, low, close = ohlcv(frame, "high", "low", "close")
    daily = calendar_frame(frame[["high", "low", "close", "open"]], "D")
    daily_open = daily["open"]
    # 上一已完成日的 H/L/C（Pine 的 high[1] with lookahead_on）。
    day_keys = frame.index.floor("D")
    prev_high = pd.Series(daily["high"].shift(1).reindex(day_keys).to_numpy(),
                          index=frame.index)
    prev_low = pd.Series(daily["low"].shift(1).reindex(day_keys).to_numpy(),
                         index=frame.index)
    prev_close = pd.Series(daily["close"].shift(1).reindex(day_keys).to_numpy(),
                           index=frame.index)
    day_open = pd.Series(daily["open"].reindex(day_keys).to_numpy(), index=frame.index)
    prev_range_daily = (daily["high"] - daily["low"]).shift(1)
    prev_range = pd.Series(prev_range_daily.reindex(day_keys).to_numpy(),
                           index=frame.index)
    adr_daily = sma(daily["high"] - daily["low"], adr_len)
    adr = pd.Series(adr_daily.reindex(day_keys).to_numpy(), index=frame.index)
    # ATR(adr_len) 在日线序列上计算再投影。
    daily_frame = daily[["high", "low", "close"]].rename(
        columns={"high": "high", "low": "low", "close": "close"})
    atr_daily = atr(daily_frame, adr_len)
    atr_d = pd.Series(atr_daily.reindex(day_keys).to_numpy(), index=frame.index)

    index = frame.index
    closes = close

    levels_per_bar: list[list[tuple[float, str]]] = []
    pivot_p = np.full(len(frame), np.nan)
    pivot_tc = np.full(len(frame), np.nan)
    pivot_bc = np.full(len(frame), np.nan)

    for bar in range(len(frame)):
        ph = prev_high.iloc[bar]
        pl = prev_low.iloc[bar]
        pc = prev_close.iloc[bar]
        if np.isnan(ph) or np.isnan(pl) or np.isnan(pc):
            levels_per_bar.append([])
            continue
        levels: list[tuple[float, str]] = []
        if show_cpr or show_pivots or show_prev_hl:
            pset = _pivot_set(ph, pl, pc, pivot_type)
            pivot_p[bar] = pset["p"]
            pivot_tc[bar] = pset["tc"]
            pivot_bc[bar] = pset["bc"]
            if show_cpr:
                levels.append((pset["p"], "P"))
                levels.append((pset["tc"], "TC"))
                levels.append((pset["bc"], "BC"))
            if show_pivots:
                for tag in ("r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"):
                    levels.append((pset[tag], tag.upper()))
        if show_prev_hl:
            levels.append((ph, "PrevH"))
            levels.append((pl, "PrevL"))
        if show_open and not np.isnan(day_open.iloc[bar]):
            levels.append((day_open.iloc[bar], "DayOpen"))
        if show_proj_025 or show_proj_050 or show_proj_100:
            base = (day_open.iloc[bar] if projection_base == "Day Open"
                    else pc if projection_base == "Previous Close"
                    else pivot_p[bar])
            rng = (adr.iloc[bar] if projection_range == "ADR"
                   else prev_range.iloc[bar] if projection_range == "Previous Day Range"
                   else atr_d.iloc[bar])
            if not (np.isnan(base) or np.isnan(rng)):
                if show_proj_025:
                    levels.append((base + rng * 0.25, "+0.25R"))
                    levels.append((base - rng * 0.25, "-0.25R"))
                if show_proj_050:
                    levels.append((base + rng * 0.50, "+0.50R"))
                    levels.append((base - rng * 0.50, "-0.50R"))
                if show_proj_100:
                    levels.append((base + rng * 1.00, "+1.00R"))
                    levels.append((base - rng * 1.00, "-1.00R"))
        levels_per_bar.append(levels)

    levels_out: list[pd.Series] = []
    names_out: list[pd.Series] = []
    for bar, levels in enumerate(levels_per_bar):
        prices = pd.Series([p for p, _ in levels], dtype=float)
        names = pd.Series([n for _, n in levels], dtype=object)
        prices.index = names.index = [index[bar]] * len(levels)
        levels_out.append(prices)
        names_out.append(names)

    atr_local = atr(frame, 14)
    cpr_width = np.abs(pivot_tc - pivot_bc)
    rel_cpr = cpr_width / atr_d.replace(0.0, np.nan)
    cpr_type = np.where(rel_cpr < 0.20, "Narrow",
                        np.where(rel_cpr > 0.55, "Wide", "Normal"))

    bias = np.full(len(frame), "Neutral", dtype=object)
    valid = ~np.isnan(pivot_tc) & ~np.isnan(pivot_bc) & ~np.isnan(day_open)
    bias = np.where(valid & (closes > pivot_tc) & (day_open > pivot_tc), "Bullish",
                    np.where(valid & (closes < pivot_bc) & (day_open < pivot_bc),
                             "Bearish", bias))

    near_above = np.full(len(frame), np.nan)
    near_below = np.full(len(frame), np.nan)
    name_above = np.full(len(frame), "", dtype=object)
    name_below = np.full(len(frame), "", dtype=object)
    confl_count = np.zeros(len(frame), dtype=int)
    confl_center = np.full(len(frame), np.nan)

    def _confl_key(name: str) -> str:
        if name.endswith(" TC") or name.endswith(" BC"):
            return name[:-3] + " CPR"
        if name.endswith(" P"):
            return name[:-2] + " CPR"
        return name

    for bar, levels in enumerate(levels_per_bar):
        if not levels:
            continue
        px = np.array([p for p, _ in levels])
        nm = [n for _, n in levels]
        above = px >= closes.iloc[bar]
        if above.any():
            pick = np.argmin(np.where(above, px, np.inf))
            near_above[bar] = px[pick]
            name_above[bar] = nm[pick]
        below = ~above
        if below.any():
            pick = np.argmax(np.where(below, px, -np.inf))
            near_below[bar] = px[pick]
            name_below[bar] = nm[pick]
        # Confluence 聚类（间隔 <= atr_local*mult 的相邻位）。
        if show_confluence and not np.isnan(atr_local.iloc[bar]):
            dist = atr_local.iloc[bar] * confl_atr_mult
            order = np.argsort(px)
            keys: list[str] = []
            cluster_max = cluster_sum = cluster_members = 0
            centers: list[float] = []

            def flush():
                if len(keys) >= confl_min_count:
                    centers.append(cluster_sum / cluster_members)

            first = True
            for i in order:
                key = _confl_key(nm[i])
                if first or px[i] - cluster_max <= dist:
                    cluster_max = px[i]
                    cluster_sum = (cluster_sum + px[i]) if not first else px[i]
                    cluster_members = cluster_members + 1 if not first else 1
                    if key not in keys:
                        keys.append(key)
                    first = False
                else:
                    flush()
                    cluster_max = px[i]
                    cluster_sum = px[i]
                    cluster_members = 1
                    keys = [key]
            flush()
            confl_count[bar] = len(centers)
            if centers:
                confl_center[bar] = centers[0]

    dist_above = pd.Series(
        np.where(np.isnan(near_above), np.nan,
                 (near_above - closes.to_numpy()) / atr_local.replace(0.0, np.nan)),
        index=index)
    dist_below = pd.Series(
        np.where(np.isnan(near_below), np.nan,
                 (closes.to_numpy() - near_below) / atr_local.replace(0.0, np.nan)),
        index=index)

    # 位集合逐 bar 长度不一：以"最近的上方/下方位"作为核心输出，
    # 全集合按名称展开成固定列（只在有值的日子有效）。
    all_names: list[str] = []
    for levels in levels_per_bar:
        for _, name in levels:
            if name not in all_names:
                all_names.append(name)
    data: dict[str, pd.Series] = {
        "pivot": pd.Series(pivot_p, index=index),
        "tc": pd.Series(pivot_tc, index=index),
        "bc": pd.Series(pivot_bc, index=index),
        "prev_high": prev_high,
        "prev_low": prev_low,
        "day_open": day_open,
        "adr": adr,
        "cpr_type": pd.Series(np.where(np.isnan(pivot_tc), "", cpr_type),
                              index=index, dtype=object),
        "bias": pd.Series(bias, index=index, dtype=object),
        "near_above": pd.Series(near_above, index=index),
        "near_below": pd.Series(near_below, index=index),
        "name_above": pd.Series(name_above, index=index, dtype=object),
        "name_below": pd.Series(name_below, index=index, dtype=object),
        "dist_above_atr": dist_above,
        "dist_below_atr": dist_below,
        "confluence_count": pd.Series(confl_count, index=index, dtype="int64"),
        "confluence_center": pd.Series(confl_center, index=index),
    }
    for name in all_names:
        column = np.full(len(frame), np.nan)
        for bar, levels in enumerate(levels_per_bar):
            for price, level_name in levels:
                if level_name == name:
                    column[bar] = price
        data[f"level_{name.lower()}"] = pd.Series(column, index=index)
    return pd.DataFrame(data, index=index)
