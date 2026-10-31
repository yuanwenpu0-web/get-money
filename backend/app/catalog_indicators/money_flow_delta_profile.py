"""Money Flow Delta Profile [WavesUnchained] — 量价分布 + 买卖 delta。

对 `lookback` 根 bar 做 `rows` 行价格分布：

- 每根 bar 与每行的重叠比例 vPOR（bar 区间 ∩ 行区间的覆盖率）；
- 行流 = volume × vPOR ×（Money Flow 模式再乘价格：Row Mid/HL2/Typical/Close）
  ×（可选指数衰减权重）；
- **delta 拆分**：Proportional（close 在 bar 区间内的位置 = buyRatio）或
  Binary（Bar Polarity：阳线 1/阴线 0/十字 0.5；Close Location 同 Proportional）；
- 行 delta = 2×bullFlow − totFlow，|delta| 最大的行为 Delta POC；
- **POC** = totFlow 最大的行；**Value Area** = 从 POC 向两侧贪心扩展到
  `va_pct`×总量（同价取下方）；
- **节点分类**：行 share > `hvn_thresh`×行最大 → HVN，< `lvn_thresh` → LVN；
- **LVN 区**：相邻 LVN 行合并（≥ `lvn_min_rows` 行），POC 上方 Supply、下方
  Demand。

丢弃：全部 box/line/label 绘图、intrabar LTF delta（保持 close 位置近似）、
absorption wick profile（可选关）。

Source: pine-collection/indicators/money_flow/money_flow_delta_profile/money_flow_delta_profile.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    boolean,
    choice,
    highest,
    integer,
    lowest,
    number,
    ohlcv,
)

__all__ = ["money_flow_delta_profile"]


def money_flow_delta_profile(
    frame: pd.DataFrame,
    *,
    lookback: int = 200,
    rows: int = 25,
    src_kind: str = "Money Flow",
    mf_price: str = "Row Mid",
    delta_mode: str = "Proportional",
    binary_polarity: str = "Bar Polarity",
    use_decay: bool = False,
    decay_factor: float = 0.015,
    va_pct: int = 70,
    hvn_thresh: float = 0.80,
    lvn_thresh: float = 0.20,
    lvn_min_rows: int = 1,
) -> pd.DataFrame:
    """计算量价分布轮廓并在最后一根 bar 输出行级数据。"""
    lookback = integer("lookback", lookback, 10, 1500)
    rows = integer("rows", rows, 10, 100)
    src_kind = choice("src_kind", src_kind, ("Money Flow", "Volume"))
    mf_price = choice("mf_price", mf_price, ("Row Mid", "HL2", "Typical", "Close"))
    delta_mode = choice("delta_mode", delta_mode, ("Proportional", "Binary"))
    binary_polarity = choice("binary_polarity", binary_polarity,
                             ("Bar Polarity", "Close Location"))
    use_decay = boolean("use_decay", use_decay)
    decay_factor = number("decay_factor", decay_factor, 0.001, 0.10)
    va_pct = integer("va_pct", va_pct, 10, 95)
    hvn_thresh = number("hvn_thresh", hvn_thresh, 0.50, 0.99)
    lvn_thresh = number("lvn_thresh", lvn_thresh, 0.01, 0.50)
    lvn_min_rows = integer("lvn_min_rows", lvn_min_rows, 1, 20)

    high, low, open_, close, volume = ohlcv(frame, "high", "low", "open", "close",
                                            "volume")
    n = len(frame)
    index = frame.index
    empty = pd.DataFrame(index=index)

    # 逐 bar 滚动窗口（Pine 只在最后 bar 画图，这里每根 bar 输出当时的窗口轮廓）。
    p_lo_arr = lowest(low, lookback).to_numpy(dtype=float)
    p_hi_arr = highest(high, lookback).to_numpy(dtype=float)

    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    vols = volume.to_numpy(dtype=float)

    out = {
        "poc": np.full(n, np.nan),
        "vah": np.full(n, np.nan),
        "val": np.full(n, np.nan),
        "delta_poc": np.full(n, np.nan),
        "total_flow": np.full(n, np.nan),
        "net_delta": np.full(n, np.nan),
        "bull_pct_at_poc": np.full(n, np.nan),
        "hvn_count": np.zeros(n, dtype=int),
        "lvn_count": np.zeros(n, dtype=int),
        "supply_zone_top": np.full(n, np.nan),
        "supply_zone_bottom": np.full(n, np.nan),
        "demand_zone_top": np.full(n, np.nan),
        "demand_zone_bottom": np.full(n, np.nan),
        "profile_ready": np.zeros(n, dtype=bool),
    }

    for bar in range(lookback - 1, n):
        p_hi = p_hi_arr[bar]
        p_lo = p_lo_arr[bar]
        if not (p_hi > p_lo):
            continue
        step = (p_hi - p_lo) / rows
        start = max(0, bar - lookback + 1)
        tot = np.zeros(rows)
        bull = np.zeros(rows)
        for b in range(start, bar + 1):
            h_, l_, c_, o_, v_ = highs[b], lows[b], closes[b], opens[b], vols[b]
            if not (v_ > 0 and h_ > l_):
                continue
            cl_ratio = min(max((c_ - l_) / (h_ - l_), 0.0), 1.0)
            buy_ratio = cl_ratio
            if delta_mode == "Binary" and binary_polarity == "Bar Polarity":
                buy_ratio = 1.0 if c_ > o_ else 0.0 if c_ < o_ else 0.5
            weight = np.exp(-decay_factor * (bar - b)) if use_decay else 1.0
            if mf_price == "Close":
                bar_mf = c_
            elif mf_price == "HL2":
                bar_mf = (h_ + l_) / 2.0
            elif mf_price == "Typical":
                bar_mf = (h_ + l_ + c_) / 3.0
            else:
                bar_mf = 0.0
            for r in range(rows):
                row_lo = p_lo + r * step
                row_hi = row_lo + step
                if not (h_ >= row_lo and l_ < row_hi):
                    continue
                if l_ >= row_lo and h_ > row_hi:
                    v_por = (row_hi - l_) / (h_ - l_)
                elif h_ <= row_hi and l_ < row_lo:
                    v_por = (h_ - row_lo) / (h_ - l_)
                elif l_ >= row_lo and h_ <= row_hi:
                    v_por = 1.0
                else:
                    v_por = step / (h_ - l_)
                price = (p_lo + (r + 0.5) * step) if mf_price == "Row Mid" else bar_mf
                flow = v_ * v_por * (price if src_kind == "Money Flow" else 1.0) * weight
                tot[r] += flow
                bull[r] += flow * buy_ratio
        tot_max = tot.max()
        if tot_max <= 0:
            continue
        tot_sum = tot.sum()
        poc_idx = int(np.argmax(tot))
        poc_price = p_lo + (poc_idx + 0.5) * step
        delta_arr = 2.0 * bull - tot
        delta_poc_idx = int(np.argmax(np.abs(delta_arr)))
        delta_poc_price = p_lo + (delta_poc_idx + 0.5) * step
        # Value area。
        va_lo = va_hi = poc_idx
        va_acc = tot[poc_idx]
        va_target = tot_sum * va_pct / 100.0
        while va_acc < va_target:
            add_lo = tot[va_lo - 1] if va_lo > 0 else -1.0
            add_hi = tot[va_hi + 1] if va_hi < rows - 1 else -1.0
            if add_lo < 0 and add_hi < 0:
                break
            if add_lo >= add_hi:
                va_lo -= 1
                va_acc += add_lo
            else:
                va_hi += 1
                va_acc += add_hi
        vah_price = p_lo + (va_hi + 1) * step
        val_price = p_lo + va_lo * step
        shares = tot / tot_max
        hvn_count = int((shares > hvn_thresh).sum())
        lvn_count = int((shares < lvn_thresh).sum())
        # LVN 合并区（Supply 在 POC 上方 / Demand 下方）。
        supply_top = supply_bottom = demand_top = demand_bottom = np.nan
        zone_start = -1
        for r in range(rows + 1):
            is_lvn = r < rows and shares[r] < lvn_thresh
            if is_lvn and zone_start < 0:
                zone_start = r
            if not is_lvn and zone_start >= 0:
                if r - zone_start >= lvn_min_rows:
                    z_lo = p_lo + zone_start * step
                    z_hi = p_lo + r * step
                    if z_lo > poc_price:
                        supply_top, supply_bottom = z_hi, z_lo
                    elif z_hi < poc_price:
                        demand_top, demand_bottom = z_hi, z_lo
                zone_start = -1
        row_total = tot[poc_idx]
        out["poc"][bar] = poc_price
        out["vah"][bar] = vah_price
        out["val"][bar] = val_price
        out["delta_poc"][bar] = delta_poc_price
        out["total_flow"][bar] = tot_sum
        out["net_delta"][bar] = delta_arr.sum()
        out["bull_pct_at_poc"][bar] = (bull[poc_idx] / row_total * 100.0
                                       if row_total > 0 else 50.0)
        out["hvn_count"][bar] = hvn_count
        out["lvn_count"][bar] = lvn_count
        out["supply_zone_top"][bar] = supply_top
        out["supply_zone_bottom"][bar] = supply_bottom
        out["demand_zone_top"][bar] = demand_top
        out["demand_zone_bottom"][bar] = demand_bottom
        out["profile_ready"][bar] = True

    data: dict[str, pd.Series] = {
        "poc": pd.Series(out["poc"], index=index),
        "vah": pd.Series(out["vah"], index=index),
        "val": pd.Series(out["val"], index=index),
        "delta_poc": pd.Series(out["delta_poc"], index=index),
        "total_flow": pd.Series(out["total_flow"], index=index),
        "net_delta": pd.Series(out["net_delta"], index=index),
        "bull_pct_at_poc": pd.Series(out["bull_pct_at_poc"], index=index),
        "hvn_count": pd.Series(out["hvn_count"], index=index, dtype="int64"),
        "lvn_count": pd.Series(out["lvn_count"], index=index, dtype="int64"),
        "supply_zone_top": pd.Series(out["supply_zone_top"], index=index),
        "supply_zone_bottom": pd.Series(out["supply_zone_bottom"], index=index),
        "demand_zone_top": pd.Series(out["demand_zone_top"], index=index),
        "demand_zone_bottom": pd.Series(out["demand_zone_bottom"], index=index),
        "profile_ready": pd.Series(out["profile_ready"], index=index),
    }
    result = pd.concat([empty, pd.DataFrame(data, index=index)], axis=1)
    return result
