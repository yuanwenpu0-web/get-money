"""Crypto AMT ICT SMC Companion v1 — ICT/SMC 结构陪伴指标。

在单周期上同时跟踪四类结构并把它们合成为一个 setup 评分：

1. **Premium/Discount 区**：最近确认的 swing 高/低点构成区间（无 swing 时退回
   `range_fallback_bars` 的 highest/lowest）。中轨之上为 premium、之下为
   discount。
2. **FVG / IFVG**：三根 K 线缺口（中间实体位移 >= `fvg_min_atr_mult`×ATR(14)）；
   多头 FVG 被收盘跌破下沿 → mitigated 并生成空头 IFVG（方向反转），反之亦然。
3. **Validated OB**：FVG 触发时在附近寻找反向 K 线作为 order block，
   2 分基础 + premium/discount 2 分 + 中轨侧 1 分 + 距离 <=4 根 1 分，
   低于 `min_ob_score` 丢弃。
4. **CISD**：`last_bearish_open`/`last_bullish_open` 记录最近反向 K 开盘价，
   收盘穿越即 CISD（Change In State of Delivery）。

Bull setup 分 = discount 2 + 活跃 bull FVG 1 + 活跃 bull IFVG 2 + 最佳 bull OB 分。
丢弃：全部绘图（box/line/label/table）、dashboard、告警配置；`confirm_alerts_on_close`
语义（Pine 里信号只在 `barstate.isconfirmed` 时处理，这里逐 bar 计算等价于收盘确认）。

Source: pine-crypto-collection/latifangren_crypto_amt/indicator/crypto_amt_ict_smc_companion_v1.pine
License: 未标注（公开 TradingView 脚本，随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    flags,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
)

__all__ = ["crypto_amt_ict_smc_companion_v1"]


def _score_text(score: int) -> str:
    if score >= 7:
        return "A"
    if score >= 5:
        return "B"
    if score >= 3:
        return "C"
    return "D" if score > 0 else "-"


def crypto_amt_ict_smc_companion_v1(
    frame: pd.DataFrame,
    *,
    range_len: int = 10,
    range_fallback_bars: int = 80,
    fvg_min_atr_mult: float = 0.8,
    max_fvg_count: int = 6,
    max_ifvg_count: int = 4,
    ob_search_bars: int = 8,
    min_ob_score: int = 4,
    max_ob_count: int = 5,
    max_cisd_marks: int = 8,
    min_setup_score: int = 5,
    keep_mitigated_fvg: bool = False,
) -> pd.DataFrame:
    """计算 ICT/SMC 陪伴指标：区间、FVG/IFVG、OB、CISD 与 setup 评分。

    Parameters
    ----------
    range_len:
        Swing pivot 半径（Pine `rangeLen`，3..50）。
    range_fallback_bars:
        无 swing 时的 fallback highest/lowest 窗口（20..300）。
    fvg_min_atr_mult:
        FVG 中间实体相对 ATR(14) 的最小位移（0.1..4.0）。
    max_fvg_count / max_ifvg_count / max_ob_count:
        各结构活跃列表的容量上限，超限淘汰最旧。
    ob_search_bars:
        OB 反向 K 搜索窗口（2..20）。
    min_ob_score:
        OB 入选的最低评分（1..7）。
    max_cisd_marks:
        CISD 标记保留上限。
    min_setup_score:
        setup 触发的最低总分（3..9）。
    keep_mitigated_fvg:
        True 时被缓解的 FVG 仍参与计数（Pine `showMitigatedFvg`）。
    """
    range_len = integer("range_len", range_len, 3, 50)
    range_fallback_bars = integer("range_fallback_bars", range_fallback_bars, 20, 300)
    fvg_min_atr_mult = number("fvg_min_atr_mult", fvg_min_atr_mult, 0.1, 4.0)
    max_fvg_count = integer("max_fvg_count", max_fvg_count, 1, 20)
    max_ifvg_count = integer("max_ifvg_count", max_ifvg_count, 1, 15)
    ob_search_bars = integer("ob_search_bars", ob_search_bars, 2, 20)
    min_ob_score = integer("min_ob_score", min_ob_score, 1, 7)
    max_ob_count = integer("max_ob_count", max_ob_count, 1, 15)
    max_cisd_marks = integer("max_cisd_marks", max_cisd_marks, 1, 30)
    min_setup_score = integer("min_setup_score", min_setup_score, 3, 9)
    keep_mitigated_fvg = boolean("keep_mitigated_fvg", keep_mitigated_fvg)

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(frame)

    atr_value = atr(frame, 14)

    # -- Premium / Discount 区 -------------------------------------------------
    swing_high = pivothigh(high, range_len, range_len)
    swing_low = pivotlow(low, range_len, range_len)
    swing_high_arr = swing_high.to_numpy(dtype=float)
    swing_low_arr = swing_low.to_numpy(dtype=float)

    last_swing_high = np.nan
    last_swing_low = np.nan
    fallback_high = high.rolling(range_fallback_bars).max().to_numpy(dtype=float)
    fallback_low = low.rolling(range_fallback_bars).min().to_numpy(dtype=float)

    # -- FVG / IFVG / OB / CISD 状态 ------------------------------------------
    fvg_zones: list[dict] = []  # {born, top, bottom, bullish, mitigated}
    ifvg_zones: list[dict] = []
    ob_zones: list[dict] = []  # + score
    last_bearish_open = np.nan
    last_bullish_open = np.nan

    out: dict[str, np.ndarray] = {
        "range_high": np.full(n, np.nan),
        "range_low": np.full(n, np.nan),
        "range_mid": np.full(n, np.nan),
        "in_premium": np.zeros(n, dtype=bool),
        "in_discount": np.zeros(n, dtype=bool),
        "pd_zone": np.full(n, "NA", dtype=object),
        "new_bull_fvg": np.zeros(n, dtype=bool),
        "new_bear_fvg": np.zeros(n, dtype=bool),
        "fvg_mitigated_bull": np.zeros(n, dtype=bool),
        "fvg_mitigated_bear": np.zeros(n, dtype=bool),
        "new_bull_ifvg": np.zeros(n, dtype=bool),
        "new_bear_ifvg": np.zeros(n, dtype=bool),
        "new_bull_ob": np.zeros(n, dtype=bool),
        "new_bear_ob": np.zeros(n, dtype=bool),
        "ob_score": np.full(n, np.nan),
        "ob_grade": np.full(n, "-", dtype=object),
        "bull_cisd": np.zeros(n, dtype=bool),
        "bear_cisd": np.zeros(n, dtype=bool),
        "active_bull_fvg": np.zeros(n, dtype=int),
        "active_bear_fvg": np.zeros(n, dtype=int),
        "active_bull_ifvg": np.zeros(n, dtype=int),
        "active_bear_ifvg": np.zeros(n, dtype=int),
        "best_bull_ob_score": np.zeros(n, dtype=int),
        "best_bear_ob_score": np.zeros(n, dtype=int),
        "bull_setup_score": np.zeros(n, dtype=int),
        "bear_setup_score": np.zeros(n, dtype=int),
        "setup_state": np.full(n, "NEUTRAL", dtype=object),
        "bull_setup": np.zeros(n, dtype=bool),
        "bear_setup": np.zeros(n, dtype=bool),
    }

    for bar in range(n):
        if not np.isnan(swing_high_arr[bar]):
            last_swing_high = swing_high_arr[bar]
        if not np.isnan(swing_low_arr[bar]):
            last_swing_low = swing_low_arr[bar]
        rh = last_swing_high if not np.isnan(last_swing_high) else fallback_high[bar]
        rl = last_swing_low if not np.isnan(last_swing_low) else fallback_low[bar]
        if not (np.isnan(rh) or np.isnan(rl)) and rh > rl:
            out["range_high"][bar] = rh
            out["range_low"][bar] = rl
            out["range_mid"][bar] = (rh + rl) * 0.5
            out["in_premium"][bar] = closes[bar] > out["range_mid"][bar]
            out["in_discount"][bar] = closes[bar] < out["range_mid"][bar]
            out["pd_zone"][bar] = ("PREMIUM" if out["in_premium"][bar]
                                   else "DISCOUNT" if out["in_discount"][bar]
                                   else "EQUILIBRIUM")

        # CISD 基准：最近反向 K 开盘价（先算当根，再判穿越，与 Pine 的
        # 「收盘更新、下一根判定」等价——Pine 的 bullishCisd 使用的是
        # 之前累积的 lastBearishOpen，当根更新发生在信号判定之后）。
        out["bull_cisd"][bar] = (not np.isnan(last_bearish_open)
                                 and closes[bar] > last_bearish_open)
        out["bear_cisd"][bar] = (not np.isnan(last_bullish_open)
                                 and closes[bar] < last_bullish_open)
        if closes[bar] < opens[bar]:
            last_bearish_open = opens[bar]
        if closes[bar] > opens[bar]:
            last_bullish_open = opens[bar]

        # FVG：三根结构，以当根为第三根。
        new_bull = new_bear = False
        if bar >= 2:
            atr_prev = atr_value.iloc[bar - 1]
            if not np.isnan(atr_prev):
                mid_body = abs(closes[bar - 1] - opens[bar - 1])
                has_displacement = mid_body >= atr_prev * fvg_min_atr_mult
                bull_top = lows[bar]
                bull_bottom = highs[bar - 2]
                bear_top = lows[bar - 2]
                bear_bottom = highs[bar]
                new_bull = (bull_bottom < bull_top and closes[bar - 1] > opens[bar - 1]
                            and has_displacement)
                new_bear = (bear_top > bear_bottom and closes[bar - 1] < opens[bar - 1]
                            and has_displacement)
        if new_bull:
            fvg_zones.append({"born": bar, "top": float(lows[bar]),
                              "bottom": float(highs[bar - 2]), "bullish": True,
                              "mitigated": False})
        if new_bear:
            fvg_zones.append({"born": bar, "top": float(lows[bar - 2]),
                              "bottom": float(highs[bar]), "bullish": False,
                              "mitigated": False})
        out["new_bull_fvg"][bar] = new_bull
        out["new_bear_fvg"][bar] = new_bear

        # OB：FVG 触发时在搜索窗内找反向 K。
        if new_bull or new_bear:
            opposite_bear = opposite_bull = None
            for offset in range(1, ob_search_bars + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if opposite_bear is None and closes[idx] < opens[idx]:
                    opposite_bear = offset
                if opposite_bull is None and closes[idx] > opens[idx]:
                    opposite_bull = offset
            if new_bull and opposite_bear is not None:
                idx = bar - opposite_bear
                score = 2 + (2 if out["in_discount"][bar] else 0) \
                    + (1 if closes[bar] > (out["range_mid"][bar] if not np.isnan(out["range_mid"][bar])
                                           else closes[bar]) else 0) \
                    + (1 if opposite_bear <= 4 else 0)
                if score >= min_ob_score:
                    ob_zones.append({"born": idx, "top": float(highs[idx]),
                                     "bottom": float(lows[idx]), "bullish": True,
                                     "mitigated": False, "score": score})
                    out["new_bull_ob"][bar] = True
                    out["ob_score"][bar] = score
                    out["ob_grade"][bar] = _score_text(score)
            if new_bear and opposite_bull is not None:
                idx = bar - opposite_bull
                score = 2 + (2 if out["in_premium"][bar] else 0) \
                    + (1 if closes[bar] < (out["range_mid"][bar] if not np.isnan(out["range_mid"][bar])
                                           else closes[bar]) else 0) \
                    + (1 if opposite_bull <= 4 else 0)
                if score >= min_ob_score:
                    ob_zones.append({"born": idx, "top": float(highs[idx]),
                                     "bottom": float(lows[idx]), "bullish": False,
                                     "mitigated": False, "score": score})
                    out["new_bear_ob"][bar] = True
                    out["ob_score"][bar] = score
                    out["ob_grade"][bar] = _score_text(score)

        # FVG mitigation → IFVG。
        for zone in fvg_zones:
            if zone["mitigated"]:
                continue
            if zone["bullish"] and closes[bar] < zone["bottom"]:
                zone["mitigated"] = True
                out["fvg_mitigated_bull"][bar] = True
                ifvg_zones.append({"born": bar, "top": zone["top"],
                                   "bottom": zone["bottom"], "bullish": False,
                                   "mitigated": False})
                out["new_bear_ifvg"][bar] = True
            elif not zone["bullish"] and closes[bar] > zone["top"]:
                zone["mitigated"] = True
                out["fvg_mitigated_bear"][bar] = True
                ifvg_zones.append({"born": bar, "top": zone["top"],
                                   "bottom": zone["bottom"], "bullish": True,
                                   "mitigated": False})
                out["new_bull_ifvg"][bar] = True
        # IFVG / OB mitigation。
        for zone in ifvg_zones:
            if not zone["mitigated"]:
                if zone["bullish"] and closes[bar] < zone["bottom"]:
                    zone["mitigated"] = True
                elif not zone["bullish"] and closes[bar] > zone["top"]:
                    zone["mitigated"] = True
        for zone in ob_zones:
            if not zone["mitigated"]:
                if zone["bullish"] and closes[bar] < zone["bottom"]:
                    zone["mitigated"] = True
                elif not zone["bullish"] and closes[bar] > zone["top"]:
                    zone["mitigated"] = True
        # 容量淘汰（FVG：mitigated 且不保留的直接移除）。
        if not keep_mitigated_fvg:
            fvg_zones = [z for z in fvg_zones
                         if not z["mitigated"] or z["born"] == bar]
        ifvg_zones = [z for z in ifvg_zones if not z["mitigated"]]
        ob_zones = [z for z in ob_zones if not z["mitigated"]]
        if len(fvg_zones) > max_fvg_count:
            fvg_zones = fvg_zones[len(fvg_zones) - max_fvg_count:]
        if len(ifvg_zones) > max_ifvg_count:
            ifvg_zones = ifvg_zones[len(ifvg_zones) - max_ifvg_count:]
        if len(ob_zones) > max_ob_count:
            ob_zones = ob_zones[len(ob_zones) - max_ob_count:]

        # 活跃计数与 setup 评分。
        active_bull_f = sum(1 for z in fvg_zones if z["bullish"])
        active_bear_f = len(fvg_zones) - active_bull_f
        active_bull_i = sum(1 for z in ifvg_zones if z["bullish"])
        active_bear_i = len(ifvg_zones) - active_bull_i
        best_bull_ob = max((z["score"] for z in ob_zones if z["bullish"]),
                           default=0)
        best_bear_ob = max((z["score"] for z in ob_zones if not z["bullish"]),
                           default=0)
        bull_score = ((2 if out["in_discount"][bar] else 0) + (1 if active_bull_f > 0 else 0)
                      + (2 if active_bull_i > 0 else 0) + best_bull_ob)
        bear_score = ((2 if out["in_premium"][bar] else 0) + (1 if active_bear_f > 0 else 0)
                      + (2 if active_bear_i > 0 else 0) + best_bear_ob)
        bull_setup = bull_score >= min_setup_score and bull_score > bear_score
        bear_setup = bear_score >= min_setup_score and bear_score > bull_score

        out["active_bull_fvg"][bar] = active_bull_f
        out["active_bear_fvg"][bar] = active_bear_f
        out["active_bull_ifvg"][bar] = active_bull_i
        out["active_bear_ifvg"][bar] = active_bear_i
        out["best_bull_ob_score"][bar] = best_bull_ob
        out["best_bear_ob_score"][bar] = best_bear_ob
        out["bull_setup_score"][bar] = bull_score
        out["bear_setup_score"][bar] = bear_score
        out["bull_setup"][bar] = bull_setup
        out["bear_setup"][bar] = bear_setup
        out["setup_state"][bar] = ("BULLISH" if bull_setup else
                                   "BEARISH" if bear_setup else "NEUTRAL")

    # CISD 标记限额只影响绘图，这里把超出窗口的 CISD 边沿清零。
    for column in ("bull_cisd", "bear_cisd"):
        edges = list(np.flatnonzero(out[column]))
        if len(edges) > max_cisd_marks:
            for position in edges[:-max_cisd_marks]:
                out[column][position] = False

    index = frame.index
    data = {"range_high": pd.Series(out["range_high"], index=index),
            "range_low": pd.Series(out["range_low"], index=index),
            "range_mid": pd.Series(out["range_mid"], index=index),
            "pd_zone": pd.Series(out["pd_zone"], index=index, dtype=object),
            "new_bull_fvg": flags(pd.Series(out["new_bull_fvg"], index=index)),
            "new_bear_fvg": flags(pd.Series(out["new_bear_fvg"], index=index)),
            "fvg_mitigated_bull": flags(pd.Series(out["fvg_mitigated_bull"], index=index)),
            "fvg_mitigated_bear": flags(pd.Series(out["fvg_mitigated_bear"], index=index)),
            "new_bull_ifvg": flags(pd.Series(out["new_bull_ifvg"], index=index)),
            "new_bear_ifvg": flags(pd.Series(out["new_bear_ifvg"], index=index)),
            "new_bull_ob": flags(pd.Series(out["new_bull_ob"], index=index)),
            "new_bear_ob": flags(pd.Series(out["new_bear_ob"], index=index)),
            "ob_score": pd.Series(out["ob_score"], index=index, dtype="Int64"),
            "ob_grade": pd.Series(out["ob_grade"], index=index, dtype=object),
            "bull_cisd": flags(pd.Series(out["bull_cisd"], index=index)),
            "bear_cisd": flags(pd.Series(out["bear_cisd"], index=index))}
    for column in ("active_bull_fvg", "active_bear_fvg", "active_bull_ifvg",
                   "active_bear_ifvg", "best_bull_ob_score", "best_bear_ob_score",
                   "bull_setup_score", "bear_setup_score"):
        data[column] = pd.Series(out[column], index=index, dtype="int64")
    data["setup_state"] = pd.Series(out["setup_state"], index=index, dtype=object)
    data["bull_setup"] = flags(pd.Series(out["bull_setup"], index=index))
    data["bear_setup"] = flags(pd.Series(out["bear_setup"], index=index))
    return pd.DataFrame(data, index=index)
