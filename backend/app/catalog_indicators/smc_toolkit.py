"""SMC Toolkit [btankutt] — 结构/订单块/公允价值缺口/流动性集合。

四个独立结构引擎：

1. **结构（BOS/CHoCH）**：pivot(5,5) 维护 last_swing_high/low；收盘首次穿越
   即 break。趋势初始化后：顺势 break = BOS，逆势 break = CHoCH 并翻转趋势。
2. **Order Block**：结构 break 时在 30 根内找最近反向 K（可选成交量 > SMA20
   过滤），区间为该 K 的 high/low；之后 bull OB 被 low 触顶 / bear OB 被
   high 触底即 mitigated。
3. **FVG**：三根缺口且宽度 ≥ ATR14×mult；bull FVG（low > high[2]）被 low
   触及顶部即 filled，bear FVG 镜像。容量淘汰最旧。
4. **Liquidity**：新 pivot 与近 10 个同侧 pivot 中 ≤ lookback 根旧、价差
   ≤ ATR×tolerance 的相等高/低点 → 流动性位（BSL/SSL）；扫过即 swept
   （wick-sweep 或 close-through 都算，前者打 `swept_*` 事件标志）。

丢弃：全部 box/line/label 绘图与颜色配置。

Source: pine-crypto-collection/btankutt_smc/indicators/01-smc-toolkit/smc-toolkit.pine
License: btankutt 公开脚本（随清单收录）。
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
    sma,
)

__all__ = ["smc_toolkit"]

_ATR_LENGTH = 14
_VOLUME_SMA_LENGTH = 20
_OB_LOOKBACK_BARS = 30
_PIVOT_BUFFER_SIZE = 10


def smc_toolkit(
    frame: pd.DataFrame,
    *,
    swing_length: int = 5,
    ob_max_count: int = 5,
    ob_volume_filter: bool = True,
    fvg_max_count: int = 5,
    fvg_min_size_atr: float = 0.5,
    liq_max_count: int = 10,
    liq_lookback: int = 50,
    liq_tolerance_atr: float = 0.2,
) -> pd.DataFrame:
    """计算 SMC 结构事件、订单块、FVG 与流动性位。"""
    swing_length = integer("swing_length", swing_length, 2, 50)
    ob_max_count = integer("ob_max_count", ob_max_count, 1, 50)
    ob_volume_filter = boolean("ob_volume_filter", ob_volume_filter)
    fvg_max_count = integer("fvg_max_count", fvg_max_count, 1, 50)
    fvg_min_size_atr = number("fvg_min_size_atr", fvg_min_size_atr, 0.1, 5.0)
    liq_max_count = integer("liq_max_count", liq_max_count, 1, 50)
    liq_lookback = integer("liq_lookback", liq_lookback, 10, 200)
    liq_tolerance_atr = number("liq_tolerance_atr", liq_tolerance_atr, 0.05, 1.0)

    high, low, open_, close, volume = ohlcv(frame, "high", "low", "open", "close",
                                            "volume")
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    vols = volume.to_numpy(dtype=float)
    n = len(frame)

    atr_value = atr(frame, _ATR_LENGTH)
    atr_arr = atr_value.to_numpy(dtype=float)
    vol_sma_arr = sma(volume, _VOLUME_SMA_LENGTH).to_numpy(dtype=float)

    ph_series = pivothigh(high, swing_length, swing_length)
    pl_series = pivotlow(low, swing_length, swing_length)
    ph_arr = ph_series.to_numpy(dtype=float)
    pl_arr = pl_series.to_numpy(dtype=float)

    out = {
        "last_swing_high": np.full(n, np.nan),
        "last_swing_low": np.full(n, np.nan),
        "bos_bullish": np.zeros(n, dtype=bool),
        "bos_bearish": np.zeros(n, dtype=bool),
        "choch_bullish": np.zeros(n, dtype=bool),
        "choch_bearish": np.zeros(n, dtype=bool),
        "trend_bullish": np.zeros(n, dtype=bool),
        "break_up": np.zeros(n, dtype=bool),
        "break_down": np.zeros(n, dtype=bool),
        "new_bull_ob": np.zeros(n, dtype=bool),
        "new_bear_ob": np.zeros(n, dtype=bool),
        "ob_mitigated": np.zeros(n, dtype=bool),
        "active_bull_ob": np.zeros(n, dtype=int),
        "active_bear_ob": np.zeros(n, dtype=int),
        "new_bull_fvg": np.zeros(n, dtype=bool),
        "new_bear_fvg": np.zeros(n, dtype=bool),
        "fvg_filled": np.zeros(n, dtype=bool),
        "active_bull_fvg": np.zeros(n, dtype=int),
        "active_bear_fvg": np.zeros(n, dtype=int),
        "new_bsl": np.zeros(n, dtype=bool),
        "new_ssl": np.zeros(n, dtype=bool),
        "bsl_swept": np.zeros(n, dtype=bool),
        "ssl_swept": np.zeros(n, dtype=bool),
        "active_bsl": np.zeros(n, dtype=int),
        "active_ssl": np.zeros(n, dtype=int),
    }

    last_swing_high = np.nan
    last_swing_low = np.nan
    broke_last_high = False
    broke_last_low = False
    trend_initialized = False
    is_bullish = False

    obs: list[dict] = []  # {top, bottom, created, bullish, mitigated}
    fvgs: list[dict] = []  # {top, bottom, created, bullish, filled}
    liqs: list[dict] = []  # {price, created, is_bsl, swept}
    recent_ph: list[tuple[int, float]] = []  # (pivot_bar, price)
    recent_pl: list[tuple[int, float]] = []

    for bar in range(n):
        if not np.isnan(ph_arr[bar]):
            last_swing_high = ph_arr[bar]
            broke_last_high = False
        if not np.isnan(pl_arr[bar]):
            last_swing_low = pl_arr[bar]
            broke_last_low = False
        out["last_swing_high"][bar] = last_swing_high
        out["last_swing_low"][bar] = last_swing_low

        # -- OB mitigation ---------------------------------------------------
        for zone in obs:
            if not zone["mitigated"] and bar > zone["created"]:
                hit = lows[bar] <= zone["top"] if zone["bullish"] \
                    else highs[bar] >= zone["bottom"]
                if hit:
                    zone["mitigated"] = True
                    out["ob_mitigated"][bar] = True
        # -- FVG fill --------------------------------------------------------
        for gap in fvgs:
            if not gap["filled"] and bar > gap["created"]:
                hit = lows[bar] <= gap["top"] if gap["bullish"] \
                    else highs[bar] >= gap["bottom"]
                if hit:
                    gap["filled"] = True
                    out["fvg_filled"][bar] = True
        # -- Liquidity sweep -------------------------------------------------
        for level in liqs:
            if not level["swept"] and bar > level["created"]:
                if level["is_bsl"]:
                    if highs[bar] > level["price"] and closes[bar] < level["price"]:
                        level["swept"] = True
                        out["bsl_swept"][bar] = True
                    elif closes[bar] > level["price"]:
                        level["swept"] = True
                else:
                    if lows[bar] < level["price"] and closes[bar] > level["price"]:
                        level["swept"] = True
                        out["ssl_swept"][bar] = True
                    elif closes[bar] < level["price"]:
                        level["swept"] = True

        # -- 结构 break ------------------------------------------------------
        break_up = (not np.isnan(last_swing_high) and closes[bar] > last_swing_high
                    and not broke_last_high)
        break_down = (not np.isnan(last_swing_low) and closes[bar] < last_swing_low
                      and not broke_last_low)
        if break_up:
            broke_last_high = True
        if break_down:
            broke_last_low = True
        if break_up or break_down:
            if not trend_initialized:
                is_bullish = break_up
                trend_initialized = True
            else:
                if break_up:
                    if is_bullish:
                        out["bos_bullish"][bar] = True
                    else:
                        out["choch_bullish"][bar] = True
                        is_bullish = True
                if break_down:
                    if is_bullish:
                        out["choch_bearish"][bar] = True
                        is_bullish = False
                    else:
                        out["bos_bearish"][bar] = True
        out["break_up"][bar] = break_up
        out["break_down"][bar] = break_down
        out["trend_bullish"][bar] = is_bullish

        # -- OB 登记 ---------------------------------------------------------
        if break_up:
            for offset in range(1, _OB_LOOKBACK_BARS + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] < opens[idx]:
                    if not ob_volume_filter or vols[idx] > vol_sma_arr[idx]:
                        obs.append({"top": highs[idx], "bottom": lows[idx],
                                    "created": bar, "bullish": True,
                                    "mitigated": False})
                        out["new_bull_ob"][bar] = True
                    break
        if break_down:
            for offset in range(1, _OB_LOOKBACK_BARS + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] > opens[idx]:
                    if not ob_volume_filter or vols[idx] > vol_sma_arr[idx]:
                        obs.append({"top": highs[idx], "bottom": lows[idx],
                                    "created": bar, "bullish": False,
                                    "mitigated": False})
                        out["new_bear_ob"][bar] = True
                    break
        obs = [z for z in obs if not z["mitigated"]]
        if len(obs) > ob_max_count:
            obs = obs[len(obs) - ob_max_count:]
        out["active_bull_ob"][bar] = sum(1 for z in obs if z["bullish"])
        out["active_bear_ob"][bar] = len(obs) - out["active_bull_ob"][bar]

        # -- FVG 登记 --------------------------------------------------------
        if bar > 2:
            min_size = atr_arr[bar] * fvg_min_size_atr
            if not np.isnan(min_size):
                if lows[bar] > highs[bar - 2] and lows[bar] - highs[bar - 2] >= min_size:
                    fvgs.append({"top": lows[bar], "bottom": highs[bar - 2],
                                 "created": bar, "bullish": True, "filled": False})
                    out["new_bull_fvg"][bar] = True
                if highs[bar] < lows[bar - 2] and lows[bar - 2] - highs[bar] >= min_size:
                    fvgs.append({"top": lows[bar - 2], "bottom": highs[bar],
                                 "created": bar, "bullish": False, "filled": False})
                    out["new_bear_fvg"][bar] = True
        fvgs = [g for g in fvgs if not g["filled"]]
        if len(fvgs) > fvg_max_count:
            fvgs = fvgs[len(fvgs) - fvg_max_count:]
        out["active_bull_fvg"][bar] = sum(1 for g in fvgs if g["bullish"])
        out["active_bear_fvg"][bar] = len(fvgs) - out["active_bull_fvg"][bar]

        # -- Liquidity 登记 --------------------------------------------------
        if not np.isnan(ph_arr[bar]):
            pivot_bar = bar - swing_length
            while recent_ph and pivot_bar - recent_ph[0][0] > liq_lookback:
                recent_ph.pop(0)
            equal = any(abs(ph_arr[bar] - price) <= atr_arr[bar] * liq_tolerance_atr
                        for _, price in recent_ph)
            if equal:
                liqs.append({"price": ph_arr[bar], "created": bar,
                             "is_bsl": True, "swept": False})
                out["new_bsl"][bar] = True
            recent_ph.append((pivot_bar, ph_arr[bar]))
            if len(recent_ph) > _PIVOT_BUFFER_SIZE:
                recent_ph.pop(0)
        if not np.isnan(pl_arr[bar]):
            pivot_bar = bar - swing_length
            while recent_pl and pivot_bar - recent_pl[0][0] > liq_lookback:
                recent_pl.pop(0)
            equal = any(abs(pl_arr[bar] - price) <= atr_arr[bar] * liq_tolerance_atr
                        for _, price in recent_pl)
            if equal:
                liqs.append({"price": pl_arr[bar], "created": bar,
                             "is_bsl": False, "swept": False})
                out["new_ssl"][bar] = True
            recent_pl.append((pivot_bar, pl_arr[bar]))
            if len(recent_pl) > _PIVOT_BUFFER_SIZE:
                recent_pl.pop(0)
        liqs = [l for l in liqs if not l["swept"]]
        if len(liqs) > liq_max_count:
            liqs = liqs[len(liqs) - liq_max_count:]
        out["active_bsl"][bar] = sum(1 for l in liqs if l["is_bsl"])
        out["active_ssl"][bar] = len(liqs) - out["active_bsl"][bar]

    index = frame.index
    data: dict[str, pd.Series] = {
        "last_swing_high": pd.Series(out["last_swing_high"], index=index),
        "last_swing_low": pd.Series(out["last_swing_low"], index=index),
        "trend_bullish": flags(pd.Series(out["trend_bullish"], index=index)),
    }
    for column in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
                   "break_up", "break_down", "new_bull_ob", "new_bear_ob",
                   "ob_mitigated", "new_bull_fvg", "new_bear_fvg", "fvg_filled",
                   "new_bsl", "new_ssl", "bsl_swept", "ssl_swept"):
        data[column] = flags(pd.Series(out[column], index=index))
    for column in ("active_bull_ob", "active_bear_ob", "active_bull_fvg",
                   "active_bear_fvg", "active_bsl", "active_ssl"):
        data[column] = pd.Series(out[column], index=index, dtype="int64")
    return pd.DataFrame(data, index=index)
