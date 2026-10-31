"""Time-to-React + Volatility-Time (BOS / Sweep / OB) — 事件反应速度评分。

核心流程：

1. **Preset 档位**：按基础周期分钟数选预设（Auto：≤20m→0、≤90m→1、≤300m→2、
   ≤10080m→3、否则 4），每档给反应窗/验证窗/ATR 乘数/权重组；Custom 用显式参数。
2. **事件源**：
   - BOS：收盘穿越前一 rolling swing 高/低（`ta.highest(high,N)[1]`）且前收盘
     未穿越（新突破）。
   - Sweep：影线穿越 swing 但收盘收回。
   - OB：位移 K（实体 > mult×ATR 且创高/低）登记反向 K 为 order block，
     之后任意 bar 的 range 与 OB 区间重叠算 tap。
   按 `priority` 只挑一个事件。
3. **反应速度**：事件后第一根"反应 K"（顺势实体或更高低点/更低高点）出现于
   `barsToReact = reacBar − eventBar − 1`，speed=100×(1−barsToReact/(reactBars−1))。
4. **验证**：A=反应后 `validBars` 内收盘越过反应 K 高/低（布尔）；
   B=事件后朝向位移 / (atrMult×事件时 ATR) 的进度。val=100×(wA·A + wB·B)。
5. **reactionScore** = clamp(0..100, wSpeed·speed + wVal·val)；事件结束或超窗后
   3 根内仍输出分数。
6. **ActivityScore**：50 + 50·tanh((ATR/ SMA(ATR, base) − 1)·k)。

丢弃：全部标签绘图与蜡烛染色。

Source: pine-collection/indicators/volatility/time_to_react_volatility_time/time_to_react_volatility_time.pine
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
    highest,
    lowest,
)

__all__ = ["time_to_react_volatility_time_bos_sweep_ob"]

# tier -> (reactMin, validMin, atrMult, wA, wB, wSpeed, wVal)
_PRESETS = (
    (150, 420, 1.2, 0.60, 0.40, 0.45, 0.55),
    (480, 1200, 1.6, 0.70, 0.30, 0.40, 0.60),
    (2160, 5760, 2.0, 0.75, 0.25, 0.35, 0.65),
    (7200, 17280, 2.4, 0.80, 0.20, 0.30, 0.70),
    (86400, 172800, 3.0, 0.85, 0.15, 0.25, 0.75),
)


def _base_step_minutes(frame: pd.DataFrame) -> float:
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        deltas = np.diff(index.view("i8"))
        positive = deltas[deltas > 0]
        if positive.size:
            return positive.min() / 60e6  # pandas 3: µs -> minutes
    raise ValueError("需要 DatetimeIndex 才能推断基础周期")


def time_to_react_volatility_time_bos_sweep_ob(
    frame: pd.DataFrame,
    *,
    preset: str = "Auto",
    custom_react_min: int = 150,
    custom_valid_min: int = 420,
    atr_len: int = 14,
    atr_base_len: int = 50,
    activity_k: float = 1.2,
    swing_lookback: int = 10,
    atr_move_mult: float = 1.2,
    w_break: float = 0.6,
    w_atr_move: float = 0.4,
    w_speed: float = 0.45,
    w_validation: float = 0.55,
    disp_body_atr: float = 1.0,
    ob_search_bars: int = 12,
    priority: str = "BOS>SWEEP>OB",
    show_labels: bool = True,
    label_alpha: int = 0,
) -> pd.DataFrame:
    """计算事件驱动的反应速度/验证/活动度评分。"""
    preset = choice("preset", preset, ("Auto", "15m", "1h", "4h", "D", "M", "Custom"))
    custom_react_min = integer("custom_react_min", custom_react_min, 5)
    custom_valid_min = integer("custom_valid_min", custom_valid_min, 10)
    atr_len = integer("atr_len", atr_len, 1)
    atr_base_len = integer("atr_base_len", atr_base_len, 5)
    activity_k = number("activity_k", activity_k, 0.1)
    swing_lookback = integer("swing_lookback", swing_lookback, 2, 100)
    atr_move_mult = number("atr_move_mult", atr_move_mult, 0.1)
    w_break = number("w_break", w_break, 0.0, 1.0)
    w_atr_move = number("w_atr_move", w_atr_move, 0.0, 1.0)
    w_speed = number("w_speed", w_speed, 0.0, 1.0)
    w_validation = number("w_validation", w_validation, 0.0, 1.0)
    disp_body_atr = number("disp_body_atr", disp_body_atr, 0.1)
    ob_search_bars = integer("ob_search_bars", ob_search_bars, 2, 50)
    priority = choice("priority", priority,
                      ("BOS>SWEEP>OB", "SWEEP>BOS>OB", "OB>BOS>SWEEP"))
    show_labels = boolean("show_labels", show_labels)
    integer("label_alpha", label_alpha, 0, 90)  # 纯展示参数，仅校验

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    closes = close.to_numpy(dtype=float)
    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(frame)

    tf_min = _base_step_minutes(frame)
    if preset == "Custom":
        react_min, valid_min, p_mult = custom_react_min, custom_valid_min, atr_move_mult
        p_wa, p_wb, p_ws, p_wv = w_break, w_atr_move, w_speed, w_validation
    else:
        tier = {"15m": 0, "1h": 1, "4h": 2, "D": 3, "M": 4}.get(
            preset, None)
        if tier is None:  # Auto
            tier = (0 if tf_min <= 20 else 1 if tf_min <= 90
                    else 2 if tf_min <= 300 else 3 if tf_min <= 10080 else 4)
        react_min, valid_min, p_mult, p_wa, p_wb, p_ws, p_wv = _PRESETS[tier]

    react_bars = max(1, int(np.ceil(react_min * 60.0 / (tf_min * 60.0))))
    valid_bars = max(1, int(np.ceil(valid_min * 60.0 / (tf_min * 60.0))))
    w_sum = max(1e-4, p_wa + p_wb)
    wa_n, wb_n = p_wa / w_sum, p_wb / w_sum
    wf_sum = max(1e-4, p_ws + p_wv)
    ws_n, wv_n = p_ws / wf_sum, p_wv / wf_sum

    atr_value = atr(frame, atr_len)
    atr_arr = atr_value.to_numpy(dtype=float)

    swing_high_ref = highest(high, swing_lookback).shift(1)
    swing_low_ref = lowest(low, swing_lookback).shift(1)
    sh_arr = swing_high_ref.to_numpy(dtype=float)
    sl_arr = swing_low_ref.to_numpy(dtype=float)

    out = {
        "event_type": np.full(n, "", dtype=object),
        "event_dir": np.zeros(n, dtype=int),
        "event_active": np.zeros(n, dtype=bool),
        "reacted": np.zeros(n, dtype=bool),
        "speed_score": np.full(n, np.nan),
        "val_break": np.zeros(n, dtype=bool),
        "val_progress": np.full(n, np.nan),
        "val_score": np.full(n, np.nan),
        "reaction_score": np.full(n, np.nan),
        "activity_score": np.full(n, np.nan),
        "bos_up": np.zeros(n, dtype=bool),
        "bos_down": np.zeros(n, dtype=bool),
        "sweep_up": np.zeros(n, dtype=bool),
        "sweep_down": np.zeros(n, dtype=bool),
        "ob_tap_bull": np.zeros(n, dtype=bool),
        "ob_tap_bear": np.zeros(n, dtype=bool),
    }

    bull_ob: tuple[float, float] | None = None  # (high, low)
    bear_ob: tuple[float, float] | None = None
    event: dict | None = None

    for bar in range(n):
        sh = sh_arr[bar]
        sl = sl_arr[bar]
        bos_up = bos_dn = swp_up = swp_dn = False
        if not np.isnan(sh):
            bos_up = (closes[bar] > sh
                      and (bar == 0 or closes[bar - 1] <= (sh)))
            swp_up = highs[bar] > sh and closes[bar] < sh
        if not np.isnan(sl):
            bos_dn = (closes[bar] < sl
                      and (bar == 0 or closes[bar - 1] >= (sl)))
            swp_dn = lows[bar] < sl and closes[bar] > sl
        out["bos_up"][bar] = bos_up
        out["bos_down"][bar] = bos_dn
        out["sweep_up"][bar] = swp_up
        out["sweep_down"][bar] = swp_dn

        # OB 登记与 tap。
        is_disp_up = (closes[bar] > opens[bar]
                      and abs(closes[bar] - opens[bar]) > disp_body_atr * atr_arr[bar]
                      and closes[bar] > highs[bar - 1]) if bar >= 1 else False
        is_disp_dn = (closes[bar] < opens[bar]
                      and abs(closes[bar] - opens[bar]) > disp_body_atr * atr_arr[bar]
                      and closes[bar] < lows[bar - 1]) if bar >= 1 else False
        if is_disp_up:
            for offset in range(1, ob_search_bars + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] < opens[idx]:
                    bull_ob = (highs[idx], lows[idx])
                    break
        if is_disp_dn:
            for offset in range(1, ob_search_bars + 1):
                idx = bar - offset
                if idx < 0:
                    break
                if closes[idx] > opens[idx]:
                    bear_ob = (highs[idx], lows[idx])
                    break
        ob_tap_bull = (bull_ob is not None and lows[bar] <= bull_ob[0]
                       and highs[bar] >= bull_ob[1])
        ob_tap_bear = (bear_ob is not None and highs[bar] >= bear_ob[1]
                       and lows[bar] <= bear_ob[0])
        out["ob_tap_bull"][bar] = ob_tap_bull
        out["ob_tap_bear"][bar] = ob_tap_bear

        # 事件挑选（仅在空闲时）。
        if event is None:
            c_bos = bos_up or bos_dn
            c_swp = swp_up or swp_dn
            c_ob = ob_tap_bull or ob_tap_bear
            fire_type = ""
            fire_dir = 0
            if priority == "BOS>SWEEP>OB":
                if c_bos:
                    fire_type, fire_dir = "BOS", 1 if bos_up else -1
                elif c_swp:
                    fire_type, fire_dir = "SWEEP", -1 if swp_up else 1
                elif c_ob:
                    fire_type = "OB"
                    fire_dir = (1 if ob_tap_bull and not (ob_tap_bear and closes[bar] < opens[bar])
                                else -1)
            elif priority == "SWEEP>BOS>OB":
                if c_swp:
                    fire_type, fire_dir = "SWEEP", -1 if swp_up else 1
                elif c_bos:
                    fire_type, fire_dir = "BOS", 1 if bos_up else -1
                elif c_ob:
                    fire_type = "OB"
                    fire_dir = (1 if ob_tap_bull and not (ob_tap_bear and closes[bar] < opens[bar])
                                else -1)
            else:
                if c_ob:
                    fire_type = "OB"
                    fire_dir = (1 if ob_tap_bull and not (ob_tap_bear and closes[bar] < opens[bar])
                                else -1)
                elif c_bos:
                    fire_type, fire_dir = "BOS", 1 if bos_up else -1
                elif c_swp:
                    fire_type, fire_dir = "SWEEP", -1 if swp_up else 1
            if fire_type:
                event = {"type": fire_type, "dir": fire_dir, "bar": bar,
                         "ref": closes[bar], "atr": atr_arr[bar],
                         "reacted": False, "reac_bar": None,
                         "reac_hi": np.nan, "reac_lo": np.nan,
                         "val_a": False, "val_b": 0.0,
                         "speed": np.nan, "val": np.nan, "score": np.nan}
                out["event_type"][bar] = fire_type
                out["event_dir"][bar] = fire_dir

        # 事件进行中：找反应 K、算 speed/val。
        if event is not None:
            age = bar - event["bar"]
            out["event_type"][bar] = event["type"]
            out["event_dir"][bar] = event["dir"]
            out["event_active"][bar] = True
            if not event["reacted"] and age >= 1:
                bull_reac = closes[bar] > opens[bar] or lows[bar] >= lows[bar - 1]
                bear_reac = closes[bar] < opens[bar] or highs[bar] <= highs[bar - 1]
                if (event["dir"] == 1 and bull_reac) or (event["dir"] == -1 and bear_reac):
                    event["reacted"] = True
                    event["reac_bar"] = bar
                    event["reac_hi"] = highs[bar]
                    event["reac_lo"] = lows[bar]
            if event["reacted"]:
                out["reacted"][bar] = True
                if np.isnan(event["speed"]):
                    bars_to_react = event["reac_bar"] - event["bar"] - 1
                    event["speed"] = 100.0 * max(0.0, min(
                        1.0, 1.0 - bars_to_react / max(1.0, react_bars - 1)))
                since_reac = bar - event["reac_bar"]
                if since_reac <= valid_bars:
                    if not event["val_a"]:
                        event["val_a"] = (closes[bar] > event["reac_hi"]
                                          if event["dir"] == 1
                                          else closes[bar] < event["reac_lo"])
                    move_target = p_mult * event["atr"]
                    move_now = (highs[bar] - event["ref"] if event["dir"] == 1
                                else event["ref"] - lows[bar])
                    event["val_b"] = max(0.0, min(1.0, move_now / move_target
                                                  if move_target > 0 else 0.0))
                    event["val"] = 100.0 * (wa_n * (1.0 if event["val_a"] else 0.0)
                                            + wb_n * event["val_b"])
                    event["score"] = max(0.0, min(100.0, ws_n * event["speed"]
                                                  + wv_n * event["val"]))
                if since_reac > valid_bars:
                    event = None
            elif age > react_bars:
                event["speed"] = event["val"] = event["score"] = 0.0
                event = None
            if event is not None:
                out["speed_score"][bar] = event["speed"]
                out["val_break"][bar] = bool(event["val_a"])
                out["val_progress"][bar] = event["val_b"]
                out["val_score"][bar] = event["val"]
                out["reaction_score"][bar] = event["score"]
        elif bar > 0:
            # 事件刚结束的 3 根余晖（Pine 的 plotScore 窗口）。
            pass

        # Activity score。
        atr_base_arr = sma(atr_value, atr_base_len).to_numpy(dtype=float)
        ratio = atr_base_arr[bar] > 0 if not np.isnan(atr_base_arr[bar]) else False
        if ratio:
            r = atr_arr[bar] / atr_base_arr[bar]
            out["activity_score"][bar] = max(0.0, min(
                100.0, 50.0 + 50.0 * np.tanh((r - 1.0) * activity_k)))

    index = frame.index
    data: dict[str, pd.Series] = {
        "event_type": pd.Series(out["event_type"], index=index, dtype=object),
        "event_dir": pd.Series(out["event_dir"], index=index, dtype="int64"),
        "event_active": flags(pd.Series(out["event_active"], index=index)),
        "reacted": flags(pd.Series(out["reacted"], index=index)),
        "speed_score": pd.Series(out["speed_score"], index=index),
        "val_break": flags(pd.Series(out["val_break"], index=index)),
        "val_progress": pd.Series(out["val_progress"], index=index),
        "val_score": pd.Series(out["val_score"], index=index),
        "reaction_score": pd.Series(out["reaction_score"], index=index),
        "activity_score": pd.Series(out["activity_score"], index=index),
        "bos_up": flags(pd.Series(out["bos_up"], index=index)),
        "bos_down": flags(pd.Series(out["bos_down"], index=index)),
        "sweep_up": flags(pd.Series(out["sweep_up"], index=index)),
        "sweep_down": flags(pd.Series(out["sweep_down"], index=index)),
        "ob_tap_bull": flags(pd.Series(out["ob_tap_bull"], index=index)),
        "ob_tap_bear": flags(pd.Series(out["ob_tap_bear"], index=index)),
    }
    return pd.DataFrame(data, index=index)
