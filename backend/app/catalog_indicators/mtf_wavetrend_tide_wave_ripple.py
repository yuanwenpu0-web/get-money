"""MTF WaveTrend Tide/Wave/Ripple [WavesUnchained] — 三层周期 WaveTrend 共振。

三层时间架构（Ripple = 图表周期，Wave = 中周期，Tide = 长周期）：

- **WaveTrend**：`esa = smooth(hlc3, n1)`、`dev = smooth(|hlc3−esa|, n1)`、
  `ci = (hlc3−esa)/(0.015·dev)`、`wt1 = smooth(ci, n2)`、`wt2 = smooth(wt1, signal)`。
  `wt_smooth` 支持 EMA / HMA / SuperSmoother / JMA Approx（后两种自实现）。
- **缩放**：Classic 原值 / Clamp ±100 / Adaptive ±100（用 `scale_lookback` 内
  `max(|wt1|,|wt2|)` 归一，交叉判定始终用**原始**值，缩放不改变交叉）。
- **相位**：`wt1>wt2` 时 oversold→Bottom(0) 否则 Up(1)；`wt1<wt2` 时
  overbought→Top(2) 否则 Down(3)。

**触发**（Ripple 层）：WT-Cross（wt1 穿 wt2）或 WT-Turn（wt1 斜率翻转），
并要求 `extreme_lookback` 内 `|WT|` 曾进入 `zone_level` 区域。

**评级 0..5**：深度（exhausted 区域 1 分 / deep 极值 2 分）+ Wave 对齐 1 +
Tide 对齐 1 + Wave-StochRSI 1 + 背离 1（上限按模块开关动态裁剪）。
`min_grade` 门控 + 同向冷却 `cooldown_bars`。

**背离**（WT pivot 版）：WT 与价格 pivot 同步，价格创新低而 WT 抬高（bull）
/ 镜像（bear），且两端至少一端 `|WT| ≥ div_min_level`。

**持仓状态机**：多/空单持仓在反向 Ripple 极值交叉或反向正式信号时平仓。

丢弃：全部绘图（hline/plot/fill/shape/table/label）、日志与 watch 标记样式。

Source: pine-collection/indicators/momentum/mtf_wavetrend_confluence/mtf_wavetrend_confluence.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    boolean,
    choice,
    crossunder,
    crossover,
    ema,
    flags,
    highest,
    hma,
    integer,
    lowest,
    number,
    ohlcv,
    pivotlow,
    pivothigh,
    rsi,
    sma,
    stoch,
    valuewhen,
)
from ._tf import calendar_frame, htf_frame

__all__ = ["mtf_wavetrend_tide_wave_ripple"]


def _supersmoother(series: pd.Series, length: int) -> pd.Series:
    """Ehlers SuperSmoother（二阶 IIR，Pine 逐 bar 递推）。"""
    s = series.to_numpy(dtype=float)
    n = len(s)
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2.0 * a1 * np.cos(1.414 * np.pi / length)
    c2, c3 = b1, -(a1 * a1)
    c1 = 1.0 - c2 - c3
    out = np.zeros(n)
    for i in range(n):
        x0 = s[i] if not np.isnan(s[i]) else 0.0
        x1 = s[i - 1] if i >= 1 and not np.isnan(s[i - 1]) else 0.0
        o1 = out[i - 1] if i >= 1 else 0.0
        o2 = out[i - 2] if i >= 2 else 0.0
        out[i] = c1 * (x0 + x1) / 2.0 + c2 * o1 + c3 * o2
    return pd.Series(out, index=series.index)


def _jma_approx(series: pd.Series, length: int, phase: float,
                power: float) -> pd.Series:
    """Pine 的 JMA 近似（e0/e1/e2 三步递推）。"""
    s = series.to_numpy(dtype=float)
    n = len(s)
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100.0 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha = beta ** power
    jma = np.full(n, np.nan)
    e0 = np.full(n, np.nan)
    e1 = np.full(n, np.nan)
    e2 = np.full(n, np.nan)
    for i in range(n):
        x = s[i] if not np.isnan(s[i]) else 0.0
        e0_prev = e0[i - 1] if i >= 1 and not np.isnan(e0[i - 1]) else x
        e1_prev = e1[i - 1] if i >= 1 and not np.isnan(e1[i - 1]) else 0.0
        jma_prev = jma[i - 1] if i >= 1 and not np.isnan(jma[i - 1]) else x
        e2_prev = e2[i - 1] if i >= 1 and not np.isnan(e2[i - 1]) else 0.0
        e0[i] = (1.0 - alpha) * x + alpha * e0_prev
        e1[i] = (x - e0[i]) * (1.0 - beta) + beta * e1_prev
        e2[i] = ((e0[i] + phase_ratio * e1[i] - jma_prev) * (1.0 - alpha) ** 2
                 + alpha ** 2 * e2_prev)
        jma[i] = jma_prev + e2[i]
    return pd.Series(jma, index=series.index)


def _smooth(series: pd.Series, length: int, mode: str, phase: float,
            power: float) -> pd.Series:
    if mode == "HMA":
        return hma(series, length)
    if mode == "SuperSmoother":
        return _supersmoother(series, length)
    if mode == "JMA Approx":
        return _jma_approx(series, length, phase, power)
    return ema(series, length)


def _wave_trend(frame: pd.DataFrame, n1: int, n2: int, signal_len: int,
                mode: str, phase: float, power: float) -> tuple[pd.Series, pd.Series]:
    """返回 (wt1_raw, wt2_raw)。"""
    hlc3 = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    esa = _smooth(hlc3, n1, mode, phase, power)
    dev_raw = _smooth((hlc3 - esa).abs(), n1, mode, phase, power)
    dev = dev_raw.abs().clip(lower=1e-10)
    ci = (hlc3 - esa) / (0.015 * dev)
    wt1 = _smooth(ci, n2, mode, phase, power)
    wt2 = _smooth(wt1, signal_len, mode, phase, power)
    return wt1, wt2


def _scale_series(values: pd.Series, mode: str, lookback: int) -> pd.Series:
    if mode == "Clamp ±100":
        return values.clip(-100.0, 100.0)
    if mode == "Adaptive ±100":
        abs_max = values.abs().rolling(lookback).max()
        return (100.0 * values / abs_max.where(abs_max != 0)).fillna(0.0)
    return values


def _phase(wt1: float, wt2: float, os_level: float, ob_level: float) -> int:
    if np.isnan(wt1) or np.isnan(wt2):
        return 3
    up = wt1 > wt2
    os_ = wt1 <= os_level
    ob = wt1 >= ob_level
    if up:
        return 0 if os_ else 1
    return 2 if ob else 3


def mtf_wavetrend_tide_wave_ripple(
    frame: pd.DataFrame,
    *,
    n1: int = 10,
    n2: int = 21,
    auto_len_htf: bool = True,
    n1_htf: int = 5,
    n2_htf: int = 11,
    signal_len: int = 4,
    signal_len_htf: int = 3,
    wt_smooth: str = "EMA",
    jma_phase: int = 50,
    jma_power: float = 2.0,
    ob_level1: float = 60.0,
    os_level1: float = -60.0,
    ob_level2: float = 100.0,
    os_level2: float = -100.0,
    scale_mode: str = "Adaptive ±100",
    scale_lookback: int = 200,
    mtf_mode: str = "Auto",
    horizon_mode: str = "Fixed",
    tide_tf_minutes: int = 1440,
    wave_tf_minutes: int = 240,
    wave_mult: int = 4,
    tide_mult: int = 24,
    trigger_mode: str = "WT-Cross",
    zone_level: float = 25.0,
    extreme_lookback: int = 8,
    use_exhaustion: bool = True,
    exhaustion_pullback: float = 8.0,
    signal_level_source: str = "Scaled ±100",
    cooldown_bars: int = 8,
    min_grade: int = 2,
    use_wave_stoch: bool = True,
    use_div: bool = True,
    use_div_grade: bool = True,
    div_lb: int = 3,
    use_price_piv_div: bool = False,
    div_min_level: float = 25.0,
    div_win: int = 6,
    sr_rsi_len: int = 14,
    sr_stoch_len: int = 14,
    sr_k: int = 3,
    sr_d: int = 3,
    sr_os: float = 20.0,
    sr_ob: float = 80.0,
) -> pd.DataFrame:
    """计算三层 WaveTrend 相位、触发、评级与持仓状态。"""
    n1 = integer("n1", n1, 2)
    n2 = integer("n2", n2, 2)
    auto_len_htf = boolean("auto_len_htf", auto_len_htf)
    n1_htf = integer("n1_htf", n1_htf, 2)
    n2_htf = integer("n2_htf", n2_htf, 2)
    signal_len = integer("signal_len", signal_len, 1)
    signal_len_htf = integer("signal_len_htf", signal_len_htf, 1)
    wt_smooth = choice("wt_smooth", wt_smooth,
                       ("EMA", "HMA", "SuperSmoother", "JMA Approx"))
    jma_phase = integer("jma_phase", jma_phase, -100, 100)
    jma_power = number("jma_power", jma_power, 1.0, 4.0)
    number("ob_level1", ob_level1, -1000.0, 1000.0)
    number("os_level1", os_level1, -1000.0, 1000.0)
    number("ob_level2", ob_level2, -1000.0, 1000.0)
    number("os_level2", os_level2, -1000.0, 1000.0)
    scale_mode = choice("scale_mode", scale_mode,
                        ("Classic", "Clamp ±100", "Adaptive ±100"))
    scale_lookback = integer("scale_lookback", scale_lookback, 20)
    mtf_mode = choice("mtf_mode", mtf_mode, ("Auto", "On", "Off"))
    horizon_mode = choice("horizon_mode", horizon_mode, ("Fixed", "Relative"))
    tide_tf_minutes = integer("tide_tf_minutes", tide_tf_minutes, 1)
    wave_tf_minutes = integer("wave_tf_minutes", wave_tf_minutes, 1)
    wave_mult = integer("wave_mult", wave_mult, 2)
    tide_mult = integer("tide_mult", tide_mult, 3)
    trigger_mode = choice("trigger_mode", trigger_mode, ("WT-Cross", "WT-Turn"))
    zone_level = number("zone_level", zone_level, 0.0, 80.0)
    extreme_lookback = integer("extreme_lookback", extreme_lookback, 1, 30)
    use_exhaustion = boolean("use_exhaustion", use_exhaustion)
    exhaustion_pullback = number("exhaustion_pullback", exhaustion_pullback, 0.0, 40.0)
    signal_level_source = choice("signal_level_source", signal_level_source,
                                 ("Scaled ±100", "Raw WT"))
    cooldown_bars = integer("cooldown_bars", cooldown_bars, 0, 100)
    min_grade = integer("min_grade", min_grade, 0, 5)
    use_wave_stoch = boolean("use_wave_stoch", use_wave_stoch)
    use_div = boolean("use_div", use_div)
    use_div_grade = boolean("use_div_grade", use_div_grade)
    div_lb = integer("div_lb", div_lb, 2, 20)
    use_price_piv_div = boolean("use_price_piv_div", use_price_piv_div)
    div_min_level = number("div_min_level", div_min_level, 0.0, 100.0)
    div_win = integer("div_win", div_win, 1, 30)
    sr_rsi_len = integer("sr_rsi_len", sr_rsi_len, 1)
    sr_stoch_len = integer("sr_stoch_len", sr_stoch_len, 1)
    sr_k = integer("sr_k", sr_k, 1)
    sr_d = integer("sr_d", sr_d, 1)
    sr_os = number("sr_os", sr_os, 0.0, 100.0)
    sr_ob = number("sr_ob", sr_ob, 0.0, 100.0)

    high, low, close = ohlcv(frame, "high", "low", "close")
    n = len(frame)
    index = frame.index
    base_minutes = 0
    if isinstance(index, pd.DatetimeIndex) and n > 2:
        deltas = np.diff(index.view("i8"))
        positive = deltas[deltas > 0]
        base_minutes = int(positive.min() / 60e6) if positive.size else 0

    is_htf = base_minutes > 240  # Pine: chartSec > 14400 -> >4h
    n1_eff = n1_htf if (auto_len_htf and is_htf) else n1
    n2_eff = n2_htf if (auto_len_htf and is_htf) else n2
    sig_eff = signal_len_htf if (auto_len_htf and is_htf) else signal_len

    if horizon_mode == "Relative" and base_minutes:
        wave_minutes = base_minutes * wave_mult
        tide_minutes = base_minutes * tide_mult
    else:
        wave_minutes = wave_tf_minutes
        tide_minutes = tide_tf_minutes

    # Ripple（图表周期）。
    r_w1, r_w2 = _wave_trend(frame, n1_eff, n2_eff, sig_eff, wt_smooth,
                             float(jma_phase), jma_power)
    r_w1_scaled = _scale_series(r_w1, scale_mode, scale_lookback)
    r_w2_scaled = _scale_series(r_w2, scale_mode, scale_lookback)
    r_w1_arr = r_w1_scaled.to_numpy(dtype=float)
    r_w2_arr = r_w2_scaled.to_numpy(dtype=float)
    r_w1_raw = r_w1.to_numpy(dtype=float)
    r_w2_raw = r_w2.to_numpy(dtype=float)

    # Wave / Tide：上一已完成周期投影（Pine lookahead_off 的因果替代）。
    def _htf_wt(minutes: int, want_stoch: bool):
        htf = htf_frame(frame, minutes).reset_index(names="__ts")
        w1, w2 = _wave_trend(htf, n1_eff, n2_eff, sig_eff, wt_smooth,
                             float(jma_phase), jma_power)
        w1_s = _scale_series(w1, scale_mode, scale_lookback)
        w2_s = _scale_series(w2, scale_mode, scale_lookback)
        key = pd.DatetimeIndex(
            frame.index.floor(f"{minutes}min") - pd.Timedelta(minutes=minutes))
        w1_lift = pd.Series(w1_s.to_numpy(), index=htf.index).reindex(key)
        w2_lift = pd.Series(w2_s.to_numpy(), index=htf.index).reindex(key)
        stoch_k = stoch_d = None
        if want_stoch:
            htf_close = htf["close"]
            r = rsi(htf_close, sr_rsi_len)
            st = stoch(r, r, r, sr_stoch_len)
            k = sma(st, sr_k)
            dd = sma(k, sr_d)
            stoch_k = pd.Series(k.to_numpy(), index=htf.index).reindex(key)
            stoch_d = pd.Series(dd.to_numpy(), index=htf.index).reindex(key)
        return (pd.Series(w1_lift.to_numpy(), index=frame.index),
                pd.Series(w2_lift.to_numpy(), index=frame.index),
                None if stoch_k is None else
                pd.Series(stoch_k.to_numpy(), index=frame.index),
                None if stoch_d is None else
                pd.Series(stoch_d.to_numpy(), index=frame.index))

    w_w1, w_w2, w_k, w_d = _htf_wt(wave_minutes, True)
    t_w1, t_w2, _, _ = _htf_wt(tide_minutes, False)
    w_w1_arr = w_w1.to_numpy(dtype=float)
    w_w2_arr = w_w2.to_numpy(dtype=float)
    t_w1_arr = t_w1.to_numpy(dtype=float)
    t_w2_arr = t_w2.to_numpy(dtype=float)
    w_k_arr = w_k.to_numpy(dtype=float)
    w_d_arr = w_d.to_numpy(dtype=float)

    use_mtf = (True if mtf_mode == "On" else False if mtf_mode == "Off"
               else (horizon_mode == "Relative" or base_minutes <= 240))
    wave_is_htf = not (base_minutes and wave_minutes
                       and base_minutes >= wave_minutes)

    # 触发。
    cross_bull = crossover(r_w1, r_w2).to_numpy(dtype=bool)
    cross_bear = crossunder(r_w1, r_w2).to_numpy(dtype=bool)
    turn_bull = np.zeros(n, dtype=bool)
    turn_bear = np.zeros(n, dtype=bool)
    for bar in range(2, n):
        if np.isnan(r_w1_raw[bar - 1]) or np.isnan(r_w1_raw[bar - 2]):
            continue
        turn_bull[bar] = (r_w1_raw[bar] > r_w1_raw[bar - 1]
                          and r_w1_raw[bar - 1] < r_w1_raw[bar - 2])
        turn_bear[bar] = (r_w1_raw[bar] < r_w1_raw[bar - 1]
                          and r_w1_raw[bar - 1] > r_w1_raw[bar - 2])
    turn_mode = trigger_mode == "WT-Turn"
    ripple_bull = turn_bull if turn_mode else cross_bull
    ripple_bear = turn_bear if turn_mode else cross_bear

    r_sig = r_w1_raw if signal_level_source == "Raw WT" else r_w1_arr
    r_sig_series = pd.Series(r_sig, index=index)
    zone_low = lowest(r_sig_series, extreme_lookback).to_numpy(dtype=float)
    zone_high = highest(r_sig_series, extreme_lookback).to_numpy(dtype=float)

    # 背离（WT pivot 版）。
    osc = pd.Series(r_w1_arr, index=index)
    osc_pl = pivotlow(osc, div_lb, div_lb)
    osc_ph = pivothigh(osc, div_lb, div_lb)
    price_pl = pivotlow(low, div_lb, div_lb)
    price_ph = pivothigh(high, div_lb, div_lb)
    osc_at_pl = osc.shift(div_lb)
    osc_at_ph = osc.shift(div_lb)
    low_at_pl = low.shift(div_lb)
    high_at_ph = high.shift(div_lb)
    cur_osc_l = valuewhen(osc_pl.notna(), osc_at_pl, 0).to_numpy(dtype=float)
    prev_osc_l = valuewhen(osc_pl.notna(), osc_at_pl, 1).to_numpy(dtype=float)
    cur_low = valuewhen(osc_pl.notna(), low_at_pl, 0).to_numpy(dtype=float)
    prev_low = valuewhen(osc_pl.notna(), low_at_pl, 1).to_numpy(dtype=float)
    cur_osc_h = valuewhen(osc_ph.notna(), osc_at_ph, 0).to_numpy(dtype=float)
    prev_osc_h = valuewhen(osc_ph.notna(), osc_at_ph, 1).to_numpy(dtype=float)
    cur_high = valuewhen(osc_ph.notna(), high_at_ph, 0).to_numpy(dtype=float)
    prev_high = valuewhen(osc_ph.notna(), high_at_ph, 1).to_numpy(dtype=float)

    # 价格 pivot 版（实验开关）。
    price_osc_l_cur = valuewhen(price_pl.notna(), osc_at_pl, 0).to_numpy(dtype=float)
    price_osc_l_prev = valuewhen(price_pl.notna(), osc_at_pl, 1).to_numpy(dtype=float)
    price_low_cur = valuewhen(price_pl.notna(), low_at_pl, 0).to_numpy(dtype=float)
    price_low_prev = valuewhen(price_pl.notna(), low_at_pl, 1).to_numpy(dtype=float)
    price_osc_h_cur = valuewhen(price_ph.notna(), osc_at_ph, 0).to_numpy(dtype=float)
    price_osc_h_prev = valuewhen(price_ph.notna(), osc_at_ph, 1).to_numpy(dtype=float)
    price_high_cur = valuewhen(price_ph.notna(), high_at_ph, 0).to_numpy(dtype=float)
    price_high_prev = valuewhen(price_ph.notna(), high_at_ph, 1).to_numpy(dtype=float)

    out = {
        "wt1": np.full(n, np.nan), "wt2": np.full(n, np.nan),
        "wt1_raw": np.full(n, np.nan), "wt2_raw": np.full(n, np.nan),
        "wave_wt1": np.full(n, np.nan), "wave_wt2": np.full(n, np.nan),
        "tide_wt1": np.full(n, np.nan), "tide_wt2": np.full(n, np.nan),
        "ripple_phase": np.zeros(n, dtype=int),
        "wave_phase": np.zeros(n, dtype=int),
        "tide_phase": np.zeros(n, dtype=int),
        "trigger_bull": np.zeros(n, dtype=bool),
        "trigger_bear": np.zeros(n, dtype=bool),
        "long_grade": np.zeros(n, dtype=int),
        "short_grade": np.zeros(n, dtype=int),
        "long_signal": np.zeros(n, dtype=bool),
        "short_signal": np.zeros(n, dtype=bool),
        "long_watch": np.zeros(n, dtype=bool),
        "short_watch": np.zeros(n, dtype=bool),
        "bull_divergence": np.zeros(n, dtype=bool),
        "bear_divergence": np.zeros(n, dtype=bool),
        "long_type": np.full(n, "", dtype=object),
        "short_type": np.full(n, "", dtype=object),
        "position_dir": np.zeros(n, dtype=int),
        "max_grade": np.full(n, 0, dtype=int),
        "active_min_grade": np.full(n, 0, dtype=int),
        "has_room_long": np.zeros(n, dtype=bool),
        "has_room_short": np.zeros(n, dtype=bool),
    }

    last_long_bar = -100000
    last_short_bar = -100000
    pos_dir = 0
    bars_since_bull_div = 10 ** 9
    bars_since_bear_div = 10 ** 9

    for bar in range(n):
        out["wt1"][bar] = r_w1_arr[bar]
        out["wt2"][bar] = r_w2_arr[bar]
        out["wt1_raw"][bar] = r_w1_raw[bar]
        out["wt2_raw"][bar] = r_w2_raw[bar]
        out["wave_wt1"][bar] = w_w1_arr[bar]
        out["wave_wt2"][bar] = w_w2_arr[bar]
        out["tide_wt1"][bar] = t_w1_arr[bar]
        out["tide_wt2"][bar] = t_w2_arr[bar]

        ripple_ph = _phase(r_w1_arr[bar], r_w2_arr[bar], os_level1, ob_level1)
        wave_ph = _phase(w_w1_arr[bar], w_w2_arr[bar], os_level1, ob_level1)
        tide_ph = _phase(t_w1_arr[bar], t_w2_arr[bar], os_level1, ob_level1)
        out["ripple_phase"][bar] = ripple_ph
        out["wave_phase"][bar] = wave_ph
        out["tide_phase"][bar] = tide_ph

        wave_up = (not np.isnan(w_w1_arr[bar]) and not np.isnan(w_w2_arr[bar])
                   and w_w1_arr[bar] > w_w2_arr[bar])
        tide_up = (not np.isnan(t_w1_arr[bar]) and not np.isnan(t_w2_arr[bar])
                   and t_w1_arr[bar] > t_w2_arr[bar])
        has_room_long = (not np.isnan(w_w1_arr[bar]) and w_w1_arr[bar] < zone_level
                         and not np.isnan(t_w1_arr[bar]) and t_w1_arr[bar] < zone_level)
        has_room_short = (not np.isnan(w_w1_arr[bar]) and w_w1_arr[bar] > -zone_level
                          and not np.isnan(t_w1_arr[bar]) and t_w1_arr[bar] > -zone_level)
        out["has_room_long"][bar] = has_room_long
        out["has_room_short"][bar] = has_room_short

        # 背离激活。
        bull_osc = (not np.isnan(osc_pl.iloc[bar])
                    and not np.isnan(cur_osc_l[bar]) and not np.isnan(prev_osc_l[bar])
                    and not np.isnan(cur_low[bar]) and not np.isnan(prev_low[bar])
                    and cur_osc_l[bar] > prev_osc_l[bar]
                    and cur_low[bar] < prev_low[bar]
                    and min(cur_osc_l[bar], prev_osc_l[bar]) <= -div_min_level)
        bull_price = (use_price_piv_div and not np.isnan(price_pl.iloc[bar])
                      and not np.isnan(price_osc_l_cur[bar])
                      and not np.isnan(price_osc_l_prev[bar])
                      and price_osc_l_cur[bar] > price_osc_l_prev[bar]
                      and price_low_cur[bar] < price_low_prev[bar]
                      and min(price_osc_l_cur[bar], price_osc_l_prev[bar])
                      <= -div_min_level)
        bear_osc = (not np.isnan(osc_ph.iloc[bar])
                    and not np.isnan(cur_osc_h[bar]) and not np.isnan(prev_osc_h[bar])
                    and not np.isnan(cur_high[bar]) and not np.isnan(prev_high[bar])
                    and cur_osc_h[bar] < prev_osc_h[bar]
                    and cur_high[bar] > prev_high[bar]
                    and max(cur_osc_h[bar], prev_osc_h[bar]) >= div_min_level)
        bear_price = (use_price_piv_div and not np.isnan(price_ph.iloc[bar])
                      and not np.isnan(price_osc_h_cur[bar])
                      and not np.isnan(price_osc_h_prev[bar])
                      and price_osc_h_cur[bar] < price_osc_h_prev[bar]
                      and price_high_cur[bar] > price_high_prev[bar]
                      and max(price_osc_h_cur[bar], price_osc_h_prev[bar])
                      >= div_min_level)
        bull_div = bull_osc or bull_price
        bear_div = bear_osc or bear_price
        bars_since_bull_div = 0 if bull_div else bars_since_bull_div + 1
        bars_since_bear_div = 0 if bear_div else bars_since_bear_div + 1
        out["bull_divergence"][bar] = bull_div
        out["bear_divergence"][bar] = bear_div

        # 深度评分。
        long_deep = (not np.isnan(zone_low[bar]) and zone_low[bar] <= os_level1)
        short_deep = (not np.isnan(zone_high[bar]) and zone_high[bar] >= ob_level1)
        long_zone = (not np.isnan(zone_low[bar]) and zone_low[bar] <= -zone_level)
        short_zone = (not np.isnan(zone_high[bar]) and zone_high[bar] >= zone_level)
        recent_low = zone_low[bar]
        recent_high = zone_high[bar]
        long_exhausted = (not use_exhaustion or turn_mode
                          or (not np.isnan(recent_low)
                              and r_sig[bar] - recent_low >= exhaustion_pullback))
        short_exhausted = (not use_exhaustion or turn_mode
                           or (not np.isnan(recent_high)
                               and recent_high - r_sig[bar] >= exhaustion_pullback))
        long_depth = 2 if long_deep else (1 if (long_exhausted and long_zone) else 0)
        short_depth = 2 if short_deep else (1 if (short_exhausted and short_zone) else 0)

        bull_prem = use_div and bars_since_bull_div <= div_win
        bear_prem = use_div and bars_since_bear_div <= div_win
        long_div_boost = use_div_grade and bull_prem
        short_div_boost = use_div_grade and bear_prem

        wave_stoch_bull = (not np.isnan(w_k_arr[bar]) and not np.isnan(w_d_arr[bar])
                           and w_k_arr[bar] > w_d_arr[bar]
                           and w_k_arr[bar] > (w_k_arr[bar - 1] if bar > 0 else np.nan)
                           and w_k_arr[bar] < sr_ob)
        wave_stoch_bear = (not np.isnan(w_k_arr[bar]) and not np.isnan(w_d_arr[bar])
                           and w_k_arr[bar] < w_d_arr[bar]
                           and w_k_arr[bar] < (w_k_arr[bar - 1] if bar > 0 else np.nan)
                           and w_k_arr[bar] > sr_os)
        wave_in_grade = use_mtf and wave_is_htf
        max_grade = ((2 if not use_mtf else (5 if use_wave_stoch else 4))
                     - (1 if (use_mtf and not wave_is_htf) else 0)
                     + (1 if use_div_grade else 0))
        active_min = min(min_grade, max_grade)
        long_grade = (long_depth + (1 if (wave_in_grade and wave_up) else 0)
                      + (1 if (use_mtf and tide_up) else 0)
                      + (1 if (use_mtf and use_wave_stoch and wave_stoch_bull) else 0)
                      + (1 if long_div_boost else 0))
        short_grade = (short_depth + (1 if (wave_in_grade and not wave_up) else 0)
                       + (1 if (use_mtf and not tide_up) else 0)
                       + (1 if (use_mtf and use_wave_stoch and wave_stoch_bear) else 0)
                       + (1 if short_div_boost else 0))
        out["long_grade"][bar] = long_grade
        out["short_grade"][bar] = short_grade
        out["max_grade"][bar] = max_grade
        out["active_min_grade"][bar] = active_min

        long_trig = bool(ripple_bull[bar] and long_zone)
        short_trig = bool(ripple_bear[bar] and short_zone)
        out["trigger_bull"][bar] = long_trig
        out["trigger_bear"][bar] = short_trig

        long_cool = bar - last_long_bar >= cooldown_bars
        short_cool = bar - last_short_bar >= cooldown_bars
        long_signal = long_trig and long_grade >= active_min and long_cool
        short_signal = short_trig and short_grade >= active_min and short_cool
        out["long_signal"][bar] = long_signal
        out["short_signal"][bar] = short_signal
        if long_signal:
            last_long_bar = bar
        if short_signal:
            last_short_bar = bar

        out["long_type"][bar] = (
            "Reversal" if (long_depth >= 2 or long_div_boost)
            else ("Continuation" if (use_mtf and long_depth <= 1
                                     and (not wave_is_htf or wave_up) and tide_up)
                  else "Mixed"))
        out["short_type"][bar] = (
            "Reversal" if (short_depth >= 2 or short_div_boost)
            else ("Continuation" if (use_mtf and short_depth <= 1
                                     and (not wave_is_htf or not wave_up) and not tide_up)
                  else "Mixed"))

        # watch：穿越极值水平。
        out["long_watch"][bar] = bool(
            bar > 0 and not np.isnan(r_sig[bar]) and not np.isnan(r_sig[bar - 1])
            and r_sig[bar] < os_level1 and r_sig[bar - 1] >= os_level1)
        out["short_watch"][bar] = bool(
            bar > 0 and not np.isnan(r_sig[bar]) and not np.isnan(r_sig[bar - 1])
            and r_sig[bar] > ob_level1 and r_sig[bar - 1] <= ob_level1)

        # 持仓状态机。
        rip_bear_top = bool(ripple_bear[bar] and short_zone)
        rip_bull_bot = bool(ripple_bull[bar] and long_zone)
        if pos_dir == 1 and (rip_bear_top or short_signal):
            pos_dir = 0
        if pos_dir == -1 and (rip_bull_bot or long_signal):
            pos_dir = 0
        if long_signal and pos_dir == 0:
            pos_dir = 1
        if short_signal and pos_dir == 0:
            pos_dir = -1
        out["position_dir"][bar] = pos_dir

    data: dict[str, pd.Series] = {
        "wt1": pd.Series(out["wt1"], index=index),
        "wt2": pd.Series(out["wt2"], index=index),
        "wt1_raw": pd.Series(out["wt1_raw"], index=index),
        "wt2_raw": pd.Series(out["wt2_raw"], index=index),
        "wave_wt1": pd.Series(out["wave_wt1"], index=index),
        "wave_wt2": pd.Series(out["wave_wt2"], index=index),
        "tide_wt1": pd.Series(out["tide_wt1"], index=index),
        "tide_wt2": pd.Series(out["tide_wt2"], index=index),
        "ripple_phase": pd.Series(out["ripple_phase"], index=index, dtype="int64"),
        "wave_phase": pd.Series(out["wave_phase"], index=index, dtype="int64"),
        "tide_phase": pd.Series(out["tide_phase"], index=index, dtype="int64"),
        "long_grade": pd.Series(out["long_grade"], index=index, dtype="int64"),
        "short_grade": pd.Series(out["short_grade"], index=index, dtype="int64"),
        "max_grade": pd.Series(out["max_grade"], index=index, dtype="int64"),
        "active_min_grade": pd.Series(out["active_min_grade"], index=index,
                                      dtype="int64"),
        "long_type": pd.Series(out["long_type"], index=index, dtype=object),
        "short_type": pd.Series(out["short_type"], index=index, dtype=object),
        "position_dir": pd.Series(out["position_dir"], index=index, dtype="int64"),
    }
    for column in ("trigger_bull", "trigger_bear", "long_signal", "short_signal",
                   "long_watch", "short_watch", "bull_divergence", "bear_divergence",
                   "has_room_long", "has_room_short"):
        data[column] = flags(pd.Series(out[column], index=index))
    return pd.DataFrame(data, index=index)
