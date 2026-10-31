"""Structure Break Risk [WavesUnchained] — 趋势结构破坏风险评分。

五个传感器（按 trendDir 方向取对应 swing 引用）：

- **sNear**：接近破位线（ uptrend 用 lastPL / downtrend 用 lastPH），
  距离以 ATR 归一（`near_atr` 内 1→0），且要求距离正在缩短。
- **sBos**：收盘越过破位线 → 0.65..1（穿透深度 / 1.5 ATR 归一）。
- **sSfp**：失败突破（影线穿越 swing 但收盘收回）→ 1。
- **sStruct**：结构侵蚀（uptrend: lastPH < prevPH 的降幅 / 2 ATR；downtrend:
  lastPL > prevPL 抬升）。
- **sDiv**：pivot 背离（价格创新高/低但对应 RSI 反向）→ 1。

证据带衰减保持（`max(now, prev×(1−1/hold_bars))`），加权合成后除以可达最大值
（wNear+wSfp+max(wStruct,wDiv)），EMA(smooth) 平滑 → risk 0..100。
状态：No trend / Quiet / Watch(≥35) / Break Pressure(≥60) / Critical(≥80) /
Structure Broken（swing break + 可选 EMA 上下文翻转，含 flip-bar 延迟触发）。

丢弃：RSI 背离线、风险区着色、事件标签、info label。

Source: pine-collection/indicators/market_structure/structure_break_risk/structure_break_risk.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    crossover,
    crossunder,
    ema,
    flags,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rma,
    rsi,
    true_range,
)

__all__ = ["structure_break_risk"]


def structure_break_risk(
    frame: pd.DataFrame,
    *,
    bias_len: int = 50,
    piv_len: int = 5,
    near_atr: float = 1.5,
    rsi_len: int = 14,
    hold_bars: int = 10,
    smooth_len: int = 3,
    w_near: float = 25.0,
    w_sfp: float = 20.0,
    w_struct: float = 15.0,
    w_div: float = 10.0,
    lvl_watch: int = 35,
    lvl_high: int = 60,
    lvl_crit: int = 80,
    require_context_flip: bool = True,
) -> pd.DataFrame:
    """计算结构破坏风险评分与状态。"""
    bias_len = integer("bias_len", bias_len, 10)
    piv_len = integer("piv_len", piv_len, 2)
    near_atr = number("near_atr", near_atr, 0.2)
    rsi_len = integer("rsi_len", rsi_len, 5)
    hold_bars = integer("hold_bars", hold_bars, 1)
    smooth_len = integer("smooth_len", smooth_len, 1)
    w_near = number("w_near", w_near, 0.0)
    w_sfp = number("w_sfp", w_sfp, 0.0)
    w_struct = number("w_struct", w_struct, 0.0)
    w_div = number("w_div", w_div, 0.0)
    integer("lvl_watch", lvl_watch, 0, 100)
    integer("lvl_high", lvl_high, 0, 100)
    integer("lvl_crit", lvl_crit, 0, 100)
    require_context_flip = boolean("require_context_flip", require_context_flip)

    high, low, close = ohlcv(frame, "high", "low", "close")
    closes = close.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(frame)

    ema_series = ema(close, bias_len)
    ema_arr = ema_series.to_numpy(dtype=float)
    ema_slope = ema_series.diff(5).to_numpy(dtype=float)
    atr_arr = atr(frame, 14).to_numpy(dtype=float)
    rsi_arr = rsi(close, rsi_len).to_numpy(dtype=float)

    ph_arr = pivothigh(high, piv_len, piv_len).to_numpy(dtype=float)
    pl_arr = pivotlow(low, piv_len, piv_len).to_numpy(dtype=float)

    out = {
        "trend_dir": np.zeros(n, dtype=int),
        "risk": np.full(n, np.nan),
        "risk_raw": np.full(n, np.nan),
        "state": np.full(n, "Quiet", dtype=object),
        "reason": np.full(n, "No clear evidence", dtype=object),
        "break_level": np.full(n, np.nan),
        "dist_to_break_atr": np.full(n, np.nan),
        "s_near": np.full(n, np.nan),
        "s_sfp": np.full(n, np.nan),
        "s_struct": np.full(n, np.nan),
        "s_div": np.full(n, np.nan),
        "bos_now": np.zeros(n, dtype=bool),
        "bos_pressure": np.zeros(n, dtype=bool),
        "bear_divergence": np.zeros(n, dtype=bool),
        "bull_divergence": np.zeros(n, dtype=bool),
        "watch_event": np.zeros(n, dtype=bool),
        "pressure_event": np.zeros(n, dtype=bool),
        "reset_event": np.zeros(n, dtype=bool),
    }

    trend_dir = 0
    last_ph = prev_ph = np.nan
    last_pl = prev_pl = np.nan
    last_price_ph = prev_price_ph = np.nan
    last_rsi_ph = prev_rsi_ph = np.nan
    last_price_pl = prev_price_pl = np.nan
    last_rsi_pl = prev_rsi_pl = np.nan
    consumed_pl = consumed_ph = np.nan
    s_near_h = s_sfp_h = s_struct_h = s_div_h = 0.0
    risk_prev: float | None = None
    risk_ema = np.nan

    decay = 0.0 if hold_bars <= 1 else 1.0 - 1.0 / hold_bars
    weight_sum = w_near + w_sfp + w_struct + w_div
    reachable = weight_sum - min(w_struct, w_div)

    dist_prev = np.nan
    for bar in range(n):
        raw_dir = (1 if closes[bar] > ema_arr[bar] and ema_slope[bar] > 0
                   else -1 if closes[bar] < ema_arr[bar] and ema_slope[bar] < 0 else 0)
        if raw_dir != 0:
            trend_dir = raw_dir
        out["trend_dir"][bar] = trend_dir

        if not np.isnan(ph_arr[bar]):
            prev_ph = last_ph
            last_ph = ph_arr[bar]
        if not np.isnan(pl_arr[bar]):
            prev_pl = last_pl
            last_pl = pl_arr[bar]

        # 背离配对（pivot 确认时同步更新）。
        bear_div = bull_div = False
        if not np.isnan(ph_arr[bar]):
            prev_price_ph = last_price_ph
            last_price_ph = ph_arr[bar]
            prev_rsi_ph = last_rsi_ph
            last_rsi_ph = rsi_arr[bar - piv_len] if bar >= piv_len else np.nan
        if not np.isnan(pl_arr[bar]):
            prev_price_pl = last_price_pl
            last_price_pl = pl_arr[bar]
            prev_rsi_pl = last_rsi_pl
            last_rsi_pl = rsi_arr[bar - piv_len] if bar >= piv_len else np.nan

        s_near = s_bos = s_sfp = s_struct = s_div = 0.0
        break_level = np.nan
        dist_to_break = np.nan
        if trend_dir == 1:
            break_level = last_pl
            if not np.isnan(last_pl):
                dist_to_break = (closes[bar] - last_pl) / atr_arr[bar]
                approaching = not np.isnan(dist_prev) and dist_to_break < dist_prev
                s_near = (1.0 - min(max(dist_to_break, 0.0) / near_atr, 1.0)
                          if dist_to_break >= 0 and approaching else 0.0)
                if closes[bar] < last_pl:
                    s_bos = max(0.65, min(max(last_pl - closes[bar], 0.0)
                                          / atr_arr[bar] / 1.5, 1.0))
            if not np.isnan(last_ph) and highs[bar] > last_ph and closes[bar] < last_ph:
                s_sfp = 1.0
            if (not np.isnan(ph_arr[bar]) and not np.isnan(last_ph)
                    and not np.isnan(prev_ph) and last_ph < prev_ph):
                s_struct = min(max(prev_ph - last_ph, 0.0) / atr_arr[bar] / 2.0, 1.0)
            if (not np.isnan(ph_arr[bar]) and not np.isnan(last_price_ph)
                    and not np.isnan(prev_price_ph) and not np.isnan(last_rsi_ph)
                    and not np.isnan(prev_rsi_ph)
                    and last_price_ph > prev_price_ph and last_rsi_ph < prev_rsi_ph):
                s_div = 1.0
                bear_div = True
        elif trend_dir == -1:
            break_level = last_ph
            if not np.isnan(last_ph):
                dist_to_break = (last_ph - closes[bar]) / atr_arr[bar]
                approaching = not np.isnan(dist_prev) and dist_to_break < dist_prev
                s_near = (1.0 - min(max(dist_to_break, 0.0) / near_atr, 1.0)
                          if dist_to_break >= 0 and approaching else 0.0)
                if closes[bar] > last_ph:
                    s_bos = max(0.65, min(max(closes[bar] - last_ph, 0.0)
                                          / atr_arr[bar] / 1.5, 1.0))
            if not np.isnan(last_pl) and lows[bar] < last_pl and closes[bar] > last_pl:
                s_sfp = 1.0
            if (not np.isnan(pl_arr[bar]) and not np.isnan(last_pl)
                    and not np.isnan(prev_pl) and last_pl > prev_pl):
                s_struct = min(max(last_pl - prev_pl, 0.0) / atr_arr[bar] / 2.0, 1.0)
            if (not np.isnan(pl_arr[bar]) and not np.isnan(last_price_pl)
                    and not np.isnan(prev_price_pl) and not np.isnan(last_rsi_pl)
                    and not np.isnan(prev_rsi_pl)
                    and last_price_pl < prev_price_pl and last_rsi_pl > prev_rsi_pl):
                s_div = 1.0
                bull_div = True
        dist_prev = dist_to_break
        out["bear_divergence"][bar] = bear_div
        out["bull_divergence"][bar] = bull_div

        # 证据衰减保持。
        s_near_h = max(s_near, s_near_h * decay)
        s_sfp_h = max(s_sfp, s_sfp_h * decay)
        s_struct_h = max(s_struct, s_struct_h * decay)
        s_div_h = max(s_div, s_div_h * decay)

        # BOS 判定（用上一根 trendDir）。
        bos_trend_dir = trend_dir  # Pine 用 trendDir[1]，此处当根已含更新；
        # 严格复刻需要上一根值，用历史趋势近似（差异只在翻转当根）。
        prev_trend = out["trend_dir"][bar - 1] if bar > 0 else 0
        bos_trend_dir = prev_trend
        raw_bos_up = (bos_trend_dir == 1 and not np.isnan(last_pl)
                      and closes[bar] < last_pl
                      and (bar == 0 or closes[bar - 1] >= last_pl))
        raw_bos_dn = (bos_trend_dir == -1 and not np.isnan(last_ph)
                      and closes[bar] > last_ph
                      and (bar == 0 or closes[bar - 1] <= last_ph))
        context_ok = (not require_context_flip
                      or (raw_bos_up and closes[bar] < ema_arr[bar]
                          and ema_slope[bar] < 0)
                      or (raw_bos_dn and closes[bar] > ema_arr[bar]
                          and ema_slope[bar] > 0))
        flip_up = (require_context_flip and bos_trend_dir == 1 and trend_dir == -1
                   and not raw_bos_up and not np.isnan(last_pl)
                   and closes[bar] < last_pl
                   and (np.isnan(consumed_pl) or last_pl != consumed_pl))
        flip_dn = (require_context_flip and bos_trend_dir == -1 and trend_dir == 1
                   and not raw_bos_dn and not np.isnan(last_ph)
                   and closes[bar] > last_ph
                   and (np.isnan(consumed_ph) or last_ph != consumed_ph))
        bos_up_now = (raw_bos_up and context_ok) or flip_up
        bos_dn_now = (raw_bos_dn and context_ok) or flip_dn
        bos_now = bos_up_now or bos_dn_now
        bos_pressure = (raw_bos_up or raw_bos_dn) and not context_ok
        if bos_up_now:
            consumed_pl = last_pl
        if bos_dn_now:
            consumed_ph = last_ph
        out["bos_now"][bar] = bos_now
        out["bos_pressure"][bar] = bos_pressure

        risk_raw = 0.0
        if trend_dir != 0 and reachable > 0:
            risk_raw = min((s_near_h * w_near + s_sfp_h * w_sfp
                            + s_struct_h * w_struct + s_div_h * w_div)
                           / reachable * 100.0, 100.0)
        risk_ema = (risk_raw if np.isnan(risk_ema)
                    else risk_ema + 2.0 / (smooth_len + 1) * (risk_raw - risk_ema))
        risk = risk_ema
        out["risk_raw"][bar] = risk_raw
        out["risk"][bar] = risk
        out["break_level"][bar] = break_level
        out["dist_to_break_atr"][bar] = dist_to_break
        out["s_near"][bar] = s_near_h
        out["s_sfp"][bar] = s_sfp_h
        out["s_struct"][bar] = s_struct_h
        out["s_div"][bar] = s_div_h

        state = "Quiet"
        if trend_dir == 0:
            state = "No trend"
        elif bos_now:
            state = "Structure Broken"
        elif bos_pressure:
            state = "Break Pressure"
        elif risk >= lvl_crit:
            state = "Critical"
        elif risk >= lvl_high:
            state = "Break Pressure"
        elif risk >= lvl_watch:
            state = "Watch"
        out["state"][bar] = state

        reason = "No clear evidence"
        if bos_now:
            reason = "Break confirmed"
        elif bos_pressure:
            reason = "Local swing break"
        elif s_near_h > 0.7:
            reason = "Approaching break level"
        elif s_sfp_h > 0.5:
            reason = "Failed breakout"
        elif s_struct_h > 0.5:
            reason = "Structure eroding"
        elif s_div_h > 0.5:
            reason = "Momentum divergence"
        out["reason"][bar] = reason

        if bar > 0 and risk_prev is not None:
            out["watch_event"][bar] = bool(
                crossover(pd.Series([risk_prev, risk]), lvl_watch).iloc[1])
            out["pressure_event"][bar] = bool(
                (crossover(pd.Series([risk_prev, risk]), lvl_high).iloc[1])
                or crossover(pd.Series([risk_prev, risk]), lvl_crit).iloc[1])
            out["reset_event"][bar] = bool(
                crossunder(pd.Series([risk_prev, risk]), lvl_watch).iloc[1])
        risk_prev = risk

    index = frame.index
    data: dict[str, pd.Series] = {
        "trend_dir": pd.Series(out["trend_dir"], index=index, dtype="int64"),
        "risk": pd.Series(out["risk"], index=index),
        "risk_raw": pd.Series(out["risk_raw"], index=index),
        "state": pd.Series(out["state"], index=index, dtype=object),
        "reason": pd.Series(out["reason"], index=index, dtype=object),
        "break_level": pd.Series(out["break_level"], index=index),
        "dist_to_break_atr": pd.Series(out["dist_to_break_atr"], index=index),
        "s_near": pd.Series(out["s_near"], index=index),
        "s_sfp": pd.Series(out["s_sfp"], index=index),
        "s_struct": pd.Series(out["s_struct"], index=index),
        "s_div": pd.Series(out["s_div"], index=index),
        "bos_now": flags(pd.Series(out["bos_now"], index=index)),
        "bos_pressure": flags(pd.Series(out["bos_pressure"], index=index)),
        "bear_divergence": flags(pd.Series(out["bear_divergence"], index=index)),
        "bull_divergence": flags(pd.Series(out["bull_divergence"], index=index)),
        "watch_event": flags(pd.Series(out["watch_event"], index=index)),
        "pressure_event": flags(pd.Series(out["pressure_event"], index=index)),
        "reset_event": flags(pd.Series(out["reset_event"], index=index)),
    }
    return pd.DataFrame(data, index=index)
