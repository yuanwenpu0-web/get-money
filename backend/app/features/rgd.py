"""RGD v2 纯因果特征：通道 / 效率 / HTF 张力 / 相关量能 / 防守 / 状态机 / 入场价位。

全部列以 `rgd_` 前缀输出（契约 §6.4），只依赖 `<= i` 的 OHLCV 与已注入的 HTF 列。
两处刻意不放进这里：`stopLvl/tp1Lvl/entryRef/tp1Hit/lastExit` 是成交依赖状态，
归专用 broker（契约 §5.1）；HTF 帧由 `features/rgd_htf.attach_htf_levels` 预注入。

`rgd_state` 是纯状态机（契约 §8 R9）：转移只读 `prevState + longBrk/shortBrk +
close/defL/defS`，不含任何成交态，因此整帧与 `frame.iloc[:k]` 截断重算逐位一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import interval_ms
from .rgd_indicators import (
    percentrank_preceding,
    prior_extreme,
    rolling_pearson,
    state_machine,
    tension_series,
    wilder_atr,
)

# `syminfo.mintick` 在项目内没有每币数据，固定 1e-8 作除零/退化保护（契约 §8 R5）。
MINTICK = 1e-8
# `htf_in` 枚举 -> 项目周期；`autoTF()` 的 Pine 返回 -> 项目周期（契约 §4.1）。
HTF_INTERVALS = {0: "1h", 1: "4h", 2: "1d"}
HTF_BY_PINE = {"60": "1h", "240": "4h", "1D": "1d"}
# 会被 broker 消费的决策列。随机基线必须循环位移这一整组，而不是只翻 direction。
DECISION_COLUMNS = (
    "rgd_long_break", "rgd_short_break", "rgd_long_pull_core", "rgd_short_pull_core",
    "rgd_allow_long", "rgd_allow_short", "rgd_state",
    "rgd_def_long", "rgd_def_short",
    "rgd_stp_long", "rgd_rd_long", "rgd_tp1_long",
    "rgd_stp_short", "rgd_rd_short", "rgd_tp1_short",
)


def auto_tf(base_interval: str) -> str:
    """Pine `autoTF()`：按 base 周期秒数返回 HTF 字符串（契约 §4.1 完整表）。"""
    seconds = interval_ms(base_interval) / 1000.0
    if seconds < 900:
        return "60"
    if seconds < 3600:
        return "240"
    if seconds < 86400:
        return "1D"
    if seconds < 604800:
        return "1W"
    return "1M"


def htf_interval_for(base_interval: str, auto_htf: bool, htf_in: int) -> str:
    """解析实际 HTF 周期；不支持时显式 raise，不静默退回 base（契约 §8 R1）。"""
    if auto_htf:
        pine = auto_tf(base_interval)
        if pine not in HTF_BY_PINE:
            raise ValueError(
                f"base={base_interval} 的 autoTF()={pine} 不在项目周期白名单"
                f"（1m/5m/15m/1h/4h/1d），请改 auto_htf=False + htf_in∈{{0,1,2}}"
            )
        return HTF_BY_PINE[pine]
    try:
        return HTF_INTERVALS[int(htf_in)]
    except KeyError as exc:
        raise ValueError(f"htf_in 枚举越界: {htf_in}（0=1h 1=4h 2=1d）") from exc


def _ohlcv(data: pd.DataFrame):
    def num(name: str) -> pd.Series:
        return pd.to_numeric(data[name], errors="coerce").astype(float)

    volume = (
        num("volume") if "volume" in data.columns
        else pd.Series(np.nan, index=data.index, dtype=float)
    )
    return num("open"), num("high"), num("low"), num("close"), volume


def _channel(high, low, high_vol, params) -> dict:
    up_fast = prior_extreme(high, int(params["brk_fast"]), "max")
    lo_fast = prior_extreme(low, int(params["brk_fast"]), "min")
    up_slow = prior_extreme(high, int(params["brk_slow"]), "max")
    lo_slow = prior_extreme(low, int(params["brk_slow"]), "min")
    ent_up = np.where(high_vol, up_slow, up_fast).astype(float)
    ent_lo = np.where(high_vol, lo_slow, lo_fast).astype(float)
    return {
        "rgd_up_fast": up_fast, "rgd_lo_fast": lo_fast,
        "rgd_up_slow": up_slow, "rgd_lo_slow": lo_slow,
        "rgd_ent_up": ent_up, "rgd_ent_lo": ent_lo,
        "rgd_mid": (ent_up + ent_lo) * 0.5,
        "rgd_chan_w": np.maximum(ent_up - ent_lo, MINTICK),
    }


def _efficiency(close: pd.Series, er_len: int) -> dict:
    noise = close.diff().abs().rolling(er_len, min_periods=er_len).sum()
    # noise==0 时 np.where 仍会求值除法分支：取值由 where 选 0.0（正确），
    # 但会在预热段刷 divide/invalid warning 污染回测日志，这里显式静默。
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(
            (noise > MINTICK).to_numpy(dtype=bool),
            (close - close.shift(er_len)).abs().to_numpy() / noise.to_numpy(),
            0.0,
        )
    er = pd.Series(raw, index=close.index, dtype=float).ewm(span=3, adjust=False).mean()
    return {"rgd_noise": noise.to_numpy(dtype=float), "rgd_er": er.to_numpy(dtype=float)}


def _tension(close: pd.Series, er: np.ndarray, data: pd.DataFrame, params) -> dict:
    """L89-99：hh/ll 来自上一根已完成 HTF bar，`np` 却用 base 当前 close（混合口径）。"""
    n = len(close)
    if {"rgd_htf_hh", "rgd_htf_ll"}.issubset(data.columns):
        hh = pd.to_numeric(data["rgd_htf_hh"], errors="coerce").to_numpy(dtype=float)
        ll = pd.to_numeric(data["rgd_htf_ll"], errors="coerce").to_numpy(dtype=float)
    else:
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
    hr = hh - ll
    h_ok = np.isfinite(hh) & np.isfinite(ll) & (hr > 0)
    hm = (hh + ll) * 0.5
    close_v = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        np_v = np.where(h_ok, 2.0 * (close_v - hm) / np.maximum(hr, MINTICK), np.nan)
    max_ten = float(params["max_ten"])
    # `not na(np) ? np*er : 0.0`：er 为 na 时乘积为 na，Pine 的 min/max 会让 tg 变 na，
    # 下一根由 nz(tension[1],0) 自愈（契约 A2）。
    with np.errstate(invalid="ignore"):
        tg = np.clip(np.where(np.isfinite(np_v), np_v * er, 0.0), -max_ten, max_ten)
    tension = tension_series(
        tg, int(params["build_len"]), int(params["rel_len"]), max_ten
    )
    ten_th = float(params["ten_th"])
    regime = np.where(tension >= ten_th, 1, np.where(tension <= -ten_th, -1, 0))
    return {
        "rgd_htf_hh": hh, "rgd_htf_ll": ll, "rgd_htf_ok": h_ok,
        "rgd_hr": hr, "rgd_hm": hm, "rgd_np": np_v, "rgd_tg": tg,
        "rgd_tension": tension, "rgd_rg": regime.astype(int),
    }


def _corr_volume(close: pd.Series, volume: pd.Series, params) -> dict:
    # `ta.correlation(close, bar_index, len)`：段内 0-based 索引与 Pine 绝对索引仿射等价。
    bar_index = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    corr = rolling_pearson(close, bar_index, int(params["corr_len"])).fillna(0.0).to_numpy(float)
    volume_v = volume.to_numpy(dtype=float)
    sma = volume.rolling(20, min_periods=1).mean().to_numpy(dtype=float)
    has_vol = np.isfinite(volume_v) & (volume_v > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_vol = np.where(has_vol, volume_v / sma, 1.0)
    vol_ok = np.ones(len(close), dtype=bool)
    if params["use_vol"]:
        # `not useVol or not hasVol or relVol >= volMin`：无成交量的币直接 bypass。
        vol_ok = (~has_vol) | (rel_vol >= float(params["vol_min"]))
    return {"rgd_corr": corr, "rgd_rel_vol": rel_vol, "rgd_vol_ok": vol_ok}


def _defense(high, low, close, ent_lo, chan_w, params) -> dict:
    f_h = prior_extreme(high, int(params["def_fast"]), "max")
    f_l = prior_extreme(low, int(params["def_fast"]), "min")
    s_h = prior_extreme(high, int(params["def_slow"]), "max")
    s_l = prior_extreme(low, int(params["def_slow"]), "min")
    close_v = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.clip((close_v - ent_lo) / chan_w, 0.0, 1.0)
    acceptance = pd.Series(pos, index=close.index).ewm(
        span=int(params["acc_len"]), adjust=False
    ).mean().to_numpy(dtype=float)
    bull_q = np.clip((acceptance - 0.35) / 0.55, 0.0, 1.0)
    bear_q = np.clip((0.65 - acceptance) / 0.55, 0.0, 1.0)
    depth = float(params["max_depth"])
    return {
        "rgd_f_h": f_h, "rgd_f_l": f_l, "rgd_s_h": s_h, "rgd_s_l": s_l,
        "rgd_pos": pos, "rgd_acceptance": acceptance,
        "rgd_bull_q": bull_q, "rgd_bear_q": bear_q,
        "rgd_def_long": np.maximum(f_l, s_l) * (1 - depth * bull_q)
        + np.minimum(f_l, s_l) * (depth * bull_q),
        "rgd_def_short": np.minimum(f_h, s_h) * (1 - depth * bear_q)
        + np.maximum(f_h, s_h) * (depth * bear_q),
    }


def _entry_levels(close, atr, def_long, def_short, params) -> dict:
    close_v = close.to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)
    max_stop, min_stop = float(params["max_stop_atr"]), float(params["min_stop_atr"])
    stp_long = np.minimum(np.maximum(def_long, close_v - max_stop * atr_v), close_v - min_stop * atr_v)
    stp_short = np.maximum(np.minimum(def_short, close_v + max_stop * atr_v), close_v + min_stop * atr_v)
    rd_long = close_v - stp_long
    rd_short = stp_short - close_v
    return {
        "rgd_stp_long": stp_long, "rgd_rd_long": rd_long,
        "rgd_tp1_long": close_v + float(params["tp1_r"]) * rd_long,
        "rgd_stp_short": stp_short, "rgd_rd_short": rd_short,
        "rgd_tp1_short": close_v - float(params["tp1_r"]) * rd_short,
    }


def build_features(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """产出契约 §6.4 的全部 `rgd_*` 列；HTF 列缺失时降级为 `rgd_htf_ok=False`。

    降级路径只为 live 单根直调保留（`PrepareContext.service is None`）；回测链路
    必须先在整帧 attach HTF 再调用本函数，脚本层用断言拦住静默零成交（§7.3.1）。
    """
    data = frame.copy()
    open_, high, low, close, volume = _ohlcv(data)
    atr = wilder_atr(high, low, close, int(params["atr_len"]))
    data["rgd_atr"] = atr.to_numpy(dtype=float)
    data["atr_pct"] = (atr / close.replace(0, np.nan)).to_numpy(dtype=float)

    vol_rank = percentrank_preceding(
        (atr / close.replace(0, np.nan)).to_numpy(dtype=float), int(params["rank_len"])
    )
    # `volRank > rankCut` 严格大于；预热期 na 比较为 False（保守，不入场）。
    high_vol = np.asarray(vol_rank > float(params["rank_cut"]), dtype=bool)
    data["rgd_vol_rank"] = vol_rank
    data["rgd_high_vol"] = high_vol

    for group in (_channel(high, low, high_vol, params), _efficiency(close, int(params["er_len"]))):
        for key, value in group.items():
            data[key] = value
    er = data["rgd_er"].to_numpy(dtype=float)
    er_ok = er >= float(params["er_min"])
    data["rgd_er_ok"] = er_ok

    for key, value in _tension(close, er, data, params).items():
        data[key] = value
    for key, value in _corr_volume(close, volume, params).items():
        data[key] = value

    regime = data["rgd_rg"].to_numpy(dtype=int)
    corr = data["rgd_corr"].to_numpy(dtype=float)
    corr_min = float(params["corr_min"])
    bull_ok = ((regime == 1) | (not params["use_ten"])) & (
        (corr >= corr_min) | (not params["use_corr"])
    )
    bear_ok = ((regime == -1) | (not params["use_ten"])) & (
        (corr <= -corr_min) | (not params["use_corr"])
    )
    data["rgd_bull_ok"] = bull_ok
    data["rgd_bear_ok"] = bear_ok

    for key, value in _defense(high, low, close, data["rgd_ent_lo"].to_numpy(float),
                               data["rgd_chan_w"].to_numpy(float), params).items():
        data[key] = value

    ent_up = data["rgd_ent_up"].to_numpy(dtype=float)
    ent_lo = data["rgd_ent_lo"].to_numpy(dtype=float)
    mid = data["rgd_mid"].to_numpy(dtype=float)
    def_long = data["rgd_def_long"].to_numpy(dtype=float)
    def_short = data["rgd_def_short"].to_numpy(dtype=float)
    close_v, open_v = close.to_numpy(dtype=float), open_.to_numpy(dtype=float)
    low_v, high_v = low.to_numpy(dtype=float), high.to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)
    ready = (
        np.isfinite(ent_up) & np.isfinite(ent_lo) & np.isfinite(def_long)
        & np.isfinite(def_short) & np.isfinite(er) & np.isfinite(atr_v)
    )
    buf = float(params["brk_buf"])
    brk_long = ready & (close_v > ent_up + buf * atr_v) & er_ok & bull_ok & data["rgd_vol_ok"].to_numpy(bool)
    brk_short = ready & (close_v < ent_lo - buf * atr_v) & er_ok & bear_ok & data["rgd_vol_ok"].to_numpy(bool)
    states, prevs, long_brk, short_brk = state_machine(brk_long, brk_short, close_v, def_long, def_short)
    pull_on = bool(params["use_pull"])
    long_pull = (pull_on & ready & (prevs == 1) & bull_ok & er_ok
                 & (low_v <= mid) & (close_v > mid) & (close_v > open_v))
    short_pull = (pull_on & ready & (prevs == -1) & bear_ok & er_ok
                  & (high_v >= mid) & (close_v < mid) & (close_v < open_v))
    dir_mode = int(params["dir_mode"])
    n = len(data)
    data["rgd_ready"] = ready
    data["rgd_allow_long"] = np.full(n, dir_mode != 2, dtype=bool)
    data["rgd_allow_short"] = np.full(n, dir_mode != 1, dtype=bool)
    data["rgd_prev_state"] = prevs
    data["rgd_state"] = states
    data["rgd_long_break"] = long_brk
    data["rgd_short_break"] = short_brk
    data["rgd_long_pull_core"] = long_pull
    data["rgd_short_pull_core"] = short_pull

    for key, value in _entry_levels(close, atr, def_long, def_short, params).items():
        data[key] = value
    return data
