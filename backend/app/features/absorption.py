"""Absorption Reversal（吸收反转）特征层 — Pine 脚本 `Absorption Reversal` 的本地适配。

原始脚本（TradingView, © blitz_locked）的核心思想：**吸收 = 一边主动成交被另一边全部吃掉**。
放量单向进攻（买 or 卖）却推不出价格、反而留下长影线 → 进攻方被困，价格向反方向反转。

判定一根 bar 是「吸收 bar」需同时满足五个条件（脚本原文 c1..c5）：
1. |delta| 位于最近 `delta_window`(500) 根的 `delta_pct`(95) 最近秩百分位之上（强弱自适应）；
2. 实体极小（|close-open| <= body_mult × ATR）——价格没有进展；或 delta 与 K 线方向不一致；
3. 出现在摆动极值处（买吸收 high >= 近 N 根最高，卖吸收 low <= 近 N 根最低）；
4. 有拒绝影线：影线占全幅比例 >= wick_mult；
5. 单边支配度：|delta| >= dominance_pct % × volume（可关：dominance_pct=0 等价关闭）。

方向语义（`abs_dir`）：**买方被吸收 → 后续做空**；**卖方被吸收 → 后续做多**。
入场不是见到吸收 bar 就进（原脚本默认 `Close beyond body`）：要等后续最多
`conf_window_bars` 根内收盘重新穿越吸收 bar 的实体边（进攻方被困），才算信号；
期间若价格先打穿吸收极值，或窗口超时，信号作废。

本地实现与脚本的三处已知差异（均已显式化，不做隐式改动）：
1. **delta 来源**：脚本优先用更低周期 intrabar delta（`request.security_lower_tf`），
   本地没有逐笔/更低周期净主动成交数据，因此走脚本自带的降级分支
   `na(ltfDelta) ? estDelta : ltfDelta`，即用 K 线形态估计
   `delta = volume * (2*close - high - low) / (high - low)`。脚本在无 lower-tf 数据
   （如部分品种/周期）时同样退化为该分支，因此本实现等价于脚本的「bar shape (est.)」模式。
2. **信号间隔节流的优先级**：脚本原文为
   `bool isAbs = absRaw and (na(lastAbs)) or bar_index - lastAbs > absGap`，
   Pine 中 `and` 优先级高于 `or`，实际语义是 `(absRaw and na(lastAbs)) or (gap > absGap)`，
   即首个吸收之后，**只要距上次信号超过 absGap 根就判定为吸收 bar，与 absRaw 无关**——
   这会把策略退化成「每 11 根随机入场一次」。本模块默认按修正语义实现
   （`absRaw and (na(lastAbs) or gap > absGap)`）；`legacy_gap_logic=True` 可 1:1 复现脚本行为。
3. **百分位**：Pine `ta.percentile_nearest_rank` 未公开实现，本模块采用 NIST
   nearest-rank 定义（第 ceil(p/100×N) 小值，窗口包含当前 bar），为 order statistic，
   rank 定义差异 ±1 位对阈值影响可忽略。

所有派生列以 `abs_` 前缀命名；`evaluate_signals` 只读单行，回测/实盘都先对整帧调用本模块。
状态机虽是逐 bar 推进（等确认 N 根），但每根只依赖当期与已保存的历史状态，
因此截断到最后一根重算的结果完全一致（见 tests/test_absorption_reversal.py 的因果性用例）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 脚本的 Sensitivity 三档预设（保留在代码里供脚本/调参映射，参数本身是平铺的数值）
SENSITIVITY_PRESETS = {
    "low": {"delta_pct": 97.5, "body_mult": 0.20, "extreme_lookback": 40, "wick_mult": 0.30},
    "medium": {"delta_pct": 95.0, "body_mult": 0.25, "extreme_lookback": 20, "wick_mult": 0.20},
    "high": {"delta_pct": 90.0, "body_mult": 0.35, "extreme_lookback": 10, "wick_mult": 0.15},
}

ABSORPTION_COLUMNS = (
    "abs_delta",        # 估计净主动成交（正=买方主动占优）
    "abs_abs_delta",    # |delta|
    "abs_dominance",    # |delta| / volume × 100
    "abs_percentile",   # 滚动窗口的最近秩百分位阈值
    "abs_atr",          # ATR(14) 绝对值（由 atr_pct × close 还原，与全局指标同口径）
    "abs_body_top",     # max(open, close)
    "abs_body_bottom",  # min(open, close)
    "abs_wick_ratio",   # 影线占全幅比例
    "abs_extreme_high", # 近 extreme_lookback 根最高
    "abs_extreme_low",  # 近 extreme_lookback 根最低
    "abs_trend_ema",    # 趋势过滤用的 EMA(trend_len)
    "abs_raw",          # 满足 c1..c5 的原始吸收 bar（未经间隔节流）
    "abs_bar",          # 经 abs_gap_bars 节流后的吸收 bar（进入状态机）
    "abs_dir",          # 吸收 bar 的待入场方向：+1 做多 / -1 做空
    "abs_extreme",      # 吸收 bar 的极值（yAbs：买吸收=high，卖吸收=low）
    "abs_level",        # 确认需收盘穿越的关卡（做多=实体上沿，做空=实体下沿）
    "abs_signal",       # 该 bar 发出的入场方向（0 = 无信号）
    "abs_sl",           # 该信号的止损绝对价（= 极值 ∓ sl_buffer_atr × ATR）
    "abs_risk",         # |确认 bar close - 止损价|
)


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def percentile_nearest_rank(
    values: np.ndarray, window: int, pct: float
) -> np.ndarray:
    """滚动最近秩百分位（NIST nearest-rank，窗口含当前 bar）。

    窗口不足 (i+1 < window) 的位置返回 NaN（对应 Pine 预热期返回 na）。
    实现：滑窗视图 + np.partition 取第 k 小的 order statistic（每窗口 O(N)）。
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if n < window or window <= 0:
        return out
    k = int(math.ceil(pct / 100.0 * window)) - 1
    k = max(0, min(window - 1, k))
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = np.partition(windows, k, axis=1)[:, k]
    return out


def intrabar_delta(lower: pd.DataFrame) -> pd.Series:
    """脚本 `f_ibDelta()` 的逐 bar 版：lower-tf 每根 bar 的带符号成交量。

    规则 1:1 照搬脚本：
        d = close > open ? v : close < open ? -v : close > close[1] ? v : close < close[1] ? -v : 0
    即实体定方向，十字星退化到"与前一根收盘比较"，再平则为 0。
    """
    if lower.empty:
        return pd.Series(dtype=float)
    o = _num(lower, "open").to_numpy()
    c = _num(lower, "close").to_numpy()
    v = _num(lower, "volume", 0.0).fillna(0.0).to_numpy()
    prev_c = np.concatenate([[np.nan], c[:-1]])
    values = np.where(
        c > o,
        v,
        np.where(
            c < o,
            -v,
            np.where(c > prev_c, v, np.where(c < prev_c, -v, 0.0)),
        ),
    )
    return pd.Series(np.nan_to_num(values, nan=0.0), index=lower.index)


def bucket_delta(
    lower_delta: pd.Series, parent_index: pd.DatetimeIndex
) -> pd.Series:
    """把 lower-tf delta 按所属 parent bar 求和（等价 `request.security_lower_tf` 的 array.sum()）。

    parent 桶边界由 parent_index 自身给出：每个 lower bar 归入 open_time <= t < 下一 open_time 的桶；
    早于首个 parent bar 的数据丢弃（不在回测帧内），晚于最后一个的归入最后一个桶。
    """
    if len(lower_delta) == 0 or len(parent_index) == 0:
        return pd.Series(0.0, index=parent_index, dtype=float)
    stamps = parent_index.as_unit("ms").asi8
    lower_stamps = lower_delta.index.as_unit("ms").asi8
    pos = np.searchsorted(stamps, lower_stamps, side="right") - 1
    keep = pos >= 0
    pos = np.clip(pos, 0, len(stamps) - 1)
    totals = np.zeros(len(stamps), dtype=float)
    np.add.at(totals, pos[keep], lower_delta.to_numpy()[keep])
    return pd.Series(totals, index=parent_index)


# 脚本的 lower-tf 映射（tfSec → ltf）：低于 1h 的图一律用 1m，4h 用 5m，日线用 15m。
# 每个条目给出「首选 + 逐级回退」的候选顺序：首选完全对齐脚本，缺失时退化到有权可用的更粗数据。
LOWER_TF_BY_INTERVAL: dict[str, tuple[str, ...]] = {
    "1m": ("1m",),
    "5m": ("1m",),
    "15m": ("1m", "5m"),
    "1h": ("1m", "5m", "15m"),
    "4h": ("5m", "15m", "1h"),
    "1d": ("15m", "1h", "4h"),
}


def load_lower_tf_delta(service, symbol: str, parent_index: pd.DatetimeIndex) -> pd.Series | None:
    """按脚本的 ltf 映射加载更低周期 K 线，聚合出每个 parent bar 的 intrabar delta。

    返回 None 表示拿不到任何 lower-tf 数据 → 调用方回退到脚本自带的 estDelta 分支
    （`float delta = na(ltfDelta) ? estDelta : ltfDelta`）。
    """
    if len(parent_index) < 2:
        return None
    from ..utils import interval_ms

    interval = _infer_interval(parent_index)
    start = int(parent_index[0].timestamp() * 1000)
    end = int(parent_index[-1].timestamp() * 1000) + interval_ms(interval)
    minimum = max(4, len(parent_index))  # lower-tf 数据至少要能覆盖 parent 帧
    for source_interval in LOWER_TF_BY_INTERVAL.get(interval, ("1m",)):
        try:
            frame = service.load_frame(symbol, source_interval, start_ms=start, end_ms=end, limit=None)
        except Exception:
            continue
        if frame is None or len(frame) < minimum:
            continue
        return bucket_delta(intrabar_delta(frame), parent_index)
    return None


def _infer_interval(index: pd.DatetimeIndex) -> str:
    """从时间索引推断 parent 周期（用于加载窗口的右端点）。"""
    deltas = np.diff(index.as_unit("ms").asi8)
    step = int(np.median(deltas)) if len(deltas) else 60_000
    mapping = [
        (60_000, "1m"), (300_000, "5m"), (900_000, "15m"), (3_600_000, "1h"),
        (14_400_000, "4h"), (86_400_000, "1d"),
    ]
    best = min(mapping, key=lambda item: abs(item[0] - step))
    return best[1]


def add_absorption_context(
    frame: pd.DataFrame, params: dict | None = None,
    lower_tf_delta: pd.Series | None = None,
) -> pd.DataFrame:
    """给 OHLCV 帧附加吸收反转所需的全部派生列（幂等、纯因果）。

    消费的参数（缺省值 = 脚本 Medium 档默认值）：
    delta_window / delta_pct / body_mult / extreme_lookback / wick_mult /
    dominance_pct / abs_gap_bars / legacy_gap_logic / conf_mode / conf_window_bars /
    sl_buffer_atr / trend_len。

    delta 来源优先级（对应脚本 `na(ltfDelta) ? estDelta : ltfDelta`）：
    1. 显式传入的 lower_tf_delta；
    2. frame 上的 `intrabar_delta` 列（回测/实盘由 load_lower_tf_delta 从更低周期聚合后附上）；
    3. 都没有 → 脚本自带的 bar shape 估计（estDelta）。
    """
    params = params or {}
    if lower_tf_delta is None and "intrabar_delta" in getattr(frame, "columns", []):
        lower_tf_delta = pd.to_numeric(frame["intrabar_delta"], errors="coerce")
    window = max(2, int(params.get("delta_window", 500)))
    pct = float(params.get("delta_pct", 95.0))
    body_mult = float(params.get("body_mult", 0.25))
    extreme_look = max(2, int(params.get("extreme_lookback", 20)))
    wick_mult = float(params.get("wick_mult", 0.20))
    dominance_pct = float(params.get("dominance_pct", 20.0))
    gap_bars = max(0, int(params.get("abs_gap_bars", 10)))
    legacy_gap = bool(params.get("legacy_gap_logic", False))
    conf_mode = int(params.get("conf_mode", 1))
    conf_wait = max(1, int(params.get("conf_window_bars", 3)))
    sl_buffer_atr = float(params.get("sl_buffer_atr", 0.25))
    trend_len = max(2, int(params.get("trend_len", 200)))

    data = frame.copy()
    if data.empty:
        for col in ABSORPTION_COLUMNS:
            data[col] = pd.Series(dtype=float)
        return data

    open_ = _num(data, "open").to_numpy()
    high = _num(data, "high").to_numpy()
    low = _num(data, "low").to_numpy()
    close = _num(data, "close").to_numpy()
    volume = _num(data, "volume", 0.0).fillna(0.0).clip(lower=0.0).to_numpy()

    n = len(data)
    rng = high - low
    # delta 优先走脚本的 lower-tf intrabar 分支；没有 lower-tf 数据时退化为脚本自带的
    # bar shape 估计（宁可注明数据限制，也不悄悄换成别的算法）
    if lower_tf_delta is not None:
        delta = (
            pd.to_numeric(lower_tf_delta, errors="coerce")
            .reindex(data.index)
            .fillna(0.0)
            .to_numpy()
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(rng > 0, volume * (2.0 * close - high - low) / np.where(rng > 0, rng, 1.0), 0.0)
    abs_delta = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        dominance = np.where(volume > 0, abs_delta / np.where(volume > 0, volume, 1.0) * 100.0, 0.0)

    # ATR(14)：由全局指标的 atr_pct 还原绝对值，保证与 pipeline/实盘同一口径
    atr_series = _num(data, "atr_pct") * _num(data, "close")
    atr = atr_series.to_numpy()
    atr_fallback = np.where(np.isfinite(atr), atr, rng)  # 脚本的 nz(atr, rng)

    p_abs = percentile_nearest_rank(abs_delta, window, pct)

    close_series = _num(data, "close")
    hi_n = (
        pd.Series(high, index=data.index)
        .rolling(extreme_look, min_periods=extreme_look)
        .max()
        .to_numpy()
    )
    lo_n = pd.Series(low, index=data.index).rolling(extreme_look, min_periods=extreme_look).min().to_numpy()
    trend_ema = (
        close_series.ewm(span=trend_len, adjust=False).mean().to_numpy()
    )

    b_top = np.maximum(open_, close)
    b_bot = np.minimum(open_, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        wick_pos = np.where(rng > 0, (high - b_top) / np.where(rng > 0, rng, 1.0), 0.0)
        wick_neg = np.where(rng > 0, (b_bot - low) / np.where(rng > 0, rng, 1.0), 0.0)

    dir_agree = ((delta > 0) & (close > open_)) | ((delta < 0) & (close < open_))
    body = np.abs(close - open_)

    # ---- 五个吸收条件（脚本 c1..c5） ----
    with np.errstate(invalid="ignore"):
        c1 = np.isfinite(p_abs) & (abs_delta >= np.where(np.isfinite(p_abs), p_abs, np.inf))
    c2 = (body <= body_mult * atr_fallback) | (~dir_agree)
    c3 = np.where(delta > 0, high >= np.where(np.isfinite(hi_n), hi_n, np.inf), low <= np.where(np.isfinite(lo_n), lo_n, -np.inf))
    c4 = np.where(delta > 0, wick_pos, wick_neg) >= wick_mult
    c5 = (dominance >= dominance_pct) if dominance_pct > 0 else np.ones(n, dtype=bool)
    finite = np.isfinite(delta) & np.isfinite(body) & np.isfinite(atr_fallback)
    abs_raw = finite & (delta != 0) & c1 & c2 & c3 & c4 & np.asarray(c5, dtype=bool)

    # ---- 信号间隔节流（脚本 and/or 优先级问题，见模块 docstring 差异 #2） ----
    abs_bar = np.zeros(n, dtype=bool)
    last_abs = -1
    for i in range(n):
        if legacy_gap:
            # 脚本原式：(absRaw and na(lastAbs)) or (bar_index - lastAbs > absGap)
            hit = (last_abs < 0 and abs_raw[i]) or (
                last_abs >= 0 and i - last_abs > gap_bars
            )
        else:
            hit = bool(abs_raw[i]) and (last_abs < 0 or i - last_abs > gap_bars)
        if hit:
            abs_bar[i] = True
            last_abs = i

    # ---- 待处理信号状态机（脚本 SIGNAL STATE MACHINE） ----
    signal = np.zeros(n, dtype=float)
    stop = np.full(n, np.nan, dtype=float)
    s_dir = np.zeros(n, dtype=float)      # 吸收 bar 的待入场方向
    s_extreme = np.full(n, np.nan, dtype=float)
    s_level = np.full(n, np.nan, dtype=float)

    p_dir, p_ext, p_lvl, p_bar = 0, np.nan, np.nan, -1
    for i in range(n):
        if abs_bar[i]:
            direction = -1.0 if delta[i] > 0 else 1.0  # 买方被吸收→做空，卖方被吸收→做多
            extreme = high[i] if delta[i] > 0 else low[i]
            level = b_top[i] if direction == 1 else b_bot[i]
            s_dir[i], s_extreme[i], s_level[i] = direction, extreme, level
            p_dir, p_ext, p_lvl, p_bar = direction, extreme, level, i
            if conf_mode == 0:  # "None"：吸收 bar 收盘即入场
                signal[i] = p_dir
                stop[i] = p_ext - direction * atr_fallback[i] * sl_buffer_atr
                p_dir = 0
        elif p_dir != 0:
            invalid = (low[i] < p_ext) if p_dir == 1 else (high[i] > p_ext)
            expired = (i - p_bar) > conf_wait
            if invalid or expired:
                p_dir = 0
            elif (p_dir == 1 and close[i] > p_lvl) or (p_dir == -1 and close[i] < p_lvl):
                signal[i] = p_dir
                stop[i] = p_ext - p_dir * atr_fallback[i] * sl_buffer_atr
                p_dir = 0

    risk = np.abs(close - stop)

    data["abs_delta"] = delta
    data["abs_abs_delta"] = abs_delta
    data["abs_dominance"] = dominance
    data["abs_percentile"] = p_abs
    data["abs_atr"] = atr_fallback
    data["abs_body_top"] = b_top
    data["abs_body_bottom"] = b_bot
    data["abs_wick_ratio"] = np.where(delta > 0, wick_pos, wick_neg)
    data["abs_extreme_high"] = hi_n
    data["abs_extreme_low"] = lo_n
    data["abs_trend_ema"] = trend_ema
    data["abs_raw"] = abs_raw.astype(float)
    data["abs_bar"] = abs_bar.astype(float)
    data["abs_dir"] = s_dir
    data["abs_extreme"] = s_extreme
    data["abs_level"] = s_level
    data["abs_signal"] = signal
    data["abs_sl"] = stop
    data["abs_risk"] = risk
    return data
