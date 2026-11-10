"""趋势反转入场 × 预注册卖出信号：纯因果回放工具。

无 DB 访问、无生产副作用。所有臂共享同一批入场点、同一段未来 K 线、
同一成本口径，配对差只隔离「出场规则」这一个变量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEE_RATE = 0.0004
SLIPPAGE = 0.0002
ROUND_TRIP = 0.0012
HORIZON = 48              # 1h × 48 = 2 天，所有臂共享的持仓上限
FORWARD_H = 24            # Gate 1 标签前瞻跨度（与 HORIZON 同量级）
DECISION_BAR = 8          # Gate 1 决策点：入场后第 8 根（预注册，不调参）
MIN_GAP = HORIZON         # 相邻入场最小间隔，消除重叠持仓（H7）
LOOKBACK = 168            # 触发阈值的滚动历史窗口（7 天，只用 shift(1) 之前）
MIN_COVERAGE = 0.90       # 币覆盖率门槛（H8）
TRIGGER_Q = 0.80          # 横截面拥挤信号的固定分位阈值（预注册）
PURGE_MS = 24 * 3_600_000  # 标签 24h 前瞻，样本外切分边界清除跨度

SELL_SIGNALS = (
    "mom_fade",              # 8 根动量转负（价格类）
    "ma_break",              # 收盘跌破 EMA20（价格类）
    "vol_spike",             # ATR14 / ATR56 放大（价格类）
    "pullback_from_high",    # 距 24 根高点回撤（价格类）
    "st_flip",               # Supertrend 原生方向翻空（价格类）
    "funding_crowd",         # 资金费率横截面高分位（非价格）
    "funding_neg",           # 资金费率转负（非价格）
    "oi_crowd",              # 持仓量横截面高分位（非价格，P9 方向）
)
PRICE_SIGNALS = SELL_SIGNALS[:5]
NONPRICE_SIGNALS = SELL_SIGNALS[5:]


def wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder RMA：前 n 根 SMA 播种后递推（P-09）。"""
    v = pd.Series(series, dtype=float)
    out = np.full(len(v), np.nan)
    arr = v.to_numpy()
    start = -1
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            start = i
            break
    if start < 0 or start + n > len(arr):
        return pd.Series(out, index=v.index)
    seed = arr[start:start + n]
    if not np.all(np.isfinite(seed)):
        return pd.Series(out, index=v.index)
    prev = float(seed.mean())
    out[start + n - 1] = prev
    for i in range(start + n, len(arr)):
        x = arr[i]
        prev = (prev * (n - 1) + x) / n if np.isfinite(x) else prev
        out[i] = prev
    return pd.Series(out, index=v.index)


def net_return_pct(entry: float, exit_price: float, side: int = 1) -> float:
    """名义本金净收益 %：精确手续费 + 双边滑点（H3，与生产一致）。"""
    if entry <= 0 or exit_price <= 0 or not np.isfinite(entry) or not np.isfinite(exit_price):
        return float("nan")
    gross = side * (exit_price - entry) / entry
    cost = FEE_RATE * (1 + exit_price / entry) + 2 * SLIPPAGE
    return float((gross - cost) * 100.0)


def attach_features(df: pd.DataFrame, funding_pct=None, funding_rate=None,
                    oi_pct=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """产出卖出信号的分值与触发布尔；所有量只用当前 bar 及之前的信息。

    funding / oi 的横截面分位由调用方按「上一已结算 8h/1h 桶」对齐后传入，
    避免结算时刻的前视（H8 新鲜度坑）。
    """
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr14 = wilder(tr, 14)
    atr56 = wilder(tr, 56)
    ema20 = c.ewm(span=20, adjust=False).mean()
    den = atr14.replace(0, np.nan)

    score = pd.DataFrame(index=df.index)
    score["atr14"] = atr14
    score["mom_fade"] = -(c / c.shift(8) - 1) * c / den
    score["ma_break"] = (ema20 - c) / den
    score["vol_spike"] = atr14 / atr56.replace(0, np.nan)
    score["pullback_from_high"] = (h.rolling(24, min_periods=1).max() - c) / den
    score["st_flip"] = ((df["st_direction"].astype(float) < 0).astype(float)
                        if "st_direction" in df else 0.0)
    score["funding_crowd"] = (pd.Series(funding_pct, index=df.index, dtype=float)
                              if funding_pct is not None else np.nan)
    score["funding_neg"] = (-pd.Series(funding_rate, index=df.index, dtype=float)
                            if funding_rate is not None else np.nan)
    score["oi_crowd"] = (pd.Series(oi_pct, index=df.index, dtype=float)
                         if oi_pct is not None else np.nan)

    trig = pd.DataFrame(index=df.index)
    for name in SELL_SIGNALS:
        v = score[name]
        if name in ("funding_crowd", "oi_crowd"):
            trig[name] = v.ge(TRIGGER_Q).fillna(False).astype(bool)
        elif name == "funding_neg":
            raw = (pd.Series(funding_rate, index=df.index, dtype=float)
                   if funding_rate is not None else pd.Series(np.nan, index=df.index))
            trig[name] = raw.lt(0).fillna(False).astype(bool)
        elif name == "st_flip":
            trig[name] = v.gt(0).fillna(False).astype(bool)
        else:
            # 阈值只由「当前 bar 之前」的历史分布决定，固定 1σ，不调参
            hist = v.shift(1)
            mu = hist.rolling(LOOKBACK, min_periods=60).mean()
            sd = hist.rolling(LOOKBACK, min_periods=60).std()
            trig[name] = (v > (mu + sd)).fillna(False).astype(bool)
    return score, trig


def build_entries(df: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    """Supertrend 趋势反转的多头信号上升沿；收盘出信号、次根开盘成交。"""
    n = len(df)
    direction = np.asarray(signals["direction"], dtype=float)
    sl_pct = np.asarray(signals["sl_pct"], dtype=float)
    tp_pct = np.asarray(signals["tp_pct"], dtype=float)
    close = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    # pandas 2 的 DatetimeIndex 可能是 us/ns 单位，用 epoch 差换算 ms 才与 DB 口径一致
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    times = ((df.index - epoch).total_seconds().to_numpy() * 1000).astype(np.int64)
    out: list[dict] = []
    last = -10**9
    for t in range(n - HORIZON - 1):
        if direction[t] != 1 or (t > 0 and direction[t - 1] == 1):
            continue
        if t - last < MIN_GAP:
            continue
        slp, tpp = sl_pct[t], tp_pct[t]
        if not (np.isfinite(slp) and np.isfinite(tpp) and slp > 0 and tpp > 0):
            continue
        ep = opens[t + 1]
        if not np.isfinite(ep) or ep <= 0:
            continue
        out.append({
            "t": t, "entry_i": t + 1, "entry_price": float(ep),
            "sl": float(close[t] * (1 - slp)), "tp": float(close[t] * (1 + tpp)),
            "sl_pct": float(slp), "tp_pct": float(tpp),
            "time_ms": int(times[t + 1]),
        })
        last = t
    return out


def mirror_barriers(entry: dict) -> dict:
    """随机方向基线翻成空头时，SL/TP 镜像到另一侧（入场点与成本不变）。"""
    out = dict(entry)
    out["side"] = -1
    # 用信号 bar 收盘价还原：sl/tp 距 close_t 的比例不变
    close_t = entry["sl"] / (1 - entry["sl_pct"])
    out["sl"] = float(close_t * (1 + entry["sl_pct"]))
    out["tp"] = float(close_t * (1 - entry["tp_pct"]))
    return out


def replay(bars: np.ndarray, sl: float, tp: float, side: int = 1,
           trigger_j: int | None = None) -> tuple[float, int, str]:
    """回放一段 (H,4) = [open,high,low,close]，返回 (原始出场价, bar 索引, 原因)。

    同根 SL 优先、跳空按开盘（项目约定）；信号在 bar j 收盘触发，
    于 bar j+1 开盘平仓，避免当根信号当根成交（H2 因果性）。
    """
    H = len(bars)
    for j in range(H):
        o, hi, low, cl = (float(x) for x in bars[j])
        if side > 0:
            if o <= sl:
                return o, j, "sl_gap"
            if o >= tp:
                return o, j, "tp_gap"
            if low <= sl:
                return float(sl), j, "sl"
            if hi >= tp:
                return float(tp), j, "tp"
        else:
            if o >= sl:
                return o, j, "sl_gap"
            if o <= tp:
                return o, j, "tp_gap"
            if hi >= sl:
                return float(sl), j, "sl"
            if low <= tp:
                return float(tp), j, "tp"
        if trigger_j is not None and j == trigger_j:
            if j + 1 < H:
                return float(bars[j + 1][0]), j + 1, "signal"
            return cl, j, "signal_end"
    return float(bars[-1][3]), H - 1, "time"


def replay_from_barrier(bars: np.ndarray, barrier: tuple[float, int, str],
                        trigger_j: int | None) -> tuple[float, int, str]:
    """解析式任意触发点结算：只需一次基线重放得到 barrier。

    若信号在 barrier 之前或同根触发，按项目约定 barrier（同根 SL 优先）胜出；
    否则 bar j 收盘触发、bar j+1 开盘平仓。用于 20 种子 × 8 信号的随机早退
    对照，避免百万次逐 bar 重放。
    """
    px, bj, why = barrier
    H = len(bars)
    if trigger_j is None or trigger_j < 0:
        return px, bj, why
    # 'time' 只是窗口末强平、并非真实 barrier：同根信号仍可胜出
    cutoff = bj if why != "time" else bj + 1
    if trigger_j >= cutoff:
        return px, bj, why
    if trigger_j + 1 < H:
        return float(bars[trigger_j + 1][0]), trigger_j + 1, "signal"
    return float(bars[trigger_j][3]), trigger_j, "signal_end"


def first_trigger(trig_col: np.ndarray, start: int) -> int | None:
    seg = trig_col[start:start + HORIZON]
    hit = np.flatnonzero(seg)
    return int(hit[0]) if len(hit) else None


def decision_point(score: pd.DataFrame, entry_i: int, close_arr: np.ndarray) -> dict | None:
    """Gate 1 决策点：入场后第 DECISION_BAR 根，标签为未来 FORWARD_H 根收益。"""
    j = entry_i + DECISION_BAR
    if j + FORWARD_H >= len(close_arr) or not np.isfinite(close_arr[j]):
        return None
    fwd = float((close_arr[j + FORWARD_H] / close_arr[j] - 1) * 100.0)
    vals = {name: float(score[name].iloc[j]) for name in SELL_SIGNALS}
    return {"features": vals, "fwd_pct": fwd, "binary": 1 if fwd > 0 else 0,
            "oracle_fwd_pct": fwd}
