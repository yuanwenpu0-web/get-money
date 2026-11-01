"""币种画像（coin profile）：把一段 K 线压成一组可比的多维指标。

纯函数、无 IO、无全局状态。所有指标只用传入帧内的数据，因此只要传入
「决策时点之前」的帧，就天然满足反前视要求（不需要额外的 as-of 包装）。

维度按**键前缀**分组，前缀即维度名，便于后续逐维度做稳定性与判别力检验：

| 前缀 | 维度 | 对量化的用途 |
| --- | --- | --- |
| `liq_` | 流动性 | 冲击成本、可容纳资金、滑点与最小下单量建模 |
| `vol_` | 波动率 | 止损距离、仓位规模、ATR 类参数、极值事件频率 |
| `trd_` | 趋势/均值回归倾向 | 选策略族（突破跟随 vs 区间反转） |
| `tail_` | 尾部与跳跃 | 止损被击穿概率、极端单 bar 风险 |
| `life_` | 生命周期与微观结构 | 上市时长、价格粒度（tick 相对价格的影响） |
| `hist_` | 历史表现 | 买持收益、回撤、日胜率、盈亏波动不对称 |

口径约定（报告必须原样复述）：

- 收益率一律用**对数收益**（小时级），日级振幅用 `high/low - 1`（路径波动，非收益）。
- 「日」按 UTC 自然日切桶，不假设每个自然日恰好 24 根（缺 bar 的日子照实计入）。
- 所有滚动统计量在预热期内为 NaN，聚合时用 `nanmedian`/`nansum` 之类忽略，不填 0。
- 帧内 bar 数少于 `min_bars` 直接返回空字典（统计量不可信，宁可缺失也不错标）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lab_indicators import adx_di, atr_wilder

MIN_BARS = 200
ATR_PERIOD = 14
ADX_PERIOD = 14
ER_PERIOD = 24
HURST_LAGS = (1, 2, 4, 8, 16, 32)

DIMENSIONS = ("liq", "vol", "trd", "tail", "life", "hist")


def dimension_of(key: str) -> str:
    """指标键 -> 维度名（取第一个下划线前的前缀）。"""
    return key.split("_", 1)[0]


def _returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape, np.nan)
    prev = close[:-1]
    curr = close[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where((prev > 0) & (curr > 0), np.log(curr / prev), np.nan)
    return out


def _nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _nanquantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    """Kaufman 效率比：净位移 / 路径总长，1 = 纯趋势，0 = 纯噪声。

    用 `diff().abs().rolling(n).sum()` 而不是循环；分母为 0 时记 NaN（不是 0），
    否则横盘币会被误判成「无趋势」而不是「无信息」。
    """
    series = pd.Series(np.asarray(close, dtype=float))
    net = series.diff(n).abs()
    path = series.diff().abs().rolling(n, min_periods=n).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (net / path).to_numpy(dtype=float)
    return np.where(path.to_numpy(dtype=float) > 0, ratio, np.nan)


def _variance_ratio(returns: np.ndarray, q: int) -> float:
    """方差比 VR(q) = Var(q 期累加收益) / (q · Var(1 期收益))。

    >1 表示正自相关（趋势延续），<1 表示均值回归。样本不足返回 NaN。
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < q * 20:
        return float("nan")
    single = float(np.var(arr, ddof=1))
    if single <= 0:
        return float("nan")
    rolled = pd.Series(arr).rolling(q, min_periods=q).sum().to_numpy(dtype=float)
    rolled = rolled[np.isfinite(rolled)]
    if rolled.size < 20:
        return float("nan")
    return float(np.var(rolled, ddof=1) / (q * single))


def _hurst_from_vr(returns: np.ndarray, lags=HURST_LAGS) -> float:
    """用方差比对 log(q) 回归斜率估 Hurst：log VR(q) = (2H-1)·log(q) + c。

    比 R/S 法稳健且不需要长序列。斜率 = 2H-1 ⇒ H = (斜率+1)/2。
    有效样本不足 3 个 lag 返回 NaN。
    """
    points = [(lag, _variance_ratio(returns, lag)) for lag in lags]
    points = [(np.log(q), np.log(v)) for q, v in points if np.isfinite(v) and v > 0]
    if len(points) < 3:
        return float("nan")
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return (slope + 1.0) / 2.0


def _autocorr1(returns: np.ndarray) -> float:
    """一阶自相关：负值倾向均值回归，正值倾向趋势延续。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 50 or np.std(arr) <= 0:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def _max_drawdown(close: np.ndarray) -> float:
    """买持最大回撤（正数百分比，如 62.0 表示 -62%）。"""
    arr = np.asarray(close, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return float("nan")
    peak = np.maximum.accumulate(arr)
    return float((arr / peak - 1.0).min() * -100.0)


def _daily_block(frame: pd.DataFrame) -> pd.DataFrame:
    """按 UTC 自然日聚合：日振幅、日涨跌、日成交额。缺 bar 的日子不会被补齐。"""
    key = frame.index.floor("D")
    daily = frame.groupby(key).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        open=("open", "first"),
        qv=("quote_volume", "sum"),
    )
    daily["range"] = daily["high"] / daily["low"].replace(0, np.nan) - 1.0
    daily["ret"] = daily["close"] / daily["open"].replace(0, np.nan) - 1.0
    return daily


def _liquidity(frame: pd.DataFrame) -> dict:
    qv = frame["quote_volume"].to_numpy(dtype=float)
    trades = frame["trades"].to_numpy(dtype=float)
    med_qv = _nanmedian(qv)
    med_trades = _nanmedian(trades)
    trade_size = med_qv / med_trades if med_trades and med_trades > 0 else float("nan")
    return {
        "liq_qv_median_usd": med_qv,
        "liq_qv_p10_usd": _nanquantile(qv, 0.10),
        "liq_qv_p90_usd": _nanquantile(qv, 0.90),
        "liq_trades_median": med_trades,
        "liq_trade_size_usd": trade_size,
        "liq_zero_vol_frac": float(np.mean(qv <= 0)) if qv.size else float("nan"),
    }


def _volatility(frame: pd.DataFrame, daily: pd.DataFrame, rets: np.ndarray) -> dict:
    atr_pct = atr_wilder(
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        frame["close"].to_numpy(dtype=float),
        ATR_PERIOD,
    ) / np.where(frame["close"].to_numpy(dtype=float) > 0, frame["close"].to_numpy(dtype=float), np.nan)

    ranges = daily["range"].to_numpy(dtype=float)
    abs_ret = np.abs(daily["ret"].to_numpy(dtype=float))
    hourly_vol = float(np.nanstd(rets)) if np.isfinite(rets).sum() > 20 else float("nan")

    # 波动率自身的不稳定度：24 根滚动实现波动的离散度 / 均值，越大越难用固定 ATR 参数
    roll_vol = pd.Series(rets).rolling(24, min_periods=24).std().to_numpy(dtype=float)
    finite = roll_vol[np.isfinite(roll_vol)]
    vol_of_vol = float(np.std(finite) / np.mean(finite)) if finite.size > 20 and np.mean(finite) > 0 else float("nan")

    return {
        "vol_range_daily_med": _nanmedian(ranges),
        "vol_range_daily_p95": _nanquantile(ranges, 0.95),
        "vol_absret_daily_med": _nanmedian(abs_ret),
        "vol_atr_pct_med": _nanmedian(atr_pct),
        "vol_hourly_std": hourly_vol,
        "vol_of_vol": vol_of_vol,
        "vol_day_frac_gt50": float(np.mean(ranges > 0.50)) if ranges.size else float("nan"),
        "vol_day_frac_gt100": float(np.mean(ranges > 1.00)) if ranges.size else float("nan"),
        "vol_max_bar_absret": float(np.nanmax(np.abs(rets))) if np.isfinite(rets).sum() else float("nan"),
    }


def _trendiness(frame: pd.DataFrame, rets: np.ndarray) -> dict:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    adx, _, _ = adx_di(high, low, close, ADX_PERIOD)
    er = _efficiency_ratio(close, ER_PERIOD)

    arr = rets[np.isfinite(rets)]
    if arr.size > 1:
        signs = np.sign(arr)
        nonzero = signs[signs != 0]
        persist = float(np.mean(nonzero[1:] == nonzero[:-1])) if nonzero.size > 1 else float("nan")
    else:
        persist = float("nan")

    return {
        "trd_er_med": _nanmedian(er),
        "trd_adx_med": _nanmedian(adx),
        "trd_acf1": _autocorr1(rets),
        "trd_vr4": _variance_ratio(rets, 4),
        "trd_hurst": _hurst_from_vr(rets),
        "trd_sign_persist": persist,
    }


def _tail(rets: np.ndarray, daily: pd.DataFrame) -> dict:
    arr = rets[np.isfinite(rets)]
    if arr.size < 50:
        return {key: float("nan") for key in (
            "tail_kurt", "tail_skew", "tail_jump_frac", "tail_down_up_ratio", "tail_worst_day",
        )}
    centered = arr - arr.mean()
    std = arr.std(ddof=1)
    kurt = float(np.mean(centered ** 4) / std ** 4 - 3.0) if std > 0 else float("nan")
    skew = float(np.mean(centered ** 3) / std ** 3) if std > 0 else float("nan")

    mad = float(np.median(np.abs(arr)))
    jump_frac = float(np.mean(np.abs(arr) > 5.0 * mad)) if mad > 0 else float("nan")

    up = arr[arr > 0]
    down = arr[arr < 0]
    ratio = float(np.std(down) / np.std(up)) if up.size > 10 and down.size > 10 and np.std(up) > 0 else float("nan")
    worst_day = _nanquantile(daily["ret"].to_numpy(dtype=float), 0.01)

    return {
        "tail_kurt": kurt,
        "tail_skew": skew,
        "tail_jump_frac": jump_frac,
        "tail_down_up_ratio": ratio,
        "tail_worst_day": worst_day,
    }


def _lifecycle(frame: pd.DataFrame, daily: pd.DataFrame) -> dict:
    close = frame["close"].to_numpy(dtype=float)
    idx = frame.index
    step = float(pd.Series(idx).diff().median().total_seconds()) if len(idx) > 2 else float("nan")
    span_bars = (idx[-1] - idx[0]).total_seconds() / step + 1 if step and np.isfinite(step) and step > 0 else float("nan")
    coverage = float(len(frame) / span_bars) if np.isfinite(span_bars) and span_bars > 0 else float("nan")
    med_price = _nanmedian(close)
    return {
        "life_bars": float(len(frame)),
        "life_days": float(len(daily)),
        "life_coverage": coverage,
        "life_price_log10": float(np.log10(med_price)) if med_price > 0 else float("nan"),
    }


def _history(daily: pd.DataFrame) -> dict:
    rets = daily["ret"].to_numpy(dtype=float)
    close = daily["close"].to_numpy(dtype=float)
    if np.isfinite(close).sum() < 2:
        return {key: float("nan") for key in (
            "hist_bh_return_pct", "hist_bh_maxdd_pct", "hist_calmar", "hist_day_win_rate",
        )}
    total = float(close[np.isfinite(close)][-1] / close[np.isfinite(close)][0] - 1.0) * 100.0
    maxdd = _max_drawdown(close)
    calmar = total / maxdd if np.isfinite(maxdd) and maxdd > 0 else float("nan")
    win = float(np.mean(rets > 0)) if np.isfinite(rets).sum() else float("nan")
    return {
        "hist_bh_return_pct": total,
        "hist_bh_maxdd_pct": maxdd,
        "hist_calmar": calmar,
        "hist_day_win_rate": win,
    }


def profile(frame: pd.DataFrame, *, min_bars: int = MIN_BARS) -> dict[str, float]:
    """把一帧 1h（或任意周期）K 线压成币种画像字典；样本不足返回空字典。

    帧必须按时间升序、索引为 UTC DatetimeIndex，且含
    `open/high/low/close/quote_volume/trades` 列。
    """
    required = ("open", "high", "low", "close", "quote_volume", "trades")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"画像需要这些列: {', '.join(missing)}")
    if len(frame) < min_bars:
        return {}

    frame = frame.sort_index()
    daily = _daily_block(frame)
    rets = _returns(frame["close"].to_numpy(dtype=float))

    out: dict[str, float] = {}
    out.update(_liquidity(frame))
    out.update(_volatility(frame, daily, rets))
    out.update(_trendiness(frame, rets))
    out.update(_tail(rets, daily))
    out.update(_lifecycle(frame, daily))
    out.update(_history(daily))
    return {key: float(value) for key, value in out.items()}


def dimension_keys(profiles: dict[str, dict[str, float]], dimension: str) -> list[str]:
    """某维度下的全部指标键（跨币种取并集，顺序按字母）。"""
    keys: set[str] = set()
    for row in profiles.values():
        keys.update(k for k in row if dimension_of(k) == dimension)
    return sorted(keys)


def cross_section(profiles: dict[str, dict[str, float]], key: str) -> pd.Series:
    """把某个指标拉成横截面序列（币种 -> 值），缺该键的币种记 NaN。"""
    return pd.Series({symbol: row.get(key, np.nan) for symbol, row in profiles.items()}, dtype=float)


def tercile_bins(values: pd.Series) -> pd.Series:
    """横截面三分位分档，返回 0/1/2（0 = 最低三分之一）。

    用 `rank(pct=True)` 而不是固定阈值：分档边界随市场整体漂移，
    固定阈值会让牛市全挤进高档、熊市全挤进低档。NaN 保留为 NaN（不参与分档）。
    """
    ranks = values.rank(pct=True, method="average")
    binned = np.where(ranks <= 1 / 3, 0, np.where(ranks <= 2 / 3, 1, 2))
    return pd.Series(binned, index=values.index, dtype="float").where(values.notna())
