"""指标与特征工程。优先 TA-Lib，缺失时回退 pandas-ta-classic，最后回退纯 pandas 实现，
保证在未安装本地 C 库的环境也能启动。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .funding import FUNDING_FEATURES, attach_funding
from .sentiment import SENTIMENT_FEATURES, attach_sentiment

try:
    import talib

    HAS_TALIB = True
except Exception:  # pragma: no cover - 环境差异
    talib = None
    HAS_TALIB = False

try:
    import pandas_ta_classic as pta

    HAS_PTA = True
except Exception:  # pragma: no cover - 环境差异
    pta = None
    HAS_PTA = False


FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "rsi_14",
    "rsi_7",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "atr_pct",
    "supertrend_dir",
    "supertrend_dist",
    "ema_ratio_9_21",
    "ema_ratio_21_50",
    "bb_width",
    "bb_pos",
    "volume_z",
    "range_pct",
    "close_pos",
    "adx_14",
    "di_diff",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    *FUNDING_FEATURES,
    *SENTIMENT_FEATURES,
]


def _rsi(close: pd.Series, length: int) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=length), index=close.index)
    if HAS_PTA:
        value = pta.rsi(close, length=length)
        if value is not None:
            return value
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((loss == 0) & (gain > 0), 100.)
    return result.mask((loss == 0) & (gain == 0), 50.)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    if HAS_TALIB:
        macd, signal, hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        return (
            pd.Series(macd, index=close.index),
            pd.Series(signal, index=close.index),
            pd.Series(hist, index=close.index),
        )
    if HAS_PTA:
        frame = pta.macd(close, fast=12, slow=26, signal=9)
        if frame is not None and not frame.empty:
            cols = list(frame.columns)
            macd_col = next((c for c in cols if c.startswith("MACD_")), cols[0])
            hist_col = next((c for c in cols if c.startswith("MACDh_")), cols[-1])
            sig_col = next((c for c in cols if c.startswith("MACDs_")), cols[1])
            return frame[macd_col], frame[sig_col], frame[hist_col]
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def _atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    if HAS_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=length), index=frame.index)
    if HAS_PTA:
        value = pta.atr(high, low, close, length=length)
        if value is not None:
            return value
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _supertrend(frame: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """返回 (方向 1/-1, 收盘价到 SuperTrend 线的百分比距离)。"""
    if HAS_PTA:
        result = pta.supertrend(frame["high"], frame["low"], frame["close"], length=length, multiplier=multiplier)
        if result is not None and not result.empty:
            direction_col = next((c for c in result.columns if c.startswith("SUPERTd_")), None)
            line_col = next((c for c in result.columns if c.startswith("SUPERT_")), None)
            if direction_col and line_col:
                direction = result[direction_col].fillna(0).astype(float)
                line = result[line_col]
                dist = (frame["close"] - line) / frame["close"] * 100
                return direction, dist.fillna(0)

    atr = _atr(frame, length)
    hl2 = (frame["high"] + frame["low"]) / 2
    upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    lower = (hl2 - multiplier * atr).to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)

    final_upper = np.full(len(frame), np.nan)
    final_lower = np.full(len(frame), np.nan)
    trend = np.ones(len(frame))
    line = np.full(len(frame), np.nan)
    for i in range(1, len(frame)):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            trend[i] = trend[i - 1]
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            line[i] = line[i - 1]
            continue
        final_upper[i] = min(upper[i], final_upper[i - 1]) if close[i - 1] <= final_upper[i - 1] else upper[i]
        final_lower[i] = max(lower[i], final_lower[i - 1]) if close[i - 1] >= final_lower[i - 1] else lower[i]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if close[i] > final_upper[i] else -1
        line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    direction = pd.Series(trend, index=frame.index)
    dist = (frame["close"] - pd.Series(line, index=frame.index)) / frame["close"] * 100
    return direction, dist.fillna(0)


def _adx(frame: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series]:
    if HAS_TALIB:
        adx = pd.Series(talib.ADX(frame["high"].values, frame["low"].values, frame["close"].values, timeperiod=length), index=frame.index)
        plus = pd.Series(talib.PLUS_DI(frame["high"].values, frame["low"].values, frame["close"].values, timeperiod=length), index=frame.index)
        minus = pd.Series(talib.MINUS_DI(frame["high"].values, frame["low"].values, frame["close"].values, timeperiod=length), index=frame.index)
        return adx, plus - minus
    if HAS_PTA:
        result = pta.adx(frame["high"], frame["low"], frame["close"], length=length)
        if result is not None and not result.empty:
            adx_col = next((c for c in result.columns if c.startswith("ADX_")), None)
            dmp_col = next((c for c in result.columns if c.startswith("DMP_")), None)
            dmn_col = next((c for c in result.columns if c.startswith("DMN_")), None)
            if adx_col and dmp_col and dmn_col:
                return result[adx_col], result[dmp_col] - result[dmn_col]
    high, low, close = frame['high'], frame['low'], frame['close']
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.)
    minus_dm = down.where((down > up) & (down > 0), 0.)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    def smooth(values):
        return values.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    atr = smooth(tr).replace(0, np.nan)
    plus, minus = 100 * smooth(plus_dm) / atr, 100 * smooth(minus_dm) / atr
    dx = (100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)).fillna(0.)
    return smooth(dx), (plus - minus).fillna(0.)


def _triple_barrier_label(
    data: pd.DataFrame, close: pd.Series, barrier: dict,
) -> pd.Series:
    """三分类 triple-barrier 标签，与实盘 sl/tp/max_hold 出场对齐。

    标签值：
      +1 = 未来窗口内先触上轨（tp_atr_mult*ATR）→ 实盘止盈；
      -1 = 先触下轨（sl_atr_mult*ATR）→ 实盘止损；
       0 = 到期（max_hold_bars）未触任一 barrier → 横盘，做了也是白交费。
    时间屏障 H 取实盘 max_hold_bars（封顶 300 控成本）；未设则退化为 horizon*4。
    这是相对 `future_return>0` 二分类的核心改进：被止损打掉但最后涨回的 bar，
    二分类仍标 +1（模型学"该做多"），triple-barrier 标 -1（不该做多）。
    """
    n = len(data)
    sl_m = float(barrier.get("sl_atr_mult", 2.0))
    tp_m = float(barrier.get("tp_atr_mult", 3.0))
    H = int(barrier.get("max_hold_bars", 0))
    H = H if 0 < H <= 300 else min(max(int(barrier.get("horizon", 5)) * 4, 60), 300)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    ap = data["atr_pct"].to_numpy(dtype=float)
    labels = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(ap[i]) or ap[i] <= 0 or c[i] <= 0:
            continue
        tp_price = c[i] * (1.0 + tp_m * ap[i])
        sl_price = c[i] * (1.0 - sl_m * ap[i])
        end = min(n, i + 1 + H)
        seg_h, seg_l = high[i + 1:end], low[i + 1:end]
        hit_up = seg_h >= tp_price
        hit_dn = seg_l <= sl_price
        if hit_up.any() and (not hit_dn.any() or int(np.argmax(hit_up)) < int(np.argmax(hit_dn))):
            labels[i] = 1.0
        elif hit_dn.any():
            labels[i] = -1.0
        else:
            labels[i] = 0.0
    return pd.Series(labels, index=data.index)


def build_features(frame: pd.DataFrame, with_label: bool = False, horizon: int = 5,
                   funding: pd.DataFrame | None = None,
                   sentiment: pd.DataFrame | None = None,
                   barrier: dict | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    close = data["close"]
    high, low = data["high"], data["low"]

    data["ret_1"] = close.pct_change(1)
    data["ret_3"] = close.pct_change(3)
    data["ret_5"] = close.pct_change(5)
    data["ret_10"] = close.pct_change(10)

    data["rsi_14"] = _rsi(close, 14)
    data["rsi_7"] = _rsi(close, 7)

    macd, signal, hist = _macd(close)
    data["macd_norm"] = macd / close
    data["macd_signal_norm"] = signal / close
    data["macd_hist_norm"] = hist / close

    atr = _atr(data, 14)
    data["atr_pct"] = atr / close

    direction, dist = _supertrend(data, 10, 3.0)
    data["supertrend_dir"] = direction
    data["supertrend_dist"] = dist

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    data["ema_ratio_9_21"] = ema9 / ema21 - 1
    data["ema_ratio_21_50"] = ema21 / ema50 - 1

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["bb_width"] = (std * 4) / mid
    data["bb_pos"] = ((close - (mid - 2 * std)) / ((std * 4).replace(0, np.nan))).mask(std == 0, .5)

    volume_mean = data["volume"].rolling(50).mean()
    volume_std = data["volume"].rolling(50).std()
    data["volume_z"] = ((data["volume"] - volume_mean) / volume_std.replace(0, np.nan)).mask(volume_std == 0, 0.)

    data["range_pct"] = (high - low) / close
    data["close_pos"] = ((close - low) / (high - low).replace(0, np.nan)).mask(high == low, .5)

    adx, di_diff = _adx(data, 14)
    data["adx_14"] = adx
    data["di_diff"] = di_diff

    attach_funding(data, funding)
    attach_sentiment(data, sentiment)

    if isinstance(data.index, pd.DatetimeIndex):
        minutes = data.index.hour * 60 + data.index.minute
        data["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
        data["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
        data["dow_sin"] = np.sin(2 * np.pi * data.index.day_of_week / 7)
        data["dow_cos"] = np.cos(2 * np.pi * data.index.day_of_week / 7)
    else:
        data["hour_sin"] = 0.0
        data["hour_cos"] = 0.0
        data["dow_sin"] = 0.0
        data["dow_cos"] = 0.0

    if with_label:
        data["future_return"] = close.pct_change(horizon).shift(-horizon)
        if barrier is not None:
            # triple-barrier 三分类标签：与实盘 sl/tp/max_hold 出场对齐
            data["label"] = _triple_barrier_label(data, close, barrier).astype("Int64")
        else:
            data["label"] = (data["future_return"] > 0).astype(int)
            data.loc[data["future_return"].isna(), "label"] = np.nan

    return data
