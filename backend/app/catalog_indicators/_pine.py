"""Pine Script `ta.*` primitives shared by the ported catalog indicators.

Every helper mirrors the TradingView semantics its source scripts rely on, so a
port stays line-by-line comparable with the Pine original instead of silently
falling back to pandas defaults (which seed EMA/RMA differently). Warm-up gaps
are propagated as NaN, never filled in, so callers can see where a series is not
yet defined.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ohlcv(frame: pd.DataFrame, *columns: str) -> tuple[pd.Series, ...]:
    """Return the requested columns as floats, rejecting missing or non-numeric ones."""
    missing = [name for name in columns if name not in getattr(frame, "columns", [])]
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")
    return tuple(pd.to_numeric(frame[name], errors="raise").astype(float) for name in columns)


def integer(name: str, value, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise ValueError(f"参数 {name} 超出允许范围")
    return value


def number(name: str, value, low: float, high: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < low or (high is not None and value > high)):
        raise ValueError(f"参数 {name} 超出允许范围")
    return float(value)


def boolean(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"参数 {name} 必须为布尔值")
    return value


def choice(name: str, value, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ValueError(f"参数 {name} 必须是 {'/'.join(options)} 之一")
    return value


def flags(values: pd.Series) -> pd.Series:
    """NaN-aware boolean series: comparisons involving warm-up rows are false."""
    return values.fillna(False).astype(bool)


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Pine's ta.ema seeds with the SMA of the first `length` values, then recurses."""
    values = series.astype(float).copy()
    if len(values) < length:
        return pd.Series(np.nan, index=series.index, dtype=float)
    values.iloc[: length - 1] = np.nan
    values.iloc[length - 1] = series.astype(float).iloc[:length].mean()
    return values.ewm(alpha=2 / (length + 1), adjust=False, ignore_na=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine's ta.rma: Wilder smoothing seeded by the mean of the first `length` values."""
    out = np.full(len(series), np.nan)
    count, total, current = 0, 0.0, np.nan
    for position, value in enumerate(series.to_numpy(dtype=float)):
        if not np.isnan(value):
            if count < length:
                count += 1
                total += value
                if count == length:
                    current = total / length
            else:
                current = (current * (length - 1) + value) / length
        out[position] = current
    return pd.Series(out, index=series.index)


def stdev(series: pd.Series, length: int) -> pd.Series:
    """Pine's ta.stdev is the population standard deviation of the window."""
    return series.rolling(length, min_periods=length).std(ddof=0)


def highest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).max()


def lowest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).min()


def change(series: pd.Series, length: int = 1) -> pd.Series:
    return series - series.shift(length)


def true_range(frame: pd.DataFrame) -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    previous = close.shift(1)
    return pd.concat([high - low, (high - previous).abs(), (low - previous).abs()],
                     axis=1).max(axis=1, skipna=True)


def atr(frame: pd.DataFrame, length: int) -> pd.Series:
    return rma(true_range(frame), length)


def percentrank(series: pd.Series, length: int) -> pd.Series:
    """Pine's ta.percentrank: share of the previous `length - 1` values below the current one."""
    if length < 2:
        raise ValueError("参数 length 至少为 2")
    return series.rolling(length, min_periods=length).apply(
        lambda window: (window[:-1] < window[-1]).sum() * 100.0 / (length - 1), raw=True)


def dmi(frame: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Pine's ta.dmi: (+DI, -DI, ADX) with Wilder smoothing."""
    high, low, _ = ohlcv(frame, "high", "low", "close")
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    trur = rma(true_range(frame), length).replace(0, np.nan)
    plus = 100 * rma(plus_dm, length) / trur
    minus = 100 * rma(minus_dm, length) / trur
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return plus, minus, rma(dx, length)


def _pivot(series: pd.Series, left: int, right: int, highest_pivot: bool) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    span = left + right + 1
    if len(values) < span:
        return pd.Series(out, index=series.index)
    windows = np.lib.stride_tricks.sliding_window_view(values, span)
    centre = windows[:, left]
    empty = np.full(len(windows), np.nan)
    left_side = windows[:, :left].max(axis=1) if left else empty
    right_side = windows[:, left + 1:].max(axis=1) if right else empty
    if highest_pivot:
        neighbours = np.fmax(left_side, right_side)
        found = centre > neighbours
    else:
        neighbours = np.fmin(left_side, right_side)
        found = centre < neighbours
    confirmed = np.arange(len(windows)) + left + right
    out[confirmed] = np.where(found, centre, np.nan)
    return pd.Series(out, index=series.index)


def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    """Pine's ta.pivothigh: the value appears `right` bars after the pivot bar."""
    return _pivot(series, left, right, True)


def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    """Pine's ta.pivotlow: the value appears `right` bars after the pivot bar."""
    return _pivot(series, left, right, False)


def crossover(left, right) -> pd.Series:
    """Pine's ta.crossover; either side may be a scalar level."""
    left, right = _broadcast_pair(left, right)
    return flags((left > right) & (left.shift(1) <= right.shift(1)))


def crossunder(left, right) -> pd.Series:
    """Pine's ta.crossunder; either side may be a scalar level."""
    left, right = _broadcast_pair(left, right)
    return flags((left < right) & (left.shift(1) >= right.shift(1)))


def _broadcast_pair(left, right) -> tuple[pd.Series, pd.Series]:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        return left, right
    template = left if isinstance(left, pd.Series) else right
    if not isinstance(template, pd.Series):
        raise TypeError("crossover 需要至少一侧是 Series")
    if not isinstance(left, pd.Series):
        left = pd.Series(float(left), index=template.index)
    if not isinstance(right, pd.Series):
        right = pd.Series(float(right), index=template.index)
    return left, right


# ---------------------------------------------------------------------------
# Missing-value helpers
# ---------------------------------------------------------------------------

def nz(series: pd.Series, replacement: float = 0.0) -> pd.Series:
    """Pine's `nz`: replace NaN with `replacement`."""
    return series.fillna(replacement)


def na(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fixnan(series: pd.Series) -> pd.Series:
    """Pine's `fixnan`: carry the last defined value forward."""
    return series.ffill()


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly weighted average; the newest value carries weight `length`."""
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()
    return series.rolling(length, min_periods=length).apply(
        lambda window: float(np.dot(window, weights) / weight_sum), raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull average: wma(2 * wma(n / 2) - wma(n), sqrt(n))."""
    half, root = max(1, length // 2), max(1, int(round(math.sqrt(length))))
    return wma(2 * wma(series, half) - wma(series, length), root)


def swma(series: pd.Series) -> pd.Series:
    """Symmetric weighted average over four bars: 1/6, 2/6, 2/6, 1/6."""
    weights = np.array([1, 2, 2, 1], dtype=float) / 6.0
    return series.rolling(4, min_periods=4).apply(
        lambda window: float(np.dot(window, weights)), raw=True)


def vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    numerator = (series * volume).rolling(length, min_periods=length).sum()
    denominator = volume.rolling(length, min_periods=length).sum()
    return numerator / denominator.replace(0, np.nan)


def ma(series: pd.Series, length: int, kind: str = "sma") -> pd.Series:
    """Dispatch over the moving-average families the catalog scripts expose."""
    kind = kind.upper()
    if kind == "SMA":
        return sma(series, length)
    if kind == "EMA":
        return ema(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "HMA":
        return hma(series, length)
    raise ValueError(f"未知均线类型: {kind}")


def alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux moving average as scripted by the catalog sources."""
    positions = np.arange(length, dtype=float)
    centre = offset * (length - 1)
    weights = np.exp(-((positions - centre) ** 2) / (2 * (length / sigma) ** 2))
    weights /= weights.sum()
    return series.rolling(length, min_periods=length).apply(
        lambda window: float(np.dot(window, weights)), raw=True)


# ---------------------------------------------------------------------------
# Dispersion and position
# ---------------------------------------------------------------------------

def variance(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).var(ddof=0)


def dev(series: pd.Series, length: int) -> pd.Series:
    """Pine's `ta.dev`: mean absolute deviation from the window mean."""
    return series.rolling(length, min_periods=length).apply(
        lambda window: float(np.abs(window - window.mean()).mean()), raw=True)


def median(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).median()


def range_(frame: pd.DataFrame, length: int) -> pd.Series:
    """Pine's `ta.range`: highest high minus lowest low over `length` bars."""
    high, low, _ = ohlcv(frame, "high", "low", "close")
    return highest(high, length) - lowest(low, length)


def highestbars(series: pd.Series, length: int) -> pd.Series:
    """Bars since the highest value in the window (Pine returns a negative offset)."""
    return series.rolling(length, min_periods=length).apply(
        lambda window: float(np.argmax(window[::-1]) * -1), raw=True)


def lowestbars(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).apply(
        lambda window: float(np.argmin(window[::-1]) * -1), raw=True)


def sum_(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).sum()


def avg(series: pd.Series, length: int) -> pd.Series:
    return sma(series, length)


def cum(series: pd.Series) -> pd.Series:
    return series.fillna(0).cumsum()


def mom(series: pd.Series, length: int) -> pd.Series:
    return series - series.shift(length)


def roc(series: pd.Series, length: int) -> pd.Series:
    return (series / series.shift(length) - 1) * 100


def zscore(series: pd.Series, length: int) -> pd.Series:
    window_mean = sma(series, length)
    window_dev = stdev(series, length)
    return (series - window_mean) / window_dev.replace(0, np.nan)


def rescale(series: pd.Series, old_min: float, old_max: float,
            new_min: float = 0.0, new_max: float = 100.0) -> pd.Series:
    if old_max == old_min:
        raise ValueError("参数 old_min/old_max 不能相等")
    return (series - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def clamp(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(lower=low, upper=high)


def percentile_nearest_rank(series: pd.Series, length: int, percentage: float) -> pd.Series:
    """Pine's `ta.percentile_nearest_rank`."""
    def pick(window: np.ndarray) -> float:
        ordered = np.sort(window)
        rank = int(math.ceil(percentage / 100.0 * len(ordered))) - 1
        return float(ordered[max(0, min(rank, len(ordered) - 1))])
    return series.rolling(length, min_periods=length).apply(pick, raw=True)


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, length: int) -> pd.Series:
    """Pine's `ta.rsi` (Wilder smoothing), 100 when there is no downside at all."""
    delta = series.diff()
    gain = rma(delta.clip(lower=0), length)
    loss = rma(-delta.clip(upper=0), length)
    values = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return values.mask(loss.eq(0) & gain.gt(0), 100.0)


def stoch(source: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    window_high, window_low = highest(high, length), lowest(low, length)
    return 100 * (source - window_low) / (window_high - window_low).replace(0, np.nan)


def stochrsi(series: pd.Series, length: int) -> pd.Series:
    values = rsi(series, length)
    spread = highest(values, length) - lowest(values, length)
    return (values - lowest(values, length)) / spread.replace(0, np.nan)


def cci(frame: pd.DataFrame, length: int, source: str = "hlc3") -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    typical = {"hlc3": (high + low + close) / 3, "close": close,
               "hl2": (high + low) / 2}[source]
    deviation = dev(typical, length)
    return (typical - sma(typical, length)) / (0.015 * deviation.replace(0, np.nan))


def mfi(frame: pd.DataFrame, length: int) -> pd.Series:
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    typical = (high + low + close) / 3
    flow = typical * volume
    rising = flow.where(typical > typical.shift(1), 0.0)
    falling = flow.where(typical < typical.shift(1), 0.0)
    positive = rising.rolling(length, min_periods=length).sum()
    negative = falling.rolling(length, min_periods=length).sum()
    values = 100 - 100 / (1 + positive / negative.replace(0, np.nan))
    return values.mask(negative.eq(0) & positive.gt(0), 100.0)


def williams_r(frame: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = ohlcv(frame, "high", "low", "close")
    window_high, window_low = highest(high, length), lowest(low, length)
    return -100 * (window_high - close) / (window_high - window_low).replace(0, np.nan)


def tsi(series: pd.Series, short: int = 13, long: int = 25) -> pd.Series:
    momentum = series.diff()
    smooth = ema(ema(momentum, long), short)
    absolute = ema(ema(momentum.abs(), long), short)
    return 100 * smooth / absolute.replace(0, np.nan)


def obv(frame: pd.DataFrame) -> pd.Series:
    close, volume = ohlcv(frame, "close", "volume")
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def ad_line(frame: pd.DataFrame) -> pd.Series:
    high, low, close, volume = ohlcv(frame, "high", "low", "close", "volume")
    spread = (high - low).replace(0, np.nan)
    multiplier = ((close - low) - (high - close)) / spread
    return (multiplier.fillna(0.0) * volume).cumsum()


def bb(series: pd.Series, length: int, multiplier: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = sma(series, length)
    deviation = stdev(series, length)
    return basis, basis + multiplier * deviation, basis - multiplier * deviation


def donchian(frame: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, _ = ohlcv(frame, "high", "low", "close")
    upper, lower = highest(high, length), lowest(low, length)
    return (upper + lower) / 2, upper, lower


def supertrend(frame: pd.DataFrame, factor: float, atr_length: int) -> tuple[pd.Series, pd.Series]:
    """Pine's `ta.supertrend`: (line, direction) with direction 1 up / -1 down."""
    high, low, close = ohlcv(frame, "high", "low", "close")
    average = atr(frame, atr_length).to_numpy(dtype=float)
    midpoint = ((high + low) / 2).to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    upper = midpoint + factor * average
    lower = midpoint - factor * average
    line = np.full(len(closes), np.nan)
    direction = np.full(len(closes), np.nan)
    final_upper = final_lower = np.nan
    trend = 1.0
    for position in range(len(closes)):
        if np.isnan(upper[position]) or np.isnan(lower[position]):
            continue
        if np.isnan(final_upper):
            final_upper, final_lower = upper[position], lower[position]
        else:
            final_upper = (min(upper[position], final_upper)
                           if closes[position - 1] <= final_upper else upper[position])
            final_lower = (max(lower[position], final_lower)
                           if closes[position - 1] >= final_lower else lower[position])
        if trend > 0 and closes[position] < final_lower:
            trend = -1.0
        elif trend < 0 and closes[position] > final_upper:
            trend = 1.0
        line[position] = final_lower if trend > 0 else final_upper
        direction[position] = trend
    return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)


def sar(high: pd.Series, low: pd.Series, start: float = 0.02, increment: float = 0.02,
        maximum: float = 0.2) -> pd.Series:
    """Pine's `ta.sar` parabolic stop-and-reverse."""
    highs, lows = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    out = np.full(len(highs), np.nan)
    if len(highs) < 2:
        return pd.Series(out, index=high.index)
    rising = True
    extreme, acceleration = highs[0], start
    value = lows[0]
    for position in range(1, len(highs)):
        previous = value
        value = previous + acceleration * (extreme - previous)
        if rising:
            value = min(value, lows[position - 1], lows[max(0, position - 2)])
            if lows[position] < value:
                rising, value, extreme, acceleration = False, extreme, lows[position], start
            elif highs[position] > extreme:
                extreme, acceleration = highs[position], min(acceleration + increment, maximum)
        else:
            value = max(value, highs[position - 1], highs[max(0, position - 2)])
            if highs[position] > value:
                rising, value, extreme, acceleration = True, extreme, highs[position], start
            elif lows[position] < extreme:
                extreme, acceleration = lows[position], min(acceleration + increment, maximum)
        out[position] = value
    return pd.Series(out, index=high.index)


# ---------------------------------------------------------------------------
# Regression and statistics
# ---------------------------------------------------------------------------

def linreg(series: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """Pine's `ta.linreg`: least-squares line value `offset` bars back."""
    def fit(window: np.ndarray) -> float:
        x = np.arange(len(window), dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        return float(slope * (len(window) - 1 - offset) + intercept)
    return series.rolling(length, min_periods=length).apply(fit, raw=True)


def correlation(left: pd.Series, right: pd.Series, length: int) -> pd.Series:
    return left.rolling(length, min_periods=length).corr(right)


def covariance(left: pd.Series, right: pd.Series, length: int) -> pd.Series:
    return left.rolling(length, min_periods=length).cov(right)


def correlation_matrix_hint() -> None:  # pragma: no cover - documentation helper
    """Placeholder kept out of the public surface."""
    return None


# ---------------------------------------------------------------------------
# Bar-conditional helpers
# ---------------------------------------------------------------------------

def barssince(condition: pd.Series) -> pd.Series:
    """Pine's `ta.barssince`: bars since the condition was last true."""
    out = np.full(len(condition), np.nan)
    last = None
    for position, value in enumerate(condition.fillna(False).to_numpy(dtype=bool)):
        if value:
            last = position
        if last is not None:
            out[position] = position - last
    return pd.Series(out, index=condition.index)


def valuewhen(condition: pd.Series, source: pd.Series, occurrence: int = 0) -> pd.Series:
    """Pine's `ta.valuewhen`: `source` at the n-th most recent true occurrence."""
    truthy = condition.fillna(False).to_numpy(dtype=bool)
    values = source.to_numpy(dtype=float)
    out = np.full(len(truthy), np.nan)
    history: list[float] = []
    for position in range(len(truthy)):
        if truthy[position]:
            history.insert(0, values[position])
        if len(history) > occurrence:
            out[position] = history[occurrence]
    return pd.Series(out, index=condition.index)


def rising(series: pd.Series, length: int) -> pd.Series:
    up = series > series.shift(1)
    for step in range(1, length):
        up &= series.shift(step) > series.shift(step + 1)
    return flags(up)


def falling(series: pd.Series, length: int) -> pd.Series:
    down = series < series.shift(1)
    for step in range(1, length):
        down &= series.shift(step) < series.shift(step + 1)
    return flags(down)


def cross(left: pd.Series, right: pd.Series) -> pd.Series:
    return flags(((left > right) & (left.shift(1) <= right.shift(1)))
                 | ((left < right) & (left.shift(1) >= right.shift(1))))


def edge(flag: pd.Series) -> pd.Series:
    """True on the first bar of a run: the flag is set and was clear on the prior bar."""
    clean = flags(flag)
    return clean & ~clean.shift(1, fill_value=False)


def nullable_int(values: pd.Series) -> pd.Series:
    """Round to whole numbers while keeping warm-up gaps as NA (pandas Int64)."""
    return values.round().astype("Int64")
