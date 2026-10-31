"""MTF StochRSI Pair Score — six-timeframe StochRSI sync score.

Source: pine-collection/indicators/momentum/mtf_stochrsi_pair_score/mtf_stochrsi_pair_score.pine
(WavesUnchained, MIT)

Each timeframe gets a StochRSI score in ±100: ±20 for the K/D stacking side,
±20 for K rising/falling, ±0.5×(K−D) and ±0.8×(K delta) as continuous terms,
±30/±15 bonuses for crosses inside/outside the OB/OS zones, ±10 for zone
riding, all ×0.65 when both lines are exhausted in one zone. Adjacent
timeframes form pairs weighted toward the faster member (`minor_weight`);
same-direction pairs earn a `sync_bonus`, conflicts pay a `conflict_penalty`.
The weighted pair average is smoothed into the total score. Agreement is the
net bull/bear vote across the six timeframes (±100 = unanimous); bias states
combine the score and agreement with the `min_agreement` gate, and late-entry
counts flag how many timeframes already sit inside a zone.

HTF series follow the repo convention of reading the previous completed
period (`_tf.htf_frame` + `lift_series`); the daily timeframe uses
`calendar_frame`. Timeframes coarser than the feed report 0 scores.

Dropped: table, debug table, signal markers and all colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import (choice, crossover, crossunder, flags, integer, number, ohlcv, rsi,
                    sma, stoch)
from ._tf import aligned_index, base_step_minutes, calendar_frame, htf_frame, lift_series

_PAIR_KEYS = (("p12", 0, 1), ("p23", 1, 2), ("p34", 2, 3),
              ("p45", 3, 4), ("p56", 4, 5))


def _smooth(series: pd.Series, length: int, mode: str, phase: int, power: float) -> pd.Series:
    from .commodity_flowtrend import _smooth as flow_smooth
    if mode == "HMA":
        from ._pine import hma
        return hma(series, length)
    mapping = {"EMA": "EMA", "RMA": "RMA", "SMA": "SMA", "WMA": "WMA", "JMA": "JMA Approx"}
    return flow_smooth(series, length, mapping[mode], phase, power)


def mtf_stochrsi_pair_score(
        frame: pd.DataFrame, *, rsi_len: int = 14, stoch_len: int = 14,
        k_smooth: int = 3, d_smooth: int = 3, upper: float = 80.0,
        lower: float = 20.0, timeframes: tuple[str, ...] = ("15", "30", "60", "120",
                                                            "240", "D"),
        minor_weight: float = 0.4, sync_bonus: float = 1.15,
        conflict_penalty: float = 0.75, use_pairs: tuple[bool, ...] = (True, True, True,
                                                                      True, False),
        w_p12: float = 1.0, w_p23: float = 1.5, w_p34: float = 1.3, w_p45: float = 1.2,
        w_p56: float = 1.0, smooth_mode: str = "JMA", smooth_len: int = 5,
        jma_phase: int = 50, jma_power: float = 2.0, long_level: float = 60.0,
        short_level: float = -60.0, building_level: float = 30.0,
        min_agreement: float = 33.0, fresh_window: int = 3) -> pd.DataFrame:
    integer("rsi_len", rsi_len, 1)
    integer("stoch_len", stoch_len, 1)
    integer("k_smooth", k_smooth, 1)
    integer("d_smooth", d_smooth, 1)
    number("upper", upper, 50.0, 100.0)
    number("lower", lower, 0.0, 50.0)
    number("minor_weight", minor_weight, 0.0, 1.0)
    number("sync_bonus", sync_bonus, 1.0, 2.0)
    number("conflict_penalty", conflict_penalty, 0.1, 1.0)
    if len(use_pairs) != 5:
        raise ValueError("参数 use_pairs 需要 5 个布尔值")
    choice("smooth_mode", smooth_mode, ("EMA", "RMA", "SMA", "WMA", "HMA", "JMA"))
    integer("smooth_len", smooth_len, 1, 50)
    number("long_level", long_level, 0.0, 100.0)
    number("short_level", short_level, -100.0, 0.0)
    number("building_level", building_level, 0.0, 100.0)
    number("min_agreement", min_agreement, 0.0, 100.0)
    integer("fresh_window", fresh_window, 1)
    if len(timeframes) != 6:
        raise ValueError("参数 timeframes 需要 6 个周期标签")

    ohlcv(frame, "close")
    utc = aligned_index(frame, 1)
    base = base_step_minutes(frame)

    def timeframe_minutes(label: str) -> int | None:
        if label in ("D", "1D"):
            return 1440
        if label.endswith("min"):
            return int(label[:-3])
        try:
            return int(label)
        except ValueError as exc:
            raise ValueError(f"未知周期标签: {label}") from exc

    close = frame["close"].astype(float)

    def tf_score(series: pd.Series) -> pd.DataFrame:
        r_value = rsi(series, rsi_len)
        raw = stoch(r_value, r_value, r_value, stoch_len)
        k = sma(raw, k_smooth)
        d = sma(k, d_smooth)
        score = pd.Series(0.0, index=series.index)
        score += np.where(k > d, 20.0, -20.0)
        score += np.where(k > k.shift(1), 20.0, -20.0)
        score += (k - d) * 0.5
        score += (k - k.shift(1)) * 0.8
        cross_up = crossover(k, d)
        cross_dn = crossunder(k, d)
        score += np.where(cross_up, np.where(k < lower, 30.0, 15.0), 0.0)
        score += np.where(cross_dn, np.where(k > upper, -30.0, -15.0), 0.0)
        score += np.where((k > upper) & (k > k.shift(1)), 10.0, 0.0)
        score += np.where((k < lower) & ~(k > k.shift(1)), -10.0, 0.0)
        exhausted_high = (k > upper) & (d > upper) & ~cross_dn
        exhausted_low = (k < lower) & (d < lower) & ~cross_up
        score = score.mask(exhausted_high | exhausted_low, score * 0.65)
        return pd.DataFrame({"score": score.clip(lower=-100.0, upper=100.0), "k": k,
                             "d": d}, index=series.index)

    scores: list[pd.DataFrame] = []
    for label in timeframes:
        minutes = timeframe_minutes(label)
        if minutes is not None and minutes < base:
            scores.append(pd.DataFrame({"score": 0.0, "k": np.nan, "d": np.nan},
                                       index=frame.index))
            continue
        if label in ("D", "1D"):
            aggregated = calendar_frame(frame, "D")["close"]
            computed = tf_score(aggregated)
            lifted_score = _lift(computed["score"], frame, "D")
            lifted_k = _lift(computed["k"], frame, "D")
            lifted_d = _lift(computed["d"], frame, "D")
        else:
            aggregated = htf_frame(frame, minutes)["close"]
            computed = tf_score(aggregated)
            lifted_score = lift_series(frame, computed["score"], minutes)
            lifted_k = lift_series(frame, computed["k"], minutes)
            lifted_d = lift_series(frame, computed["d"], minutes)
        scores.append(pd.DataFrame({"score": lifted_score.fillna(0.0),
                                    "k": lifted_k, "d": lifted_d},
                                   index=frame.index))

    def pair(s_minor: pd.Series, s_major: pd.Series) -> pd.Series:
        major_weight = 1.0 - minor_weight
        base_score = s_minor * minor_weight + s_major * major_weight
        same_direction = ((s_minor > 0) & (s_major > 0)) | ((s_minor < 0) & (s_major < 0))
        conflict = ((s_minor > 0) & (s_major < 0)) | ((s_minor < 0) & (s_major > 0))
        adjusted = base_score.mask(same_direction, base_score * sync_bonus)
        adjusted = adjusted.mask(conflict, base_score * conflict_penalty)
        return adjusted.clip(lower=-100.0, upper=100.0)

    weights = {"p12": w_p12, "p23": w_p23, "p34": w_p34, "p45": w_p45, "p56": w_p56}
    pair_columns: dict[str, pd.Series] = {}
    weighted_sum = pd.Series(0.0, index=frame.index)
    weight_sum = pd.Series(0.0, index=frame.index)
    for (key, a, b), enabled in zip(_PAIR_KEYS, use_pairs):
        pair_value = pair(scores[a]["score"], scores[b]["score"])
        pair_columns[key] = pair_value
        if enabled:
            weighted_sum = weighted_sum + pair_value * weights[key]
            weight_sum = weight_sum + weights[key]
    total = pd.Series(np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan),
                      index=frame.index)
    smoothed_total = _smooth(total, smooth_len, smooth_mode, jma_phase, jma_power)

    score_matrix = pd.concat([s["score"] for s in scores], axis=1)
    bull_count = (score_matrix > 0).sum(axis=1).astype(float)
    bear_count = (score_matrix < 0).sum(axis=1).astype(float)
    agreement = (bull_count - bear_count) / 6.0 * 100.0

    long_trigger = flags(crossover(smoothed_total, long_level)
                         & (agreement >= min_agreement))
    short_trigger = flags(crossunder(smoothed_total, short_level)
                          & (agreement <= -min_agreement))

    bias_dir = pd.Series(np.where(smoothed_total > 0, 1, np.where(smoothed_total < 0, -1, 0)),
                         index=frame.index)
    agree_dir = pd.Series(np.where(agreement > 0, 1, np.where(agreement < 0, -1, 0)),
                          index=frame.index)
    conflict_flag = (bias_dir != 0) & (agree_dir != 0) & (bias_dir != agree_dir)
    bias_state = pd.Series("NO EDGE", index=frame.index, dtype=object)
    bias_state = bias_state.mask((smoothed_total >= long_level)
                                 & (agreement >= min_agreement), "LONG CONFIRMED")
    bias_state = bias_state.mask((smoothed_total <= short_level)
                                 & (agreement <= -min_agreement), "SHORT CONFIRMED")
    bias_state = bias_state.mask(conflict_flag, "CONFLICT")
    bias_state = bias_state.mask((smoothed_total > building_level) & (agree_dir >= 0),
                                 "LONG BUILDING")
    bias_state = bias_state.mask((smoothed_total < -building_level) & (agree_dir <= 0),
                                 "SHORT BUILDING")

    k_matrix = pd.concat([s["k"] for s in scores], axis=1)
    quality_state = pd.Series("WEAK", index=frame.index, dtype=object)
    quality_state = quality_state.mask(agreement.abs() >= 66, "CLEAN")
    quality_state = quality_state.mask(agreement.abs() >= 33, "MIXED")

    late_long_count = (k_matrix > upper).sum(axis=1).astype(float)
    late_short_count = (k_matrix < lower).sum(axis=1).astype(float)
    is_late_long = flags(late_long_count >= 4)
    is_late_short = flags(late_short_count >= 4)

    from ._pine import barssince
    bars_since_long = barssince(long_trigger)
    bars_since_short = barssince(short_trigger)
    timing = pd.Series("-", index=frame.index, dtype=object)
    timing = timing.mask(long_trigger, "FRESH LONG")
    timing = timing.mask(short_trigger, "FRESH SHORT")
    timing = timing.mask(bars_since_long < fresh_window, "FRESH LONG")
    timing = timing.mask(bars_since_short < fresh_window, "FRESH SHORT")
    timing = timing.mask(is_late_long, "LATE LONG")
    timing = timing.mask(is_late_short, "LATE SHORT")

    output = {f"score_{label}": scores[i]["score"] for i, label in enumerate(timeframes)}
    output.update({f"k_{label}": scores[i]["k"] for i, label in enumerate(timeframes)})
    output.update(pair_columns)
    output.update({
        "total_score": total,
        "smoothed_score": smoothed_total,
        "agreement": agreement,
        "bull_count": bull_count.astype("Int64"),
        "bear_count": bear_count.astype("Int64"),
        "long_trigger": long_trigger,
        "short_trigger": short_trigger,
        "bias_state": bias_state,
        "quality_state": quality_state,
        "late_long": is_late_long,
        "late_short": is_late_short,
        "timing": timing,
    })
    return pd.DataFrame(output, index=frame.index)


def _lift(values: pd.Series, frame: pd.DataFrame, label: str) -> pd.Series:
    """Lift a daily (calendar) series onto base bars, previous completed day."""
    utc = aligned_index(frame, 1)
    period = utc.tz_localize(None).to_period("D")
    previous = pd.PeriodIndex(period - 1, freq="D")
    starts = pd.DatetimeIndex([p.start_time for p in previous]).tz_localize("UTC")
    return pd.Series(values.reindex(starts).to_numpy(dtype=float), index=utc)
