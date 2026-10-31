"""Ports batch W3A: anchored_vwap, momentum_profile,
adaptive_arithmetic_candles_quality_score, capital41_order_block_v2,
predictability_regime_index."""
import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.adaptive_arithmetic_candles_quality_score import (
    adaptive_arithmetic_candles_quality_score,
)
from app.catalog_indicators.anchored_vwap import anchored_vwap
from app.catalog_indicators.capital41_order_block_v2 import capital41_order_block_v2
from app.catalog_indicators.momentum_profile import momentum_profile


def _frame(bars, *, start="2024-01-01", freq="1h", open_=None, high=None, low=None,
           close=None, volume=None):
    index = pd.date_range(start, periods=bars, freq=freq, tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": np.asarray(open_, dtype=float) if open_ is not None else close,
        "high": np.asarray(high, dtype=float) if high is not None else close + 1.0,
        "low": np.asarray(low, dtype=float) if low is not None else close - 1.0,
        "close": close,
        "volume": np.full(bars, 100.0) if volume is None else np.asarray(volume, dtype=float),
    }, index=index)


# ---------------------------------------------------------------------------
# anchored_vwap
# ---------------------------------------------------------------------------

def test_anchored_vwap_session_cumulative_hand_computed():
    # One session, constant volume: the VWAP is the running mean of the closes
    # 1..20, so bar t reads (t + 2) / 2.
    bars = 20
    closes = np.arange(1, bars + 1, dtype=float)
    frame = _frame(bars, close=closes, volume=np.full(bars, 10.0))
    result = anchored_vwap(frame, anchor_mode="Session", source="close")

    # The session start fires the anchor on bar 0 only.
    assert result.anchor_event.tolist() == [True] + [False] * (bars - 1)
    for bar in range(bars):
        assert result.avwap.iloc[bar] == pytest.approx((bar + 2) / 2.0)

    # 20 bars of 1..20: E[x^2] = 143.5, E[x] = 10.5 -> var = 33.25.
    variance = 143.5 - 10.5 ** 2
    sigma = math.sqrt(variance)
    assert result.sigma.iloc[19] == pytest.approx(sigma)
    assert result.upper_band_1.iloc[19] == pytest.approx(10.5 + sigma)
    assert result.lower_band_1.iloc[19] == pytest.approx(10.5 - sigma)
    assert result.upper_band_2.iloc[19] == pytest.approx(10.5 + 2.0 * sigma)
    assert result.lower_band_2.iloc[19] == pytest.approx(10.5 - 2.0 * sigma)
    assert result.distance_sigma.iloc[19] == pytest.approx((20.0 - 10.5) / sigma)
    # |distance| = 1.65σ stays below the default 2σ stretch gate.
    assert not result.stretched_up.iloc[19]
    assert not result.stretched_dn.iloc[19]


def test_anchored_vwap_swing_low_reseeds_from_pivot_bar():
    # Lows fall to bar 5 and rise afterwards, so with left=1/right=1 the only
    # swing low is bar 5 and it is confirmed on bar 6.
    bars = 20
    lows = np.concatenate([np.arange(9, 3, -1.0), np.arange(5, 19.0)])
    closes = lows + 1.0
    frame = _frame(bars, low=lows, close=closes, volume=np.full(bars, 10.0))
    result = anchored_vwap(frame, anchor_mode="Auto: Swing Low", pivot_left=1, pivot_right=1,
                           source="close", stretch_sigma=0.5)

    # No anchor event until the pivot is confirmed on bar 6; the seed on bar 0
    # comes from `na(sumV)`, not from an event.
    assert not result.anchor_event.iloc[:6].any()
    assert result.anchor_event.iloc[6]
    assert not result.anchor_event.iloc[7:].any()

    # Bar 0..5 accumulate from bar 0: running mean of 10, 9, 8, 7, 6, 5.
    assert result.avwap.iloc[0] == pytest.approx(10.0)
    assert result.avwap.iloc[5] == pytest.approx((10.0 + 9.0 + 8.0 + 7.0 + 6.0 + 5.0) / 6.0)
    # Bar 6 re-seeds from the pivot bar (5) through the confirmation bar (6).
    assert result.avwap.iloc[6] == pytest.approx((5.0 + 6.0) / 2.0)
    assert result.avwap.iloc[7] == pytest.approx((5.0 + 6.0 + 7.0) / 3.0)
    # σ over the two bars 5 and 6: E[x^2] = 30.5, E[x] = 5.5 -> var = 0.25.
    assert result.sigma.iloc[6] == pytest.approx(0.5)
    assert result.distance_sigma.iloc[6] == pytest.approx((6.0 - 5.5) / 0.5)
    assert result.stretched_up.iloc[6]
    assert not result.stretched_dn.iloc[6]


def test_anchored_vwap_without_real_volume_stays_empty():
    bars = 12
    frame = _frame(bars, close=np.linspace(10.0, 12.0, bars), volume=np.zeros(bars))
    result = anchored_vwap(frame, anchor_mode="Session", source="close")
    assert result.avwap.isna().all()
    assert result.sigma.isna().all()
    assert not result.stretched_up.any()


def test_anchored_vwap_rejects_bad_input():
    frame = _frame(20, close=np.arange(1, 21, dtype=float))
    with pytest.raises(ValueError, match="pivot_left"):
        anchored_vwap(frame, anchor_mode="Auto: Last Swing", pivot_left=0)
    with pytest.raises(ValueError, match="stretch_sigma"):
        anchored_vwap(frame, anchor_mode="Session", stretch_sigma=0.4)
    with pytest.raises(ValueError, match="anchor_mode"):
        anchored_vwap(frame, anchor_mode="Quarter")
    with pytest.raises(ValueError, match="volume"):
        anchored_vwap(frame.drop(columns="volume"), anchor_mode="Session")


# ---------------------------------------------------------------------------
# momentum_profile
# ---------------------------------------------------------------------------

def _ema(values, length):
    """Pine's ta.ema: seed with the mean of the first `length` values, then recurse."""
    out = [float("nan")] * len(values)
    alpha = 2.0 / (length + 1.0)
    seed = [value for value in values[:length] if not math.isnan(value)]
    previous = sum(seed) / len(seed) if seed else float("nan")
    out[length - 1] = previous
    for position in range(length, len(values)):
        value = values[position]
        if math.isnan(value):
            continue
        previous = value if math.isnan(previous) else alpha * value + (1.0 - alpha) * previous
        out[position] = previous
    return out


_LEVELS = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 20.0, 60.0, 40.0, 80.0, 100.0])


def _profile_frame(bars=20):
    # The level sequence repeats every 10 bars, so every 10-bar window spans
    # 9.9..100.1 and each level owns one row of the 10-row grid:
    # row = floor((level - 9.9) / 9.02), and the bar's ±0.1 range never crosses
    # a row boundary.
    level = np.tile(_LEVELS, bars // 10 + 1)[:bars]
    return _frame(bars, open_=level, high=level + 0.1, low=level - 0.1,
                  close=level, volume=np.full(bars, 100.0)), level


def _row_of(level):
    return int((level - 9.9) // 9.02)


def _expected_wavetrend(level, channel_length=3, average_length=3):
    typical = level.tolist()
    average = _ema(typical, channel_length)
    deviation = _ema([abs(typical[i] - average[i]) for i in range(len(typical))], channel_length)
    channel = [(typical[i] - average[i]) / (0.015 * deviation[i])
               for i in range(len(typical))]
    return _ema(channel, average_length)


def test_momentum_profile_rows_and_poc_hand_computed():
    bars = 20
    frame, level = _profile_frame(bars)
    rows, lookback = 10, 10
    result = momentum_profile(frame, lookback=lookback, rows=rows, wt_channel_length=3,
                              wt_average_length=3, mfi_length=2)
    wt = _expected_wavetrend(level)

    # WaveTrend itself, bar by bar (bars 0..1 are the EMA warm-up).
    assert result.wavetrend.iloc[:2].isna().all()
    for bar in range(2, bars):
        assert result.wavetrend.iloc[bar] == pytest.approx(wt[bar])

    # The 10-bar window is only complete from bar 9 on.
    assert result.profile_low.iloc[:9].isna().all()
    assert result.profile_high.iloc[:9].isna().all()
    step = (100.1 - 9.9) / rows
    for bar in range(9, bars):
        assert result.profile_low.iloc[bar] == pytest.approx(9.9)
        assert result.profile_high.iloc[bar] == pytest.approx(100.1)
        # One bar per row -> the row of the current close holds exactly one bar.
        assert result.close_row.iloc[bar] == _row_of(level[bar])
        assert result.close_row_coverage.iloc[bar] == pytest.approx(1.0)
        assert result.close_row_wavetrend.iloc[bar] == pytest.approx(wt[bar])
        assert result.close_row_mfi.iloc[bar] == pytest.approx(result.mfi.iloc[bar])
        # Pine feeds the rows with nz(wavetrend), so warm-up bars count as 0.
        window_wt = [0.0 if math.isnan(value) else value
                     for value in wt[bar - lookback + 1:bar + 1]]
        assert result.avg_wavetrend.iloc[bar] == pytest.approx(sum(window_wt) / lookback)
        # mPOC = the row with the largest |WaveTrend| inside the window.
        window = list(range(bar - lookback + 1, bar + 1))
        best = max(window, key=lambda index: abs(0.0 if math.isnan(wt[index]) else wt[index]))
        assert result.momentum_poc_row.iloc[bar] == _row_of(level[best])
        assert result.momentum_poc_wavetrend.iloc[bar] == pytest.approx(wt[best])
        assert result.momentum_poc_price.iloc[bar] == pytest.approx(
            9.9 + (_row_of(level[best]) + 0.5) * step)

    # MFI on bar 10: window bars 9 (rising, 100 * 100) and 10 (falling, 10 * 100)
    # -> 100 - 100 / (1 + 10000 / 1000) = 90.909.
    assert result.mfi.iloc[10] == pytest.approx(100.0 - 100.0 / 11.0)
    assert result.mfi.iloc[1] == pytest.approx(100.0)


def test_momentum_profile_neutral_mfi_without_volume():
    bars = 20
    frame, _ = _profile_frame(bars)
    frame["volume"] = 0.0
    result = momentum_profile(frame, lookback=10, rows=10, wt_channel_length=3,
                              wt_average_length=3, mfi_length=2)
    # MFI is gated on a real volume feed; the profile then reads every bar as 50.
    assert result.mfi.isna().all()
    assert result.avg_mfi.iloc[9:].eq(50.0).all()
    assert result.close_row_mfi.iloc[9:].eq(50.0).all()


def test_momentum_profile_rejects_bad_input():
    frame, _ = _profile_frame()
    with pytest.raises(ValueError, match="lookback"):
        momentum_profile(frame, lookback=9)
    with pytest.raises(ValueError, match="rows"):
        momentum_profile(frame, rows=101)
    with pytest.raises(ValueError, match="wt_channel_length"):
        momentum_profile(frame, wt_channel_length=0)
    with pytest.raises(ValueError, match="volume"):
        momentum_profile(frame.drop(columns="volume"))


# ---------------------------------------------------------------------------
# adaptive_arithmetic_candles_quality_score
# ---------------------------------------------------------------------------

_RAMP = [10.0, 11.0, 12.5, 12.0, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5,
         17.0, 18.0, 17.0, 19.0, 20.0, 19.5, 21.0, 22.0, 21.5, 23.0]


def _arithmetic_frame(bars=20):
    close = np.array(_RAMP[:bars], dtype=float)
    return _frame(bars, open_=close - 0.5, high=close + 1.0, low=close - 1.0, close=close)


def _expected_arithmetic(frame, efficiency_length=2, atr_length=2, regression_length=5,
                         wick_factor=0.55):
    """Replay the Pine script bar by bar from the source formulas."""
    opening = frame["open"].tolist()
    high = frame["high"].tolist()
    low = frame["low"].tolist()
    close = frame["close"].tolist()
    bars = len(close)
    nan = float("nan")

    efficiency = [nan] * bars
    for bar in range(efficiency_length, bars):
        noise = sum(abs(close[i] - close[i - 1])
                    for i in range(bar - efficiency_length + 1, bar + 1))
        travel = abs(close[bar] - close[bar - efficiency_length])
        efficiency[bar] = travel / noise if noise != 0.0 else 0.0

    true_range = [high[0] - low[0]]
    for bar in range(1, bars):
        previous = close[bar - 1]
        true_range.append(max(high[bar] - low[bar], abs(high[bar] - previous),
                              abs(low[bar] - previous)))
    average_range = [nan] * bars
    seed = sum(true_range[:atr_length]) / atr_length
    average_range[atr_length - 1] = seed
    for bar in range(atr_length, bars):
        seed = (seed * (atr_length - 1) + true_range[bar]) / atr_length
        average_range[bar] = seed

    def _linreg(length, offset):
        out = [nan] * bars
        xs = list(range(length))
        mean_x = sum(xs) / length
        denominator = sum((x - mean_x) ** 2 for x in xs)
        for bar in range(length - 1, bars):
            ys = close[bar - length + 1:bar + 1]
            mean_y = sum(ys) / length
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
            out[bar] = slope * (length - 1 - offset) + mean_y - slope * mean_x
        return out

    regression_now = _linreg(regression_length, 0)
    regression_prev = _linreg(regression_length, 1)
    slope = [nan] * bars
    body_norm = [nan] * bars
    for bar in range(bars):
        if average_range[bar] != 0.0:
            body_norm[bar] = (close[bar] - opening[bar]) / average_range[bar]
            slope[bar] = (regression_now[bar] - regression_prev[bar]) / average_range[bar]

    alpha = [(efficiency[bar] * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2.0
             for bar in range(bars)]
    typical = [(opening[bar] + high[bar] + low[bar] + close[bar]) / 4.0 for bar in range(bars)]
    weighted = [close[bar] * 0.45 + typical[bar] * 0.35 + regression_now[bar] * 0.20
                for bar in range(bars)]

    smoothed = [nan] * bars
    previous = nan
    for bar in range(bars):
        previous = weighted[bar] if math.isnan(previous) else previous + alpha[bar] * (
            weighted[bar] - previous)
        smoothed[bar] = previous
    synthetic_open = [nan] * bars
    previous = nan
    for bar in range(bars):
        previous = opening[bar] if math.isnan(previous) else previous * 0.45 + smoothed[bar - 1] * 0.55
        synthetic_open[bar] = previous

    synthetic_close = [nan] * bars
    synthetic_high = [nan] * bars
    synthetic_low = [nan] * bars
    direction = [0] * bars
    body_score = [nan] * bars
    slope_score = [nan] * bars
    quality = [None] * bars
    for bar in range(bars):
        pressure = math.tanh(slope[bar] * 1.8 + body_norm[bar] * 0.35)
        boost = average_range[bar] * pressure * efficiency[bar] * 0.18
        closing = smoothed[bar] + boost
        upper = (high[bar] - max(opening[bar], close[bar])) * (
            wick_factor + (1.0 - efficiency[bar]) * 0.25)
        lower = (min(opening[bar], close[bar]) - low[bar]) * (
            wick_factor + (1.0 - efficiency[bar]) * 0.25)
        synthetic_close[bar] = closing
        synthetic_high[bar] = np.maximum(np.maximum(synthetic_open[bar], closing), closing + upper)
        synthetic_low[bar] = np.minimum(np.minimum(synthetic_open[bar], closing), closing - lower)
        if not math.isnan(closing):
            direction[bar] = 1 if closing > synthetic_open[bar] else (
                -1 if closing < synthetic_open[bar] else 0)
            slope_score[bar] = min(abs(slope[bar]) * 140.0, 100.0)
            span = synthetic_high[bar] - synthetic_low[bar]
            ratio = abs(closing - synthetic_open[bar]) / span if span != 0.0 else 0.0
            body_score[bar] = min(ratio * 140.0, 100.0)
            follow = (close[bar] > close[bar - 1] if direction[bar] == 1 else
                      close[bar] < close[bar - 1] if direction[bar] == -1 else False) if bar else False
            raw = (efficiency[bar] * 100.0 * 0.35 + slope_score[bar] * 0.30
                   + body_score[bar] * 0.20 + (100.0 if follow else 35.0) * 0.15)
            quality[bar] = int(np.round(min(max(raw, 0.0), 100.0)))
    return {
        "efficiency": efficiency, "slope": slope, "alpha": alpha, "arith_open": synthetic_open,
        "arith_close": synthetic_close, "arith_high": synthetic_high, "arith_low": synthetic_low,
        "direction": direction, "quality": quality,
    }


def test_adaptive_arithmetic_candles_hand_computed():
    frame = _arithmetic_frame()
    result = adaptive_arithmetic_candles_quality_score(
        frame, efficiency_length=2, atr_length=2, regression_length=5, wick_factor=0.55,
        signal_length=2, min_quality=65)
    expected = _expected_arithmetic(frame)

    # ER needs `close[efficiency_length]`, ATR the Wilder seed and the regression
    # five bars, so the synthetic candles only exist from bar 4 on.
    assert result.quality.iloc[:4].isna().all()
    assert result.efficiency_ratio.iloc[:2].isna().all()
    assert result.regression_slope.iloc[:4].isna().all()

    for bar in range(len(frame)):
        assert result.efficiency_ratio.iloc[bar] == pytest.approx(expected["efficiency"][bar],
                                                                  nan_ok=True)
        assert result.regression_slope.iloc[bar] == pytest.approx(expected["slope"][bar],
                                                                  nan_ok=True)
        assert result.adaptive_alpha.iloc[bar] == pytest.approx(expected["alpha"][bar],
                                                               nan_ok=True)
        assert result.arith_open.iloc[bar] == pytest.approx(expected["arith_open"][bar],
                                                            nan_ok=True)
        assert result.arith_close.iloc[bar] == pytest.approx(expected["arith_close"][bar],
                                                             nan_ok=True)
        assert result.arith_high.iloc[bar] == pytest.approx(expected["arith_high"][bar],
                                                            nan_ok=True)
        assert result.arith_low.iloc[bar] == pytest.approx(expected["arith_low"][bar],
                                                           nan_ok=True)
        assert result.direction.iloc[bar] == expected["direction"][bar]
        if expected["quality"][bar] is None:
            assert result.quality.iloc[bar] is pd.NA or math.isnan(result.quality.iloc[bar])
        else:
            assert result.quality.iloc[bar] == expected["quality"][bar]
            assert result.dir_quality.iloc[bar] == (expected["quality"][bar]
                                                    * expected["direction"][bar])

    # Strong bull/bear and the breakout triggers, recomputed from the same data.
    closes = frame["close"].tolist()
    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    for bar in range(len(frame)):
        # ta.highest(high[1], 2) needs two defined bars, so bar 1 has no breakout.
        assert result.break_high.iloc[bar] == bool(
            bar >= 2 and closes[bar] > max(highs[bar - 2:bar]))
        assert result.break_low.iloc[bar] == bool(
            bar >= 2 and closes[bar] < min(lows[bar - 2:bar]))
        assert result.strong_bull.iloc[bar] == bool(
            expected["direction"][bar] == 1 and expected["quality"][bar] is not None
            and expected["quality"][bar] >= 65 and expected["slope"][bar] > 0)
        assert result.long_signal.iloc[bar] == bool(result.strong_bull.iloc[bar]
                                                    and result.break_high.iloc[bar])
        assert result.trend_flip_up.iloc[bar] == bool(
            bar and expected["direction"][bar] == 1 and expected["direction"][bar - 1] == -1
            and expected["quality"][bar] is not None and expected["quality"][bar] >= 55
            and expected["slope"][bar] > 0)


def test_adaptive_arithmetic_candles_rejects_bad_input():
    frame = _arithmetic_frame()
    with pytest.raises(ValueError, match="efficiency_length"):
        adaptive_arithmetic_candles_quality_score(frame, efficiency_length=1)
    with pytest.raises(ValueError, match="regression_length"):
        adaptive_arithmetic_candles_quality_score(frame, regression_length=4)
    with pytest.raises(ValueError, match="wick_factor"):
        adaptive_arithmetic_candles_quality_score(frame, wick_factor=1.6)
    with pytest.raises(ValueError, match="min_quality"):
        adaptive_arithmetic_candles_quality_score(frame, min_quality=0)
    with pytest.raises(ValueError, match="open"):
        adaptive_arithmetic_candles_quality_score(frame.drop(columns="open"))


# ---------------------------------------------------------------------------
# capital41_order_block_v2
# ---------------------------------------------------------------------------

def _order_block_frame():
    # (open, close) pairs; high = max + 0.5, low = min - 0.5. The ATR-2 impulse
    # bars are 5 (bull), 6 (bull), 8 (bear) and 11 (bear); every other bar moves
    # by 0.1 only.
    opens = [10.0, 10.1, 10.0, 10.1, 10.0, 9.9, 13.9, 19.9, 19.8, 11.8, 11.9, 11.8, 5.8, 5.9]
    closes = [10.1, 10.0, 10.1, 10.0, 9.9, 13.9, 19.9, 19.8, 13.8, 11.9, 11.8, 5.8, 5.9, 5.8]
    opens = np.array(opens, dtype=float)
    closes = np.array(closes, dtype=float)
    return _frame(len(opens), open_=opens, high=np.maximum(opens, closes) + 0.5,
                  low=np.minimum(opens, closes) - 0.5, close=closes)


def _staircase_frame():
    # Pairs of (bearish, bullish impulse) candles ten price units apart: every
    # impulse opens one bullish order block from the candle right before it.
    opens, closes = [], []
    for step in range(8):
        opens.extend([10.0 * step + 11.0, 10.0 * step + 10.0])
        closes.extend([10.0 * step + 10.0, 10.0 * step + 18.0])
    opens = np.array(opens, dtype=float)
    closes = np.array(closes, dtype=float)
    return _frame(len(opens), open_=opens, high=np.maximum(opens, closes) + 0.5,
                  low=np.minimum(opens, closes) - 0.5, close=closes)


def test_capital41_order_block_v2_creation_and_mitigation():
    frame = _order_block_frame()
    result = capital41_order_block_v2(frame, atr_length=2, impulse_mult=1.0, ob_lookback=10,
                                      max_active=25, freshness_length=80,
                                      min_mitigation_age=2)

    assert result.bull_impulse.tolist() == [False] * 5 + [True, True] + [False] * 7
    assert result.bear_impulse.tolist() == [False] * 8 + [True] + [False] * 2 + [True] + [False] * 2

    # Bar 5 takes the bearish candle of bar 4 as the bullish order block: its full
    # range 9.4..10.5, one bar back. Bar 6 finds the same candle again and the
    # overlap check skips it.
    assert result.bull_ob_created.tolist() == [False] * 5 + [True] + [False] * 8
    assert result.new_bull_ob_top.iloc[5] == pytest.approx(10.5)
    assert result.new_bull_ob_bottom.iloc[5] == pytest.approx(9.4)
    assert result.new_bull_ob_offset.iloc[5] == 1

    # Bar 8 walks back over the bearish bar 7 and takes the bullish bar 6
    # (13.4..20.4) as the supply zone; bar 11 takes the bullish bar 9 (11.3..12.4).
    assert result.bear_ob_created.tolist() == [False] * 8 + [True] + [False] * 2 + [True] + [False] * 2
    assert result.new_bear_ob_top.iloc[8] == pytest.approx(20.4)
    assert result.new_bear_ob_bottom.iloc[8] == pytest.approx(13.4)
    assert result.new_bear_ob_offset.iloc[8] == 2
    assert result.new_bear_ob_top.iloc[11] == pytest.approx(12.4)
    assert result.new_bear_ob_bottom.iloc[11] == pytest.approx(11.3)

    # The bullish zone (born bar 5) is only touched on bar 11, when the low of 5.3
    # finally trades back into 10.5 and the age of six bars clears the gate.
    assert result.bull_mitigated_now.tolist() == [False] * 11 + [True] + [False] * 2
    assert not result.bear_mitigated_now.any()

    # One bullish (mitigated) plus two live bearish zones.
    assert result.bull_active_zones.tolist() == [0] * 5 + [1] * 6 + [0] * 3
    assert result.bull_mitigated_zones.tolist() == [0] * 11 + [1] * 3
    assert result.bear_active_zones.tolist() == [0] * 8 + [1] * 3 + [2] * 3
    assert result.bear_mitigated_zones.tolist() == [0] * 14
    assert result.total_zones.tolist() == [0] * 5 + [1] * 3 + [2] * 3 + [3] * 3

    # Newest tracked zone per direction and its age in bars.
    assert result.latest_bull_ob_top.iloc[13] == pytest.approx(10.5)
    assert result.latest_bull_ob_bottom.iloc[13] == pytest.approx(9.4)
    assert result.latest_bull_ob_age.iloc[13] == 8
    assert result.latest_bear_ob_top.iloc[13] == pytest.approx(12.4)
    assert result.latest_bear_ob_bottom.iloc[13] == pytest.approx(11.3)
    assert result.latest_bear_ob_age.iloc[13] == 2


def test_capital41_order_block_v2_options():
    frame = _order_block_frame()

    # Body-only zones: bar 4's candle gives open 10.0 / close 9.9 for the bull OB.
    bodies = capital41_order_block_v2(frame, atr_length=2, impulse_mult=1.0, use_body=True)
    assert bodies.new_bull_ob_top.iloc[5] == pytest.approx(10.0)
    assert bodies.new_bull_ob_bottom.iloc[5] == pytest.approx(9.9)
    assert bodies.new_bear_ob_top.iloc[8] == pytest.approx(19.9)
    assert bodies.new_bear_ob_bottom.iloc[8] == pytest.approx(13.9)

    # A mitigated zone leaves the book immediately once removal is on.
    removed = capital41_order_block_v2(frame, atr_length=2, impulse_mult=1.0,
                                       remove_mitigated=True)
    assert removed.total_zones.tolist() == [0] * 5 + [1] * 3 + [2] * 3 + [2] * 3
    assert removed.bull_mitigated_zones.tolist() == [0] * 14

    # Eight non-overlapping bullish zones on a staircase: only the last five stay.
    stairs = _staircase_frame()
    limited = capital41_order_block_v2(stairs, atr_length=2, impulse_mult=1.0, max_active=5)
    assert limited.bull_impulse.tolist() == [False, True] * 8
    assert limited.bull_ob_created.tolist() == [False, True] * 8
    assert limited.total_zones.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5]
    assert limited.bull_active_zones.tolist() == limited.total_zones.tolist()
    assert limited.latest_bull_ob_top.iloc[15] == pytest.approx(81.5)
    assert limited.latest_bull_ob_bottom.iloc[15] == pytest.approx(79.5)
    assert limited.latest_bull_ob_age.iloc[15] == 0


def test_capital41_order_block_v2_rejects_bad_input():
    frame = _order_block_frame()
    with pytest.raises(ValueError, match="atr_length"):
        capital41_order_block_v2(frame, atr_length=0)
    with pytest.raises(ValueError, match="impulse_mult"):
        capital41_order_block_v2(frame, impulse_mult=5.5)
    with pytest.raises(ValueError, match="ob_lookback"):
        capital41_order_block_v2(frame, ob_lookback=31)
    with pytest.raises(ValueError, match="min_mitigation_age"):
        capital41_order_block_v2(frame, min_mitigation_age=21)
    with pytest.raises(ValueError, match="use_body"):
        capital41_order_block_v2(frame, use_body="yes")
    with pytest.raises(ValueError, match="high"):
        capital41_order_block_v2(frame.drop(columns="high"))
