"""Ports batch W3E: tsi_advanced, rsi_advanced, fisher_transform_advanced,
williams_pct_r_advanced, live_market_scanner_v7."""
import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.fisher_transform_advanced import fisher_transform_advanced
from app.catalog_indicators.live_market_scanner_v7 import live_market_scanner_v7
from app.catalog_indicators.rsi_advanced import rsi_advanced
from app.catalog_indicators.tsi_advanced import tsi_advanced
from app.catalog_indicators.williams_pct_r_advanced import williams_pct_r_advanced as williams_r_advanced


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
# tsi_advanced
# ---------------------------------------------------------------------------

_ZIGZAG = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]


def _zigzag_frame(bars=len(_ZIGZAG)):
    return _frame(bars, close=_ZIGZAG[:bars])


def test_tsi_advanced_hand_computed():
    # long=short=1 collapses both EMA chains onto the raw momentum, so
    # tsi = 100 * mom / |mom| = ±100, and a flat bar hits the upstream
    # zero-denominator branch returning 0.0. Signal is SMA(tsi, 3).
    result = tsi_advanced(_zigzag_frame(), long_len=1, short_len=1, signal_len=3,
                          smoothing_type="SMA", context_long_len=1, context_short_len=1,
                          extreme_lookback=2, divergence_len=4)
    expected_tsi = [float("nan"), 100.0, -100.0, 100.0, -100.0, 100.0, 0.0, 0.0, 100.0]
    expected_signal = [float("nan")] * 3 + [100.0 / 3.0, -100.0 / 3.0, 100.0 / 3.0,
                                            0.0, 100.0 / 3.0, 100.0 / 3.0]
    for bar in range(9):
        assert result.tsi.iloc[bar] == pytest.approx(expected_tsi[bar], nan_ok=True)
        assert result.tsi_signal.iloc[bar] == pytest.approx(expected_signal[bar], nan_ok=True)
        if not math.isnan(expected_tsi[bar]) and not math.isnan(expected_signal[bar]):
            assert result.tsi_histogram.iloc[bar] == pytest.approx(
                expected_tsi[bar] - expected_signal[bar])
        else:
            assert result.tsi_histogram.iloc[bar] != result.tsi_histogram.iloc[bar]

    # The zero bar returns 0.0 from the zero-denominator branch (not NaN).
    assert result.tsi.iloc[6] == 0.0

    # Context uses the same lengths, so trend_context == tsi.
    for bar in range(9):
        assert result.trend_context.iloc[bar] == pytest.approx(expected_tsi[bar], nan_ok=True)

    # Stretch zones over a 2-bar lookback.
    assert result.was_oversold.tolist() == [False, False, True, True, True, True,
                                            False, False, False]
    assert result.was_overbought.tolist() == [False, False, True, True, True, True,
                                              True, False, True]

    # Crosses: bull at 5 (tsi -100 -> 100 across the -33.3 signal) and 8;
    # bear at 4 and 7. Bar 3 has no prior defined signal, so no cross.
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, False]
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, False, False]
    # extreme gate: bar 8 crosses up but the last two TSIs (0, 100) never reached
    # the oversold zone, so only bar 5 is an extreme.
    assert result.bull_zero_cross.tolist() == [False, False, False, True, False, True,
                                               False, False, True]
    assert result.bear_zero_cross.tolist() == [False, False, True, False, True, False,
                                               False, False, False]

    # ATR(14) never warms up on 9 bars, so no stall can be detected and the
    # extremes keep their full quality score.
    assert not result.stall_bull.any()
    assert not result.stall_bear.any()
    assert not result.stalled.any()
    assert result.bull_extreme_strong.tolist() == result.bull_extreme.tolist()
    assert result.bear_extreme_strong.tolist() == result.bear_extreme.tolist()
    assert not result.bull_extreme_weak.any()
    assert not result.bear_extreme_weak.any()

    assert result.context_bullish.tolist() == [False, True, False, True, False, True,
                                               False, False, True]
    assert result.context_bearish.tolist() == [False, False, True, False, True, False,
                                               False, False, False]

    # Sentiment: tsi sits at ±100 inside a ±25 zone -> saturated ±100; 0 -> 0.
    expected_sentiment = [float("nan"), 100.0, -100.0, 100.0, -100.0, 100.0, 0.0, 0.0, 100.0]
    for bar in range(9):
        assert result.sentiment_score.iloc[bar] == pytest.approx(expected_sentiment[bar],
                                                                 nan_ok=True)

    # Quality = 50 * zone depth + 25 * context agreement + 25 * stall-free.
    # depth at bar 2..6: lowest(tsi,2) = -100 -> (−25−(−100))/25 = 3 -> capped 1.
    expected_bull_quality = [float("nan"), float("nan"), 75.0, 100.0, 75.0, 100.0,
                             25.0, 25.0, 50.0]
    expected_bear_quality = [float("nan"), float("nan"), 100.0, 75.0, 100.0, 75.0,
                             75.0, 25.0, 75.0]
    for bar in range(9):
        assert result.bull_quality.iloc[bar] == pytest.approx(expected_bull_quality[bar],
                                                              nan_ok=True)
        assert result.bear_quality.iloc[bar] == pytest.approx(expected_bear_quality[bar],
                                                              nan_ok=True)

    # Context == tsi here, so fast and slow slopes share a sign and no
    # divergence can fire.
    assert not result.trend_divergence_bull.any()
    assert not result.trend_divergence_bear.any()


def test_tsi_advanced_extreme_gate_disabled():
    # Without the stretch-zone requirement every cross is an extreme signal.
    result = tsi_advanced(_zigzag_frame(), long_len=1, short_len=1, signal_len=3,
                          smoothing_type="SMA", context_long_len=1, context_short_len=1,
                          extreme_only=False, divergence_len=4)
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, True]
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, True, False]


def test_tsi_advanced_stall_detection_hand_computed():
    # TSI moves while the price holds: closes alternate by ±0.02 (so the TSI
    # swings ±100 and its 5-bar change is ±200) while the 5-bar price change
    # stays within ±0.02 of zero. True range is a constant 2.02 (high=close+1,
    # low=close-1), so ATR(14) is 2.02 from bar 13 on and |priceChg|/ATR ~ 0.01
    # stays below the 0.5 stall gate.
    bars = 16
    closes = [10.0 + 0.02 * (bar % 2) for bar in range(bars)]
    frame = _frame(bars, close=closes)
    result = tsi_advanced(frame, long_len=1, short_len=1, signal_len=3, smoothing_type="SMA",
                          stall_len=5, stall_flat_atr=0.5, stall_tsi_min=0.5,
                          context_long_len=1, context_short_len=1)
    assert not result.stalled.iloc[:13].any()
    # Bar 13 (odd) reads tsi +100 vs tsi[8] -100 -> bearish absorption;
    # bar 14 (even) reads tsi -100 vs tsi[9] +100 -> bullish absorption.
    assert not result.stall_bull.iloc[13] and result.stall_bear.iloc[13]
    assert result.stall_bull.iloc[14] and not result.stall_bear.iloc[14]
    assert result.stalled.iloc[13] and result.stalled.iloc[14]


def test_tsi_advanced_rejects_bad_input():
    frame = _zigzag_frame()
    with pytest.raises(ValueError, match="long_len"):
        tsi_advanced(frame, long_len=0)
    with pytest.raises(ValueError, match="short_len"):
        tsi_advanced(frame, short_len=-1)
    with pytest.raises(ValueError, match="signal_len"):
        tsi_advanced(frame, signal_len=0)
    with pytest.raises(ValueError, match="smoothing_type"):
        tsi_advanced(frame, smoothing_type="FRAMA")
    with pytest.raises(ValueError, match="jma_power"):
        tsi_advanced(frame, jma_power=5.0)
    with pytest.raises(ValueError, match="jma_phase"):
        tsi_advanced(frame, jma_phase=101)
    with pytest.raises(ValueError, match="overbought"):
        tsi_advanced(frame, overbought=-1.0)
    with pytest.raises(ValueError, match="oversold"):
        tsi_advanced(frame, oversold=1.0)
    with pytest.raises(ValueError, match="stall_len"):
        tsi_advanced(frame, stall_len=0)
    with pytest.raises(ValueError, match="stall_tsi_min"):
        tsi_advanced(frame, stall_tsi_min=-1.0)
    with pytest.raises(ValueError, match="context_long_len"):
        tsi_advanced(frame, context_long_len=0)
    with pytest.raises(ValueError, match="extreme_lookback"):
        tsi_advanced(frame, extreme_lookback=0)
    with pytest.raises(ValueError, match="divergence_len"):
        tsi_advanced(frame, divergence_len=0)
    with pytest.raises(ValueError, match="divergence_min_slope"):
        tsi_advanced(frame, divergence_min_slope=-0.5)
    with pytest.raises(ValueError, match="detect_stall"):
        tsi_advanced(frame, detect_stall="yes")
    with pytest.raises(ValueError, match="close"):
        tsi_advanced(frame.drop(columns="close"))


# ---------------------------------------------------------------------------
# rsi_advanced
# ---------------------------------------------------------------------------

def _rma_ref(values, length):
    """Pine's ta.rma: skip leading NA, seed with the mean of `length` values."""
    out = [float("nan")] * len(values)
    count, total, current = 0, 0.0, float("nan")
    for position, value in enumerate(values):
        if not math.isnan(value):
            if count < length:
                count += 1
                total += value
                if count == length:
                    current = total / length
            else:
                current = (current * (length - 1) + value) / length
        out[position] = current
    return out


def _rsi_ref(closes, length):
    """Pine's ta.rsi from the closes, replayed bar by bar."""
    deltas = [float("nan")] + [closes[bar] - closes[bar - 1] for bar in range(1, len(closes))]
    gains = _rma_ref([max(value, 0.0) if not math.isnan(value) else value
                      for value in deltas], length)
    losses = _rma_ref([max(-value, 0.0) if not math.isnan(value) else value
                       for value in deltas], length)
    out = []
    for gain, loss in zip(gains, losses):
        if math.isnan(gain) or math.isnan(loss):
            out.append(float("nan"))
        elif loss == 0.0 and gain > 0.0:
            out.append(100.0)
        elif loss == 0.0:
            out.append(float("nan"))
        else:
            out.append(100.0 - 100.0 / (1.0 + gain / loss))
    return out


def test_rsi_advanced_hand_computed():
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = rsi_advanced(frame, rsi_len=2, avg_len=1, signal_len=2, smoothing_type="SMA",
                          mid_line=50.0, overbought=75.0, oversold=40.0, context_len=3,
                          extreme_lookback=2, divergence_len=2)

    # RSI(2) by hand: the Wilder gains/losses of the delta sequence
    # [_, +1, -1, +1, -1, +2, 0, 0, +1]. The Wilder seed needs two defined
    # deltas, so RSI starts on bar 2 (50) and reads 75 / 37.5 afterwards.
    expected_rsi = _rsi_ref(closes, 2)
    assert math.isnan(expected_rsi[0]) and math.isnan(expected_rsi[1])
    assert expected_rsi[2] == 50.0 and expected_rsi[3] == 75.0 and expected_rsi[4] == 37.5
    for bar in range(len(closes)):
        assert result.rsi.iloc[bar] == pytest.approx(expected_rsi[bar], nan_ok=True)
    # avg_len=1 with SMA leaves the RSI untouched, so the signal is SMA(rsi, 2).
    assert result.rsi_signal.iloc[3] == pytest.approx((50.0 + 75.0) / 2.0)
    assert result.rsi_signal.iloc[4] == pytest.approx((75.0 + 37.5) / 2.0)
    assert result.rsi_signal.iloc[5] == pytest.approx((37.5 + 100.0 - 100.0 / 4.8) / 2.0)

    # Trend context is RSI(3) with the same one-bar smoothing.
    expected_ctx = _rsi_ref(closes, 3)
    for bar in range(len(closes)):
        assert result.trend_context.iloc[bar] == pytest.approx(expected_ctx[bar], nan_ok=True)

    # Stretch zones over two bars: RSI dips to 37.5 (below 40) on bars 4-5 and
    # the raw RSI(2) touches 75 from bar 3 on.
    assert result.was_oversold.tolist() == [False, False, False, False, True, True,
                                            False, False, False]
    assert result.was_overbought.tolist() == [False, False, False, True, True, True,
                                              True, True, True]

    # Crosses of RSI against its SMA(2) signal, gated on the stretch zone:
    # bull crosses at 5 and 8 (bar 3 has no prior defined signal), only bar 5
    # had RSI <= 40 recently; bear at 4.
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, False]
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, False, False]
    assert result.bull_mid_cross.tolist() == [False, False, False, True, False, True,
                                              False, False, False]
    assert result.bear_mid_cross.tolist() == [False, False, False, False, True, False,
                                              False, False, False]

    # ATR(14) is undefined on 9 bars, so stalls never fire and the extremes
    # stay strong.
    assert not result.stalled.any()
    assert result.bull_extreme_strong.tolist() == result.bull_extreme.tolist()
    assert result.bear_extreme_strong.tolist() == result.bear_extreme.tolist()

    # RSI(3) context: bullish above 50 from bar 3 on except bar 4 (44.4).
    assert result.context_bullish.tolist() == [False, False, False, True, False, True,
                                               True, True, True]
    assert result.context_bearish.tolist() == [False, False, False, False, True, False,
                                               False, False, False]

    # Sentiment inside [40, 50, 75]: saturated on every defined bar except 50.
    expected_sentiment = [float("nan"), float("nan"), 0.0, 100.0, -100.0, 100.0,
                          100.0, 100.0, 100.0]
    for bar in range(len(closes)):
        assert result.sentiment_score.iloc[bar] == pytest.approx(expected_sentiment[bar],
                                                                 nan_ok=True)

    # Quality = 50 * zone depth + 25 * context agreement + 25 * stall-free.
    expected_bull_quality = [float("nan"), float("nan"), float("nan"), 50.0, 37.5, 62.5,
                             50.0, 50.0, 50.0]
    expected_bear_quality = [float("nan"), float("nan"), float("nan"), 25.0, 50.0,
                             100.0 / 3.0, 100.0 / 3.0, 100.0 / 3.0,
                             (94.3181818 - 75.0) / 25.0 * 50.0 + 25.0]
    for bar in range(len(closes)):
        assert result.bull_quality.iloc[bar] == pytest.approx(expected_bull_quality[bar],
                                                              nan_ok=True)
        assert result.bear_quality.iloc[bar] == pytest.approx(expected_bear_quality[bar],
                                                              nan_ok=True)

    # Fast and slow slopes never point opposite ways on this series.
    assert not result.trend_divergence_bull.any()
    assert not result.trend_divergence_bear.any()


def test_rsi_advanced_context_switch_off():
    # Pine grades quality against the context line even when `useCtx` is off
    # (the gate only affects the counter-trend demotion and the divergence),
    # so the 25-point agreement bonus stays.
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = rsi_advanced(frame, rsi_len=2, avg_len=1, signal_len=2, smoothing_type="SMA",
                          mid_line=50.0, overbought=75.0, oversold=40.0, context_len=3,
                          extreme_lookback=2, show_context=False)
    assert result.bull_quality.iloc[5] == pytest.approx(12.5 + 25.0 + 25.0)
    assert not result.trend_divergence_bull.any()
    assert not result.trend_divergence_bear.any()
    assert result.bull_extreme_strong.tolist() == result.bull_extreme.tolist()


def test_rsi_advanced_rejects_bad_input():
    frame = _frame(9, close=[10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="rsi_len"):
        rsi_advanced(frame, rsi_len=0)
    with pytest.raises(ValueError, match="avg_len"):
        rsi_advanced(frame, avg_len=0)
    with pytest.raises(ValueError, match="signal_len"):
        rsi_advanced(frame, signal_len=0)
    with pytest.raises(ValueError, match="smoothing_type"):
        rsi_advanced(frame, smoothing_type="LSMA")
    with pytest.raises(ValueError, match="mid_line"):
        rsi_advanced(frame, mid_line=101.0)
    with pytest.raises(ValueError, match="overbought"):
        rsi_advanced(frame, overbought=49.0)
    with pytest.raises(ValueError, match="oversold"):
        rsi_advanced(frame, oversold=51.0)
    with pytest.raises(ValueError, match="stall_rsi_min"):
        rsi_advanced(frame, stall_rsi_min=-1.0)
    with pytest.raises(ValueError, match="context_len"):
        rsi_advanced(frame, context_len=1)
    with pytest.raises(ValueError, match="extreme_lookback"):
        rsi_advanced(frame, extreme_lookback=0)
    with pytest.raises(ValueError, match="divergence_min_slope"):
        rsi_advanced(frame, divergence_min_slope=-1.0)
    with pytest.raises(ValueError, match="show_divergence"):
        rsi_advanced(frame, show_divergence=1)
    with pytest.raises(ValueError, match="high"):
        rsi_advanced(frame.drop(columns="high"))


# ---------------------------------------------------------------------------
# fisher_transform_advanced
# ---------------------------------------------------------------------------

def _fisher_ref(source, length):
    """Pine's f_fisher replayed bar by bar from the source values."""
    out = [float("nan")] * len(source)
    value_previous = fish_previous = float("nan")
    for bar in range(len(source)):
        if bar < length - 1:
            continue
        window = source[bar - length + 1:bar + 1]
        span = max(window) - min(window)
        normalized = (source[bar] - min(window)) / span - 0.5 if span != 0.0 else 0.0
        if math.isnan(value_previous):
            value = 0.66 * normalized
        else:
            value = 0.66 * normalized + 0.67 * value_previous
        value = min(max(value, -0.999), 0.999)
        fish = (0.5 * math.log((1.0 + value) / (1.0 - value))
                + 0.5 * (0.0 if math.isnan(fish_previous) else fish_previous))
        out[bar] = fish
        value_previous, fish_previous = value, fish
    return out


def test_fisher_transform_advanced_hand_computed():
    # high = close + 1 and low = close - 1 put hl2 exactly on the close, so the
    # fisher recursion can be replayed on the close sequence alone.
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = fisher_transform_advanced(frame, fisher_len=3, avg_len=1, signal_len=2,
                                       smoothing_type="SMA", mid_line=0.0, overbought=0.8,
                                       oversold=-0.25, context_len=3, extreme_lookback=2,
                                       divergence_len=2)

    expected_fisher = _fisher_ref(closes, 3)
    assert expected_fisher[2] == pytest.approx(-0.3428283)
    for bar in range(len(closes)):
        assert result.fisher.iloc[bar] == pytest.approx(expected_fisher[bar], nan_ok=True)
        assert result.trend_context.iloc[bar] == pytest.approx(expected_fisher[bar],
                                                               nan_ok=True)
    assert result.fisher.isna().tolist() == [True, True] + [False] * 7

    # Signal is SMA(fisher, 2).
    assert result.fisher_signal.iloc[3] == pytest.approx(
        (expected_fisher[2] + expected_fisher[3]) / 2.0)
    assert result.fisher_signal.iloc[8] == pytest.approx(
        (expected_fisher[7] + expected_fisher[8]) / 2.0)

    # Stretch zones with [-0.25, 0.8]: the fisher dips below -0.25 on bars 2-4
    # (2-bar windows), and only bar 8's window reaches above 0.8.
    assert result.was_oversold.tolist() == [False, False, False, True, True, True,
                                            False, False, False]
    assert result.was_overbought.tolist() == [False] * 8 + [True]

    # One bull cross (bar 5, out of the trough) and one bear cross (bar 4);
    # only the bull cross lands inside the stretch zone.
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, False]
    assert result.bear_extreme.tolist() == [False] * 9
    # Zero-line cross: the fisher leaves the negative half on bar 5 and never
    # falls back.
    assert result.bull_zero_cross.tolist() == [False, False, False, False, False, True,
                                               False, False, False]
    assert not result.bear_zero_cross.any()

    # No ATR warm-up on 9 bars -> no stall; context == fisher here.
    assert not result.stalled.any()
    assert result.context_bullish.tolist() == [False, False, False, False, False, True,
                                               True, True, True]
    assert result.context_bearish.tolist() == [False, False, True, True, True, False,
                                               False, False, False]
    assert result.bull_extreme_strong.tolist() == result.bull_extreme.tolist()

    # Sentiment inside [-0.25, 0, 0.8]: saturated where the fisher sits beyond.
    expected_sentiment = [float("nan")] * 9
    expected_sentiment[2] = -100.0  # bear pct = 0.3428 / 0.25 -> clipped
    expected_sentiment[3] = -(0.0 - expected_fisher[3]) * 100.0 / 0.25
    expected_sentiment[4] = -100.0  # bear pct = 0.2938 / 0.25 -> clipped
    expected_sentiment[5] = expected_fisher[5] * 100.0 / 0.8
    expected_sentiment[6] = expected_fisher[6] * 100.0 / 0.8
    expected_sentiment[7] = expected_fisher[7] * 100.0 / 0.8
    expected_sentiment[8] = 100.0  # 0.8526 / 0.8 -> clipped
    for bar in range(len(closes)):
        assert result.sentiment_score.iloc[bar] == pytest.approx(expected_sentiment[bar],
                                                                 nan_ok=True)

    # Quality = 50 * zone depth + 25 * context agreement + 25 * stall-free.
    assert result.bull_quality.iloc[3] == pytest.approx(((-0.25) - expected_fisher[2])
                                                        / 0.25 * 50.0 + 25.0)
    assert result.bull_quality.iloc[5] == pytest.approx(((-0.25) - expected_fisher[4])
                                                        / 0.25 * 50.0 + 25.0 + 25.0)
    assert result.bear_quality.iloc[8] == pytest.approx((expected_fisher[8] - 0.8) / 0.8
                                                        * 50.0 + 25.0)
    assert result.bear_quality.iloc[4] == pytest.approx(50.0)

    # Context equals the fisher, so fast and slow slopes share a sign.
    assert not result.trend_divergence_bull.any()
    assert not result.trend_divergence_bear.any()


def test_fisher_transform_advanced_flat_source_zero_branch():
    # A constant hl2 makes the range zero, and the upstream ternary returns a
    # normalized value of 0.0 (fish 0.0), not NaN, once the window is full.
    bars = 6
    frame = _frame(bars, close=np.full(bars, 10.0))
    result = fisher_transform_advanced(frame, fisher_len=3, avg_len=1, signal_len=2,
                                       smoothing_type="SMA", context_len=3)
    assert result.fisher.iloc[:2].isna().all()
    assert result.fisher.iloc[2:].eq(0.0).all()
    assert result.trend_context.iloc[2:].eq(0.0).all()


def test_fisher_transform_advanced_extreme_gate_disabled():
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = fisher_transform_advanced(frame, fisher_len=3, avg_len=1, signal_len=2,
                                       smoothing_type="SMA", overbought=0.9, oversold=-0.3,
                                       context_len=3, extreme_only=False)
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, False]
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, False, False]


def test_fisher_transform_advanced_rejects_bad_input():
    frame = _frame(9, close=[10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="fisher_len"):
        fisher_transform_advanced(frame, fisher_len=1)
    with pytest.raises(ValueError, match="avg_len"):
        fisher_transform_advanced(frame, avg_len=0)
    with pytest.raises(ValueError, match="smoothing_type"):
        fisher_transform_advanced(frame, smoothing_type="ZLEMA")
    with pytest.raises(ValueError, match="overbought"):
        fisher_transform_advanced(frame, overbought=-0.5)
    with pytest.raises(ValueError, match="oversold"):
        fisher_transform_advanced(frame, oversold=0.5)
    with pytest.raises(ValueError, match="stall_fisher_min"):
        fisher_transform_advanced(frame, stall_fisher_min=-1.0)
    with pytest.raises(ValueError, match="context_len"):
        fisher_transform_advanced(frame, context_len=1)
    with pytest.raises(ValueError, match="divergence_min_slope"):
        fisher_transform_advanced(frame, divergence_min_slope=-0.1)
    with pytest.raises(ValueError, match="jma_power"):
        fisher_transform_advanced(frame, jma_power=0.5)
    with pytest.raises(ValueError, match="low"):
        fisher_transform_advanced(frame.drop(columns="low"))


# ---------------------------------------------------------------------------
# williams_r_advanced
# ---------------------------------------------------------------------------

def test_williams_r_advanced_hand_computed():
    # high = close + 1 / low = close - 1, so the normalized position reads
    # 100 * (close - lowest(close,3)) / (highest(close,3) - lowest(close,3) + 2).
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = williams_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
                                 smoothing_type="SMA", mid_line=50.0, overbought=75.0,
                                 oversold=35.0, context_len=3, extreme_lookback=2,
                                 divergence_len=2)

    # Hand-computed raw positions; bar 7's window is flat at 12, so the price
    # range collapses to the 2-point candle body (high=13, low=11).
    expected_raw = [float("nan"), float("nan"), 100.0 / 3.0, 200.0 / 3.0, 100.0 / 3.0,
                    75.0, 75.0, 50.0, 200.0 / 3.0]
    for bar in range(len(closes)):
        assert result.williams_r.iloc[bar] == pytest.approx(expected_raw[bar], nan_ok=True)
        # context_len == wpr_len, so the context line is the same series.
        assert result.trend_context.iloc[bar] == pytest.approx(expected_raw[bar], nan_ok=True)

    # Signal is SMA(wpr, 2): 50, 50, 54.17, 75, 62.5, 58.33.
    assert result.williams_r_signal.iloc[3] == pytest.approx(50.0)
    assert result.williams_r_signal.iloc[4] == pytest.approx(50.0)
    assert result.williams_r_signal.iloc[5] == pytest.approx((100.0 / 3.0 + 75.0) / 2.0)
    assert result.williams_r_signal.iloc[7] == pytest.approx((75.0 + 50.0) / 2.0)
    assert result.williams_r_signal.iloc[8] == pytest.approx((50.0 + 200.0 / 3.0) / 2.0)

    # Stretch zones with [35, 75]: the position dips to 33.3 on bars 3-5 and
    # rides at 75 on bars 5-7 (bar 8's window no longer reaches 75).
    assert result.was_oversold.tolist() == [False, False, False, True, True, True,
                                            False, False, False]
    assert result.was_overbought.tolist() == [False, False, False, False, False, True,
                                              True, True, False]

    # Bull crosses at 5 and 8, only bar 5 sits in the stretch zone; bear
    # crosses at 4 and 7, only bar 7 sits in the overbought zone.
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, False]
    assert result.bear_extreme.tolist() == [False, False, False, False, False, False,
                                            False, True, False]
    # Bar 7's context (50) is neither above nor below the midline, so the
    # extreme stays strong; no counter-trend demotion fires.
    assert result.bear_extreme_strong.tolist() == [False] * 7 + [True, False]
    assert not result.bear_extreme_weak.any()
    assert result.bull_extreme_strong.tolist() == result.bull_extreme.tolist()
    assert not result.bull_extreme_weak.any()

    assert result.bull_mid_cross.tolist() == [False, False, False, True, False, True,
                                              False, False, True]
    assert result.bear_mid_cross.tolist() == [False, False, False, False, True, False,
                                              False, False, False]

    # No ATR warm-up on 9 bars -> no stalls.
    assert not result.stalled.any()

    assert result.context_bullish.tolist() == [False, False, False, True, False, True,
                                               True, False, True]
    assert result.context_bearish.tolist() == [False, False, True, False, True, False,
                                               False, False, False]

    # Sentiment inside [35, 50, 75]: the 33.3 reading saturates the bear side,
    # the 50 reading is exactly neutral.
    expected_sentiment = [float("nan"), float("nan"), -100.0, 200.0 / 3.0, -100.0,
                          100.0, 100.0, 0.0, 200.0 / 3.0]
    for bar in range(len(closes)):
        assert result.sentiment_score.iloc[bar] == pytest.approx(expected_sentiment[bar],
                                                                 nan_ok=True)

    # Quality uses the williams-specific depth denominators: oversold for the
    # bull side, 100 - overbought for the bear side.
    expected_bull_quality = [float("nan"), float("nan"), float("nan"),
                             (35.0 - 100.0 / 3.0) / 35.0 * 50.0 + 25.0 + 25.0,
                             (35.0 - 100.0 / 3.0) / 35.0 * 50.0 + 25.0,
                             (35.0 - 100.0 / 3.0) / 35.0 * 50.0 + 25.0 + 25.0,
                             50.0, 25.0, 50.0]
    expected_bear_quality = [float("nan"), float("nan"), float("nan"), 25.0, 50.0, 25.0,
                             25.0, 25.0, 25.0]
    for bar in range(len(closes)):
        assert result.bull_quality.iloc[bar] == pytest.approx(expected_bull_quality[bar],
                                                              nan_ok=True)
        assert result.bear_quality.iloc[bar] == pytest.approx(expected_bear_quality[bar],
                                                              nan_ok=True)

    # Fast and slow slopes share a sign everywhere on this series.
    assert not result.trend_divergence_bull.any()
    assert not result.trend_divergence_bear.any()


def test_williams_r_advanced_extreme_gate_disabled():
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = williams_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
                                 smoothing_type="SMA", overbought=75.0, oversold=35.0,
                                 context_len=3, extreme_only=False)
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, True]
    # Without the stretch gate the bar-4 and bar-7 bear crosses count too.
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, True, False]


def test_williams_r_advanced_flat_range_fifty_branch():
    # Constant high/low makes the price range zero and the upstream ternary
    # returns 50.0 (not NaN) once the window is full.
    bars = 6
    frame = _frame(bars, close=np.full(bars, 10.0))
    result = williams_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
                                 smoothing_type="SMA", context_len=3)
    assert result.williams_r.iloc[:2].isna().all()
    assert result.williams_r.iloc[2:].eq(50.0).all()
    assert result.trend_context.iloc[2:].eq(50.0).all()


def test_williams_r_advanced_rejects_bad_input():
    frame = _frame(9, close=[10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="wpr_len"):
        williams_r_advanced(frame, wpr_len=0)
    with pytest.raises(ValueError, match="avg_len"):
        williams_r_advanced(frame, avg_len=0)
    with pytest.raises(ValueError, match="smoothing_type"):
        williams_r_advanced(frame, smoothing_type="VWMA")
    with pytest.raises(ValueError, match="mid_line"):
        williams_r_advanced(frame, mid_line=-1.0)
    with pytest.raises(ValueError, match="overbought"):
        williams_r_advanced(frame, overbought=101.0)
    with pytest.raises(ValueError, match="oversold"):
        williams_r_advanced(frame, oversold=-1.0)
    with pytest.raises(ValueError, match="stall_wpr_min"):
        williams_r_advanced(frame, stall_wpr_min=-5.0)
    with pytest.raises(ValueError, match="context_len"):
        williams_r_advanced(frame, context_len=1)
    with pytest.raises(ValueError, match="extreme_only"):
        williams_r_advanced(frame, extreme_only="no")
    with pytest.raises(ValueError, match="divergence_len"):
        williams_r_advanced(frame, divergence_len=0)
    with pytest.raises(ValueError, match="high"):
        williams_r_advanced(frame.drop(columns="high"))


# ---------------------------------------------------------------------------
# live_market_scanner_v7
# ---------------------------------------------------------------------------

def _scanner_frame(bars=60):
    # Gentle +1 uptrend, a -2.5 flush, then a +2 recovery; every bar opens at
    # the previous close. Volume is flat 100 except a spike on the last bar.
    closes = ([100.0 + bar for bar in range(30)]
              + [129.0 - 2.5 * (bar - 29) for bar in range(30, 35)]
              + [119.0 + 2.0 * (bar - 34) for bar in range(35, bars)])
    closes = np.asarray(closes[:bars], dtype=float)
    closes[-1] = closes[-2] + 4.0  # accelerating final candle: MACD hist expands
    opens = np.concatenate([[closes[0] - 1.0], closes[:-1]])
    volume = np.full(bars, 100.0)
    volume[-1] = 1000.0
    return _frame(bars, open_=opens, high=np.maximum(opens, closes) + 0.5,
                  low=np.minimum(opens, closes) - 0.5, close=closes, volume=volume)


def _scanner_scores(frame, *, min_confidence):
    """Second, loop-based composition of the Pine score rules."""
    from app.catalog_indicators._pine import atr, crossover, crossunder, ema, linreg, rsi, sma
    open_, close, volume = (frame["open"], frame["close"], frame["volume"])
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    rsi_line = rsi(close, 14)
    ema9, ema21, ema50 = ema(close, 9), ema(close, 21), ema(close, 50)
    volume_average = sma(volume, 20)
    average_range = atr(frame, 14)

    bars = len(frame)
    longs, shorts, statuses = [], [], []
    for bar in range(bars):
        def flag(value):
            return bool(value) if value is not None else False

        def at(series, index):
            value = series.iloc[index]
            return None if pd.isna(value) else float(value)

        def cmp_flag(left, right):
            return left is not None and right is not None and left > right

        macd_now, sig_now, hist_now = at(macd_line, bar), at(signal_line, bar), at(hist, bar)
        macd_prev_hist = at(hist, bar - 1) if bar else None
        macd_bull = cmp_flag(macd_now, sig_now) and hist_now is not None and \
            macd_prev_hist is not None and hist_now > macd_prev_hist
        macd_bear = sig_now is not None and macd_now is not None and macd_now < sig_now and \
            hist_now is not None and macd_prev_hist is not None and hist_now < macd_prev_hist
        cross_up = flag(crossover(macd_line, signal_line).iloc[bar])
        cross_down = flag(crossunder(macd_line, signal_line).iloc[bar])

        rsi_now = at(rsi_line, bar)
        rsi_1, rsi_2 = at(rsi_line, bar - 1) if bar else None, \
            at(rsi_line, bar - 2) if bar > 1 else None
        rising = None not in (rsi_now, rsi_1, rsi_2) and rsi_now > rsi_1 and rsi_now > rsi_2
        falling = None not in (rsi_now, rsi_1, rsi_2) and rsi_now < rsi_1 and rsi_now < rsi_2

        e9, e21 = at(ema9, bar), at(ema21, bar)
        bull_aligned = cmp_flag(e9, e21)
        bear_aligned = e9 is not None and e21 is not None and e9 < e21
        ema_up = flag(crossover(ema9, ema21).iloc[bar])
        ema_down = flag(crossunder(ema9, ema21).iloc[bar])

        vol_now, vol_avg = at(volume, bar), at(volume_average, bar)
        vol_prev = at(volume, bar - 1) if bar else None
        spike = vol_avg is not None and vol_now > vol_avg * 1.3
        vol_rising = vol_prev is not None and vol_now > vol_prev

        price_up = (close.iloc[bar] > open_.iloc[bar]
                    and bar > 0 and close.iloc[bar] > close.iloc[bar - 1])
        price_down = (close.iloc[bar] < open_.iloc[bar]
                      and bar > 0 and close.iloc[bar] < close.iloc[bar - 1])
        atr_now = at(average_range, bar)
        strong_bull = atr_now is not None and (close.iloc[bar] - open_.iloc[bar]) > atr_now * 0.3
        strong_bear = atr_now is not None and (open_.iloc[bar] - close.iloc[bar]) > atr_now * 0.3

        trend_now = at(ema50, bar)
        bull_trend = trend_now is not None and close.iloc[bar] > trend_now
        bear_trend = trend_now is not None and close.iloc[bar] < trend_now

        long_score = 0
        long_score += 30 if cross_up else 20 if macd_bull else 0
        long_score += 25 if rsi_now is not None and rsi_now < 35 and rising else \
            15 if rising else 0
        long_score += 25 if ema_up else 15 if bull_aligned else 0
        long_score += 15 if spike and price_up else 10 if vol_rising and price_up else 0
        long_score += 10 if strong_bull else 5 if price_up else 0
        long_score += 5 if bull_trend else 0

        short_score = 0
        short_score += 30 if cross_down else 20 if macd_bear else 0
        short_score += 25 if rsi_now is not None and rsi_now > 65 and falling else \
            15 if falling else 0
        short_score += 25 if ema_down else 15 if bear_aligned else 0
        short_score += 15 if spike and price_down else 10 if vol_rising and price_down else 0
        short_score += 10 if strong_bear else 5 if price_down else 0
        short_score += 5 if bear_trend else 0

        longs.append(long_score)
        shorts.append(short_score)
        if long_score >= min_confidence:
            statuses.append("LONG")
        elif short_score >= min_confidence:
            statuses.append("SHORT")
        elif long_score > short_score:
            statuses.append("WAIT (Bullish)")
        elif short_score > long_score:
            statuses.append("WAIT (Bearish)")
        else:
            statuses.append("NEUTRAL")
    return longs, shorts, statuses


def test_live_market_scanner_v7_scores_hand_computed():
    frame = _scanner_frame()
    result = live_market_scanner_v7(frame)
    longs, shorts, statuses = _scanner_scores(frame, min_confidence=65)

    assert result.long_score.tolist() == longs
    assert result.short_score.tolist() == shorts
    assert result.market_status.tolist() == statuses

    # Warm-up gaps: MACD(12/26) needs 26 bars, its signal chains onto that,
    # ATR(14) starts at 13, SMA(volume, 20) at 19, linreg at 19, EMA50 at 49.
    assert result.macd_line.iloc[:25].isna().all() and not result.macd_line.iloc[25:].isna().any()
    assert result.macd_signal.iloc[:25].isna().all()
    assert result.atr.iloc[:13].isna().all() and not result.atr.iloc[13:].isna().any()
    assert result.volume_average.iloc[:19].isna().all()
    assert result.linreg.iloc[:19].isna().all()
    assert result.linreg_slope.iloc[:20].isna().all()
    assert result.ema50.iloc[:49].isna().all() and not result.ema50.iloc[49:].isna().any()
    # The volume spike on the last bar (1000 vs ~100 average) is a real flag,
    # and the score columns stay whole numbers.
    assert result.volume_spike.iloc[-1]
    assert not result.volume_spike.iloc[:-1].any()
    assert result.long_score.dtype == pd.Int64Dtype()

    # Last bar: +2 recovery candle with a volume spike. Hand-checked rules:
    # strong bull candle (+10), volume spike with price up (+15), rising RSI
    # (+15), bullish EMA alignment (+15), MACD momentum (+20), EMA50 trend (+5).
    assert result.strong_bull_candle.iloc[-1]
    assert result.price_up.iloc[-1]
    assert result.rsi_rising.iloc[-1]
    assert result.ema_bullish.iloc[-1]
    assert result.macd_bullish.iloc[-1]
    assert result.bull_trend.iloc[-1]
    assert result.long_score.iloc[-1] == 20 + 15 + 15 + 15 + 10 + 5
    assert result.valid_long.iloc[-1]
    assert result.market_status.iloc[-1] == "LONG"

    # Mid-flush bar: price falls but nothing extreme enough for a short.
    flush = 32
    assert result.price_down.iloc[flush]
    assert result.market_status.iloc[flush] != "SHORT"


def test_live_market_scanner_v7_sensitivity_thresholds():
    frame = _scanner_frame()
    # High sensitivity lowers the bar to 55, so the signal set can only grow;
    # Low raises it to 75 and can only shrink. NEUTRAL/WAIT mix shifts too.
    high = live_market_scanner_v7(frame, sensitivity="High")
    low = live_market_scanner_v7(frame, sensitivity="Low")
    medium = live_market_scanner_v7(frame, sensitivity="Medium")
    assert (high.valid_long | ~medium.valid_long).all()
    assert (medium.valid_long | ~low.valid_long).all()
    assert (high.long_score == medium.long_score).all()
    assert (low.long_score == medium.long_score).all()
    assert (medium.market_status == "LONG").sum() >= (low.market_status == "LONG").sum()


def test_live_market_scanner_v7_rejects_bad_input():
    frame = _scanner_frame()
    with pytest.raises(ValueError, match="sensitivity"):
        live_market_scanner_v7(frame, sensitivity="Extreme")
    with pytest.raises(ValueError, match="volume"):
        live_market_scanner_v7(frame.drop(columns="volume"))
    with pytest.raises(ValueError, match="open"):
        live_market_scanner_v7(frame.drop(columns="open"))
    with pytest.raises(ValueError, match="high"):
        live_market_scanner_v7(frame.drop(columns="high"))
