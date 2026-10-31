"""Ports batch W4A: williams_pct_r_advanced, live_market_scanner_v7."""
import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.live_market_scanner_v7 import live_market_scanner_v7
from app.catalog_indicators.williams_pct_r_advanced import williams_pct_r_advanced


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


def test_williams_pct_r_advanced_hand_computed():
    # high = close + 1 / low = close - 1, so the normalized position reads
    # 100 * (close - lowest(close,3)) / (highest(close,3) - lowest(close,3) + 2).
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = williams_pct_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
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

    # Signal is SMA(wpr, 2).
    assert result.williams_r_signal.iloc[3] == pytest.approx(50.0)
    assert result.williams_r_signal.iloc[4] == pytest.approx(50.0)
    assert result.williams_r_signal.iloc[5] == pytest.approx((100.0 / 3.0 + 75.0) / 2.0)
    assert result.williams_r_signal.iloc[7] == pytest.approx((75.0 + 50.0) / 2.0)
    assert result.williams_r_signal.iloc[8] == pytest.approx((50.0 + 200.0 / 3.0) / 2.0)
    # Histogram = main line minus signal.
    for bar in range(len(closes)):
        main, sig = result.williams_r.iloc[bar], result.williams_r_signal.iloc[bar]
        if pd.notna(main) and pd.notna(sig):
            assert result.williams_r_histogram.iloc[bar] == pytest.approx(main - sig)

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


def test_williams_pct_r_advanced_extreme_gate_disabled():
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0]
    frame = _frame(len(closes), close=closes)
    result = williams_pct_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
                                     smoothing_type="SMA", overbought=75.0, oversold=35.0,
                                     context_len=3, extreme_only=False)
    assert result.bull_extreme.tolist() == [False, False, False, False, False, True,
                                            False, False, True]
    # Without the stretch gate the bar-4 and bar-7 bear crosses count too.
    assert result.bear_extreme.tolist() == [False, False, False, False, True, False,
                                            False, True, False]


def test_williams_pct_r_advanced_flat_range_fifty_branch():
    # Constant high/low makes the price range zero and the upstream ternary
    # returns 50.0 (not NaN) once the window is full.
    bars = 6
    frame = _frame(bars, close=np.full(bars, 10.0))
    result = williams_pct_r_advanced(frame, wpr_len=3, avg_len=1, signal_len=2,
                                     smoothing_type="SMA", context_len=3)
    assert result.williams_r.iloc[:2].isna().all()
    assert result.williams_r.iloc[2:].eq(50.0).all()
    assert result.trend_context.iloc[2:].eq(50.0).all()


def test_williams_pct_r_advanced_rejects_bad_input():
    frame = _frame(9, close=[10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 12.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="wpr_len"):
        williams_pct_r_advanced(frame, wpr_len=0)
    with pytest.raises(ValueError, match="avg_len"):
        williams_pct_r_advanced(frame, avg_len=0)
    with pytest.raises(ValueError, match="smoothing_type"):
        williams_pct_r_advanced(frame, smoothing_type="VWMA")
    with pytest.raises(ValueError, match="mid_line"):
        williams_pct_r_advanced(frame, mid_line=-1.0)
    with pytest.raises(ValueError, match="overbought"):
        williams_pct_r_advanced(frame, overbought=101.0)
    with pytest.raises(ValueError, match="oversold"):
        williams_pct_r_advanced(frame, oversold=-1.0)
    with pytest.raises(ValueError, match="stall_wpr_min"):
        williams_pct_r_advanced(frame, stall_wpr_min=-5.0)
    with pytest.raises(ValueError, match="context_len"):
        williams_pct_r_advanced(frame, context_len=1)
    with pytest.raises(ValueError, match="extreme_only"):
        williams_pct_r_advanced(frame, extreme_only="no")
    with pytest.raises(ValueError, match="divergence_len"):
        williams_pct_r_advanced(frame, divergence_len=0)
    with pytest.raises(ValueError, match="high"):
        williams_pct_r_advanced(frame.drop(columns="high"))


# ---------------------------------------------------------------------------
# live_market_scanner_v7
# ---------------------------------------------------------------------------

def test_live_market_scanner_v7_flat_series_scores_zero():
    # Constant series: every rule's inequality is an equality, so every flag
    # stays false and both scores hand-collapse to 0 -> NEUTRAL everywhere.
    bars = 60
    frame = _frame(bars, close=np.full(bars, 100.0), open_=np.full(bars, 100.0),
                   high=np.full(bars, 100.5), low=np.full(bars, 99.5))
    result = live_market_scanner_v7(frame)

    assert result.long_score.tolist() == [0] * bars
    assert result.short_score.tolist() == [0] * bars
    assert result.market_status.tolist() == ["NEUTRAL"] * bars
    assert result.long_score.dtype == pd.Int64Dtype()

    # MACD(12/26) defines from bar 25, its signal chains onto that; ATR(14)
    # from bar 0 (true_range uses high-low when the prior close is NaN) -> no,
    # rma needs 14 TRs, so ATR defines from bar 13; SMA(volume, 20) from 19;
    # linreg from 19; EMA50 from 49.
    assert result.macd_line.iloc[:25].isna().all() and not result.macd_line.iloc[25:].isna().any()
    assert result.macd_signal.iloc[:25].isna().all()
    assert result.atr.iloc[:13].isna().all() and not result.atr.iloc[13:].isna().any()
    assert result.volume_average.iloc[:19].isna().all()
    assert result.linreg.iloc[:19].isna().all()
    assert result.linreg_slope.iloc[:20].isna().all()
    assert result.ema50.iloc[:49].isna().all() and not result.ema50.iloc[49:].isna().any()

    # Flat RSI has no downside at all -> the gain/loss ratio is undefined and
    # the series stays NaN (warm-up semantics), so no OB/OS flag can fire.
    assert result.rsi.isna().all()
    assert not result.volume_spike.any()

    # A single rising step on the last bar of a flat series fires every
    # trend-following rule at once, structurally: ema12 outpaces ema26 so the
    # MACD line lifts off a zero signal with an expanding histogram (+30);
    # ema9 > ema21 with flat-equal priors is a fresh cross (+25); the 1.0
    # candle body exceeds 0.3 * ATR (= 0.3, TR stays high-low = 1.0) (+10);
    # close 101 clears the barely-moved ema50 (+5). RSI sits at 100 (no
    # downside in the whole series), volume is flat, so nothing else adds.
    frame = _frame(bars, close=np.concatenate([np.full(bars - 1, 100.0), [101.0]]),
                   open_=np.concatenate([np.full(bars - 1, 100.0), [100.0]]),
                   high=np.full(bars, 100.5), low=np.full(bars, 99.5))
    stepped = live_market_scanner_v7(frame)
    assert stepped.price_up.iloc[-1]
    assert stepped.strong_bull_candle.iloc[-1]
    assert not stepped.volume_spike.iloc[-1]
    assert stepped.macd_cross_up.iloc[-1]
    assert stepped.ema_cross_up.iloc[-1]
    assert stepped.bull_trend.iloc[-1]
    assert stepped.rsi.iloc[-1] == 100.0
    assert stepped.long_score.iloc[-1] == 30 + 25 + 10 + 5
    assert stepped.short_score.iloc[-1] == 0
    assert stepped.valid_long.iloc[-1]
    assert stepped.market_status.iloc[-1] == "LONG"


def test_live_market_scanner_v7_rejects_bad_input():
    frame = _frame(60, close=np.full(60, 100.0))
    with pytest.raises(ValueError, match="sensitivity"):
        live_market_scanner_v7(frame, sensitivity="Extreme")
    with pytest.raises(ValueError, match="volume"):
        live_market_scanner_v7(frame.drop(columns="volume"))
    with pytest.raises(ValueError, match="open"):
        live_market_scanner_v7(frame.drop(columns="open"))
    with pytest.raises(ValueError, match="high"):
        live_market_scanner_v7(frame.drop(columns="high"))
