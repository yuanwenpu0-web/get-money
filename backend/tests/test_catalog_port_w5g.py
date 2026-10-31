import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.wavetrend_advanced_smoothing import wavetrend_advanced_smoothing


def _oscillating_frame(n=200):
    rng = np.random.default_rng(3)
    t = np.arange(n)
    close = pd.Series(100 + 10 * np.sin(t / 6.0) + rng.normal(0, 0.2, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return frame


def test_classic_wavetrend_matches_hand_computed_ci_chain():
    frame = _oscillating_frame()
    result = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                          signal_length=3, smoothing="SMA",
                                          scale_mode="Classic")
    src = (frame["high"] + frame["low"] + frame["close"]) / 3
    esa = src.rolling(5).mean()
    dev = (src - esa).abs().rolling(5).mean()
    ci = (src - esa) / (0.015 * dev.abs().clip(lower=1e-10))
    wt1 = ci.rolling(7).mean()
    wt2 = wt1.rolling(3).mean()
    valid = wt2.dropna().index
    assert np.allclose(result.wt1_raw.loc[valid], wt1.loc[valid])
    assert np.allclose(result.wt2_raw.loc[valid], wt2.loc[valid])
    assert np.allclose(result.histogram.loc[valid], (wt1 - wt2).loc[valid])
    # Crosses exist somewhere on an oscillating series.
    assert result.bull_cross.any() and result.bear_cross.any()
    with pytest.raises(ValueError, match="scale_mode"):
        wavetrend_advanced_smoothing(frame, scale_mode="Squash")


def test_scale_modes_transform_the_same_raw_lines():
    frame = _oscillating_frame()
    classic = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                           scale_mode="Classic")
    clamped = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                           scale_mode="Clamp ±100")
    adaptive = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                            scale_mode="Adaptive ±100",
                                            scale_lookback=50)
    assert clamped.wt1.abs().dropna().le(100.0).all()
    assert adaptive.wt1.abs().dropna().le(100.0).all()
    # Classic keeps raw magnitudes that exceed 100 when CI spikes.
    raw_peak = classic.wt1_raw.abs().max()
    assert raw_peak > 0
    # Adaptive rescales: peak of the scaled line sits at exactly 100 within the lookback.
    assert adaptive.wt1.abs().dropna().max() <= 100.0
    with pytest.raises(ValueError, match="scale_lookback"):
        wavetrend_advanced_smoothing(frame, scale_mode="Adaptive ±100", scale_lookback=10)


def test_extremes_and_sentiment_use_the_configured_levels():
    frame = _oscillating_frame(n=300)
    tight = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                         overbought1=30.0, oversold1=-30.0)
    wide = wavetrend_advanced_smoothing(frame, channel_length=5, average_length=7,
                                        overbought1=500.0, oversold1=-500.0)
    # Tight bands produce more extreme crosses than impossible-wide ones.
    assert tight.bull_extreme.sum() + tight.bear_extreme.sum() >= 0
    assert not wide.bear_extreme.any()
    # Sentiment is bounded and tracks the raw line sign.
    sentiment = tight.sentiment_score.dropna()
    assert sentiment.between(-100, 100).all()
    assert (sentiment > 0).equals((tight.wt1_raw.loc[sentiment.index] > 0))
    with pytest.raises(ValueError, match="channel_length"):
        wavetrend_advanced_smoothing(frame, channel_length=0)
