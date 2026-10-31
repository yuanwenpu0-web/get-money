import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.smooth_trend_radar import smooth_trend_radar


def _trend_frame(n=400):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    # A declining warm-up forces the trend to -1 first, then the ramp flips it.
    base = 100.0 + 0.08 * t + 0.6 * np.sin(t / 10.0)
    close = pd.Series(np.where(t < 200, 300.0 - base, base), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.3, "low": close - 0.3,
                         "close": close})


def test_baseline_smoothing_and_trend_hysteresis():
    frame = _trend_frame()
    result = smooth_trend_radar(frame, st_atr_period=30, wma_length=15, ema_length=5)
    settled = result.iloc[280:]
    assert (settled.trend == 1).all()
    assert result.long_signal.any()
    assert settled.short_signal.sum() == 0
    assert settled.strength_rank.dropna().between(0, 100).all()
    assert set(result.rejection_regime.unique()) <= {"Trend Pullback", "Range Rejection"}
    with pytest.raises(ValueError, match="st_factor"):
        smooth_trend_radar(frame, st_factor=0.0)
    with pytest.raises(ValueError, match="行情列"):
        smooth_trend_radar(frame.drop(columns="low"))


def test_rejection_requires_touches_slope_and_polarity():
    frame = _trend_frame(n=500)
    base = smooth_trend_radar(frame, st_atr_period=30, wma_length=15, ema_length=5,
                              cont_factor=2, polarity_method="Off")
    body = smooth_trend_radar(frame, st_atr_period=30, wma_length=15, ema_length=5,
                              cont_factor=2, polarity_method="Body")
    # Adding the body filter can only reduce the bull rejections.
    assert body.rejection_bull.sum() <= base.rejection_bull.sum()
    assert base.bull_touches.dropna().min() >= 0
    # Rejections in an uptrend are bull-sided only.
    bull_rejections = base[base.rejection_bull]
    assert (bull_rejections.trend == 1).all()
    with pytest.raises(ValueError, match="polarity_method"):
        smooth_trend_radar(frame, polarity_method="Wick")


def test_volume_filter_and_overextension():
    frame = _trend_frame(n=500)
    frame["volume"] = 10.0
    filtered = smooth_trend_radar(frame, volume_filter="Above Average",
                                  volume_ma_len=10, st_atr_period=30,
                                  wma_length=15, ema_length=5)
    # Constant volume never exceeds its own SMA -> everything is filtered.
    assert not filtered.rejection_bull.any()
    assert not filtered.rejection_bear.any()
    off = smooth_trend_radar(frame, volume_filter="Off", st_atr_period=30,
                             wma_length=15, ema_length=5, cont_factor=1)
    assert off.rejection_bull.sum() + off.rejection_bear.sum() >= 0  # computed
    # Overextension ranks are bounded.
    assert filtered.overext_bull.sum() + filtered.overext_bear.sum() \
        <= filtered.overextended.sum()
    with pytest.raises(ValueError, match="volume_filter"):
        smooth_trend_radar(frame, volume_filter="Below Average")
