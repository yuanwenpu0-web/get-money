import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.cci_advanced import cci_advanced


def _oscillating_frame(n=300):
    rng = np.random.default_rng(12)
    t = np.arange(n)
    close = pd.Series(100 + 6 * np.sin(t / 7.0) + rng.normal(0, 0.3, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return frame


def test_cci_chain_matches_hand_computed_values():
    frame = _oscillating_frame()
    result = cci_advanced(frame, cci_len=10, avg_len=2, sig_len=3, smoothing="SMA",
                          scale_mode="Classic")
    src = (frame["high"] + frame["low"] + frame["close"]) / 3
    avg = src.rolling(10).mean()
    mad = src.rolling(10).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    raw = (src - avg) / (0.015 * mad.where(mad > 0))
    line = raw.rolling(2).mean()
    sig = line.rolling(3).mean()
    valid = sig.dropna().index
    assert np.allclose(result.cci_raw.loc[valid], line.loc[valid])
    assert np.allclose(result.signal_raw.loc[valid], sig.loc[valid])
    assert np.allclose(result.histogram.loc[valid], (line - sig).loc[valid])
    assert result.bull_cross.any() and result.bear_cross.any()
    with pytest.raises(ValueError, match="scale_mode"):
        cci_advanced(frame, scale_mode="Squash")
    with pytest.raises(ValueError, match="行情列"):
        cci_advanced(frame.drop(columns="close"))


def test_scale_modes_change_only_the_displayed_columns():
    frame = _oscillating_frame()
    classic = cci_advanced(frame, cci_len=10, avg_len=2, scale_mode="Classic")
    clamped = cci_advanced(frame, cci_len=10, avg_len=2, scale_mode="Clamp ±200")
    adaptive = cci_advanced(frame, cci_len=10, avg_len=2, scale_mode="Adaptive ±100",
                            scale_lookback=50)
    assert clamped.cci.abs().dropna().le(200.0).all()
    assert adaptive.cci.abs().dropna().le(100.0).all()
    # Raw values are identical regardless of scale.
    assert np.allclose(classic.cci_raw.dropna(), clamped.cci_raw.dropna())


def test_extremes_context_and_quality():
    frame = _oscillating_frame(n=400)
    result = cci_advanced(frame, cci_len=10, avg_len=2, overbought1=80.0,
                          oversold1=-80.0)
    assert (result.bull_extreme == (result.bull_strong | result.bull_weak)).all()
    assert (result.bear_extreme == (result.bear_strong | result.bear_weak)).all()
    assert result.sentiment_score.dropna().between(-100, 100).all()
    assert result.bull_quality.dropna().between(0, 100).all()
    assert result.bear_quality.dropna().between(0, 100).all()
    off = cci_advanced(frame, cci_len=10, avg_len=2, use_extreme_only=False)
    assert off.bull_extreme.sum() >= result.bull_extreme.sum()
    with pytest.raises(ValueError, match="cci_len"):
        cci_advanced(frame, cci_len=0)
