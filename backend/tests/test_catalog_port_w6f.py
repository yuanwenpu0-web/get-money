import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mfi_advanced import mfi_advanced


def _flow_frame(n=200):
    rng = np.random.default_rng(14)
    t = np.arange(n)
    close = pd.Series(100 + 5 * np.sin(t / 6.0) + rng.normal(0, 0.4, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3, "close": close})
    frame["volume"] = np.abs(rng.normal(50, 10, n))
    frame.index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return frame


def test_mfi_chain_matches_hand_computed_values():
    frame = _flow_frame()
    result = mfi_advanced(frame, mfi_len=7, avg_len=2, sig_len=3, smoothing="SMA")
    src = (frame["high"] + frame["low"] + frame["close"]) / 3
    flow = src * frame["volume"]
    pos = flow.where(src > src.shift(1), 0.0)
    neg = flow.where(src < src.shift(1), 0.0)
    pos_sum = pos.rolling(7).sum()
    neg_sum = neg.rolling(7).sum()
    raw = pd.Series(np.where(neg_sum == 0, 100.0,
                             100.0 - 100.0 / (1.0 + pos_sum / neg_sum)), index=frame.index)
    line = raw.rolling(2).mean()
    sig = line.rolling(3).mean()
    valid = sig.dropna().index
    assert np.allclose(result.mfi.loc[valid], line.loc[valid])
    assert np.allclose(result.signal.loc[valid], sig.loc[valid])
    assert result.mfi.dropna().between(0, 100).all()
    assert result.bull_cross.any()
    with pytest.raises(ValueError, match="mfi_len"):
        mfi_advanced(frame, mfi_len=0)
    with pytest.raises(ValueError, match="行情列"):
        mfi_advanced(frame.drop(columns="volume"))


def test_zero_volume_feed_goes_inert():
    frame = _flow_frame()
    frame["volume"] = 0.0
    result = mfi_advanced(frame)
    assert result.mfi.isna().all()
    assert not result.bull_extreme.any()
    assert not result.stall_bull.any()
    # The chain still computes the price-only context line? No — MFI needs volume.
    assert result.context_line.isna().all()


def test_extremes_quality_and_partition():
    frame = _flow_frame(n=300)
    strict = mfi_advanced(frame, mfi_len=7, avg_len=2, use_extreme_only=True)
    loose = mfi_advanced(frame, mfi_len=7, avg_len=2, use_extreme_only=False)
    assert loose.bull_extreme.sum() >= strict.bull_extreme.sum()
    assert (strict.bull_extreme == (strict.bull_strong | strict.bull_weak)).all()
    assert (strict.bear_extreme == (strict.bear_strong | strict.bear_weak)).all()
    assert strict.sentiment_score.dropna().between(-100, 100).all()
    assert strict.bull_quality.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="os_level"):
        mfi_advanced(frame, os_level=60.0)
