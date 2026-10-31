import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.williams_vix_fix_advanced import williams_vix_fix_advanced


def _calm_then_crash_frame():
    # 40 quiet bars then a sharp crash: WVF (distance below the rolling high)
    # spikes on the crash.
    n = 60
    closes = [100.0] * 40 + [100.0 - 2.0 * step for step in range(1, n - 39)]
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return frame


def test_wvf_matches_the_definition_and_spikes_on_crash():
    frame = _calm_then_crash_frame()
    result = williams_vix_fix_advanced(frame, wvf_len=22, band_len=20, band_mult=2.0,
                                       use_band=True, use_percentile=False)
    hc = frame["close"].rolling(22).max()
    expected_wvf = (hc - frame["low"]) / hc * 100.0
    valid = expected_wvf.dropna().index
    assert np.allclose(result.wvf.loc[valid], expected_wvf.loc[valid])
    # The crash drives WVF far above the quiet baseline and eventually spikes.
    assert result.wvf.iloc[-1] > result.wvf.iloc[30]
    assert result.spike.any()
    assert result.wvf.dropna().ge(0).all()
    with pytest.raises(ValueError, match="wvf_len"):
        williams_vix_fix_advanced(frame, wvf_len=1)
    with pytest.raises(ValueError, match="行情列"):
        williams_vix_fix_advanced(frame.drop(columns="low"))


def test_bear_wvf_mirrors_and_quality_bounded():
    frame = _calm_then_crash_frame()
    result = williams_vix_fix_advanced(frame, show_bear_wvf=True)
    assert result.bear_wvf.dropna().ge(0).all()
    assert result.bull_quality.dropna().between(0, 100).all()
    assert result.bear_quality.dropna().between(0, 100).all()
    assert result.sentiment_score.dropna().between(-100, 100).all()
    # Strong + weak partition the crosses exactly.
    assert (result.spike_cross == (result.spike_strong | result.spike_weak)).all()
    assert (result.bear_spike_cross
            == (result.bear_spike_strong | result.bear_spike_weak)).all()


def test_percentile_mode_and_stall_gating():
    frame = _calm_then_crash_frame()
    pct = williams_vix_fix_advanced(frame, use_band=False, use_percentile=True,
                                    range_len=30, range_high_pct=95.0)
    assert pct.spike.any()
    none_mode = williams_vix_fix_advanced(frame, use_band=False, use_percentile=False)
    assert not none_mode.spike.any()
    # Heavy stall tolerance turns crosses weak.
    weak = williams_vix_fix_advanced(frame, stall_flat_atr=100.0, stall_wvf_min=0.0)
    if weak.spike_cross.any():
        assert weak.spike_weak.sum() == weak.spike_cross.sum()
    with pytest.raises(ValueError, match="band_len"):
        williams_vix_fix_advanced(frame, band_len=1)
