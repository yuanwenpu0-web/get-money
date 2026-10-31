import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.btc_logarithmic_growth_curves import \
    btc_logarithmic_growth_curves


def _time_frame(days=900):
    index = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    close = pd.Series(30000.0 + np.arange(days) * 10.0, index=index)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_bands_are_deterministic_in_time_and_ordered():
    frame = _time_frame()
    result = btc_logarithmic_growth_curves(frame)
    names = [c for c in result.columns if c.startswith("band_")]
    assert len(names) == 11
    # Deterministic: recomputing gives identical values.
    again = btc_logarithmic_growth_curves(frame)
    pd.testing.assert_frame_equal(result, again)
    # Ordered top to bottom at every bar.
    ordered = result[names].dropna()
    assert (ordered.diff(axis=1).iloc[:, 1:] <= 1e-9).all().all()
    # The mid band is the geometric mean of top and bottom in log space.
    mid = (np.log(result.band_100) + np.log(result.band_000)) / 2
    assert np.allclose(np.log(result.band_mid.dropna()), mid.dropna())
    # Bands grow over time (log-growth channel slopes upward in linear space).
    assert result.band_100.iloc[-1] > result.band_100.iloc[100]
    with pytest.raises(ValueError, match="行情列"):
        btc_logarithmic_growth_curves(frame.drop(columns="close"))
    with pytest.raises(ValueError, match="索引"):
        btc_logarithmic_growth_curves(frame.reset_index(drop=True))


def test_pre_epoch_bars_clamp_to_the_epoch_constant():
    # Pine clamps the time index to 3 days before the epoch, so pre-epoch bars
    # all share one constant band level (t=3 > 2 keeps the log valid).
    index = pd.date_range("2009-01-01", periods=5, freq="1D", tz="UTC")
    close = pd.Series([1.0] * 5, index=index)
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    result = btc_logarithmic_growth_curves(frame)
    assert result.band_100.nunique() == 1
    # The constant sits far below the modern channel: the curve has grown.
    modern = btc_logarithmic_growth_curves(_time_frame(100))
    assert result.band_100.iloc[0] < modern.band_100.iloc[50]


def test_slope_parameters_move_the_channel():
    frame = _time_frame(400)
    base = btc_logarithmic_growth_curves(frame)
    steeper = btc_logarithmic_growth_curves(frame, high_slope=0.001)
    assert steeper.band_100.iloc[-1] > base.band_100.iloc[-1]
    with pytest.raises(ValueError, match="high_intercept"):
        btc_logarithmic_growth_curves(frame, high_intercept=20.0)
