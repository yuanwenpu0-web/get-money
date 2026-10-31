import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.regime_transition_engine import regime_transition_engine


def _trending_frame(n=600):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.08 * t + 0.4 * np.sin(t / 9.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                         "low": np.minimum(open_, close) - 0.3, "close": close})


def test_states_come_from_the_seven_labelled_regimes():
    frame = _trending_frame()
    result = regime_transition_engine(frame, len_mid=50, pct_len=120)
    assert set(result.state_name.unique()) <= {
        "Noise", "Compression", "Expansion", "Trend Up", "Trend Down",
        "Exhaustion", "Reversion"}
    assert result.state.dropna().between(0, 6).all()
    # All scores live in 0..1.
    for column in ("compression_score", "expansion_score", "trend_up_score",
                   "trend_down_score", "exhaustion_score", "reversion_score",
                   "noise_score", "active_score"):
        assert result[column].dropna().between(0, 1).all(), column
    # A clean ramp should spend most settled bars in a trend state.
    settled = result.iloc[300:]
    assert settled.state.isin([3, 4]).mean() > 0.3
    # Dwell: state changes are rarer than every bar.
    assert result.state_changed.sum() < len(frame) / 4
    with pytest.raises(ValueError, match="len_mid"):
        regime_transition_engine(frame, len_mid=10)
    with pytest.raises(ValueError, match="行情列"):
        regime_transition_engine(frame.drop(columns="close"))


def test_hysteresis_and_dwell_govern_switching():
    frame = _trending_frame()
    strict = regime_transition_engine(frame, dwell_bars=50, hysteresis=0.9)
    loose = regime_transition_engine(frame, dwell_bars=1, hysteresis=0.0)
    assert strict.state_changed.sum() <= loose.state_changed.sum()
    # state_age resets on switch and counts up otherwise.
    ages = strict.state_age.to_numpy(dtype=float)
    assert (ages >= 1).all()
    with pytest.raises(ValueError, match="hysteresis"):
        regime_transition_engine(frame, hysteresis=1.5)


def test_directional_pressure_follows_state_and_trend():
    frame = _trending_frame()
    result = regime_transition_engine(frame, len_mid=50, pct_len=120)
    up_bars = result.state == 3
    down_bars = result.state == 4
    assert (result.directional_pressure[up_bars] >= 0).all()
    assert (result.directional_pressure[down_bars] <= 0).all()
    # Transition score is an EMA of a 0..1 raw score scaled to 0..100.
    assert result.transition_score.dropna().between(0, 100).all()
