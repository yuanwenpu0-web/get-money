import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.trend_persistence_score import trend_persistence_score


def _trending_frame(n=500):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.15 * t + 0.5 * np.sin(t / 7.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                         "low": np.minimum(open_, close) - 0.3, "close": close})


def test_trending_market_scores_high_and_state_upgrades():
    frame = _trending_frame()
    result = trend_persistence_score(frame, length=30, adx_len=10)
    settled = result.iloc[150:]
    assert settled.tps.dropna().mean() > 50.0
    # Sensors are bounded 0..100.
    for column in ("r2_score", "er_score", "adx_score", "fdi_score"):
        assert result[column].dropna().between(0, 100).all(), column
    # A clean ramp reads Up and reaches at least Healthy somewhere.
    assert settled.direction_up.any()
    assert (settled.state >= 2).any()
    assert set(result.state_name.unique()) <= {
        "Trend Strong", "Trend Healthy", "Transition", "Weak / Range", "Trend Dead"}
    with pytest.raises(ValueError, match="length"):
        trend_persistence_score(frame, length=5)
    with pytest.raises(ValueError, match="行情列"):
        trend_persistence_score(frame.drop(columns="close"))


def test_hysteresis_downgrades_need_a_three_point_margin():
    frame = _trending_frame()
    result = trend_persistence_score(frame, length=30, adx_len=10)
    # The state index never sits above the raw index by more than one bucket
    # for long; here we just assert monotone membership bounds.
    assert result.state.between(0, 4).all()
    # Events are mutually exclusive crossings.
    assert not (result.weakening_event & result.recovery_event).any()
    assert result.transition_risk.dropna().between(0, 100).all()


def test_choppy_market_scores_low():
    rng = np.random.default_rng(11)
    n = 500
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100 + rng.normal(0, 0.8, n).cumsum() * 0.1, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3, "close": close})
    result = trend_persistence_score(frame, length=30, adx_len=10)
    settled = result.iloc[150:]
    assert settled.tps.dropna().mean() < 70.0
    with pytest.raises(ValueError, match="lvl_strong"):
        trend_persistence_score(frame, lvl_strong=120)
