import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.regime_classifier import regime_classifier


def _trending_frame():
    t = np.arange(240)
    close = pd.Series(100 + 0.5 * t + 2.0 * np.sin(t / 4.0))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.4,
                          "low": np.minimum(open_, close) - 0.4, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(t), freq="15min", tz="UTC")
    return frame


def _choppy_frame():
    rng = np.random.default_rng(4)
    t = np.arange(240)
    close = pd.Series(100 + 2.0 * np.sin(t * 1.7) + rng.normal(0, 0.8, t.size))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.4,
                          "low": np.minimum(open_, close) - 0.4, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(t), freq="15min", tz="UTC")
    return frame


def test_strong_trend_scores_high_and_labels_trend():
    frame = _trending_frame()
    result = regime_classifier(frame, er_len=10, fdi_len=20, chop_len=10,
                               include_htf=False)
    settled = result.trend_score.iloc[30:]
    assert (settled >= 0.62).mean() > 0.8  # a clean ramp is almost always Trend
    assert (result.regime.iloc[30:] == "Trend").mean() > 0.8
    assert result.trend_score.dropna().between(0, 1).all()
    assert set(result.regime.unique()) <= {"Trend", "Range", "Chaos"}
    assert result.efficiency_ratio.dropna().between(0, 1).all()
    assert result.fractal_dimension.dropna().between(1, 2).all()
    assert result.choppiness.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="er_len"):
        regime_classifier(frame, er_len=1)
    with pytest.raises(ValueError, match="行情列"):
        regime_classifier(frame.drop(columns="close"))


def test_choppy_market_scores_low_and_chaos_override_can_fire():
    frame = _choppy_frame()
    result = regime_classifier(frame, er_len=10, fdi_len=20, chop_len=10,
                               include_htf=False)
    # Directionless noise: the mean trendiness is far below the trend threshold.
    assert result.trend_score.iloc[30:].mean() < 0.62
    # With very loose chaos thresholds the override fires somewhere.
    chaotic = regime_classifier(frame, er_len=10, fdi_len=20, chop_len=10,
                                chop_chaos=0.0, er_chaos=1.0, fdi_chaos=1.0,
                                include_htf=False)
    assert chaotic.chaos.all()
    # Chaos damp suppresses both permissions to their damped share.
    bar = 60
    assert chaotic.trend_permission.iloc[bar] == pytest.approx(
        chaotic.trend_score.iloc[bar] * 0.30)
    assert chaotic.reversal_permission.iloc[bar] == pytest.approx(
        (1 - chaotic.trend_score.iloc[bar]) * 0.30)


def test_hysteresis_holds_between_the_thresholds():
    frame = _trending_frame()
    # Tight thresholds push the whole mid band into "hold previous".
    result = regime_classifier(frame, er_len=10, fdi_len=20, chop_len=10,
                               trend_hi=0.99, range_lo=0.01, include_htf=False)
    # First bar decides at 0.5, then the state persists until a threshold is
    # crossed — with 0.99/0.01 almost nothing changes after the start.
    assert result.regime_state.iloc[1:].nunique() <= 2


def test_htf_regime_is_the_previour_completed_period():
    frame = _trending_frame()
    result = regime_classifier(frame, er_len=10, fdi_len=20, chop_len=10,
                               include_htf=True)
    assert "htf_score" in result.columns
    # The first 4 bars cannot have a completed 1h period.
    assert result.htf_score.iloc[:4].isna().all()
    assert set(result.htf_regime.dropna().unique()) <= {"Trend", "Range", "Neutral"}
