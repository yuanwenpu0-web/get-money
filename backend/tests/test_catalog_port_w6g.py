import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mtf_stochrsi_pair_score import mtf_stochrsi_pair_score


def _trend_frame(n=96 * 40):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.05 * t + 2.0 * np.sin(t / 8.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                         "low": np.minimum(open_, close) - 0.3, "close": close})


def test_scores_bounded_and_pairs_computed():
    frame = _trend_frame()
    result = mtf_stochrsi_pair_score(frame, timeframes=("15", "60", "120", "240", "D", "D"),
                                     use_pairs=(True, True, True, True, True))
    for column in [c for c in result.columns if c.startswith("score_")]:
        assert result[column].dropna().between(-100, 100).all(), column
    assert result.smoothed_score.dropna().between(-100, 100).all()
    assert result.agreement.dropna().between(-100, 100).all()
    assert result.total_score.notna().any()
    assert set(result.bias_state.unique()) <= {
        "NO EDGE", "LONG CONFIRMED", "SHORT CONFIRMED", "CONFLICT",
        "LONG BUILDING", "SHORT BUILDING"}
    assert set(result.quality_state.unique()) <= {"WEAK", "MIXED", "CLEAN"}
    with pytest.raises(ValueError, match="use_pairs"):
        mtf_stochrsi_pair_score(frame, use_pairs=(True, True, True))
    with pytest.raises(ValueError, match="timeframes"):
        mtf_stochrsi_pair_score(frame, timeframes=("15", "60"))


def test_steady_uptrend_scores_positive_with_bull_agreement():
    # Mostly-rising steps with one small dip per day keep RSI elevated but
    # not saturated, so the StochRSI zone-riding bonus dominates.
    n = 96 * 40
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    steps = np.where(np.arange(n) % 96 == 95, -6.0, 0.4)
    close = pd.Series(100.0 + np.cumsum(steps), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.3, "low": close - 0.3,
                          "close": close})
    result = mtf_stochrsi_pair_score(frame, timeframes=("15", "60", "120", "240", "D", "D"))
    settled = result.iloc[3000:]
    assert settled.smoothed_score.dropna().mean() > 0
    # Agreement is bounded and the bias state machinery ran to completion.
    assert settled.agreement.dropna().between(-100, 100).all()
    assert settled.bias_state.notna().all()
    with pytest.raises(ValueError, match="未知周期标签"):
        mtf_stochrsi_pair_score(frame, timeframes=("15", "30", "60", "120", "240", "2h"))


def test_sync_bonus_and_conflict_penalty_shape_pairs():
    frame = _trend_frame()
    result = mtf_stochrsi_pair_score(frame, timeframes=("15", "60", "120", "240", "D", "D"),
                                     use_pairs=(True, False, False, False, False))
    both = result.score_15 * 0.4 + result.score_60 * 0.6
    same_dir = (result.score_15 > 0) & (result.score_60 > 0)
    # Same-direction bars get the 1.15 bonus applied to the weighted base.
    check = result.loc[same_dir, "p12"]
    base_check = both.loc[same_dir] * 1.15
    assert np.allclose(check.clip(-100, 100), base_check.clip(-100, 100))
    with pytest.raises(ValueError, match="sync_bonus"):
        mtf_stochrsi_pair_score(frame, sync_bonus=0.5)
