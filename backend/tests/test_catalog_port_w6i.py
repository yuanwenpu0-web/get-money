import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.signal_quality_engine import signal_quality_engine


def _range_frame(n=500):
    # Ornstein-Uhlenbeck mean reversion: a genuine range with DI+ ≈ DI-.
    rng = np.random.default_rng(7)
    x = np.zeros(n)
    x[0] = 100.0
    for i in range(1, n):
        x[i] = x[i - 1] + (100.0 - x[i - 1]) * 0.05 + rng.normal(0, 0.15)
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series(x, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.1, "low": close - 0.1,
                          "close": close})
    frame["volume"] = 10.0
    return frame


def test_scores_bounded_and_edge_direction_sane():
    frame = _range_frame()
    result = signal_quality_engine(frame)
    for column in ("bottom_score", "top_score"):
        assert result[column].dropna().between(0, 100).all(), column
    assert set(result.edge_dir.dropna().unique()) <= {-1, 0, 1}
    # Edge direction 1 implies bottom score at least as large as top.
    bull = result[result.edge_dir == 1]
    assert (bull.bottom_score >= bull.top_score).all()
    assert set(result.bottom_grade.unique()) <= {"A+", "A", "B", "C", "—"}
    with pytest.raises(ValueError, match="profile"):
        signal_quality_engine(frame, profile="Wheat")
    with pytest.raises(ValueError, match="strictness"):
        signal_quality_engine(frame, strictness="Loose")
    with pytest.raises(ValueError, match="行情列"):
        signal_quality_engine(frame.drop(columns="volume"))


def test_profile_constants_change_the_regime_gate():
    frame = _range_frame()
    crypto = signal_quality_engine(frame, profile="Crypto")
    # The OU range sits inside the Crypto gate (ADX < 24, slope < 0.45).
    assert crypto.range_ok.mean() > 0.5
    # The tighter Stocks slope limit admits fewer bars.
    stocks = signal_quality_engine(frame, profile="Stocks / Indices")
    assert stocks.range_ok.mean() <= crypto.range_ok.mean()


def test_strictness_gates_and_trigger_construction():
    frame = _range_frame()
    strict = signal_quality_engine(frame, strictness="Very Strict")
    normal = signal_quality_engine(frame, strictness="Normal")
    # Strict needs a higher score for an extreme: fewer or equal extremes.
    assert strict.bottom_extreme.sum() <= normal.bottom_extreme.sum()
    assert strict.top_extreme.sum() <= normal.top_extreme.sum()
    # Triggers imply the armed window and the range-side condition.
    fired = normal[normal.long_trigger]
    assert (fired.range_position < 0.35).all()
    fired_short = normal[normal.short_trigger]
    assert (fired_short.range_position > 0.65).all()
    # Watch events never coincide with their trigger.
    assert not (normal.long_trigger & normal.long_watch).any()
    with pytest.raises(ValueError, match="confirm_window"):
        signal_quality_engine(frame, confirm_window=0)
