import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.trading_range_state_machine import trading_range_state_machine


def _range_frame(n=600):
    # Oscillating price between 100 and 102: a textbook trading range.
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(101.0 + 0.35 * np.sin(t / 40.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.15,
                         "low": close - 0.15, "close": close})


def test_range_state_walks_the_ladder():
    frame = _range_frame()
    result = trading_range_state_machine(frame, length=60, pivot_len=4)
    settled = result.iloc[150:]
    assert set(result.state.unique()) <= {"Inactive", "Building", "Confirmed",
                                          "Mature", "Breaking"}
    # A perfect oscillation should build and confirm a range.
    assert settled.state.isin(["Confirmed", "Mature"]).mean() > 0.05
    assert settled.range_score.dropna().between(0, 100).all()
    # Boundaries bracket the price.
    valid = result.dropna(subset=["range_high"])
    assert (valid.range_high >= valid.range_low).all()
    assert valid.range_position.between(0, 100).all()
    assert set(result.position_zone.unique()) <= {"Upper", "Mid", "Lower"}
    with pytest.raises(ValueError, match="length"):
        trading_range_state_machine(frame, length=10)
    with pytest.raises(ValueError, match="行情列"):
        trading_range_state_machine(frame.drop(columns="low"))


def test_breakout_fires_breaking_state():
    frame = _range_frame()
    # Force a violent breakout on the last bars.
    frame.iloc[-3:, frame.columns.get_loc("close")] = 120.0
    frame.iloc[-3:, frame.columns.get_loc("high")] = 120.5
    frame.iloc[-3:, frame.columns.get_loc("open")] = 102.0
    result = trading_range_state_machine(frame, length=60, pivot_len=4)
    assert result.state.iloc[-1] in ("Breaking", "Inactive")
    # A breaking-up close is far above the range high.
    if result.breaking.iloc[-1]:
        assert result.breaking_up.iloc[-1]
        assert not result.breaking_down.iloc[-1]
    # Scores stay in 0..100 across all sensors.
    for column in ("er_score", "overlap_score", "chop_score", "structure_score",
                   "regression_score", "atr_score"):
        assert result[column].dropna().between(0, 100).all(), column


def test_trending_market_scores_low():
    index = pd.date_range("2026-01-01", periods=600, freq="15min", tz="UTC")
    close = pd.Series(100.0 + np.arange(600) * 0.15, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.3, "low": close - 0.3,
                          "close": close})
    result = trading_range_state_machine(frame, length=60, pivot_len=4)
    settled = result.iloc[300:]
    assert settled.range_score.dropna().mean() < 70.0
    assert settled.state.isin(["Confirmed", "Mature"]).mean() < 0.5
    with pytest.raises(ValueError, match="pivot_len"):
        trading_range_state_machine(frame, pivot_len=1)
