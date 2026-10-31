import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.markov_state_engine import markov_state_engine


def _updown_frame(n=900):
    # Alternating regimes: long ramps up/down with flat pauses between.
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    level = 100.0
    closes = []
    for bar in range(n):
        phase = (bar // 150) % 2
        if phase == 0:
            level += 0.08
        elif bar % 150 < 50:
            pass  # flat
        else:
            level -= 0.08
        closes.append(level)
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.15, "low": close - 0.15,
                         "close": close})


def test_states_named_and_probabilities_sum():
    frame = _updown_frame()
    result = markov_state_engine(frame, lookback=300)
    assert set(result.state_name.unique()) <= {
        "Up Trend", "Up Weak", "Compression", "Down Trend", "Down Weak",
        "Expansion/Chaos"}
    prob_sum = result.p_bull + result.p_bear + result.p_neutral
    assert prob_sum.dropna().round(6).eq(100.0).all()
    assert result.persistence.dropna().between(0, 100).all()
    assert result.reliability.dropna().between(0, 100).all()
    assert result.net_forecast.dropna().between(-100, 100).all()
    assert set(result.bias.unique()) <= {-1, 0, 1}
    assert set(result.lean_type.unique()) <= {
        "—", "Reversal", "Breakout", "Continuation"}
    with pytest.raises(ValueError, match="lookback"):
        markov_state_engine(frame, lookback=10)
    with pytest.raises(ValueError, match="行情列"):
        markov_state_engine(frame.drop(columns="close"))


def test_debounce_reduces_state_flicker():
    frame = _updown_frame()
    loose = markov_state_engine(frame, lookback=300, min_dwell=1)
    strict = markov_state_engine(frame, lookback=300, min_dwell=10)
    assert strict.state_changed.sum() <= loose.state_changed.sum()
    assert strict.bars_in_state.min() >= 0


def test_lean_types_match_state_and_bias():
    frame = _updown_frame()
    result = markov_state_engine(frame, lookback=300)
    bull_reversal = result[result.lean_type == "Reversal"]
    assert (bull_reversal.bias == 1).all()
    # Edges only fire with confident+reliable bias.
    assert not (result.bull_edge & (result.bias != 1)).any()
    assert not (result.bear_edge & (result.bias != -1)).any()
    with pytest.raises(ValueError, match="min_dwell"):
        markov_state_engine(frame, min_dwell=0)
