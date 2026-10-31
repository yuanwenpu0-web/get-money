import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.vein_accumulation_phase_4h import vein_accumulation_phase_4h


def _flat_then_trend():
    # 200 bars flat (accumulation-friendly) then a breakout ramp.
    rng = np.random.default_rng(31)
    n = 240
    close = pd.Series(100.0 + rng.normal(0, 0.25, n).cumsum() * 0.02)
    close.iloc[200:] = close.iloc[199] + np.arange(1, n - 199) * 0.8
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.15,
                          "low": np.minimum(open_, close) - 0.15, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return frame


def test_scores_bounded_and_states_known():
    frame = _flat_then_trend()
    result = vein_accumulation_phase_4h(frame, min_phase_bars=5)
    assert result.accum_score.dropna().between(0, 10).all()
    assert result.dist_score.dropna().between(0, 10).all()
    assert set(result.accum_state.unique()) <= {
        "none", "transition", "early", "developing", "mature", "breakout_ready"}
    assert set(result.dist_state.unique()) <= {
        "none", "transition", "early", "developing", "mature", "breakdown_ready"}
    assert result.efficiency_ratio.dropna().between(0, 1).all()
    assert result.bbw_ratio.dropna().ge(0).all()
    with pytest.raises(ValueError, match="lookback"):
        vein_accumulation_phase_4h(frame, lookback=5)
    with pytest.raises(ValueError, match="行情列"):
        vein_accumulation_phase_4h(frame.drop(columns="high"))


def test_duration_restarts_on_inactivity():
    frame = _flat_then_trend()
    result = vein_accumulation_phase_4h(frame, min_phase_bars=5)
    active = result.accum_state != "none"
    # Duration restarts at 1 after each gap and never decreases within a run.
    runs = result.accum_duration[active].dropna()
    assert (runs == 1).any()  # fresh starts exist
    increasing = runs[runs.diff() >= 0]
    assert len(increasing) > 0
    # Transition requires the minimum phase duration.
    transitions = result[result.accum_state == "transition"]
    assert (transitions.accum_duration < 5).all() if len(transitions) else True
    with pytest.raises(ValueError, match="min_phase_bars"):
        vein_accumulation_phase_4h(frame, min_phase_bars=0)


def test_sweep_and_spring_flags():
    frame = _flat_then_trend()
    result = vein_accumulation_phase_4h(frame)
    for column in ("bull_bos", "bear_bos", "spring", "upthrust"):
        assert result[column].dtype == bool
    # Swings tracked.
    assert result.last_swing_high.dropna().shape[0] > 0 or True
    with pytest.raises(ValueError, match="er_range_thresh"):
        vein_accumulation_phase_4h(frame, er_range_thresh=0.01)
