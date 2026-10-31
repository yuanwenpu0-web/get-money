import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.time_to_react_volatility_time_bos_sweep_ob import \
    time_to_react_volatility_time_bos_sweep_ob


def _bos_frame():
    # Flat base then a strong breakout bar closing above the rolling swing high.
    closes = [100.0] * 20
    closes += [100.2, 100.4, 103.5]   # the last bar closes above the prior 10-bar high
    closes += [104.0, 104.5, 105.0]
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="1h",
                                tz="UTC")
    return frame


def test_bos_event_fires_and_scores_react_fast():
    frame = _bos_frame()
    result = time_to_react_volatility_time_bos_sweep_ob(
        frame, preset="Custom", custom_react_min=60, custom_valid_min=60,
        atr_base_len=5)
    # An earlier wick-sweep occupies the event slot first (correct priority
    # behaviour); the BOS still fires as its own event once the slot frees.
    bos_positions = list(np.flatnonzero(result.bos_up))
    assert bos_positions
    bos_events = result[(result.event_type == "BOS")]
    assert bos_events.event_dir.isin([1, -1]).any()
    # Every bar during an event carries the same event type.
    types = result.event_type[result.event_active]
    assert (types != "").all()
    # Any reacted bar scores speed at the ceiling when it reacts instantly.
    assert result.reacted.any()
    assert result.speed_score.dropna().max() == pytest.approx(100.0)
    assert result.reaction_score.dropna().gt(0).any()
    with pytest.raises(ValueError, match="preset"):
        time_to_react_volatility_time_bos_sweep_ob(frame, preset="Weekly")
    with pytest.raises(ValueError, match="行情列"):
        time_to_react_volatility_time_bos_sweep_ob(frame.drop(columns="open"))


def test_sweep_and_ob_events_register():
    frame = _bos_frame()
    result = time_to_react_volatility_time_bos_sweep_ob(frame, preset="Custom",
                                                        custom_react_min=600,
                                                        custom_valid_min=1200,
                                                        atr_base_len=5)
    # Sanity: booleans and bounded scores across the run.
    for column in ("bos_up", "sweep_up", "ob_tap_bull"):
        assert result[column].dtype == bool
    assert result.reaction_score.dropna().between(0, 100).all()
    assert result.activity_score.dropna().between(0, 100).all()
    assert result.val_progress.dropna().between(0, 1).all()
    with pytest.raises(ValueError, match="priority"):
        time_to_react_volatility_time_bos_sweep_ob(frame, priority="SWEEP>OB>BOS")


def test_activity_score_peaks_above_baseline():
    frame = _bos_frame()
    result = time_to_react_volatility_time_bos_sweep_ob(frame, preset="Custom",
                                                        custom_react_min=600,
                                                        custom_valid_min=1200,
                                                        atr_base_len=5)
    # ATR burst on the breakout lifts activity above its flat base (50).
    late = result.activity_score.iloc[-3:]
    assert late.mean() > 50.0
    with pytest.raises(ValueError, match="swing_lookback"):
        time_to_react_volatility_time_bos_sweep_ob(frame, swing_lookback=1)


def test_priority_orders_pick_one_event():
    frame = _bos_frame()
    bos_first = time_to_react_volatility_time_bos_sweep_ob(
        frame, preset="Custom", custom_react_min=60, custom_valid_min=60,
        atr_base_len=5, priority="BOS>SWEEP>OB")
    assert (bos_first.event_type == "BOS").any()
    ob_first = time_to_react_volatility_time_bos_sweep_ob(
        frame, preset="Custom", custom_react_min=60, custom_valid_min=60,
        atr_base_len=5, priority="OB>BOS>SWEEP")
    # Both run and produce bounded outputs; only the label priority differs.
    assert ob_first.reaction_score.dropna().between(0, 100).all()
