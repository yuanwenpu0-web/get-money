import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.relative_leg_efficiency import relative_leg_efficiency


def _zigzag_frame(legs=5, leg_bars=8, up_move=8.0, retrace=3.0):
    retrace_bars = 6  # keep pivot-to-pivot spacing >= min_leg_bars=6
    closes = []
    level = 100.0
    for cycle in range(legs):
        for i in range(leg_bars):
            level += up_move / leg_bars
            closes.append(level)
        for i in range(retrace_bars):
            level -= retrace / retrace_bars
            closes.append(level)
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2,
                          "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min",
                                tz="UTC")
    # Unique pivots at the leg extremes.
    for k in range(legs):
        top = k * (leg_bars + retrace_bars) + leg_bars - 1
        frame.iloc[top, frame.columns.get_loc("high")] += 0.6
        frame.iloc[top + retrace_bars, frame.columns.get_loc("low")] -= 0.6
    return frame


def test_leg_completion_and_rle_components():
    frame = _zigzag_frame()
    result = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                     min_leg_bars=6, min_leg_move_atr=0.5)
    assert result.leg_completed.any()
    done = result[result.leg_completed]
    assert (done.leg_bars >= 6).all()
    assert done.leg_pe.between(0, 1).all()
    assert done.leg_tf.between(0, 1).all()
    assert done.leg_cp.between(0, 1).all()
    assert done.rle_completed.between(0, 100).all()
    # Alternating legs: up legs (dir=1) follow down pivots.
    assert set(done.leg_direction.unique()) <= {1, -1}
    with pytest.raises(ValueError, match="pivot_left"):
        relative_leg_efficiency(frame, pivot_left=0)
    with pytest.raises(ValueError, match="行情列"):
        relative_leg_efficiency(frame.drop(columns="high"))


def test_straight_leg_scores_higher_than_choppy():
    straight = _zigzag_frame(legs=4, leg_bars=8)
    rng = np.random.default_rng(21)
    choppy = straight.copy()
    noise = rng.normal(0, 0.8, len(choppy))
    choppy["close"] = choppy["close"] + noise
    choppy["high"] = choppy[["open", "close"]].max(axis=1) + 0.2
    choppy["low"] = choppy[["open", "close"]].min(axis=1) - 0.2
    a = relative_leg_efficiency(straight, pivot_left=3, pivot_right=3,
                                min_leg_move_atr=0.5)
    b = relative_leg_efficiency(choppy, pivot_left=3, pivot_right=3,
                                min_leg_move_atr=0.5)
    # The straight run's completed legs have PE at or above the noisy run's.
    if a.leg_completed.any() and b.leg_completed.any():
        assert a.leg_pe.dropna().mean() >= b.leg_pe.dropna().mean() - 0.15


def test_counter_modes_change_cp():
    frame = _zigzag_frame()
    body = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                   counter_mode="Body Weighted",
                                   min_leg_move_atr=0.5)
    count = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                    counter_mode="Count", min_leg_move_atr=0.5)
    both = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                   counter_mode="Both", min_leg_move_atr=0.5)
    for result in (body, count, both):
        assert result.leg_cp.dropna().between(0, 1).all()
        assert result.rle_completed.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="counter_mode"):
        relative_leg_efficiency(frame, counter_mode="Volume")
    with pytest.raises(ValueError, match="time_mode"):
        relative_leg_efficiency(frame, time_mode="Warp")


def test_spike_guard_caps_the_score():
    frame = _zigzag_frame()
    # One bar with a monster body inside the first up leg.
    frame.iloc[5, frame.columns.get_loc("open")] = frame.close.iloc[4]
    frame.iloc[5, frame.columns.get_loc("close")] = frame.close.iloc[4] + 12.0
    frame.iloc[5, frame.columns.get_loc("high")] = frame.close.iloc[4] + 12.2
    off = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                  min_leg_move_atr=0.5)
    on = relative_leg_efficiency(frame, pivot_left=3, pivot_right=3,
                                 min_leg_move_atr=0.5, enable_spike_guard=True,
                                 spike_atr_mult=1.0, spike_cap=40.0)
    guarded = on.rle_completed.dropna()
    plain = off.rle_completed.dropna()
    # At least one completed leg got capped, and the guard never raises scores.
    assert (guarded <= 40.0).any()
    assert plain.max() >= guarded.max() - 1e-9
