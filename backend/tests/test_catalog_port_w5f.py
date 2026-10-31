import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_structure_advanced import market_structure_advanced


def _zigzag_frame():
    # Rising zigzag: each cycle climbs 8 then retraces 4, so every high is a
    # strict HH and every low a strict HL.
    closes = []
    level = 100.0
    for cycle in range(5):
        for _ in range(4):
            level += 2.0
            closes.append(level)
        for _ in range(4):
            level -= 1.0
            closes.append(level)
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2,
                          "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    # Make the leg extremes unique strict pivots.
    for k in range(5):
        frame.iloc[k * 8 + 3, frame.columns.get_loc("high")] = close.iloc[k * 8 + 3] + 0.5
        frame.iloc[k * 8 + 7, frame.columns.get_loc("low")] = close.iloc[k * 8 + 7] - 0.5
    return frame


def test_score_tracks_pivot_events_and_normalises_to_pm100():
    frame = _zigzag_frame()
    result = market_structure_advanced(frame, pivot_left=2, pivot_right=2,
                                       score_window=4, signal_length=2)
    # Every high after the first is a HH (+2) and every low after the first a
    # HL (+1); the first pivots have no same-side predecessor.
    assert result.higher_high.sum() == 4
    assert result.higher_low.sum() == 7  # double-counted plateau lows
    assert not result.lower_high.iloc[20:].any()
    assert not result.lower_low.iloc[20:].any()
    # Ring of (1,2,1,2) sums to 6 -> 75; (2,1,1,2) and (1,1,2,2) -> 62.5/75.
    settled = result.score.dropna()
    assert settled.tail(20).isin([62.5, 75.0]).all()
    assert result.score.dropna().between(-100, 100).all()
    with pytest.raises(ValueError, match="score_window"):
        market_structure_advanced(frame, score_window=1)
    with pytest.raises(ValueError, match="smoothing"):
        market_structure_advanced(frame, smoothing="VIP")


def test_bearish_structure_scores_double_negative():
    # Falling zigzag: each cycle drops 8 then bounces 4, so every high is a LH
    # (-1) and every low a LL (-2).
    closes = []
    level = 140.0
    for cycle in range(5):
        for _ in range(4):
            level -= 2.0
            closes.append(level)
        for _ in range(4):
            level += 1.0
            closes.append(level)
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2,
                          "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    for k in range(5):
        frame.iloc[k * 8 + 3, frame.columns.get_loc("low")] = close.iloc[k * 8 + 3] - 0.5
        frame.iloc[k * 8 + 7, frame.columns.get_loc("high")] = close.iloc[k * 8 + 7] + 0.5
    result = market_structure_advanced(frame, pivot_left=2, pivot_right=2,
                                       score_window=4, signal_length=2)
    assert result.lower_high.sum() >= 3
    assert result.lower_low.sum() >= 3
    assert not result.higher_high.any()
    # Rings mix (-1,-2) pairs: -6 -> -75, -4 -> -50, -3 -> -37.5.
    settled = result.score.dropna()
    assert settled.tail(20).isin([-12.5, -50.0, -37.5, -75.0]).all()
    assert (settled.tail(20) <= -12.5).all()


def test_score_window_limits_the_memory():
    frame = _zigzag_frame()
    short = market_structure_advanced(frame, pivot_left=2, pivot_right=2,
                                      score_window=2, signal_length=1,
                                      smoothing="SMA")
    # Rings of (2,1)/(1,2)/(1,1) sums to 3-4 -> 62.5/50/75; with
    # signal_length=1 the smoothed signal equals the score wherever defined.
    settled = short.score.dropna()
    assert settled.tail(20).isin([62.5, 50.0, 75.0]).all()
    assert (settled.tail(20) >= 50.0).all()
    valid = short.signal.dropna()
    assert np.allclose(valid.to_numpy(), settled.loc[valid.index].to_numpy())
    with pytest.raises(ValueError, match="pivot_right"):
        market_structure_advanced(frame, pivot_right=0)
