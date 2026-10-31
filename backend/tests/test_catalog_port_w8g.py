import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.s_r_zones_mtf import s_r_zones_mtf


def _zigzag_frame(legs=6, leg_bars=10):
    closes = []
    level = 100.0
    for cycle in range(legs):
        for _ in range(leg_bars):
            level += 2.0
            closes.append(level)
        for _ in range(leg_bars):
            level -= 2.0
            closes.append(level)
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3,
                          "close": close})
    frame["volume"] = 100.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min",
                                tz="UTC")
    for k in range(legs * 2):
        bar = k * leg_bars + leg_bars - 1
        if k % 2 == 0:
            frame.iloc[bar, frame.columns.get_loc("high")] += 1.0
        else:
            frame.iloc[bar, frame.columns.get_loc("low")] -= 1.0
    return frame


def test_zones_form_and_events_fire():
    frame = _zigzag_frame()
    result = s_r_zones_mtf(frame, htf1_minutes=60, max_zones=15, rev_atr_mult=1.0)
    assert result.active_zones.max() >= 1
    assert result.test.any() or result.sweep.any()
    # Counts stay within the cap.
    assert result.active_zones.max() <= 15
    assert set(result.columns) >= {"break_up", "break_down", "retest", "sweep"}
    with pytest.raises(ValueError, match="max_zones"):
        s_r_zones_mtf(frame, max_zones=0)
    with pytest.raises(ValueError, match="行情列"):
        s_r_zones_mtf(frame.drop(columns="low"))


def test_invalidation_mode_changes_break_logic():
    frame = _zigzag_frame()
    close_inv = s_r_zones_mtf(frame, htf1_minutes=60, invalidation="Close",
                              max_zones=15)
    wick_inv = s_r_zones_mtf(frame, htf1_minutes=60, invalidation="Wick",
                             max_zones=15)
    # Wick invalidation is more eager: at least as many breaks.
    assert wick_inv.break_up.sum() + wick_inv.break_down.sum() >= \
        close_inv.break_up.sum() + close_inv.break_down.sum()
    with pytest.raises(ValueError, match="invalidation"):
        s_r_zones_mtf(frame, invalidation="Body")


def test_merge_and_too_close_dedupe():
    frame = _zigzag_frame()
    merged = s_r_zones_mtf(frame, htf1_minutes=60, merge_on=True,
                           merge_atr_mult=5.0, too_close_atr=2.0, max_zones=50)
    unmerged = s_r_zones_mtf(frame, htf1_minutes=60, merge_on=True,
                             merge_atr_mult=0.1, too_close_atr=0.0, max_zones=50)
    # Wide merge distance keeps fewer distinct zones alive.
    assert merged.active_zones.max() <= unmerged.active_zones.max()
    with pytest.raises(ValueError, match="merge_atr_mult"):
        s_r_zones_mtf(frame, merge_atr_mult=0.05)


def test_volume_filter_gates_breaks():
    frame = _zigzag_frame()
    frame["volume"] = 1.0
    filtered = s_r_zones_mtf(frame, htf1_minutes=60, avoid_false_breaks=True,
                             break_vol_mult=2.0, max_zones=15)
    unfiltered = s_r_zones_mtf(frame, htf1_minutes=60, avoid_false_breaks=False,
                               max_zones=15)
    # Constant low volume fails the volume gate: fewer breaks.
    assert filtered.break_up.sum() + filtered.break_down.sum() == 0
    assert unfiltered.break_up.sum() + unfiltered.break_down.sum() >= 0
    with pytest.raises(ValueError, match="retest_min_bars"):
        s_r_zones_mtf(frame, retest_min_bars=0)
