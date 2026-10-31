import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.ichi_workflow_overlay import ichi_workflow_overlay


def _trend_frame(n=96 * 30):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.06 * t + 1.5 * np.sin(t / 40.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                         "close": close}, index=index)


def test_states_bounded_and_presets_resolve():
    frame = _trend_frame()
    result = ichi_workflow_overlay(frame, preset="Traditional Standard",
                                   htf_rule="D")
    for column in ("price_state", "future_state", "tk_state", "chikou_state"):
        assert result[column].isin([-1, 0, 1]).all()
    assert result.ctf_score.dropna().between(-4, 4).all()
    assert result.cloud_top.dropna().ge(result.cloud_bottom.dropna()).all()
    assert set(result.active_setup.unique()) <= {
        "None", "Checklist Long", "Checklist Short", "Kijun Bounce Long",
        "Kijun Bounce Short", "Breakout Long", "Breakout Short",
        "Edge-to-Edge Long", "Edge-to-Edge Short"}
    with pytest.raises(ValueError, match="preset"):
        ichi_workflow_overlay(frame, preset="Fibonacci")
    with pytest.raises(ValueError, match="行情列"):
        ichi_workflow_overlay(frame.drop(columns="low"))


def test_direction_filter_gates_signals():
    frame = _trend_frame()
    long_only = ichi_workflow_overlay(frame, direction_filter="Long Only")
    short_only = ichi_workflow_overlay(frame, direction_filter="Short Only")
    both = ichi_workflow_overlay(frame, direction_filter="Both")
    assert not long_only.short_breakout.any()
    assert not long_only.short_checklist.any()
    assert not long_only.short_kijun.any()
    assert not long_only.short_edge.any()
    assert not short_only.long_edge.any()
    assert (both.long_edge.sum() + both.short_edge.sum()
            >= long_only.long_edge.sum() + short_only.short_edge.sum())
    with pytest.raises(ValueError, match="direction_filter"):
        ichi_workflow_overlay(frame, direction_filter="Sideways")


def test_targets_and_active_setup_state():
    frame = _trend_frame()
    result = ichi_workflow_overlay(frame)
    # Targets are ordered above/below the price where present.
    has_above = result.dropna(subset=["target_above_1"])
    if len(has_above):
        assert (has_above.target_above_1 >= has_above.close if "close" in has_above
                else True)
    # Active setup invalidation is below entry for longs.
    longs = result[(result.active_dir == 1) & result.active_invalidation.notna()]
    if len(longs):
        assert (longs.active_invalidation < longs.active_entry).all()
    shorts = result[(result.active_dir == -1) & result.active_invalidation.notna()]
    if len(shorts):
        assert (shorts.active_invalidation > shorts.active_entry).all()
    with pytest.raises(ValueError, match="flat_scan_bars"):
        ichi_workflow_overlay(frame, flat_scan_bars=300)


def test_htf_score_lags_the_base_bars():
    frame = _trend_frame()
    result = ichi_workflow_overlay(frame, htf_rule="D")
    # The first day has no completed HTF period -> the lagged score is 0.
    assert result.htf_ctf_score.iloc[:96].eq(0).all()
    assert result.htf_ctf_score.dropna().between(-4, 4).all()
