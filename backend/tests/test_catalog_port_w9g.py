"""Fib Reaction Memory 移植验证 — 参考腿、微观反应、六档分数/状态/NEXT。

按公式手算的期望值校验：腿几何、反应触发、分数边界、状态合法集、NEXT 选择。
"""

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.fib_reaction_memory import fib_reaction_memory
from app.catalog_indicators._pine import atr


def _frame(closes, spread=0.4):
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    close = pd.Series(np.asarray(closes, dtype=float), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    data = {"open": open_, "high": close + spread, "low": close - spread, "close": close}
    return pd.DataFrame(data, index=index)


def _bull_leg_frame():
    """Clean bull leg: a trough at ~100 then a rise to ~150, then a pullback.

    The trough/high values are made unique (101/148 neighbours) so the strict
    structural pivots actually confirm instead of being swallowed by equal bars.
    """
    closes = []
    closes += list(np.linspace(120, 100, 12))   # dip to the structural low (100)
    closes += list(np.linspace(101, 150, 20))   # impulse up to the structural high (150)
    closes += list(np.linspace(148, 130, 10))   # pullback so the high confirms
    return _frame(closes)


def _bear_reaction_frame():
    """Bear leg 150->116 with a bounce (130->134->down) that forms a micro reaction."""
    closes = []
    closes += list(np.linspace(100, 150, 30))
    closes += [148, 144, 138, 132, 130, 134, 136, 128, 122, 116]
    closes += list(np.linspace(119, 140, 12))
    return _frame(closes)


VALID_STATES = {"", "APPROACHING", "TESTING", "REJECTED", "ACCEPTED", "RECLAIMED"}


def test_fib_defaults_run_and_columns():
    frame = _bull_leg_frame()
    out = fib_reaction_memory(frame)
    expected = {
        "leg_active", "leg_new", "leg_invalidated", "leg_a", "leg_a_bar", "leg_b",
        "leg_b_bar", "leg_direction", "leg_range", "current_retracement",
        "pullback_deepest", "reaction_high", "reaction_low", "next_index", "next_fib",
        "next_score", "strong_score", "medium_score",
    }
    for name in ("23.6", "38.2", "50.0", "61.8", "78.6", "100.0"):
        expected |= {f"score_{name}", f"hold_rate_{name}", f"price_{name}",
                     f"state_{name}", f"is_next_{name}"}
    assert expected.issubset(set(out.columns))
    assert len(out) == len(frame)


def test_fib_bull_leg_geometry():
    frame = _bull_leg_frame()
    out = fib_reaction_memory(frame, swing_pivot_len=4)
    assert out.leg_new.any()
    active = out[out.leg_active]
    assert (active.leg_direction == 1).any()
    leg = active[active.leg_direction == 1].iloc[0]
    assert leg.leg_a < leg.leg_b
    assert leg.leg_range > 0
    # current retracement is normalized 0..1 along the leg
    valid = out.current_retracement.dropna()
    assert valid.between(-0.2, 1.2).all()


def test_fib_reaction_triggers_and_scores():
    frame = _bear_reaction_frame()
    out = fib_reaction_memory(
        frame, swing_pivot_len=4, reaction_pivot_len=2,
        min_reaction_atr=0.05, fib_tolerance=0.06)
    reactions = int(out.reaction_high.sum() + out.reaction_low.sum())
    assert reactions >= 1
    # the 38.2% level sits near the bounce, so its memory score rises above 0
    assert out["score_38.2"].max() > 0
    # every reaction flag lands on a bar (flag column is boolean)
    assert out.reaction_high.dtype == bool and out.reaction_low.dtype == bool


def test_fib_scores_bounded_and_states_valid():
    frame = _bear_reaction_frame()
    out = fib_reaction_memory(
        frame, swing_pivot_len=4, reaction_pivot_len=2,
        min_reaction_atr=0.05, fib_tolerance=0.06)
    for name in ("23.6", "38.2", "50.0", "61.8", "78.6", "100.0"):
        scores = out[f"score_{name}"].dropna()
        assert scores.between(0, 100, inclusive="both").all()
        states = set(out[f"state_{name}"].dropna().unique())
        assert states.issubset(VALID_STATES)


def test_fib_next_level_selected_inside_leg():
    frame = _bull_leg_frame()
    out = fib_reaction_memory(frame, swing_pivot_len=4, highlight_next=True)
    active = out[out.leg_active]
    assert active.next_index.notna().any()
    # next fib must stay ahead of the current retracement
    sel = active.dropna(subset=["next_index", "current_retracement"])
    assert (sel.next_fib >= sel.current_retracement - 1e-9).all()
    assert out["is_next_23.6"].dtype == bool


def test_fib_disable_levels_drops_them_from_next():
    frame = _bull_leg_frame()
    full = fib_reaction_memory(frame, swing_pivot_len=4)
    partial = fib_reaction_memory(
        frame, swing_pivot_len=4, show_236=False, show_382=False,
        show_500=False, show_618=False, show_786=False, show_100=False)
    # with every level disabled, no NEXT can be selected
    assert partial.next_index.notna().sum() == 0
    assert full.next_index.notna().any()
