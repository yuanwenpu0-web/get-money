import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.market_structure_pivot_map import \
    market_structure_pivot_map


def _frame(days=4):
    index = pd.date_range("2026-01-01", periods=96 * days, freq="15min", tz="UTC")
    t = np.arange(len(index))
    close = pd.Series(100.0 + 0.05 * t + 3.0 * np.sin(t / 20.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.5, "low": close - 0.5,
                         "close": close}, index=index)


def test_classic_pivots_match_the_formula():
    frame = _frame()
    result = market_structure_pivot_map(frame)
    # Day 2 onwards: yesterday's H/L/C feed the classic formulas.
    day0 = frame.iloc[:96]
    ph, pl = day0["high"].max(), day0["low"].min()
    pc = day0.close.iloc[-1]
    p = (ph + pl + pc) / 3.0
    rng = ph - pl
    tc_raw = 2 * p - (ph + pl) / 2.0
    tc, bc = max(tc_raw, (ph + pl) / 2.0), min(tc_raw, (ph + pl) / 2.0)
    assert result["pivot"].iloc[96 + 10] == pytest.approx(p)
    assert result.tc.iloc[96 + 10] == pytest.approx(tc)
    assert result.bc.iloc[96 + 10] == pytest.approx(bc)
    assert result.level_r1.iloc[96 + 10] == pytest.approx(2 * p - pl)
    assert result.level_s1.iloc[96 + 10] == pytest.approx(2 * p - ph)
    assert result.level_r2.iloc[96 + 10] == pytest.approx(p + rng)
    # Levels published for the whole second day.
    assert result["pivot"].iloc[96:192].notna().all()
    with pytest.raises(ValueError, match="pivot_type"):
        market_structure_pivot_map(frame, pivot_type="Woodie")
    with pytest.raises(ValueError, match="行情列"):
        market_structure_pivot_map(frame.drop(columns="close"))


def test_fibonacci_kind_uses_ratio_multipliers():
    frame = _frame()
    fib = market_structure_pivot_map(frame, pivot_type="Fibonacci")
    day0 = frame.iloc[:96]
    ph, pl = day0["high"].max(), day0["low"].min()
    pc = day0.close.iloc[-1]
    p = (ph + pl + pc) / 3.0
    rng = ph - pl
    assert fib.level_r2.iloc[96 + 10] == pytest.approx(p + rng * 0.618)
    assert fib.level_s3.iloc[96 + 10] == pytest.approx(p - rng)
    # CPR is the same regardless of kind.
    assert fib["pivot"].iloc[96 + 10] == pytest.approx(p)


def test_bias_cpr_width_and_projections():
    frame = _frame()
    result = market_structure_pivot_map(frame)
    settled = result.iloc[96:]
    assert set(result.cpr_type.unique()) <= {"", "Narrow", "Normal", "Wide"}
    assert set(result.bias.unique()) <= {"Neutral", "Bullish", "Bearish"}
    # Near levels bracket the close.
    valid = settled.dropna(subset=["near_above", "near_below"])
    assert (valid.near_above >= valid.near_below).all()
    assert valid.dist_above_atr.ge(0).all() and valid.dist_below_atr.ge(0).all()
    # Projection levels exist by name.
    assert result.level_dayopen.notna().any()
    with pytest.raises(ValueError, match="projection_range"):
        market_structure_pivot_map(frame, projection_range="Weekly")
    with pytest.raises(ValueError, match="adr_len"):
        market_structure_pivot_map(frame, adr_len=0)


def test_confluence_clusters_count_members():
    frame = _frame()
    tight = market_structure_pivot_map(frame, confl_atr_mult=0.01,
                                       confl_min_count=2)
    loose = market_structure_pivot_map(frame, confl_atr_mult=5.0,
                                       confl_min_count=2)
    # A wider cluster tolerance merges at least as many bars into zones.
    assert loose.confluence_count.sum() >= tight.confluence_count.sum()
    assert tight.confluence_count.min() >= 0
    off = market_structure_pivot_map(frame, show_confluence=False)
    assert off.confluence_count.eq(0).all()
