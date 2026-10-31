import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.ichi_2d_overlay_profiles import (
    SYMBOL_PROFILES_2D, ichi_2d_overlay_profiles)


def _frame(n=96 * 20):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.05 * t + 1.5 * np.sin(t / 40.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                         "close": close}, index=index)


def test_2d_profile_overrides_and_confirmation_rule():
    frame = _frame()
    btc = ichi_2d_overlay_profiles(frame, symbol="BTCUSD")
    assert btc.using_profile.iloc[-1]
    assert btc.profile_label.iloc[-1] == "BTC 2D Long"
    assert btc.effective_conversion.iloc[-1] == 20
    assert btc.effective_base.iloc[-1] == 73
    assert btc.effective_span_b.iloc[-1] == 96
    assert btc.effective_displacement.iloc[-1] == 36
    assert btc.confirmation_rule.iloc[-1] == "D"
    with pytest.raises(ValueError, match="preset"):
        ichi_2d_overlay_profiles(frame, preset="Fibonacci")


def test_dbc_and_lqd_use_weekly_confirmation():
    frame = _frame()
    dbc = ichi_2d_overlay_profiles(frame, symbol="DBC")
    lqd = ichi_2d_overlay_profiles(frame, symbol="LQD")
    assert dbc.confirmation_rule.iloc[-1] == "W"
    assert lqd.confirmation_rule.iloc[-1] == "W"
    assert lqd.effective_direction.iloc[-1] == "Short Only"
    # Short-only profile blocks longs.
    assert not lqd.long_edge.any()
    assert not lqd.long_checklist.any()


def test_fallback_uses_input_rule_and_preset():
    frame = _frame()
    other = ichi_2d_overlay_profiles(frame, symbol="XYZUSD",
                                     preset="Traditional Doubled",
                                     htf_rule="M", direction_filter="Both")
    assert not other.using_profile.iloc[-1]
    assert other.profile_label.iloc[-1] == "Manual"
    assert other.confirmation_rule.iloc[-1] == "M"
    assert other.effective_conversion.iloc[-1] == 18
    assert other.effective_direction.iloc[-1] == "Both"
    with pytest.raises(ValueError, match="use_symbol_profiles"):
        ichi_2d_overlay_profiles(frame, use_symbol_profiles=0)


def test_setup_code_and_two_day_flag():
    frame = _frame()
    result = ichi_2d_overlay_profiles(frame, symbol="GLD")
    assert result.setup_code.isin([0, 1, -1, 2, -2, 3, -3, 4, -4]).all()
    assert result.chart_is_two_day.all()
    off = ichi_2d_overlay_profiles(frame, symbol="GLD", chart_is_two_day=False)
    assert not off.chart_is_two_day.any()
    with pytest.raises(ValueError, match="chart_is_two_day"):
        ichi_2d_overlay_profiles(frame, chart_is_two_day="yes")


def test_2d_table_is_consistent_and_differs_from_1d():
    from app.catalog_indicators.ichi_workflow_overlay_profiles import \
        SYMBOL_PROFILES
    for ticker, row in SYMBOL_PROFILES_2D.items():
        label, conv, base, span_b, disp, rule, direction = row
        assert ticker.isupper()
        assert rule in ("D", "W", "M")
        assert label.split()[0] == ticker.replace("USD", "") and "2D" in label
        if label.endswith("Short"):
            assert direction == "Short Only"
        else:
            assert direction == "Long Only"
    # 2D 档案与 1D 档案的长度必须不同（专为 2D 拟合）。
    shared = set(SYMBOL_PROFILES_2D) & set(SYMBOL_PROFILES)
    assert shared
    for ticker in shared:
        one_d = SYMBOL_PROFILES[ticker][1:5]
        two_d = SYMBOL_PROFILES_2D[ticker][1:5]
        assert one_d != two_d, ticker
