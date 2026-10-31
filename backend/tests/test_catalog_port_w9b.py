import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.ichi_workflow_overlay_profiles import (
    SYMBOL_PROFILES, ichi_workflow_overlay_profiles)


def _frame(n=96 * 25):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 0.05 * t + 1.5 * np.sin(t / 40.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                         "close": close}, index=index)


def test_profile_overrides_lengths_and_direction():
    frame = _frame()
    btc = ichi_workflow_overlay_profiles(frame, symbol="BTCUSD")
    assert btc.using_profile.iloc[-1]
    assert btc.profile_label.iloc[-1] == "BTC Long"
    assert btc.effective_conversion.iloc[-1] == 22
    assert btc.effective_base.iloc[-1] == 40
    assert btc.effective_span_b.iloc[-1] == 104
    assert btc.effective_displacement.iloc[-1] == 64
    assert btc.effective_direction.iloc[-1] == "Long Only"
    with pytest.raises(ValueError, match="preset"):
        ichi_workflow_overlay_profiles(frame, preset="Fibonacci")


def test_lqd_profile_flips_to_short_only():
    frame = _frame()
    lqd = ichi_workflow_overlay_profiles(frame, symbol="LQD")
    assert lqd.profile_label.iloc[-1] == "LQD Short"
    assert lqd.effective_direction.iloc[-1] == "Short Only"
    # Short-only profile blocks every long setup.
    assert not lqd.long_breakout.any()
    assert not lqd.long_checklist.any()
    assert not lqd.long_kijun.any()
    assert not lqd.long_edge.any()


def test_unknown_symbol_falls_back_to_preset():
    frame = _frame()
    other = ichi_workflow_overlay_profiles(frame, symbol="XYZUSDT",
                                           preset="Traditional Standard",
                                           direction_filter="Both")
    assert not other.using_profile.iloc[-1]
    assert other.profile_label.iloc[-1] == "Manual"
    assert other.effective_conversion.iloc[-1] == 9
    assert other.effective_direction.iloc[-1] == "Both"
    # Disabling profiles has the same effect even for a known symbol.
    off = ichi_workflow_overlay_profiles(frame, symbol="BTCUSD",
                                         use_symbol_profiles=False)
    assert not off.using_profile.iloc[-1]
    with pytest.raises(ValueError, match="use_symbol_profiles"):
        ichi_workflow_overlay_profiles(frame, use_symbol_profiles=1)


def test_setup_code_maps_every_label():
    frame = _frame()
    result = ichi_workflow_overlay_profiles(frame, symbol="ETHUSD")
    assert result.setup_code.isin([0, 1, -1, 2, -2, 3, -3, 4, -4]).all()
    # Code sign matches the setup direction.
    longs = result[result.active_setup.str.endswith("Long")]
    if len(longs):
        assert (longs.setup_code > 0).all()
    shorts = result[result.active_setup.str.endswith("Short")]
    if len(shorts):
        assert (shorts.setup_code < 0).all()
    with pytest.raises(ValueError, match="flat_scan_bars"):
        ichi_workflow_overlay_profiles(frame, flat_scan_bars=300)


def test_profile_table_is_consistent():
    # 每个档案的 label 前缀与 direction 自洽。
    for ticker, (label, conv, base, span_b, disp, direction) in SYMBOL_PROFILES.items():
        assert ticker.isupper()
        assert conv >= 1 and base >= 2 and span_b >= 2 and disp >= 1
        if label.endswith("Short"):
            assert direction == "Short Only"
        else:
            assert direction == "Long Only"
