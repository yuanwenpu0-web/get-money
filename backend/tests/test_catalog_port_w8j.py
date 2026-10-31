import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.oscillator_divergence_zones import \
    oscillator_divergence_zones


def _divergence_frame():
    # Two clean V-troughs separated by a rally. The second trough is LOWER in
    # price but its RSI is HIGHER (regular bullish divergence).
    closes = [100.0 + 0.3 * i for i in range(40)]          # gentle ramp
    closes += [112.0 - 2.5 * i for i in range(1, 9)]       # trough 1 at 92.0
    closes += [92.0 + 2.5 * i for i in range(1, 9)]        # back to 112
    closes += [112.0 - 2.8 * i for i in range(1, 8)]       # trough 2 at 92.4 -> lower
    closes += [90.4 + 2.0 * i for i in range(1, 30)]       # strong recovery
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.3,
                          "low": np.minimum(open_, close) - 0.3, "close": close})
    frame["volume"] = 100.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="1h",
                                tz="UTC")
    return frame


def test_oscillator_values_and_levels():
    frame = _divergence_frame()
    for osc in ("RSI", "CCI", "MFI", "Fisher", "TSI", "STC", "DPO", "Roofing",
                "Cyber"):
        result = oscillator_divergence_zones(frame, osc_type=osc, osc_len=14,
                                             piv_left=5, piv_right=5,
                                             max_range=200, min_range=5,
                                             filter_mode="Off")
        assert result.osc_value.notna().any(), osc
    with pytest.raises(ValueError, match="osc_type"):
        oscillator_divergence_zones(frame, osc_type="Stochastic")
    with pytest.raises(ValueError, match="行情列"):
        oscillator_divergence_zones(frame.drop(columns="close"))


def test_regular_bull_divergence_fires_and_creates_zone():
    frame = _divergence_frame()
    result = oscillator_divergence_zones(frame, osc_type="RSI", osc_len=14,
                                         piv_left=5, piv_right=5, min_range=5,
                                         max_range=200, filter_mode="Off")
    assert result.bull_divergence.any()
    assert result.zone_created.any()
    assert result.bull_zone_count.max() >= 1
    assert result.bull_score.dropna().between(0, 100).all()
    assert set(result.columns) >= {"bear_divergence", "zone_touches_max"}
    with pytest.raises(ValueError, match="max_range"):
        oscillator_divergence_zones(frame, max_range=3)


def test_filter_modes_gate_the_signals():
    frame = _divergence_frame()
    off = oscillator_divergence_zones(frame, filter_mode="Off", min_range=5,
                                      max_range=200)
    fixed = oscillator_divergence_zones(frame, filter_mode="Fixed",
                                        bull_filter_pct=50.0, min_range=5,
                                        max_range=200)
    dynamic = oscillator_divergence_zones(frame, filter_mode="Dynamic Zones",
                                          dyn_sample_len=60, min_range=5,
                                          max_range=200)
    # A tighter fixed threshold can only reduce the signal count.
    assert fixed.bull_divergence.sum() <= off.bull_divergence.sum()
    assert dynamic.bull_divergence.sum() <= off.bull_divergence.sum()
    with pytest.raises(ValueError, match="filter_mode"):
        oscillator_divergence_zones(frame, filter_mode="Adaptive")


def test_hidden_divergence_and_std_filter():
    frame = _divergence_frame()
    hidden = oscillator_divergence_zones(frame, show_hidden=True, min_range=5,
                                         max_range=200, filter_mode="Off")
    no_hidden = oscillator_divergence_zones(frame, show_hidden=False,
                                            min_range=5, max_range=200,
                                            filter_mode="Off")
    assert hidden.bull_hidden.sum() >= no_hidden.bull_hidden.sum()
    std = oscillator_divergence_zones(frame, use_std_filter=True, std_period=50,
                                      std_mult=1.0, min_range=5, max_range=200,
                                      filter_mode="Off")
    assert std.bull_divergence.sum() <= no_hidden.bull_divergence.sum()
    with pytest.raises(ValueError, match="std_period"):
        oscillator_divergence_zones(frame, std_period=5)
