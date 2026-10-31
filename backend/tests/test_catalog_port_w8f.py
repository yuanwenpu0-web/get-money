import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.money_flow_delta_profile import money_flow_delta_profile


def _oscillating_frame(n=240):
    index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 4.0 * np.sin(t / 25.0), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": close + 0.4, "low": close - 0.4,
                          "close": close}, index=index)
    frame["volume"] = 100.0
    return frame


def test_poc_vah_val_bracket_and_levels_ordered():
    frame = _oscillating_frame()
    result = money_flow_delta_profile(frame, lookback=100, rows=20)
    last = result.iloc[-1]
    assert last.vah > last.poc > last.val
    assert result.vah.dropna().gt(result.poc.dropna()).all()
    assert result.poc.dropna().gt(result.val.dropna()).all()
    assert result.hvn_count.iloc[-1] >= 1  # POC row itself is an HVN
    assert result.profile_ready.iloc[-1]
    # Early bars have no profile.
    assert not result.profile_ready.iloc[:50].any()
    with pytest.raises(ValueError, match="lookback"):
        money_flow_delta_profile(frame, lookback=5)
    with pytest.raises(ValueError, match="行情列"):
        money_flow_delta_profile(frame.drop(columns="volume"))


def test_binary_bar_polarity_flips_the_delta():
    frame = _oscillating_frame()
    prop = money_flow_delta_profile(frame, lookback=100, rows=20,
                                    delta_mode="Proportional")
    binary = money_flow_delta_profile(frame, lookback=100, rows=20,
                                      delta_mode="Binary",
                                      binary_polarity="Bar Polarity")
    # Both modes produce profiles; the delta split differs.
    assert not prop.net_delta.dropna().equals(binary.net_delta.dropna())
    assert prop.net_delta.dropna().between(-1e12, 1e12).all()
    with pytest.raises(ValueError, match="delta_mode"):
        money_flow_delta_profile(frame, delta_mode="Hybrid")


def test_volume_mode_and_decay_change_total_flow():
    frame = _oscillating_frame()
    flow = money_flow_delta_profile(frame, lookback=100, rows=20,
                                    src_kind="Money Flow")
    vol = money_flow_delta_profile(frame, lookback=100, rows=20,
                                   src_kind="Volume")
    decayed = money_flow_delta_profile(frame, lookback=100, rows=20,
                                       use_decay=True, decay_factor=0.05)
    # Money-flow totals scale with price (~100x) while volume totals do not.
    assert flow.total_flow.dropna().mean() > vol.total_flow.dropna().mean() * 10
    assert decayed.total_flow.dropna().mean() < flow.total_flow.dropna().mean()
    with pytest.raises(ValueError, match="src_kind"):
        money_flow_delta_profile(frame, src_kind="Delta")
    with pytest.raises(ValueError, match="decay_factor"):
        money_flow_delta_profile(frame, decay_factor=0.5)


def test_lvn_zones_split_around_poc():
    frame = _oscillating_frame()
    result = money_flow_delta_profile(frame, lookback=100, rows=20,
                                      lvn_thresh=0.5, lvn_min_rows=2)
    last = result.iloc[-1]
    # With a very high LVN threshold almost every row is an LVN, so the
    # merged zones hug the POC from both sides.
    if not np.isnan(last.supply_zone_top):
        assert last.supply_zone_bottom > last.poc
    if not np.isnan(last.demand_zone_top):
        assert last.demand_zone_top < last.poc
    with pytest.raises(ValueError, match="hvn_thresh"):
        money_flow_delta_profile(frame, hvn_thresh=0.2)
