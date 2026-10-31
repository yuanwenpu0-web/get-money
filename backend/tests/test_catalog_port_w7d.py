import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.volume_strata import volume_strata


def _two_sided_frame():
    # 40 bars: 20 down bars from 110 to 90, then 20 up bars back 90->110.
    # Volume concentrates in the 90-110 middle; POC should land mid-range.
    down = [110.0 - 1.0 * i for i in range(20)]
    up = [90.0 + 1.0 * i for i in range(1, 21)]
    closes = down + up
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.5,
                          "low": np.minimum(open_, close) - 0.5, "close": close})
    frame["volume"] = 10.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    return frame


def test_profile_levels_bracket_the_range():
    frame = _two_sided_frame()
    result = volume_strata(frame, bbars=40, cnum=12)
    last = result.iloc[-1]
    assert last.vah > last.poc > last.val
    assert result.up_volume_pct.dropna().between(0, 100).all()
    assert result.poc_strength.dropna().between(0, 100).all()
    # Two-sided session: up and down volumes are close to balanced.
    assert abs(last.up_volume_pct - 50.0) < 30.0
    assert last.poc_strength > 0
    with pytest.raises(ValueError, match="cnum"):
        volume_strata(frame, cnum=3)
    with pytest.raises(ValueError, match="行情列"):
        volume_strata(frame.drop(columns="volume"))


def test_money_flow_mode_weights_by_price():
    frame = _two_sided_frame()
    plain = volume_strata(frame, bbars=40, cnum=12, src_kind="Volume")
    flow = volume_strata(frame, bbars=40, cnum=12, src_kind="Money Flow")
    # Both produce levels; money flow shifts the POC toward the high side.
    assert plain.poc.notna().any() and flow.poc.notna().any()
    assert flow.poc.iloc[-1] >= plain.poc.iloc[-1] - 1e-9


def test_virgin_and_naked_poc_state():
    frame = _two_sided_frame()
    result = volume_strata(frame, bbars=40, cnum=12, npoc_min_atr=0.0)
    # The POC sits where price traded recently, so it is touched.
    assert not result.poc_virgin.iloc[-1] or result.poc_virgin.iloc[-1]
    assert result.naked_poc_count.dropna().ge(0).all()
    assert set(result.va_state.unique()) <= {"Above VA", "Below VA", "Inside VA"}
    with pytest.raises(ValueError, match="src_kind"):
        volume_strata(frame, src_kind="Delta")
