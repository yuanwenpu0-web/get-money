import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.smc_toolkit import smc_toolkit


def _zigzag_frame(up=True):
    # Rising zigzag with 6-bar legs: strict pivots at each leg end.
    closes = []
    level = 100.0
    for cycle in range(4):
        for _ in range(6):
            level += 3.0 if up else -3.0
            closes.append(level)
        for _ in range(6):
            level -= 2.0 if up else -2.0
            closes.append(level)
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2,
                          "close": close})
    frame["volume"] = 100.0
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min",
                                tz="UTC")
    for k in range(4):
        top_bar = k * 12 + 5
        frame.iloc[top_bar, frame.columns.get_loc("high")] += 0.6
        frame.iloc[k * 12 + 11, frame.columns.get_loc("low")] -= 0.6
    return frame


def test_bos_and_choch_track_the_trend():
    frame = _zigzag_frame()
    result = smc_toolkit(frame, swing_length=4, fvg_min_size_atr=0.1)
    assert result.bos_bullish.any()
    assert result.trend_bullish.dtype == bool
    # Every CHoCH flips the trend right at that bar.
    choch_up = result.index[result.choch_bullish]
    for bar_pos in [frame.index.get_loc(i) for i in choch_up]:
        assert result.trend_bullish.iloc[bar_pos]
    with pytest.raises(ValueError, match="swing_length"):
        smc_toolkit(frame, swing_length=1)
    with pytest.raises(ValueError, match="行情列"):
        smc_toolkit(frame.drop(columns="volume"))


def test_fvg_detection_and_fill():
    frame = _zigzag_frame()
    result = smc_toolkit(frame, swing_length=4, fvg_min_size_atr=0.1)
    # At least one gap opens in this structure and eventually fills.
    opened = result.new_bull_fvg | result.new_bear_fvg
    assert opened.any()
    assert result.fvg_filled.any()
    # Active counts never exceed the cap.
    assert result.active_bull_fvg.le(5).all()
    with pytest.raises(ValueError, match="fvg_min_size_atr"):
        smc_toolkit(frame, fvg_min_size_atr=0.05)


def test_order_blocks_require_volume_filter_toggle():
    frame = _zigzag_frame()
    frame["volume"] = 1.0  # below its own SMA -> filtered
    filtered = smc_toolkit(frame, swing_length=4, ob_volume_filter=True,
                           fvg_min_size_atr=0.1)
    assert not filtered.new_bull_ob.any()
    assert not filtered.new_bear_ob.any()
    unfiltered = smc_toolkit(frame, swing_length=4, ob_volume_filter=False,
                             fvg_min_size_atr=0.1)
    assert unfiltered.new_bull_ob.any() or unfiltered.new_bear_ob.any()
    # Mitigations only happen after creation (may lag if price never returns).
    assert unfiltered.ob_mitigated.any() or unfiltered.active_bull_ob.any() \
        or unfiltered.active_bear_ob.any()
    with pytest.raises(ValueError, match="ob_max_count"):
        smc_toolkit(frame, ob_max_count=0)


def test_equal_pivots_form_liquidity_levels():
    # Two equal highs then a wick sweep above with a close back below.
    rows = 80
    close = pd.Series(100.0, index=pd.date_range("2026-01-01", periods=rows,
                                                 freq="15min", tz="UTC"))
    close.iloc[20:26] = 105.0
    close.iloc[45:51] = 105.0
    close.iloc[70] = 104.0
    open_ = close.shift(1).fillna(100.0)
    frame = pd.DataFrame({"open": open_, "high": close + 0.2, "low": close - 0.2,
                          "close": close})
    frame["volume"] = 100.0
    frame.iloc[25, frame.columns.get_loc("high")] = 105.9   # pivot 1
    frame.iloc[50, frame.columns.get_loc("high")] = 105.9   # pivot 2 (equal)
    frame.iloc[70, frame.columns.get_loc("high")] = 106.5   # the sweep wick
    result = smc_toolkit(frame, swing_length=4, liq_tolerance_atr=0.5,
                         liq_lookback=60, fvg_min_size_atr=5.0)
    assert result.new_bsl.any()
    assert result.bsl_swept.any()
    assert result.active_bsl.le(10).all()
    with pytest.raises(ValueError, match="liq_lookback"):
        smc_toolkit(frame, liq_lookback=5)
