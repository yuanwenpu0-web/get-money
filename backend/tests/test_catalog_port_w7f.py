import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.crypto_amt_ict_smc_companion_v1 import \
    crypto_amt_ict_smc_companion_v1


def _setup_frame():
    # Base 100 for 30 bars, then a displacement bull candle opening an FVG,
    # an OB window with an earlier bear candle, and a later mitigation.
    closes = [100.0] * 30
    closes += [100.0, 106.0, 108.0]     # bars 30-32: bull displacement opens FVG
    closes += [108.5] * 4               # ride above the gap
    closes += [103.0]                   # bar 37: close below FVG bottom -> mitigated
    closes += [109.0] * 4               # reclaim
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    # Bar 27 is a bear candle (the OB candidate) sitting in the search window.
    opens = open_.to_numpy(dtype=float).copy()
    closes_arr = close.to_numpy(dtype=float).copy()
    frame = pd.DataFrame({"open": open_, "high": np.maximum(opens, closes_arr) + 0.2,
                          "low": np.minimum(opens, closes_arr) - 0.2,
                          "close": close})
    # Carve a swing high at bar 12 (unique strict pivot both sides).
    frame.iloc[12, frame.columns.get_loc("high")] = 101.5
    frame.iloc[11, frame.columns.get_loc("high")] = 100.3
    frame.iloc[13, frame.columns.get_loc("high")] = 100.3
    # Carve a swing low at bar 18.
    frame.iloc[18, frame.columns.get_loc("low")] = 98.5
    frame.iloc[17, frame.columns.get_loc("low")] = 99.7
    frame.iloc[19, frame.columns.get_loc("low")] = 99.7
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min",
                                tz="UTC")
    return frame


def test_fvg_opening_mitigation_and_ifvg_flip():
    frame = _setup_frame()
    result = crypto_amt_ict_smc_companion_v1(frame, fvg_min_atr_mult=0.1)
    # Bar 32 (third candle of the displacement leg) opens a bull FVG whose
    # bottom is the high of bar 30.
    assert result.new_bull_fvg.iloc[32]
    assert result.range_low.notna().any() or result.range_high.notna().any()
    # Bar 37 closes below the FVG bottom -> mitigation + bear IFVG born.
    assert result.fvg_mitigated_bull.iloc[37]
    assert result.new_bear_ifvg.iloc[37]
    with pytest.raises(ValueError, match="range_len"):
        crypto_amt_ict_smc_companion_v1(frame, range_len=2)
    with pytest.raises(ValueError, match="行情列"):
        crypto_amt_ict_smc_companion_v1(frame.drop(columns="open"))


def test_premium_discount_state_and_scores():
    frame = _setup_frame()
    result = crypto_amt_ict_smc_companion_v1(frame, fvg_min_atr_mult=0.1)
    settled = result.iloc[35:]
    assert set(result.pd_zone.unique()) <= {"PREMIUM", "DISCOUNT", "EQUILIBRIUM", "NA"}
    # Riding above the range midpoint reads premium.
    assert settled.pd_zone.isin(["PREMIUM", "EQUILIBRIUM"]).all()
    assert result.bull_setup_score.dropna().ge(0).all()
    assert result.bear_setup_score.dropna().ge(0).all()
    # Setups are exclusive by construction.
    assert not (result.bull_setup & result.bear_setup).any()
    with pytest.raises(ValueError, match="min_setup_score"):
        crypto_amt_ict_smc_companion_v1(frame, min_setup_score=2)


def test_cisd_crosses_the_last_opposite_open():
    # Flat then a bear candle (records its open), then a close above it.
    closes = [100.0] * 10 + [99.0, 100.5, 101.0]
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_,
                          "high": np.maximum(open_, close) + 0.2,
                          "low": np.minimum(open_, close) - 0.2,
                          "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="15min",
                                tz="UTC")
    result = crypto_amt_ict_smc_companion_v1(frame)
    # Bar 10 is the bear candle (open 100.0); bar 11 closes above it -> BULL CISD.
    assert result.bull_cisd.iloc[11]
    assert not result.bear_cisd.any()
    with pytest.raises(ValueError, match="max_cisd_marks"):
        crypto_amt_ict_smc_companion_v1(frame, max_cisd_marks=0)
