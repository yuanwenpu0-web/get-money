import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.coilforge_zones_v1_2_2 import coilforge_zones_v1_2_2


def _coiling_frame():
    # 300 bars: sideways coil (±0.6 noise around 100) then a violent breakout.
    rng = np.random.default_rng(17)
    n = 300
    close = pd.Series(100.0 + rng.normal(0, 0.3, n).cumsum() * 0.05)
    breakout = np.full(n, 0.0)
    breakout[260:] = np.arange(260, n) - 259  # ramp up at the end
    close = close + breakout * 1.2
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.15,
                          "low": np.minimum(open_, close) - 0.15, "close": close})
    frame["volume"] = np.where(np.arange(n) > 260, 200.0, 100.0)
    frame.index = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return frame


def test_zone_forms_in_a_coil_and_ends_on_trend():
    frame = _coiling_frame()
    result = coilforge_zones_v1_2_2(frame, sensitivity="Aggressive",
                                    min_zone_bars=3, avoid_contrary_read=False)
    assert result.in_zone.any()
    assert result.zone_score.dropna().between(0, 80).all()
    valid = result.dropna(subset=["zone_high"])
    assert (valid.zone_high >= valid.zone_low).all()
    # Breakout fires at the end of the ramp.
    assert result.breakout_up.any()
    # Bias labels stay in the known set.
    assert set(result.zone_bias.unique()) <= {
        "", "Neutral", "Up", "Down", "Up / Continuation", "Down / Continuation"}
    with pytest.raises(ValueError, match="sensitivity"):
        coilforge_zones_v1_2_2(frame, sensitivity="Turbo")
    with pytest.raises(ValueError, match="行情列"):
        coilforge_zones_v1_2_2(frame.drop(columns="volume"))


def test_sensitivity_presets_change_the_thresholds():
    frame = _coiling_frame()
    strict = coilforge_zones_v1_2_2(frame, sensitivity="Strict", min_zone_bars=3)
    loose = coilforge_zones_v1_2_2(frame, sensitivity="Aggressive", min_zone_bars=3)
    # Loose preset admits at least as many candidate bars.
    assert loose.in_zone.sum() >= strict.in_zone.sum()
    with pytest.raises(ValueError, match="custom_min_score"):
        coilforge_zones_v1_2_2(frame, sensitivity="Custom", custom_min_score=101)


def test_module_toggles_rescore():
    frame = _coiling_frame()
    all_on = coilforge_zones_v1_2_2(frame, sensitivity="Aggressive",
                                    min_zone_bars=3)
    modules_off = coilforge_zones_v1_2_2(frame, sensitivity="Aggressive",
                                         min_zone_bars=3, use_atr_compression=False,
                                         use_adx_filter=False, use_volume=False,
                                         use_impulse=False, use_historical=False)
    # Disabling modules floors their contributions: scores never decrease
    # below the all-on run's module share.
    assert modules_off.zone_score.dropna().min() >= all_on.zone_score.dropna().min()
    # Impulse direction column stays integer and bounded.
    assert all_on.impulse_dir.isin([-1, 0, 1]).all()
    with pytest.raises(ValueError, match="breakout_quality"):
        coilforge_zones_v1_2_2(frame, breakout_quality="Loose")
