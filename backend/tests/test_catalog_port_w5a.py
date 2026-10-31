import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.elder_ray_pressure_engine import elder_ray_pressure_engine


def _frame(closes):
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 10.0})


def test_classic_elder_pressures_match_the_definition():
    # Rising closes put every bar fully above a lagging SMA basis.
    closes = [100.0 + 2.0 * step for step in range(20)]
    frame = _frame(closes)
    result = elder_ray_pressure_engine(frame, engine="Classic Elder", length=3,
                                       ma_type="SMA", normalization="Raw",
                                       smooth_length=1, extreme_lookback=20)
    expected_basis = pd.Series(closes).rolling(3).mean()  # = close - 2 on linear data
    assert result.basis.iloc[10] == pytest.approx(expected_basis.iloc[10])
    # bull = high - basis = 2.5, bear = low - basis = 1.5 (still positive: the
    # whole bar sits above the lagging MA), net = 4 and constant -> impulse 0.
    assert result.bull.iloc[10] == pytest.approx(2.5)
    assert result.bear.iloc[10] == pytest.approx(1.5)
    assert result.net.iloc[10] == pytest.approx(4.0)
    assert result.bar_position.iloc[-1] == "Above"
    assert result.regime.iloc[-1] == "Bull Confirmed"  # basis rising and net > 0
    assert result.pressure_state.iloc[-1] == "Balanced"  # impulse is exactly 0
    # Convex acceleration turns the same uptrend into a Bull Expansion state.
    curve = _frame([100.0 + 0.4 * step * step for step in range(20)])
    convex = elder_ray_pressure_engine(curve, length=3, ma_type="SMA",
                                       normalization="Raw", smooth_length=1,
                                       extreme_lookback=20)
    assert convex.pressure_state.iloc[-1] == "Bull Expansion"
    assert convex.impulse.iloc[-1] > 0
    with pytest.raises(ValueError, match="engine"):
        elder_ray_pressure_engine(frame, engine="Turbo")
    with pytest.raises(ValueError, match="length"):
        elder_ray_pressure_engine(frame, length=1)


def test_atr_normalization_scales_by_true_range():
    closes = [100.0 + step for step in range(30)]
    frame = _frame(closes)
    raw = elder_ray_pressure_engine(frame, length=3, ma_type="SMA", normalization="Raw",
                                    norm_length=5, smooth_length=1, extreme_lookback=20)
    scaled = elder_ray_pressure_engine(frame, length=3, ma_type="SMA",
                                       normalization="ATR", norm_length=5,
                                       smooth_length=1, extreme_lookback=20)
    atr5 = (pd.concat([raw.bull, raw.bear], axis=1))  # placeholder to keep imports tidy
    ratio = scaled.bull.iloc[-1] / raw.bull.iloc[-1]
    assert 0.0 < ratio < 1.0  # ATR is positive and larger than 1 price unit here
    assert scaled.bear.iloc[-1] == pytest.approx(raw.bear.iloc[-1] * ratio, rel=1e-9)


def test_directional_elder_clamps_and_close_weight_weights():
    closes = [100.0, 104.0, 108.0, 104.0, 100.0, 96.0, 100.0, 104.0]
    frame = _frame(closes)
    classic = elder_ray_pressure_engine(frame, engine="Classic Elder", length=2,
                                        ma_type="SMA", normalization="Raw",
                                        smooth_length=1, extreme_lookback=20)
    directional = elder_ray_pressure_engine(frame, engine="Directional Elder", length=2,
                                            ma_type="SMA", normalization="Raw",
                                            smooth_length=1, extreme_lookback=20)
    # Directional keeps only the positive bull side and the negative bear side.
    assert directional.bull.iloc[5] == pytest.approx(max(classic.bull.iloc[5], 0.0))
    assert directional.bear.iloc[5] == pytest.approx(min(classic.bear.iloc[5], 0.0))
    weighted = elder_ray_pressure_engine(frame, engine="Close-Weighted Elder", length=2,
                                         ma_type="SMA", normalization="Raw",
                                         smooth_length=1, extreme_lookback=20)
    # Bar 5 is a down bar with clv = 0 (close at the low half-boundary): bull is
    # killed by the 0 clamp, bear keeps weight (1 - 0) / 2 = 0.5.
    assert weighted.bull.iloc[5] == pytest.approx(0.0)
    assert weighted.bear.iloc[5] == pytest.approx(directional.bear.iloc[5] * 0.5)
    with pytest.raises(ValueError, match="normalization"):
        elder_ray_pressure_engine(frame, normalization="ZScore")


def test_exhaustion_and_divergence_flags_exist_and_are_exclusive():
    rng = np.random.default_rng(2)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    frame = _frame(closes)
    result = elder_ray_pressure_engine(frame, extreme_lookback=50, pivot_left=2,
                                       pivot_right=2)
    # Exhaustion flags are mutually exclusive by construction.
    assert not (result.bull_exhaustion & result.bear_exhaustion).any()
    assert result.bull_rank.dropna().between(0, 100).all()
    states = {"Bull Expansion", "Bull Deceleration", "Bear Expansion",
              "Bear Deceleration", "Balanced", "Bull Exhaustion",
              "Bear Exhaustion", "Dual Exhaustion"}
    assert set(result.pressure_state.unique()) <= states
    assert set(result.regime.unique()) <= {"Bull Confirmed", "Bull Pullback",
                                           "Bear Confirmed", "Bear Rally", "Neutral"}
    off = elder_ray_pressure_engine(frame, show_divergence=False, pivot_left=2,
                                    pivot_right=2, extreme_lookback=50)
    assert not off.bullish_divergence.any() and not off.bearish_divergence.any()
    with pytest.raises(ValueError, match="extreme_lookback"):
        elder_ray_pressure_engine(frame, extreme_lookback=5)
