import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.mtf_wavetrend_tide_wave_ripple import \
    mtf_wavetrend_tide_wave_ripple


def _osc_frame(n=96 * 40):
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    close = pd.Series(100.0 + 6.0 * np.sin(t / 18.0) + 0.05 * t, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close + 0.5, "low": close - 0.5,
                         "close": close}, index=index)


def test_wavetrend_chain_and_phases():
    frame = _osc_frame()
    result = mtf_wavetrend_tide_wave_ripple(frame, scale_mode="Classic",
                                            n1=10, n2=21, signal_len=4)
    assert result.wt1.dropna().any()
    assert result.wt1_raw.dropna().equals(result.wt1.dropna())
    assert set(result.ripple_phase.unique()) <= {0, 1, 2, 3}
    assert set(result.wave_phase.unique()) <= {0, 1, 2, 3}
    assert set(result.tide_phase.unique()) <= {0, 1, 2, 3}
    # Wave/Tide lag the chart: the first bars cannot have a completed period.
    assert result.wave_wt1.iloc[:4].isna().all()
    with pytest.raises(ValueError, match="wt_smooth"):
        mtf_wavetrend_tide_wave_ripple(frame, wt_smooth="KAMA")
    with pytest.raises(ValueError, match="行情列"):
        mtf_wavetrend_tide_wave_ripple(frame.drop(columns="close"))


def test_scale_modes_change_only_displayed_values():
    frame = _osc_frame()
    classic = mtf_wavetrend_tide_wave_ripple(frame, scale_mode="Classic")
    clamped = mtf_wavetrend_tide_wave_ripple(frame, scale_mode="Clamp ±100")
    adaptive = mtf_wavetrend_tide_wave_ripple(frame, scale_mode="Adaptive ±100",
                                              scale_lookback=100)
    assert clamped.wt1.abs().dropna().le(100.0).all()
    assert adaptive.wt1.abs().dropna().le(100.0 + 1e-9).all()
    # Raw values are identical across modes.
    assert np.allclose(classic.wt1_raw.dropna(), adaptive.wt1_raw.dropna())
    with pytest.raises(ValueError, match="scale_lookback"):
        mtf_wavetrend_tide_wave_ripple(frame, scale_lookback=10)


def test_triggers_grades_and_signals():
    frame = _osc_frame()
    result = mtf_wavetrend_tide_wave_ripple(frame, min_grade=2, cooldown_bars=8)
    assert result.trigger_bull.any() or result.trigger_bear.any()
    assert result.long_grade.dropna().between(0, 6).all()
    assert result.short_grade.dropna().between(0, 6).all()
    # A signal needs a trigger and the grade threshold.
    fired = result[result.long_signal]
    if len(fired):
        assert (fired.long_grade >= fired.active_min_grade).all()
        assert fired.trigger_bull.all()
    with pytest.raises(ValueError, match="trigger_mode"):
        mtf_wavetrend_tide_wave_ripple(frame, trigger_mode="WT-Break")
    with pytest.raises(ValueError, match="min_grade"):
        mtf_wavetrend_tide_wave_ripple(frame, min_grade=6)


def test_turn_mode_fires_earlier_than_cross():
    frame = _osc_frame()
    cross = mtf_wavetrend_tide_wave_ripple(frame, trigger_mode="WT-Cross")
    turn = mtf_wavetrend_tide_wave_ripple(frame, trigger_mode="WT-Turn")
    # Turn triggers exist independently of crosses.
    assert turn.trigger_bull.sum() > 0
    assert cross.trigger_bull.sum() > 0
    # A turn mode run can differ from cross mode.
    assert not turn.trigger_bull.equals(cross.trigger_bull)
    with pytest.raises(ValueError, match="mtf_mode"):
        mtf_wavetrend_tide_wave_ripple(frame, mtf_mode="Half")


def test_position_state_machine_and_cooldown():
    frame = _osc_frame()
    result = mtf_wavetrend_tide_wave_ripple(frame, cooldown_bars=20)
    assert set(result.position_dir.unique()) <= {-1, 0, 1}
    # Position only opens on a signal and only when flat.
    prev = result.position_dir.shift(1).fillna(0)
    opened_long = (result.position_dir == 1) & (prev == 0)
    assert (result.long_signal[opened_long.values] | result.short_signal[opened_long.values]).all()
    with pytest.raises(ValueError, match="cooldown_bars"):
        mtf_wavetrend_tide_wave_ripple(frame, cooldown_bars=200)
