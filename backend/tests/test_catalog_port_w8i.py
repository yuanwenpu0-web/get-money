import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.vein_reversal_zones import vein_reversal_zones


def _reversal_frame():
    # 100 bars flat, a sharp V bottom at bar 120, then a rally: the V bar is
    # labelled a strong support after the lookahead window completes.
    n = 200
    closes = [100.0] * 100
    closes += [100.0 - 2.0 * i for i in range(1, 21)]   # crash to 60
    closes += [60.0 + 2.0 * i for i in range(1, 21)]    # V recovery to 100
    closes += [100.0 + 0.5 * i for i in range(60)]      # rally to 130
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.5,
                          "low": np.minimum(open_, close) - 0.5, "close": close})
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="1h",
                                tz="UTC")
    return frame


def test_labels_are_lookahead_evaluated_and_zones_form():
    frame = _reversal_frame()
    result = vein_reversal_zones(frame, forward_bars=8, extended_bars=16,
                                 target_atr=0.5, strong_atr=1.5,
                                 max_adverse_atr=3.0)
    # Labels only appear once the lookahead window has closed.
    warm = 16 + 1
    assert (result.reversal_label.iloc[:warm] == 0).all()
    assert (result.reversal_label != 0).any()
    assert result.zone_created.any()
    assert result.zone_count.max() >= 1
    assert set(result.reversal_label.unique()) <= {-2, -1, 0, 1, 2}
    with pytest.raises(ValueError, match="forward_bars"):
        vein_reversal_zones(frame, forward_bars=0)
    with pytest.raises(ValueError, match="行情列"):
        vein_reversal_zones(frame.drop(columns="low"))


def test_signal_modes_and_requirement():
    frame = _reversal_frame()
    known = vein_reversal_zones(frame, signal_mode="Known Zones")
    candidates = vein_reversal_zones(frame, signal_mode="Candidates")
    both = vein_reversal_zones(frame, signal_mode="Both")
    # Candidates mode never gates on proximity, so it fires at least as often.
    assert candidates.long_signal.sum() >= known.long_signal.sum()
    assert both.long_signal.sum() >= candidates.long_signal.sum()
    strict = vein_reversal_zones(frame, sig_requirement="2+",
                                 signal_mode="Candidates")
    assert strict.long_signal.sum() <= candidates.long_signal.sum()
    assert strict.role_count_long.dropna().ge(0).all()
    with pytest.raises(ValueError, match="signal_mode"):
        vein_reversal_zones(frame, signal_mode="Zones")
    with pytest.raises(ValueError, match="sig_requirement"):
        vein_reversal_zones(frame, sig_requirement="3+")


def test_cooldown_limits_consecutive_signals():
    frame = _reversal_frame()
    tight = vein_reversal_zones(frame, signal_mode="Candidates", cooldown_bars=0)
    loose = vein_reversal_zones(frame, signal_mode="Candidates", cooldown_bars=20)
    assert loose.long_signal.sum() <= tight.long_signal.sum()
    with pytest.raises(ValueError, match="cooldown_bars"):
        vein_reversal_zones(frame, cooldown_bars=-1)


def test_triggers_are_boolean_and_bounded_scores():
    frame = _reversal_frame()
    result = vein_reversal_zones(frame, signal_mode="Candidates")
    for column in ("wt_long", "wt_short", "rsi_long", "rsi_short", "st_long",
                   "st_short", "sweep_long", "sweep_short", "ema_long",
                   "ema_short"):
        assert result[column].dtype == bool
    assert result.confl_score_long.dropna().between(0, 10).all()
    assert result.confl_score_short.dropna().between(0, 10).all()
    with pytest.raises(ValueError, match="max_stored_zones"):
        vein_reversal_zones(frame, max_stored_zones=2)
