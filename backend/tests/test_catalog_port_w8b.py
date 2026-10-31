import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.structure_break_risk import structure_break_risk


def _rise_then_break():
    # Ramp with small pullbacks (swing lows) then a collapse through the
    # last standing swing low.
    closes = []
    level = 100.0
    for cycle in range(10):
        for _ in range(6):
            level += 3.0
            closes.append(level)
        for _ in range(2):
            level -= 1.5
            closes.append(level)
    closes += [closes[-1] - 30.0, closes[-1] - 45.0, closes[-1] - 60.0]
    close = pd.Series(closes, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    highs = np.maximum(open_, close) + 0.4
    lows = np.minimum(open_, close) - 0.4
    frame = pd.DataFrame({"open": open_, "high": highs, "low": lows,
                          "close": close})
    # Nudge the pullback lows into unique strict pivots.
    for k in range(10):
        frame.iloc[k * 8 + 7, frame.columns.get_loc("low")] -= 0.5
    frame.index = pd.date_range("2026-01-01", periods=len(closes), freq="1h",
                                tz="UTC")
    return frame


def test_risk_builds_in_uptrend_and_state_flips_on_break():
    frame = _rise_then_break()
    result = structure_break_risk(frame, bias_len=20, rsi_len=5, piv_len=3,
                                  hold_bars=5, smooth_len=1,
                                  require_context_flip=False)
    # Uptrend latched during the ramp.
    assert (result.trend_dir.iloc[40:78] == 1).all()
    # After the collapse the structure-broken event fires.
    assert result.bos_now.any()
    assert (result.state == "Structure Broken").any()
    # Risk is bounded and states stay in the known set.
    assert result.risk.dropna().between(0, 100).all()
    assert set(result.state.unique()) <= {
        "Quiet", "No trend", "Watch", "Break Pressure", "Critical",
        "Structure Broken"}
    with pytest.raises(ValueError, match="bias_len"):
        structure_break_risk(frame, bias_len=5)
    with pytest.raises(ValueError, match="行情列"):
        structure_break_risk(frame.drop(columns="close"))


def test_sensors_decay_over_hold_bars():
    frame = _rise_then_break()
    short = structure_break_risk(frame, bias_len=20, rsi_len=5, piv_len=3,
                                 hold_bars=2, smooth_len=1,
                                 require_context_flip=False)
    long_h = structure_break_risk(frame, bias_len=20, rsi_len=5, piv_len=3,
                                  hold_bars=40, smooth_len=1,
                                  require_context_flip=False)
    # Longer memory keeps higher sensor peaks somewhere in the run.
    assert long_h.s_sfp.dropna().max() >= short.s_sfp.dropna().max() - 1e-9
    assert short.s_sfp.dropna().between(0, 1).all()
    with pytest.raises(ValueError, match="hold_bars"):
        structure_break_risk(frame, hold_bars=0)


def test_context_flip_gates_the_break_event():
    frame = _rise_then_break()
    gated = structure_break_risk(frame, bias_len=20, rsi_len=5, piv_len=3,
                                 hold_bars=5, smooth_len=1,
                                 require_context_flip=True)
    ungated = structure_break_risk(frame, bias_len=20, rsi_len=5, piv_len=3,
                                   hold_bars=5, smooth_len=1,
                                   require_context_flip=False)
    # Gating can only delay/reduce the broken events, never exceed them.
    assert gated.bos_now.sum() <= ungated.bos_now.sum()
    # Break pressure events still fire when the swing breaks without context.
    assert ungated.bos_pressure.any() or ungated.bos_now.any()
    with pytest.raises(ValueError, match="piv_len"):
        structure_break_risk(frame, piv_len=1)
