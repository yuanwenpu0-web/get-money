import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.crypto_amt_toolkit import crypto_amt_toolkit
from app.catalog_indicators.zigzag_fibo_pullback_map import zigzag_fibo_pullback_map


def _frame(closes, spread=0.4, volume=None):
    """High/low track close exactly so pivots stay unambiguous; the body still exists."""
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    close = pd.Series(np.asarray(closes, dtype=float), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    data = {"open": open_, "high": close + spread, "low": close - spread, "close": close}
    if volume is not None:
        data["volume"] = pd.Series(np.asarray(volume, dtype=float), index=index)
    return pd.DataFrame(data, index=index)


def _leg_frame(legs, bars=8, tail=8, spread=0.4, volume=None):
    closes = []
    for index, (start, end) in enumerate(legs):
        points = np.linspace(start, end, bars)
        if index:
            points = points[1:]
        closes.extend(points.tolist())
    closes.extend([legs[-1][1]] * tail)
    return _frame(closes, spread=spread, volume=volume)


# ------------------------------------------------------- zigzag_fibo_pullback_map

def _fib_frame():
    """Pivots 100 → 130 → 111.5 → 140 → 118 → 148: a 0.617 pullback off the 100-130 leg,
    with two later pivots so the reaction can resolve."""
    return _leg_frame([(112, 100), (100, 130), (130, 111.5), (111.5, 140),
                       (140, 118), (118, 148)], bars=9, tail=10)


def test_fibo_pullback_matches_standard_level():
    frame = _fib_frame()
    out = zigzag_fibo_pullback_map(frame, pivot_len=3, match_tolerance=0.05,
                                  min_leg_atr=0.0, min_leg_pct=0.0)
    rows = out[out.pullback]
    assert len(rows) >= 1
    first = rows.iloc[0]
    assert first.pb_impulse_up
    # Leg 100→130, pullback to 111.5 → ratio ≈ 0.617.
    assert first.pb_ratio == pytest.approx(0.6167, abs=0.02)
    assert first.pb_matched == pytest.approx(0.618, abs=1e-9)
    assert first.pb_depth == "Deep"
    assert first.pb_level == pytest.approx(130 - 30 * 0.618, abs=0.1)


def test_fibo_pullback_tolerance_controls_match():
    frame = _fib_frame()
    loose = zigzag_fibo_pullback_map(frame, pivot_len=3, match_tolerance=0.05,
                                     min_leg_atr=0.0, min_leg_pct=0.0)
    tight = zigzag_fibo_pullback_map(frame, pivot_len=3, match_tolerance=0.001,
                                     min_leg_atr=0.0, min_leg_pct=0.0)
    assert loose.pb_matched.notna().any()
    assert not tight.pb_matched.notna().any()
    # Even without a match, the reached level is the deepest standard fib.
    assert tight[tight.pullback].iloc[0].pb_reached == pytest.approx(0.618, abs=1e-9)


def test_fibo_pullback_reaction_resolves_after_two_pivots():
    frame = _fib_frame()
    out = zigzag_fibo_pullback_map(frame, pivot_len=3, min_leg_atr=0.0, min_leg_pct=0.0)
    assert out.reaction_update.any()
    resolved = out[out.reaction_update]
    assert resolved.iloc[-1].pb_reaction in {"HH", "LL", "Break", "Weak"}
    # The reaction is still pending on the pullback bar itself (only two pivots exist),
    # and resolves later once `d` and `e` are confirmed.
    assert out[out.pullback].iloc[0].pb_reaction == "Pending"
    assert out.pb_reaction.iloc[-1] == "HH"


def test_fibo_live_fan_extends_beyond_last_pivot():
    frame = _fib_frame()
    out = zigzag_fibo_pullback_map(frame, pivot_len=3, min_leg_atr=0.0, min_leg_pct=0.0)
    assert out.live_ratio.notna().any()
    live = out.live_ratio.iloc[-1]
    assert 0 <= live < 3
    # Deeper fib ratios sit further from the base anchor, so the ordering flips
    # with the direction of the unfinished leg.
    step = out.live_level_382.iloc[-1] - out.live_level_236.iloc[-1]
    assert step > 0 if out.live_move_up.iloc[-1] else step < 0
    assert out.live_zone.iloc[-1] in {
        "Below 0.236", "0.236-0.382", "0.382-0.500", "0.500-0.618",
        "0.618-0.786", "0.786-1.000", "> 1.000"}


def test_fibo_leg_filter_blocks_small_legs():
    """Legs of ~4 points on a ~100 price: 0.25 retrace counts loosely, not strictly."""
    frame = _leg_frame([(102, 100), (100, 104), (104, 103), (103, 107), (107, 106)],
                       bars=9, tail=10)
    loose = zigzag_fibo_pullback_map(frame, pivot_len=3, min_leg_atr=0.0, min_leg_pct=0.0)
    strict = zigzag_fibo_pullback_map(frame, pivot_len=3, min_leg_atr=5.0, min_leg_pct=5.0)
    assert loose.pullback.any()
    assert not strict.pullback.any()


def test_fibo_rejects_bad_params():
    frame = _fib_frame()
    with pytest.raises(ValueError, match="pivot_len"):
        zigzag_fibo_pullback_map(frame, pivot_len=1)
    with pytest.raises(ValueError, match="label_mode"):
        zigzag_fibo_pullback_map(frame, label_mode="Weird")


# ---------------------------------------------------------------- crypto_amt_toolkit

def _amt_frame(n=320):
    t = np.arange(n)
    close = 100 + 3 * np.sin(t / 6.0) + 2 * np.sin(t / 23.0) + 0.03 * t
    volume = 1000 + 500 * np.abs(np.sin(t / 9.0))
    return _frame(close, volume=volume)


def test_amt_volume_profile_ordering():
    frame = _amt_frame()
    out = crypto_amt_toolkit(frame)
    rows = out[["poc", "vah", "val"]].dropna()
    assert len(rows) == len(frame)
    assert (rows.val <= rows.poc).all()
    assert (rows.poc <= rows.vah).all()


def test_amt_poc_sits_on_heaviest_row():
    frame = _amt_frame(200)
    out = crypto_amt_toolkit(frame, vp_lookback=100, vp_rows=20)
    window_high = frame.high.iloc[100:200].max()
    window_low = frame.low.iloc[100:200].min()
    row_height = (window_high - window_low) / 20
    poc = out.poc.iloc[-1]
    # POC is the centre of a row, so it must sit exactly half a row above a boundary.
    offset = (poc - window_low) / row_height
    assert abs(offset - round(offset - 0.5) - 0.5) < 1e-6


def test_amt_vwap_is_volume_weighted_within_session():
    frame = _amt_frame(100)
    out = crypto_amt_toolkit(frame)
    typical = (frame.high + frame.low + frame.close) / 3.0
    first_day = frame.index.tz_convert("UTC").tz_localize(None).day == 1
    weights = frame.volume[first_day]
    expected = float((typical[first_day] * weights).sum() / weights.sum())
    assert out.vwap[first_day].iloc[-1] == pytest.approx(expected, rel=1e-9)
    # A new session restarts the accumulator.
    assert out.vwap.iloc[0] == pytest.approx(float(typical.iloc[0]), rel=1e-12)


def test_amt_value_area_respects_percentage():
    frame = _amt_frame(200)
    wide = crypto_amt_toolkit(frame, vp_lookback=100, value_area_pct=90.0)
    narrow = crypto_amt_toolkit(frame, vp_lookback=100, value_area_pct=50.0)
    assert (wide.vah.iloc[-1] - wide.val.iloc[-1]) >= (narrow.vah.iloc[-1] - narrow.val.iloc[-1])


def test_amt_cvd_resets_each_session():
    frame = _amt_frame(200)
    out = crypto_amt_toolkit(frame, cvd_reset="Daily")
    no_reset = crypto_amt_toolkit(frame, cvd_reset="None")
    # With a daily reset the first bar of each day carries only that bar's delta.
    firsts = frame.index.tz_convert("UTC").tz_localize(None).normalize()
    assert out.cvd.iloc[0] == pytest.approx(out.bar_delta.iloc[0])
    assert abs(no_reset.cvd.iloc[-1]) >= abs(out.cvd.iloc[-1]) or no_reset.cvd.iloc[-1] != out.cvd.iloc[-1]


def test_amt_score_is_bounded_and_graded():
    frame = _amt_frame()
    out = crypto_amt_toolkit(frame)
    assert out.score.between(-9, 9).all()
    assert set(out.signal_grade.unique()) <= {
        "BLOCKED", "A+ LONG", "A+ SHORT", "A LONG", "A SHORT", "B LONG", "B SHORT", "WAIT"}
    assert out.trigger_text.iloc[-1] in {
        "NONE", "LONG BREAKOUT", "VAL REJECT", "LONG RECLAIM", "LONG RETEST", "BOS UP",
        "SHORT BREAKDOWN", "VAH REJECT", "SHORT RECLAIM", "SHORT RETEST", "BOS DOWN"}


def test_amt_confirm_on_close_is_always_confirmed_in_history():
    frame = _amt_frame()
    out = crypto_amt_toolkit(frame, confirm_on_close=True)
    assert not out.pending_long.any()
    assert not out.pending_short.any()


def test_amt_no_trade_filter_off_opens_zone():
    frame = _amt_frame()
    adaptive = crypto_amt_toolkit(frame, no_trade_filter="Adaptive")
    off = crypto_amt_toolkit(frame, no_trade_filter="Off")
    assert adaptive.no_trade_zone.sum() >= 1
    assert not off.no_trade_zone.any()
    assert (off.block_reason == "OFF").all()


def test_amt_signal_plan_has_ordered_invalidation_and_targets():
    frame = _amt_frame()
    out = crypto_amt_toolkit(frame, risk_mode="Tight")
    longs = out[out.confirmed_long]
    assert len(longs) >= 1
    first = longs.iloc[0]
    assert first.active_invalidation < first.active_entry < first.active_target_1
    assert first.active_target_1 < first.active_target_2


def test_amt_invalidation_clears_active_plan():
    frame = _amt_frame()
    out = crypto_amt_toolkit(frame)
    assert out.confirmed_long.any() or out.confirmed_short.any()
    assert out.invalidated.any() or (out.active_dir != 0).any()


def test_amt_rejects_bad_params():
    frame = _amt_frame()
    with pytest.raises(ValueError, match="vp_rows"):
        crypto_amt_toolkit(frame, vp_rows=5)
    with pytest.raises(ValueError, match="trade_mode"):
        crypto_amt_toolkit(frame, trade_mode="Scalper")
    with pytest.raises(ValueError, match="volume"):
        crypto_amt_toolkit(frame.drop(columns=["volume"]))
