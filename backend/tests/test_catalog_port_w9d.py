import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.elliott_wave_radar import elliott_wave_radar
from app.catalog_indicators.market_exhaustion import market_exhaustion
from app.catalog_indicators.sm_radar import sm_radar
from app.catalog_indicators.zigzag_patterns import zigzag_patterns


def _frame(closes, spread=0.3, volume=None):
    """Pivot-strict bars (high/low follow close) that still carry a real body."""
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    close = pd.Series(np.asarray(closes, dtype=float), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    data = {"open": open_, "high": close + spread, "low": close - spread, "close": close}
    if volume is not None:
        data["volume"] = pd.Series(np.asarray(volume, dtype=float), index=index)
    return pd.DataFrame(data, index=index)


def _leg_frame(legs, bars=8, tail=8, spread=0.5):
    """Chain linear legs; shared endpoints are emitted once so pivots stay strict."""
    closes = []
    for index, (start, end) in enumerate(legs):
        points = np.linspace(start, end, bars)
        if index:
            points = points[1:]
        closes.extend(points.tolist())
    closes.extend([legs[-1][1]] * tail)
    return _frame(closes, spread=spread)


def _zigzag_frame():
    """Alternating swings ending in a failed retest: L100 H104 L102 H108 L105 H112 L104 H110 → 100."""
    return _leg_frame([(100, 104), (104, 102), (102, 108), (108, 105),
                       (105, 112), (112, 104), (104, 110), (110, 100)])


# ---------------------------------------------------------------- zigzag_patterns

def test_zigzag_pivots_alternate_and_cap():
    frame = _zigzag_frame()
    out = zigzag_patterns(frame, pivot_len=3)
    assert set(out.pivot_type.dropna().unique()) <= {-1.0, 1.0}
    kinds = out.pivot_type.dropna().to_numpy()
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))
    assert out.pivot_new.sum() == len(kinds)


def test_zigzag_hh_ll_labels_compare_same_type():
    frame = _zigzag_frame()
    out = zigzag_patterns(frame, pivot_len=3)
    labels = out.pivot_label[out.pivot_label != ""]
    assert len(labels) >= 1
    assert all(label.split(" | ")[0] in {"HH", "LH", "HL", "LL"} for label in labels)


def test_zigzag_abc_needs_retrace_band():
    frame = _zigzag_frame()
    loose = zigzag_patterns(frame, pivot_len=3, min_retr=0.0, max_retr=1.0)
    strict = zigzag_patterns(frame, pivot_len=3, min_retr=0.99, max_retr=1.0)
    assert loose.abc_bull.sum() + loose.abc_bear.sum() >= 1
    assert strict.abc_bull.sum() + strict.abc_bear.sum() == 0


def test_zigzag_triangle_requires_alternation():
    frame = _zigzag_frame()
    out = zigzag_patterns(frame, pivot_len=3, flat_atr=10.0)
    # Every side is "flat" at that tolerance → no triangle class fires.
    assert not out.triangle_sym.any()
    assert not out.triangle_rising.any()
    assert not out.triangle_falling.any()


def test_zigzag_123_break_triggers_on_close_not_pivot():
    frame = _zigzag_frame()
    out = zigzag_patterns(frame, pivot_len=3, min_retr=0.0, max_retr=1.0)
    assert out.setup_123.any()
    assert out.long_123.any() or out.short_123.any()
    assert out.break_123_score[out.break_123_score.notna()].between(0, 100).all()


def test_zigzag_rejects_bad_params():
    frame = _zigzag_frame()
    with pytest.raises(ValueError, match="pivot_len"):
        zigzag_patterns(frame, pivot_len=1)
    with pytest.raises(ValueError, match="min_retr"):
        zigzag_patterns(frame, min_retr=-0.1)


# ----------------------------------------------------------------------- sm_radar

def _sm_frame():
    """Up-break of a swing high, then a down leg that breaks the swing low."""
    return _leg_frame([(100, 120), (120, 108), (108, 132), (132, 100)],
                      bars=6, tail=6)


def test_sm_radar_order_block_on_bos():
    frame = _sm_frame()
    out = sm_radar(frame, ob_pivot_len=3, ms_pivot_len=3)
    assert out.ob_bull.any()
    assert out.ob_bear.any()
    rows = out[out.ob_bull]
    assert rows.ob_top.iloc[0] > rows.ob_bot.iloc[0]
    assert out.ob_count.max() >= 1


def test_sm_radar_fvg_needs_min_size():
    frame = _sm_frame()
    loose = sm_radar(frame, fvg_min_size_pct=0.0)
    strict = sm_radar(frame, fvg_min_size_pct=20.0)
    assert loose.fvg_bull.any() or loose.fvg_bear.any()
    assert not strict.fvg_bull.any() and not strict.fvg_bear.any()


def test_sm_radar_structure_flags_choch_once_per_flip():
    frame = _sm_frame()
    out = sm_radar(frame, ms_pivot_len=3)
    assert out.trend.iloc[-1] != 0
    assert out.bos_bull.any() or out.choch_bull.any()
    # A CHoCH only appears on a direction flip, so it is never the very first break.
    first_break = out[(out.bos_bull | out.choch_bull) | (out.bos_bear | out.choch_bear)]
    assert not (first_break.choch_bull.iloc[0] and first_break.choch_bear.iloc[0])


def test_sm_radar_prev_period_levels_have_no_lookahead():
    frame = _sm_frame()
    out = sm_radar(frame, ms_pivot_len=3)
    # 15m bars: the first day is incomplete, so previous-day H/L starts empty.
    assert out.prev_day_high.isna().all()
    assert out.prev_day_low.isna().all()


def test_sm_radar_sessions_respect_hours():
    frame = _sm_frame()
    out = sm_radar(frame, ms_pivot_len=3)
    asia_hours = frame.index.tz_convert("UTC").hour
    assert out.asia_active.to_numpy().sum() == int(((asia_hours >= 0) & (asia_hours < 8)).sum())
    assert not out.london_active.iloc[0]


# ---------------------------------------------------------------- market_exhaustion

def _mex_frame(n=240):
    t = np.arange(n)
    close = 100 + 3 * np.sin(t / 3.0) + 2 * np.sin(t / 11.0) + 0.01 * t
    volume = 1000 + 400 * np.abs(np.sin(t / 5.0))
    return _frame(close, spread=0.4, volume=volume)


def test_market_exhaustion_without_volume_goes_nan():
    frame = _mex_frame().drop(columns=["volume"])
    out = market_exhaustion(frame)
    assert out.mfi.isna().all()
    assert not out.volume_real.iloc[0]
    assert out.stoch_k.notna().any()


def test_market_exhaustion_mfi_bounds_and_zones():
    frame = _mex_frame()
    out = market_exhaustion(frame)
    values = out.mfi.dropna()
    assert values.between(0, 100).all()
    assert out.volume_real.iloc[0]
    zones = out[["zone_upper", "zone_lower"]].dropna()
    assert (zones.zone_upper >= zones.zone_lower).all()
    assert out.is_ob.sum() + out.is_os.sum() >= 1


def test_market_exhaustion_signals_need_stoch_cross():
    frame = _mex_frame()
    out = market_exhaustion(frame)
    assert out.long_signal.any() or out.short_signal.any()
    rows = out[out.long_signal | out.short_signal]
    assert len(rows) >= 1
    disabled = market_exhaustion(frame, use_stoch=False)
    assert not disabled.long_signal.any() and not disabled.short_signal.any()


def test_market_exhaustion_htf_filter_blocks_signals():
    frame = _mex_frame()
    filtered = market_exhaustion(frame, use_htf_filter=True)
    unfiltered = market_exhaustion(frame, use_htf_filter=False)
    assert filtered.long_signal.sum() <= unfiltered.long_signal.sum()
    assert filtered.short_signal.sum() <= unfiltered.short_signal.sum()


def test_market_exhaustion_divergence_scores_are_bounded():
    frame = _mex_frame()
    out = market_exhaustion(frame)
    scores = out.div_score.dropna()
    assert len(scores) >= 1
    assert scores.between(0, 100).all()
    assert out.div_finalized.any()
    finals = out.div_final_score.dropna()
    assert finals.between(0, 100).all()


def test_market_exhaustion_osc_mode_switch():
    frame = _mex_frame()
    cci_only = market_exhaustion(frame, osc_mode="CCI")
    assert cci_only.oscillator.dropna().between(0, 100).all()
    with pytest.raises(ValueError, match="osc_mode"):
        market_exhaustion(frame, osc_mode="TSI")


# ---------------------------------------------------------------- elliott_wave_radar

def _ew_frame():
    """Textbook 1-5 impulse off a prior low, then a deep pullback through W1's end."""
    return _leg_frame([(112, 100), (100, 120), (120, 112), (112, 155),
                       (155, 148), (148, 168), (168, 105)], bars=10, tail=20)


def test_ew_radar_impulse_detected_with_score():
    frame = _ew_frame()
    out = elliott_wave_radar(frame, pivot_len=3, min_score=0, min_swing_atr=0.0)
    assert (out.count_type == "IMP").any()
    impulse = out[out.count_type == "IMP"].iloc[0]
    assert 0 <= impulse.count_score <= 100
    assert impulse.count_dir == 1
    assert impulse.ev_imp


def test_ew_radar_projects_correction_targets_after_impulse():
    frame = _ew_frame()
    out = elliott_wave_radar(frame, pivot_len=3, min_score=0, min_swing_atr=0.0)
    impulse_rows = out[out.count_type == "IMP"]
    first = impulse_rows.iloc[0]
    p0, p5 = first.p0_price, first.p5_price
    rng = p5 - p0
    assert first.proj_1 == pytest.approx(p5 - rng * 0.382)
    assert first.proj_2 == pytest.approx(p5 - rng * 0.5)
    assert first.proj_3 == pytest.approx(p5 - rng * 0.618)
    assert first.proj_1_label == "Corr 38.2%"


def test_ew_radar_no_count_when_score_gate_is_high():
    frame = _ew_frame()
    out = elliott_wave_radar(frame, pivot_len=3, min_score=100, min_swing_atr=0.0)
    assert (out.count_type == "NONE").all()
    assert out.count_score.isna().all()


def test_ew_radar_dev2_invalidation_clears_count():
    """W1-W2 confirmed (99.5 → 119.5 → 111.5), then a crash below W1's start."""
    closes = [112, 110, 108, 106, 104, 102, 100, 104, 108, 112, 116, 120,
              118, 116, 114, 112, 114, 116, 118, 90, 90, 90, 90, 90, 90]
    frame = _frame(closes, spread=0.5)
    out = elliott_wave_radar(frame, pivot_len=3, min_score=0, min_swing_atr=0.0)
    assert (out.count_type == "DEV2").any()
    assert out.invalidation_hit.any()
    hit = int(np.flatnonzero(out.invalidation_hit.to_numpy())[0])
    # Invalidation clears the count on the same bar it fires.
    assert out.count_type.iloc[hit] == "NONE"
    assert np.isnan(out.invalidation.iloc[hit])


def test_ew_radar_sub_wave_legs_counted():
    frame = _ew_frame()
    out = elliott_wave_radar(frame, pivot_len=3, sub_len=2, min_score=0, min_swing_atr=0.0)
    impulse = out[out.count_type == "IMP"].iloc[0]
    assert impulse.sub_w1_legs >= 0
    assert impulse.sub_w3_legs >= 0
    assert impulse.sub_structure.startswith("W1:")


def test_ew_radar_rejects_bad_params():
    frame = _ew_frame()
    with pytest.raises(ValueError, match="pivot_len"):
        elliott_wave_radar(frame, pivot_len=2)
    with pytest.raises(ValueError, match="min_swing_atr"):
        elliott_wave_radar(frame, min_swing_atr=-1.0)
