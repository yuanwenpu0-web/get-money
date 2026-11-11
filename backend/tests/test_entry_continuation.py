"""Unit tests for the post-entry continuation helpers.

Guards the three things that can silently corrupt this study: wrong bar
aggregation, wrong precedence between stop-loss and continuation, and
mis-reported performance metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from entry_continuation_common import (  # noqa: E402
    arm_stats, judge, max_drawdown, obs_bar, verdict,
)

T0 = 1_760_000_000_000


def row(i: int, o: float, h: float, l: float, c: float) -> dict:
    return {"open_time": T0 + i * 60_000, "open": o, "high": h, "low": l, "close": c}


def test_single_minute_bar_passes_through():
    bars = [row(0, 100.0, 100.6, 99.8, 100.4)]
    b = obs_bar(bars, 1)
    assert (b["open"], b["high"], b["low"], b["close"]) == (100.0, 100.6, 99.8, 100.4)


def test_five_minute_bar_aggregates_ohlc():
    bars = [row(i, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(5)]
    b = obs_bar(bars, 5)
    assert b["open"] == 100.0
    assert b["high"] == 105.0
    assert b["low"] == 99.0
    assert b["close"] == 104.5


def test_missing_minutes_return_none():
    bars = [row(i, 100, 101, 99, 100) for i in range(5)]
    bars.pop(2)
    assert obs_bar(bars, 5) is None


def test_shifted_start_returns_none():
    bars = [row(i + 1, 100, 101, 99, 100) for i in range(5)]
    assert obs_bar(bars, 5, start_ms=T0) is None


def test_aligned_start_passes():
    bars = [row(i, 100, 101, 99, 100) for i in range(5)]
    assert obs_bar(bars, 5, start_ms=T0) is not None


def test_continuation_keeps_baseline_exit():
    bar = {"open": 100.0, "high": 100.9, "low": 99.9, "close": 100.7}
    px, why = judge(bar, entry=100.0, sl=98.0, base_px=106.0)
    assert (px, why) == (106.0, "continuation")


def test_no_continuation_exits_at_bar_close():
    bar = {"open": 100.0, "high": 100.4, "low": 99.6, "close": 99.7}
    px, why = judge(bar, entry=100.0, sl=98.0, base_px=106.0)
    assert (px, why) == (99.7, "no_continuation")


def test_stop_inside_window_wins_over_a_green_close():
    bar = {"open": 100.0, "high": 100.5, "low": 97.5, "close": 100.2}
    px, why = judge(bar, entry=100.0, sl=98.0, base_px=106.0)
    assert why == "sl_in_window"
    assert px == pytest.approx(98.0)


def test_gap_through_stop_fills_at_open():
    bar = {"open": 96.0, "high": 96.5, "low": 95.0, "close": 95.5}
    px, why = judge(bar, entry=100.0, sl=98.0, base_px=106.0)
    assert (px, why) == (96.0, "sl_in_window")


def test_max_drawdown_tracks_cumulative_curve():
    assert max_drawdown([1.0, 2.0, -5.0, 1.0]) == pytest.approx(-5.0)
    assert max_drawdown([-3.0, 1.0]) == pytest.approx(-3.0)
    assert max_drawdown([]) == 0.0


def test_arm_stats_reports_all_five_metrics():
    s = arm_stats([10.0, -2.0, -2.0, -2.0, 0.0])
    assert s["trades"] == 5
    assert s["total_return_pct"] == pytest.approx(4.0)
    assert s["expectancy_pct"] == pytest.approx(0.8)
    assert s["win_rate"] == pytest.approx(0.2)
    assert s["payoff_ratio"] == pytest.approx(5.0)
    assert s["profit_factor"] == pytest.approx(10.0 / 6.0)


def test_verdict_uses_the_documented_words_only():
    assert verdict([0.1, 0.2], 40) == "pass"
    assert verdict([-0.2, -0.1], 40) == "negative"
    assert verdict([-0.2, 0.2], 40) == "inconclusive"
    assert verdict([0.1, 0.2], 10) == "insufficient"
