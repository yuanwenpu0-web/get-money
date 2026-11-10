"""Unit tests for the pure-causal order-exit replay helpers (diagnosis only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from order_exit_common import (  # noqa: E402
    net_return_pct, run_rule, bootstrap_ci, verdict, bh_fdr, build_rules, diagnostics,
)


def bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_round_trip_cost_is_exactly_012_percent():
    # Flat exit: gross 0, fee 0.0004*(1+1), slippage 2*0.0002 => 0.0012.
    assert net_return_pct(100.0, 100.0, "LONG") == pytest.approx(-0.12, abs=1e-9)
    assert net_return_pct(100.0, 100.0, "SHORT") == pytest.approx(-0.12, abs=1e-9)


def test_long_side_signs():
    assert net_return_pct(100.0, 105.0, "LONG") > 0
    assert net_return_pct(100.0, 105.0, "SHORT") < 0


def test_tp_fires_at_barrier_price_not_bar_close():
    bars = [bar(100, 104, 99.5, 103.5)]
    px, idx, why = run_rule(bars, "LONG", 100, 99,
                            {"tp": 103, "sl": 99, "kind": "barrier",
                             "R": 1, "activate_R": 0, "trail_R": 0, "max_bars": 0})
    assert (px, idx, why) == (103.0, 0, "tp")


def test_same_bar_sl_takes_precedence_over_tp():
    bars = [bar(100, 104, 98, 103.5)]
    px, _, why = run_rule(bars, "LONG", 100, 99,
                          {"tp": 103, "sl": 99, "kind": "barrier",
                           "R": 1, "activate_R": 0, "trail_R": 0, "max_bars": 0})
    assert (px, why) == (99.0, "sl")


def test_gap_uses_open_price_and_sl_first():
    bars = [bar(98, 98.5, 97, 97.5)]
    px, _, why = run_rule(bars, "LONG", 100, 99,
                          {"tp": 103, "sl": 99, "kind": "barrier",
                           "R": 1, "activate_R": 0, "trail_R": 0, "max_bars": 0})
    assert (px, why) == (98.0, "sl_gap")


def test_short_side_barriers_are_mirrored():
    bars = [bar(100, 100.5, 96, 97)]
    px, _, why = run_rule(bars, "SHORT", 100, 101,
                          {"tp": 97, "sl": 101, "kind": "barrier",
                           "R": 1, "activate_R": 0, "trail_R": 0, "max_bars": 0})
    assert (px, why) == (97.0, "tp")


def test_trailing_never_fires_before_activation_and_uses_past_extremes_only():
    bars = [bar(100, 100.4, 99.8, 100.2), bar(100.2, 101.5, 100.1, 101.4)]
    _, _, why = run_rule(bars, "LONG", 100, 99,
                         {"tp": None, "sl": 99, "kind": "trailing",
                          "R": 1, "activate_R": 1.0, "trail_R": 0.5, "max_bars": 0})
    # peak 101.5 -> trail price 101.0 lies inside bar 2 => trail fires there.
    assert why == "trail"


def test_build_rules_has_baseline_and_no_duplicate_names():
    rules = build_rules(100, 103, 99, 1.0, 96, "LONG")
    names = [r["name"] for r in rules]
    assert names[0] == "baseline"
    assert len(names) == len(set(names))
    assert rules[0]["tp"] == 103


def test_diagnostics_bounds_the_post_exit_window():
    bars = [bar(100, 100 + i, 99, 100) for i in range(1, 301)]
    d = diagnostics(bars, "LONG", 100, 99, 103, 103, 0, 103, 5, post_cap=10)
    assert d["window_bars"] == 15
    assert d["post_bars_used"] == 10
    assert d["tp_dist_R"] == pytest.approx(3.0)


def test_bootstrap_and_verdict_vocabulary():
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0] * 20, seed=1)
    assert lo <= hi
    assert verdict([0.1, 0.2], 50) == "pass"
    assert verdict([-0.2, -0.1], 50) == "negative"
    assert verdict([-0.1, 0.2], 50) == "inconclusive"
    assert verdict([0.1, 0.2], 5) == "insufficient"


def test_bh_fdr_monotone():
    q = bh_fdr([0.001, 0.02, 0.4, 0.9])
    assert q == sorted(q)
    assert q[0] <= q[-1]
