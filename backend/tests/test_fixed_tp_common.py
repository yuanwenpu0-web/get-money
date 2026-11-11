"""Unit tests for the optional ABSOLUTE fixed-TP rule family.

Guards the two invariants that matter here: (1) adding the grid must not
change the historical default rule family, (2) a fixed target must be a fixed
PERCENTAGE of entry, independent of each order's own TP distance or ATR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from order_exit_common import build_rules, net_return_pct, run_rule  # noqa: E402

GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def names(entry=100.0, tp=104.0, sl=98.0, atr_value=1.0, side="LONG", grid=()):
    return [r["name"] for r in build_rules(entry, tp, sl, atr_value, 96, side,
                                           fixed_tp_pcts=grid)]


def test_default_family_is_unchanged():
    assert len(names()) == 18
    assert not any(n.startswith("fix_") for n in names())


def test_grid_adds_exactly_one_rule_per_percentage():
    n = names(grid=GRID)
    assert len(n) == 18 + len(GRID)
    assert [x for x in n if x.startswith("fix_")] == [f"fix_{p:g}pct" for p in GRID]


def test_long_target_is_fixed_percentage_of_entry():
    rules = {r["name"]: r for r in build_rules(100.0, 106.0, 98.0, 2.0, 96,
                                               "LONG", fixed_tp_pcts=GRID)}
    assert rules["fix_2pct"]["tp"] == pytest.approx(102.0)
    assert rules["fix_8pct"]["tp"] == pytest.approx(108.0)


def test_short_target_is_below_entry():
    rules = {r["name"]: r for r in build_rules(100.0, 94.0, 102.0, 2.0, 96,
                                               "SHORT", fixed_tp_pcts=GRID)}
    assert rules["fix_2pct"]["tp"] == pytest.approx(98.0)
    assert rules["fix_8pct"]["tp"] == pytest.approx(92.0)


def test_fixed_target_does_not_depend_on_atr():
    a = {r["name"]: r["tp"] for r in build_rules(100.0, 104.0, 98.0, 0.5, 96,
                                                 "LONG", fixed_tp_pcts=GRID)}
    b = {r["name"]: r["tp"] for r in build_rules(100.0, 104.0, 98.0, 9.0, 96,
                                                 "LONG", fixed_tp_pcts=GRID)}
    for p in GRID:
        assert a[f"fix_{p:g}pct"] == b[f"fix_{p:g}pct"]


def test_fixed_target_does_not_depend_on_original_tp():
    a = {r["name"]: r["tp"] for r in build_rules(100.0, 102.0, 98.0, 1.0, 96,
                                                 "LONG", fixed_tp_pcts=GRID)}
    b = {r["name"]: r["tp"] for r in build_rules(100.0, 112.0, 98.0, 1.0, 96,
                                                 "LONG", fixed_tp_pcts=GRID)}
    for p in GRID:
        assert a[f"fix_{p:g}pct"] == b[f"fix_{p:g}pct"]


def test_grid_leaves_stop_loss_untouched():
    # Historical sl_x1.5 / sl_x2.0 rules DO move SL by design; only the new
    # fixed-TP arms must be a single-variable A/B.
    for r in build_rules(100.0, 104.0, 98.0, 1.0, 96, "LONG",
                         fixed_tp_pcts=GRID):
        if r["name"].startswith("fix_"):
            assert r["sl"] == 98.0


def test_replay_hits_the_fixed_barrier_price():
    rule = next(r for r in build_rules(100.0, 104.0, 98.0, 1.0, 96, "LONG",
                                       fixed_tp_pcts=GRID)
                if r["name"] == "fix_2pct")
    bars = [{"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8},
            {"open": 100.8, "high": 102.6, "low": 100.5, "close": 102.0}]
    px, idx, why = run_rule(bars, "LONG", 100.0, rule["sl"], rule)
    assert (px, idx, why) == (102.0, 1, "tp")


def test_same_bar_stop_loss_still_wins_over_fixed_tp():
    rule = next(r for r in build_rules(100.0, 104.0, 98.0, 1.0, 96, "LONG",
                                       fixed_tp_pcts=GRID)
                if r["name"] == "fix_2pct")
    bars = [{"open": 100.0, "high": 103.0, "low": 97.0, "close": 99.0}]
    px, _, why = run_rule(bars, "LONG", 100.0, rule["sl"], rule)
    assert (px, why) == (98.0, "sl")


def test_round_trip_cost_is_twelve_basis_points_at_flat_exit():
    assert net_return_pct(100.0, 100.0, "LONG") == pytest.approx(-0.12, abs=1e-9)
