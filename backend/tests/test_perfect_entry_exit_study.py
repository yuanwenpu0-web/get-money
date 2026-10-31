import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from perfect_entry_exit_study import (
    _net,
    confirmed_pivot_lows,
    evaluate_exit,
)


def test_confirmed_bottom_requires_prior_drawdown_and_later_confirmation():
    low = np.array([100.0, 99.0, 94.0, 94.5, 95.0])
    high = np.array([101.0, 102.0, 103.0, 96.0, 97.0])
    close = np.array([100.0, 100.0, 94.5, 95.5, 96.0])

    # Lowest price is bar 2; it has a >5% prior drawdown and is only known on bar 3.
    assert confirmed_pivot_lows(low, high, close, confirm=0.01, min_gap=1, min_dd=0.05) == [(2, 3)]


def test_shallow_low_is_excluded_by_minimum_drawdown():
    low = np.array([100.0, 99.5, 99.0, 100.0])
    high = np.array([101.0, 101.0, 100.0, 101.0])
    close = np.array([100.0, 100.0, 100.0, 101.0])

    assert confirmed_pivot_lows(low, high, close, confirm=0.01, min_gap=1, min_dd=0.05) == []


def test_trailing_stop_uses_prior_bar_high_not_same_bar_high():
    # The bar rises to 120 then falls to 112.  A same-bar trail would invent
    # a 114 exit; causal handling only knows the previous high 110, so it
    # must not exit on this bar at all.
    O = np.array([100.0, 110.0])
    H = np.array([110.0, 120.0])
    L = np.array([100.0, 112.0])
    C = np.array([109.0, 115.0])
    rule = {"name": "trail_5pct", "kind": "trail_pct", "k": 0.05}

    offset, price, reason = evaluate_exit(rule, O, H, L, C, 100.0, 1.0, {})

    assert (offset, price, reason) == (1, 115.0, "cap")


def test_level_rule_treats_same_bar_stop_and_target_as_stop_first():
    O = np.array([100.0])
    H = np.array([104.0])
    L = np.array([98.0])
    C = np.array([101.0])
    rule = {"name": "tp3_sl1", "kind": "level", "tp_atr": 3.0, "sl_atr": 1.0}

    offset, price, reason = evaluate_exit(rule, O, H, L, C, 100.0, 2.0, {})

    assert (offset, price, reason) == (0, 98.0, "stop")


def test_net_return_charges_round_trip_cost():
    # Gross zero still pays 2 * (fee + slippage), with multiplicative handling.
    assert _net(100.0, 100.0) < -0.0011
    assert _net(100.0, 100.0) > -0.0013
