"""Deterministic broker replay tests for Regime-Gated Donchian v2 (RGD).

Every fill expectation below is calculated from a hand-built OHLC sequence.
The tests do not infer an expected result from broker internals: each one guards
a Pine timing or order-priority rule from the porting contract.
"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from app.backtest.execution import randomize_directions
from app.backtest.rgd_execution import rgd_path, simulate_rgd
from app.backtest.trades import aggregate_positions
from app.features.rgd import DECISION_COLUMNS
from app.strategy.rules import resolve_params
from scripts.rgd_baselines import randomize_rgd


CODE = "rgd_donchian_v2"


def _params(**overrides) -> dict:
    return resolve_params(
        CODE,
        {"fee_rate": 0.0004, "slippage": 0.0002},
        overrides or None,
        interval="15m",
    )


def _signal(**overrides) -> dict:
    row = {
        "rgd_long_break": False,
        "rgd_short_break": False,
        "rgd_long_pull_core": False,
        "rgd_short_pull_core": False,
        "rgd_allow_long": True,
        "rgd_allow_short": True,
        "rgd_state": 0,
        "rgd_def_long": 90.0,
        "rgd_def_short": 110.0,
        # A signal-close of 100 and 1% risk budget against R=10 means q=10 units.
        "rgd_stp_long": 90.0,
        "rgd_rd_long": 10.0,
        "rgd_tp1_long": 120.0,
        "rgd_stp_short": 110.0,
        "rgd_rd_short": 10.0,
        "rgd_tp1_short": 80.0,
        # RGD deliberately blocks the generic broker; its dedicated broker consumes rgd_* fields.
        "direction": 0,
        "block_reason": "stop_order_requires_dedicated_broker",
    }
    row.update(overrides)
    return row


def replay(ohlc, rows, *, fee: float = 0.0, slippage: float = 0.0, **overrides):
    index = pd.date_range("2026-02-01", periods=len(ohlc), freq="15min", tz="UTC")
    data = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=index)
    signals = pd.DataFrame(rows, index=index)
    equity, trades = simulate_rgd(
        data, signals, _params(**overrides), 10_000.0, fee, slippage, "15m"
    )
    return equity, trades, signals, index


def test_a12_tp1_closes_exactly_half_then_aggregate_combines_the_two_legs():
    equity, trades, _, index = replay(
        [
            (100, 101, 99, 100),   # submit long: stop=90, TP1=120, q=10
            (100, 110, 95, 105),   # fill long; exit orders only become armed at this close
            (105, 121, 104, 119),  # TP1 fills 5 at 120
            (119, 119, 99, 100),   # remaining 5 stops at the now-active 100 breakeven stop
        ],
        [_signal(rgd_long_break=True), _signal(), _signal(), _signal()],
    )
    assert len(trades) == 2
    assert [(item["status"], item["size"], item["exit_price"]) for item in trades] == [
        ("take_profit_1", 5.0, 120.0),
        ("stop_loss", 5.0, 100.0),
    ]
    assert all(item["entry_price"] == 100.0 for item in trades)
    assert all(item["entry_time"] == str(index[1]) for item in trades)
    assert equity.iloc[-1] == pytest.approx(10_100.0)

    combined = aggregate_positions(trades)
    assert len(combined) == 1
    assert combined[0]["size"] == pytest.approx(10.0)
    assert combined[0]["pnl"] == pytest.approx(100.0)
    assert combined[0]["margin"] == pytest.approx(1000.0)
    assert combined[0]["return_pct"] == pytest.approx(10.0)


def test_a13_breakeven_stop_is_not_active_until_the_bar_after_tp1_detection():
    equity, trades, _, index = replay(
        [
            (100, 101, 99, 100),
            (100, 110, 95, 105),
            # This wick crosses TP1 then drops below entryRef=100, but old stop remains 90 on N.
            (105, 121, 95, 105),
            # The N-close stop update is armed only now; a gap through it fills at the opening 99.
            (99, 101, 98, 99),
        ],
        [_signal(rgd_long_break=True), _signal(), _signal(), _signal()],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["size"]) for item in trades] == [
        ("take_profit_1", str(index[2]), 120.0, 5.0),
        ("stop_loss", str(index[3]), 99.0, 5.0),
    ]
    assert equity.iloc[2] == pytest.approx(10_125.0)

    # Mutation guard for R13: the entry bar is naked, so a TP1 sighting there must
    # move the next-bar stop to breakeven but must not silently cancel the TP1 limit.
    # Only the later, post-arm touch at bar 3 may close exactly half.
    _, post_entry_touch, _, touch_index = replay(
        [
            (100, 101, 99, 100),
            (100, 125, 95, 105),   # entry fill bar: reaches TP1 but cannot exit
            (105, 119, 101, 110),  # no second TP1 touch; position must remain whole
            (110, 121, 105, 119),  # re-armed TP1 limit fills 50%
        ],
        [_signal(rgd_long_break=True), _signal(), _signal(), _signal()],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["size"]) for item in post_entry_touch] == [
        ("take_profit_1", str(touch_index[3]), 120.0, 5.0)
    ]


def test_a14_trailing_stops_are_monotonic_for_long_and_short_positions():
    long_equity, long_trades, _, _ = replay(
        [
            (100, 101, 99, 100),
            (100, 110, 99, 105),   # long defense ratchets 90 -> 95
            (104, 110, 96, 100),   # defense falls to 90 but cannot pull stop down
            (100, 103, 94, 96),    # must fill at retained 95
        ],
        [
            _signal(rgd_long_break=True),
            _signal(rgd_def_long=95),
            _signal(rgd_def_long=90),
            _signal(rgd_def_long=80),
        ],
    )
    assert [(trade["status"], trade["exit_price"], trade["size"]) for trade in long_trades] == [
        ("stop_loss", 95.0, 10.0)
    ]
    assert long_equity.iloc[-1] == pytest.approx(9_950.0)

    short_equity, short_trades, _, _ = replay(
        [
            (100, 101, 99, 100),
            (100, 105, 90, 95),    # short defense ratchets 110 -> 105
            (96, 104, 90, 100),    # defense rises to 110 but cannot loosen stop upward
            (100, 106, 98, 104),   # must fill at retained 105
        ],
        [
            _signal(rgd_short_break=True),
            _signal(rgd_def_short=105),
            _signal(rgd_def_short=110),
            _signal(rgd_def_short=120),
        ],
    )
    assert [(trade["status"], trade["exit_price"], trade["size"]) for trade in short_trades] == [
        ("stop_loss", 105.0, 10.0)
    ]
    assert short_equity.iloc[-1] == pytest.approx(9_950.0)


def test_a15_intrabar_stop_and_tp1_double_touch_is_stop_first_without_partial_fill():
    _, trades, _, _ = replay(
        [
            (100, 101, 99, 100),
            (100, 110, 95, 105),
            # Open is below TP1; high crosses 120 and low crosses 90 in the same bar.
            (100, 125, 85, 100),
        ],
        [_signal(rgd_long_break=True), _signal(), _signal()],
    )
    assert [(item["status"], item["size"], item["exit_price"]) for item in trades] == [
        ("stop_loss", 10.0, 90.0)
    ]


def test_a16_open_gaps_fill_stop_and_limit_at_the_open_price():
    _, stop_trades, _, stop_index = replay(
        [(100, 101, 99, 100), (100, 110, 95, 105), (85, 86, 80, 84)],
        [_signal(rgd_long_break=True), _signal(), _signal()],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["size"]) for item in stop_trades] == [
        ("stop_loss", str(stop_index[2]), 85.0, 10.0)
    ]

    equity, limit_trades, _, limit_index = replay(
        [(100, 101, 99, 100), (100, 110, 95, 105), (125, 126, 121, 124)],
        [_signal(rgd_long_break=True), _signal(), _signal()],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["size"]) for item in limit_trades] == [
        ("take_profit_1", str(limit_index[2]), 125.0, 5.0)
    ]
    assert equity.attrs["open_position"]["qty"] == pytest.approx(5.0)


def test_a17_entry_fill_bar_is_unprotected_even_when_it_crosses_both_exit_levels():
    equity, trades, _, index = replay(
        [
            (100, 101, 99, 100),
            # The long fills at this open; this same bar pierces both TP1=120 and stop=90.
            (100, 130, 80, 100),
        ],
        [_signal(rgd_long_break=True), _signal()],
    )
    assert trades == []
    assert equity.attrs["open_position"] == {
        "side": "long", "qty": 10.0, "entry_price": 100.0, "entry_time": str(index[1]),
        "stop": 100.0, "target": 120.0, "entry_fee": 0.0, "unrealized_pnl": 0.0,
    }


def test_a18_trend_flip_is_direction_mode_only_while_bidirectional_mode_reverses():
    _, long_only, _, index = replay(
        [(100, 101, 99, 100), (100, 105, 95, 100), (101, 102, 100, 101)],
        [
            _signal(rgd_long_break=True, rgd_allow_short=False),
            _signal(rgd_state=-1, rgd_allow_short=False),
            _signal(rgd_state=-1, rgd_allow_short=False),
        ],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["side"]) for item in long_only] == [
        ("trend_flip", str(index[2]), 101.0, "long")
    ]

    _, short_only, _, index = replay(
        [(100, 101, 99, 100), (100, 105, 95, 100), (99, 100, 98, 99)],
        [
            _signal(rgd_short_break=True, rgd_allow_long=False),
            _signal(rgd_state=1, rgd_allow_long=False),
            _signal(rgd_state=1, rgd_allow_long=False),
        ],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["side"]) for item in short_only] == [
        ("trend_flip", str(index[2]), 99.0, "short")
    ]

    equity, bidirectional, _, index = replay(
        [(100, 101, 99, 100), (100, 105, 95, 100), (100, 105, 95, 100)],
        [
            _signal(rgd_long_break=True),
            _signal(rgd_short_break=True),
            _signal(),
        ],
    )
    assert [(item["status"], item["exit_time"], item["side"], item["size"]) for item in bidirectional] == [
        ("signal_reverse", str(index[2]), "long", 10.0)
    ]
    assert equity.attrs["open_position"]["side"] == "short"
    assert equity.attrs["open_position"]["qty"] == pytest.approx(10.0)


def test_a19_pyramiding_zero_ignores_same_side_and_reversal_charges_each_leg():
    equity, trades, _, _ = replay(
        [
            (100, 101, 99, 100),
            (100, 105, 95, 100),
            (100, 105, 95, 100),
            (100, 105, 95, 100),
        ],
        [
            _signal(rgd_long_break=True),
            _signal(rgd_long_break=True),      # same-side entry is ignored at next open
            _signal(rgd_short_break=True),     # opposite entry reverses at next open
            _signal(),
        ],
        fee=0.001,
    )
    assert [(item["status"], item["side"], item["size"], item["fees"]) for item in trades] == [
        ("signal_reverse", "long", 10.0, 2.0)
    ]
    # Fees: original long entry 1 + its exit 1 + newly opened short entry 1 = 3.
    assert equity.iloc[-1] == pytest.approx(9_997.0)
    assert equity.attrs["open_position"]["side"] == "short"
    # Reverse sizing uses signal-close equity after the first entry fee: 9999/100 = 9.999.
    # The old leg's close restores no price PnL, so the fill-time affordability cap does not bind.
    assert equity.attrs["open_position"]["qty"] == pytest.approx(9.999)
    assert equity.attrs["open_position"]["entry_fee"] == pytest.approx(0.9999)


def test_a22_cost_groups_have_hand_calculated_two_fee_two_slippage_round_trip():
    ohlc = [(100, 101, 99, 100), (100, 110, 95, 105), (100, 105, 85, 90)]
    rows = [_signal(rgd_long_break=True), _signal(), _signal()]
    g1_equity, g1, _, _ = replay(ohlc, rows, fee=0.0004, slippage=0.0002)
    g3_equity, g3, _, _ = replay(ohlc, rows, fee=0.0005, slippage=0.0004)

    # G1: entry=100*(1+.0002)=100.02, exit=90*(1-.0002)=89.982, q=10.
    assert g1[0]["entry_price"] == pytest.approx(100.02)
    assert g1[0]["exit_price"] == pytest.approx(89.982)
    assert g1[0]["fees"] == pytest.approx(0.40008 + 0.359928)
    assert g1[0]["pnl"] == pytest.approx(-101.140008)
    assert g1_equity.iloc[-1] == pytest.approx(9_898.859992)

    # G3 uses a distinct 0.05% fee and approximate two-tick proportional slip.
    assert g3[0]["entry_price"] == pytest.approx(100.04)
    assert g3[0]["exit_price"] == pytest.approx(89.964)
    assert g3[0]["fees"] == pytest.approx(0.5002 + 0.44982)
    assert g3[0]["pnl"] == pytest.approx(-101.71002)
    assert g3_equity.iloc[-1] == pytest.approx(9_898.28998)
    assert g1_equity.iloc[-1] != g3_equity.iloc[-1]


def test_a24_generic_direction_randomizer_is_noop_but_rgd_shift_changes_execution():
    _, original_trades, signals, _ = replay(
        [(100, 101, 99, 100), (100, 110, 95, 105), (100, 105, 85, 90)],
        [_signal(rgd_long_break=True), _signal(), _signal()],
    )
    generic = randomize_directions(signals, seed=1)
    pdt.assert_frame_equal(generic, signals)

    shifted = randomize_rgd(signals, seed=1)  # three rows + seed=1 => deterministic non-zero circular shift.
    assert not shifted.loc[:, DECISION_COLUMNS].equals(signals.loc[:, DECISION_COLUMNS])
    index = signals.index
    data = pd.DataFrame(
        [(100, 101, 99, 100), (100, 110, 95, 105), (100, 105, 85, 90)],
        columns=["open", "high", "low", "close"], index=index,
    )
    _, shifted_trades = simulate_rgd(data, shifted, _params(), 10_000.0, 0.0, 0.0, "15m")
    assert [(item["status"], item["exit_price"]) for item in original_trades] == [("stop_loss", 90.0)]
    assert shifted_trades == []


def test_a26_end_of_data_keeps_marked_open_position_without_forced_close():
    equity, trades, _, index = replay(
        [(100, 101, 99, 100), (100, 105, 95, 103)],
        [_signal(rgd_long_break=True), _signal()],
        fee=0.001,
    )
    assert trades == []
    # cash=9999 after entry fee; close 103 marks +3*10 gross floating profit.
    assert equity.iloc[-1] == pytest.approx(10_029.0)
    assert equity.attrs["open_position"] == {
        "side": "long", "qty": 10.0, "entry_price": 100.0, "entry_time": str(index[1]),
        "stop": 90.0, "target": 120.0, "entry_fee": 1.0, "unrealized_pnl": 30.0,
    }


def test_a29_equal_distance_ohlc_path_is_open_low_high_close():
    open_price, high, low, close = 100.0, 110.0, 90.0, 100.0
    assert abs(open_price - high) == abs(open_price - low)
    # A long stop lies at the lower extreme and TP1 at the upper extreme. Tie must visit low first.
    assert rgd_path(open_price, high, low, close) == (100.0, 90.0, 110.0, 100.0)


def test_a30_gap_through_tp1_then_same_bar_stop_produces_two_distinct_legs():
    equity, trades, _, index = replay(
        [
            (100, 101, 99, 100),
            (100, 110, 95, 105),
            # Existing stop=90, TP1=120. Open has already crossed TP1; low then crosses stop.
            (125, 126, 85, 90),
        ],
        [_signal(rgd_long_break=True), _signal(), _signal()],
    )
    assert [(item["status"], item["exit_time"], item["exit_price"], item["size"]) for item in trades] == [
        ("take_profit_1", str(index[2]), 125.0, 5.0),
        ("stop_loss", str(index[2]), 90.0, 5.0),
    ]
    assert sum(item["size"] for item in trades) == pytest.approx(10.0)
    assert equity.attrs["open_position"] is None
    assert equity.iloc[-1] == pytest.approx(10_075.0)
