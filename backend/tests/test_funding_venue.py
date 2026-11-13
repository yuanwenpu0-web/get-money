"""H-F Gate 1a 纯函数：现货对匹配、重平衡时点、窗口费率、选币。"""

import numpy as np
import pytest

from app.research.funding_venue import (
    DAY_MS,
    hedgeable_split,
    rebalance_times,
    select_top_k,
    spot_candidates,
    window_stats,
)


def test_spot_candidates_plain_symbol_is_base_asset_only():
    assert spot_candidates("BTCUSDT") == {"BTC"}


def test_spot_candidates_adds_thousand_stripped_variant():
    assert spot_candidates("1000BONKUSDT") == {"1000BONK", "BONK"}


def test_spot_candidates_strips_longest_multiplier_prefix_only():
    assert spot_candidates("1000000MOGUSDT") == {"1000000MOG", "MOG"}


def test_spot_candidates_keeps_thousand_prefixed_spot_base():
    assert "1000CAT" in spot_candidates("1000CATUSDT")


def test_hedgeable_split_matches_perp_to_spot_base_through_multiplier():
    out = hedgeable_split(["1000BONKUSDT", "BTCUSDT"], {"BONK", "BTC"})
    assert out == {"hedgeable": ["1000BONKUSDT", "BTCUSDT"], "non_hedgeable": []}


def test_hedgeable_split_is_exhaustive_disjoint_and_sorted():
    perps = ["ZZZUSDT", "1000BONKUSDT", "AAAUSDT", "BTCUSDT", "AAAUSDT"]
    out = hedgeable_split(perps, {"BTC"})
    assert out["hedgeable"] == ["BTCUSDT"]
    assert out["non_hedgeable"] == ["1000BONKUSDT", "AAAUSDT", "ZZZUSDT"]
    assert not set(out["hedgeable"]) & set(out["non_hedgeable"])
    assert len(out["hedgeable"]) + len(out["non_hedgeable"]) == len(set(perps))


def test_hedgeable_split_spot_bases_are_case_insensitive():
    assert hedgeable_split(["BTCUSDT"], {"btc"})["hedgeable"] == ["BTCUSDT"]


def test_rebalance_times_steps_by_hold_and_keeps_full_forward_window():
    start, end = 0, 20 * DAY_MS
    times = rebalance_times(start, end, lookback_days=7, hold_days=3)
    assert times[0] == 7 * DAY_MS
    assert all(b - a == 3 * DAY_MS for a, b in zip(times, times[1:]))
    assert times[-1] + 3 * DAY_MS <= end
    assert times[-1] + 6 * DAY_MS > end


def test_rebalance_times_empty_when_span_shorter_than_lookback_plus_hold():
    assert rebalance_times(0, 9 * DAY_MS, lookback_days=7, hold_days=3) == []


def test_window_stats_is_half_open_on_both_ends():
    times = np.array([10, 20, 30, 40], dtype="int64")
    rates = np.array([1.0, 2.0, 4.0, 8.0])
    mean, total, count = window_stats(times, rates, 20, 40)
    assert count == 2
    assert total == pytest.approx(6.0)
    assert mean == pytest.approx(3.0)


def test_window_stats_empty_window_returns_none_not_zero():
    times = np.array([10, 20], dtype="int64")
    rates = np.array([1.0, 2.0])
    assert window_stats(times, rates, 30, 40) == (None, None, 0)


def test_window_stats_counts_every_settlement_not_assumed_eight_hour():
    times = np.arange(0, 3 * DAY_MS, 4 * 3600 * 1000, dtype="int64")
    rates = np.full(times.size, 0.0001)
    mean, total, count = window_stats(times, rates, 0, 3 * DAY_MS)
    assert count == 18
    assert total == pytest.approx(18 * 0.0001)
    assert mean == pytest.approx(0.0001)


def test_select_top_k_ranks_descending_and_drops_non_positive():
    trailing = {"A": 0.001, "B": 0.003, "C": -0.002, "D": 0.0, "E": None}
    assert select_top_k(trailing, 5) == ["B", "A"]


def test_select_top_k_breaks_ties_by_symbol_ascending():
    trailing = {"Z": 0.002, "A": 0.002, "M": 0.002}
    assert select_top_k(trailing, 2) == ["A", "M"]


def test_select_top_k_returns_fewer_than_k_when_candidates_short():
    assert select_top_k({"A": 0.001}, 10) == ["A"]


def test_select_top_k_non_positive_k_is_empty():
    assert select_top_k({"A": 0.001}, 0) == []