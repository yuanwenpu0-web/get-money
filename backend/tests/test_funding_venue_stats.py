"""H-F Gate 1a 统计层：逐期毛费率、奖金占比、横截面 IC、年化与 D1–D3 判据。"""

import numpy as np
import pytest

from app.research.funding_venue import DAY_MS
from app.research.funding_venue_stats import (
    annualize_gross,
    block_bootstrap_ci,
    bonus_share,
    cross_section_ic,
    gate_checks,
    gross_period_return,
    paired_increment_ci,
    period_snapshot,
    pooled_share_ci,
    random_baseline,
    summarize_ic,
)

T = 100 * DAY_MS


def _book(step_hours, rate, start=0, end=200 * DAY_MS):
    times = np.arange(start, end, step_hours * 3600 * 1000, dtype="int64")
    return times, np.full(times.size, rate)


def test_period_snapshot_trailing_excludes_t_and_forward_starts_at_t():
    book = {"A": _book(8, 0.001)}
    snap = period_snapshot(book, T)
    assert snap["trailing"]["A"] == pytest.approx(0.001)
    assert snap["forward_events"]["A"] == 9
    assert snap["forward"]["A"] == pytest.approx(9 * 0.001)


def test_period_snapshot_counts_four_hour_settlements_as_eighteen_events():
    snap = period_snapshot({"A": _book(4, 0.0002)}, T)
    assert snap["forward_events"]["A"] == 18
    assert snap["forward"]["A"] == pytest.approx(18 * 0.0002)


def test_period_snapshot_missing_window_is_none_not_zero():
    times = np.array([T - 20 * DAY_MS], dtype="int64")
    snap = period_snapshot({"A": (times, np.array([0.001]))}, T)
    assert snap["trailing"]["A"] is None
    assert snap["forward"]["A"] is None
    assert snap["forward_events"]["A"] == 0


def test_gross_period_return_is_equal_weight_mean_of_forward_sums():
    trailing = {"A": 0.003, "B": 0.002, "C": 0.001}
    forward = {"A": 0.009, "B": 0.006, "C": 0.003}
    out = gross_period_return(trailing, forward, ["A", "B", "C"], 2)
    assert out["selected"] == ["A", "B"]
    assert out["return"] == pytest.approx(0.0075)


def test_gross_period_return_cannot_select_outside_its_universe():
    trailing = {"A": 0.003, "H": 0.009}
    forward = {"A": 0.009, "H": 0.027}
    out = gross_period_return(trailing, forward, ["A"], 5)
    assert out["selected"] == ["A"]
    assert out["return"] == pytest.approx(0.009)


def test_gross_period_return_drops_selected_without_forward_and_reports_them():
    trailing = {"A": 0.003, "B": 0.002}
    forward = {"A": 0.009, "B": None}
    out = gross_period_return(trailing, forward, ["A", "B"], 2)
    assert out["missing_forward"] == ["B"]
    assert out["return"] == pytest.approx(0.009)


def test_gross_period_return_none_when_no_selected_has_forward():
    out = gross_period_return({"A": 0.003}, {"A": None}, ["A"], 1)
    assert out["return"] is None


def test_bonus_share_pools_non_hedgeable_forward_over_total():
    selected = ["A", "B", "C"]
    forward = {"A": 0.010, "B": 0.006, "C": 0.004}
    out = bonus_share(selected, forward, {"B", "C"})
    assert out["total"] == pytest.approx(0.020)
    assert out["non_hedgeable"] == pytest.approx(0.010)
    assert out["share"] == pytest.approx(0.5)
    assert out["n_non_hedgeable"] == 2


def test_bonus_share_none_when_total_not_positive():
    out = bonus_share(["A"], {"A": -0.001}, {"A"})
    assert out["share"] is None


def test_cross_section_ic_requires_thirty_symbols():
    trailing = {f"S{i}": float(i) for i in range(29)}
    forward = {f"S{i}": float(i) for i in range(29)}
    ic, n = cross_section_ic(trailing, forward, list(trailing))
    assert ic is None
    assert n == 29


def test_cross_section_ic_perfect_monotone_is_one():
    symbols = [f"S{i}" for i in range(40)]
    trailing = {s: float(i) for i, s in enumerate(symbols)}
    forward = {s: float(i) * 2 for i, s in enumerate(symbols)}
    ic, n = cross_section_ic(trailing, forward, symbols)
    assert n == 40
    assert ic == pytest.approx(1.0)


def test_cross_section_ic_ignores_symbols_missing_either_side():
    symbols = [f"S{i}" for i in range(40)]
    trailing = {s: float(i) for i, s in enumerate(symbols)}
    forward = {s: None for s in symbols}
    forward["S0"] = 1.0
    ic, n = cross_section_ic(trailing, forward, symbols)
    assert ic is None
    assert n == 1


def test_annualize_gross_compounds_over_covered_days():
    assert annualize_gross([0.10], hold_days=365) == pytest.approx(0.10)
    assert annualize_gross([0.10, 0.10], hold_days=365) == pytest.approx(0.10)


def test_annualize_gross_skips_none_periods_and_scales_by_actual_coverage():
    assert annualize_gross([0.10, None], hold_days=365) == pytest.approx(0.10)


def test_annualize_gross_empty_is_none():
    assert annualize_gross([], hold_days=3) is None
    assert annualize_gross([None, None], hold_days=3) is None


def test_annualize_gross_three_day_period_scales_up():
    value = annualize_gross([0.003], hold_days=3)
    assert value == pytest.approx(1.003 ** (365 / 3) - 1)
    assert value > 0.40


def test_summarize_ic_reports_mean_positive_share_and_skips():
    out = summarize_ic([0.6, 0.4, -0.1, None])
    assert out["n_periods"] == 3
    assert out["n_skipped"] == 1
    assert out["mean"] == pytest.approx(0.3)
    assert out["positive_share"] == pytest.approx(2 / 3)


def test_gate_checks_all_thresholds_met():
    out = gate_checks(0.60, {"mean": 0.50, "positive_share": 0.90}, 25.0)
    assert out == {"D1": True, "D2": True, "D3": True, "pass": True}


@pytest.mark.parametrize("share,ic,d3,failed", [
    (0.59, {"mean": 0.9, "positive_share": 1.0}, 90.0, "D1"),
    (0.99, {"mean": 0.49, "positive_share": 1.0}, 90.0, "D2"),
    (0.99, {"mean": 0.9, "positive_share": 0.89}, 90.0, "D2"),
    (0.99, {"mean": 0.9, "positive_share": 1.0}, 24.9, "D3"),
])
def test_gate_checks_any_single_failure_closes_the_gate(share, ic, d3, failed):
    out = gate_checks(share, ic, d3)
    assert out[failed] is False
    assert out["pass"] is False


def test_gate_checks_missing_statistics_are_not_a_pass():
    out = gate_checks(None, {"mean": None, "positive_share": None}, None)
    assert out["pass"] is False
    assert not any(out[k] for k in ("D1", "D2", "D3"))


def test_gate_checks_does_not_mutate_caller_ic_summary():
    ic = {"mean": 0.6, "positive_share": 0.95}
    gate_checks(0.7, ic, 30.0)
    assert ic == {"mean": 0.6, "positive_share": 0.95}


def test_block_bootstrap_ci_needs_two_full_blocks():
    out = block_bootstrap_ci([0.004] * 10, block=8)
    assert out["ci_low"] is None and out["ci_high"] is None
    assert out["n"] == 10


def test_block_bootstrap_ci_constant_series_collapses():
    out = block_bootstrap_ci([0.004] * 40, block=8)
    assert out["ci_low"] == pytest.approx(0.004)
    assert out["ci_high"] == pytest.approx(0.004)
    assert out["mean"] == pytest.approx(0.004)


def test_block_bootstrap_ci_is_reproducible_and_seed_sensitive():
    values = list(np.sin(np.arange(60)) * 0.01)
    first = block_bootstrap_ci(values, block=8, seed=20261112)
    assert first == block_bootstrap_ci(values, block=8, seed=20261112)
    other = block_bootstrap_ci(values, block=8, seed=7)
    assert first["ci_low"] != other["ci_low"]
    assert first["ci_low"] < first["mean"] < first["ci_high"]


def test_block_bootstrap_ci_skips_none_periods():
    out = block_bootstrap_ci([0.004] * 40 + [None] * 5, block=8)
    assert out["n"] == 40


def test_paired_increment_ci_positive_when_full_dominates_every_period():
    out = paired_increment_ci([0.003] * 40, [0.0005] * 40, block=8, hold_days=3)
    expected = ((1.003 ** (365 / 3)) - (1.0005 ** (365 / 3))) * 100
    assert out["ci_low_pp"] == pytest.approx(expected, rel=1e-6)
    assert out["ci_high_pp"] == pytest.approx(expected, rel=1e-6)
    assert out["ci_low_pp"] > 0
    assert out["n_pairs"] == 40


def test_paired_increment_ci_identical_arms_give_zero_increment():
    rng = np.random.default_rng(3)
    full = list(rng.normal(0.0, 0.01, 60))
    out = paired_increment_ci(full, full, block=8, hold_days=3)
    assert out["ci_low_pp"] == pytest.approx(0.0)
    assert out["ci_high_pp"] == pytest.approx(0.0)


def test_paired_increment_ci_none_when_too_few_pairs():
    out = paired_increment_ci([0.003] * 5, [0.001] * 5, block=8)
    assert out["ci_low_pp"] is None and out["ci_high_pp"] is None


def test_paired_increment_ci_drops_periods_missing_either_arm():
    out = paired_increment_ci([0.003] * 40 + [None], [0.0005] * 41, block=8)
    assert out["n_pairs"] == 40


def test_pooled_share_ci_constant_share_collapses():
    bonus = [{"total": 0.02, "non_hedgeable": 0.015} for _ in range(40)]
    out = pooled_share_ci(bonus, block=8)
    assert out["ci_low"] == pytest.approx(0.75)
    assert out["ci_high"] == pytest.approx(0.75)


def test_pooled_share_ci_none_when_too_few_periods():
    bonus = [{"total": 0.02, "non_hedgeable": 0.015} for _ in range(4)]
    assert pooled_share_ci(bonus, block=8)["ci_low"] is None


def test_random_baseline_takes_whole_pool_when_k_exceeds_eligible():
    eligible = [["A", "B", "C", "D"]] * 30
    forward = [{"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04}] * 30
    out = random_baseline(eligible, forward, 10, n_seeds=5, hold_days=3)
    expected = (1.025 ** (365 / 3) - 1) * 100
    assert out["n_seeds"] == 5
    assert all(v == pytest.approx(expected) for v in out["samples_pct"])


def test_random_baseline_is_reproducible_and_varies_with_k():
    eligible = [[f"S{i}" for i in range(20)]] * 30
    forward = [{f"S{i}": 0.001 * (i + 1) for i in range(20)}] * 30
    first = random_baseline(eligible, forward, 3, n_seeds=20, seed=20261112)
    assert first == random_baseline(eligible, forward, 3, n_seeds=20, seed=20261112)
    assert len(set(first["samples_pct"])) > 1
    assert first["p05_pct"] <= first["median_pct"] <= first["p95_pct"]


def test_random_baseline_excludes_symbols_without_forward():
    eligible = [["A", "B"]] * 30
    forward = [{"A": 0.01, "B": None}] * 30
    out = random_baseline(eligible, forward, 5, n_seeds=3, hold_days=3)
    expected = (1.01 ** (365 / 3) - 1) * 100
    assert all(v == pytest.approx(expected) for v in out["samples_pct"])


def test_random_baseline_empty_eligible_is_not_fabricated():
    out = random_baseline([[]] * 30, [{}] * 30, 10, n_seeds=3)
    assert out["samples_pct"] == [None, None, None]
    assert out["mean_pct"] is None