"""H-A 仓位开关组合、成本、随机基线与事前终止条件的测试。

成本口径固定名义本金双边 0.12%（fee 0.0004 + slippage 0.0002，单腿 0.0006），
精确按 fee*(1+exit/entry) 计，不用固定 2×fee 近似。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ha_portfolio import (  # noqa: E402
    FEE_RATE, SLIPPAGE, constant_full_returns, fit_markov,
    matched_random_switches, paired_bootstrap_ci, percentile_rank, spearman,
    switch_from_signal, switch_net_returns, termination_checks,
    time_segment_sign_share,
)

C = FEE_RATE + SLIPPAGE


def test_switch_is_on_when_factor_below_threshold():
    """预登记方向：OI z 高＝杠杆拥挤→空仓；低于阈值才满仓做多市场。"""
    on = switch_from_signal(np.array([-1.0, 0.0, 1.0, np.nan]), threshold=0.5)
    np.testing.assert_array_equal(on, [True, True, False, False])


def test_zero_cost_net_equals_position_times_return():
    ret = np.array([0.01, -0.02, 0.03, 0.0])
    on = np.array([True, True, False, True])
    net = switch_net_returns(ret, on, fee=0.0, slippage=0.0)
    np.testing.assert_allclose(net, ret * on, atol=1e-15)


def test_all_off_costs_nothing():
    net = switch_net_returns(np.array([0.05, -0.05, 0.02]), np.zeros(3, bool))
    np.testing.assert_allclose(net, np.zeros(3), atol=1e-15)


def test_single_full_segment_cost_uses_exact_exit_notional():
    ret = np.array([0.10, 0.10])
    net = switch_net_returns(ret, np.ones(2, bool))
    growth = float(np.prod(1 + ret))
    assert net.sum() == pytest.approx(ret.sum() - C * (1 + growth), abs=1e-12)


def test_two_round_trips_cost_two_times_round_trip_when_flat_returns():
    ret = np.zeros(4)
    on = np.array([True, False, True, False])
    net = switch_net_returns(ret, on)
    assert net.sum() == pytest.approx(-2 * 2 * C, abs=1e-12)


def test_constant_full_is_single_round_trip():
    ret = np.array([0.01, 0.02, -0.01, 0.03])
    net = constant_full_returns(ret)
    growth = float(np.prod(1 + ret))
    assert net.sum() == pytest.approx(ret.sum() - C * (1 + growth), abs=1e-12)


def test_fit_markov_counts_stay_transitions():
    p_on, p_off = fit_markov(np.array([True, True, False, False, True, True]))
    assert p_on == pytest.approx(2 / 3)
    assert p_off == pytest.approx(0.5)


def test_fit_markov_absorbing_when_constant():
    p_on, _ = fit_markov(np.ones(5, bool))
    assert p_on == pytest.approx(1.0)


def test_matched_baseline_is_reproducible_and_preserves_exposure_exactly():
    rng = np.random.default_rng(3)
    on = rng.random(400) < 0.4
    a = matched_random_switches(on, n_seeds=200, seed=20261112)
    b = matched_random_switches(on, n_seeds=200, seed=20261112)
    assert a.shape == (200, 400)
    assert a.dtype == bool
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a.sum(axis=1), np.full(200, on.sum()))


def test_matched_baseline_preserves_turnover_exactly():
    """随机基线必须与真实信号逐序列同换手，否则成本差会伪造出 alpha。"""
    on = np.array([True] * 7 + [False] * 2 + [True] * 3 + [False] * 5
                  + [True] * 4 + [False] * 1 + [True] * 6 + [False] * 2)
    draws = matched_random_switches(on, n_seeds=200, seed=7)
    real_sw = int(np.sum(on[1:] != on[:-1]))
    rand_sw = np.sum(draws[:, 1:] != draws[:, :-1], axis=1)
    np.testing.assert_array_equal(rand_sw, np.full(200, real_sw))
    assert not np.all(np.array_equal(draws[0], draws[1]))  # 确实随机化了时点


def test_matched_baseline_degenerates_to_real_signal_when_single_run():
    on = np.ones(30, bool)
    draws = matched_random_switches(on, n_seeds=20, seed=1)
    np.testing.assert_array_equal(draws, np.tile(on, (20, 1)))


def test_paired_bootstrap_ci_collapses_on_constant_series():
    out = paired_bootstrap_ci(np.full(60, 0.002), n_boot=500, seed=1)
    assert out["ci_low"] == pytest.approx(0.002)
    assert out["ci_high"] == pytest.approx(0.002)
    assert out["n"] == 60


def test_paired_bootstrap_ci_is_reproducible():
    x = np.random.default_rng(5).normal(size=80)
    a = paired_bootstrap_ci(x, n_boot=300, seed=42)
    b = paired_bootstrap_ci(x, n_boot=300, seed=42)
    assert a == b


def test_spearman_perfect_negative_and_nan_handling():
    assert spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, np.nan, 3]), np.array([1.0, 2, np.nan])) is None


def test_percentile_rank_counts_strictly_below():
    assert percentile_rank(5.0, np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])) == pytest.approx(0.4)


def test_time_segment_sign_share_all_positive():
    assert time_segment_sign_share(np.ones(40), n_segments=4) == pytest.approx(1.0)


def test_time_segment_sign_share_mixed():
    x = np.concatenate([np.ones(10), np.ones(10), np.ones(10), -np.ones(10)])
    assert time_segment_sign_share(x, n_segments=4) == pytest.approx(0.75)


_BASE = dict(train_rho=-0.30, oos_rho=-0.25, switch_net=0.05, baseline_net=0.01,
             ci_low=0.002, random_p90=0.02, n_blocks=60, segment_sign_share=1.0)


def test_termination_passes_when_all_conditions_hold():
    out = termination_checks(**{**_BASE, "switch_net": 0.05})
    assert out["terminated"] is False
    assert out["reasons"] == []


def test_termination_triggers_on_oos_sign_flip():
    out = termination_checks(**{**_BASE, "oos_rho": 0.25})
    assert out["terminated"] is True
    assert any("sign" in r for r in out["reasons"])


def test_termination_triggers_on_weak_oos_rho():
    out = termination_checks(**{**_BASE, "oos_rho": -0.05})
    assert out["terminated"] is True
    assert any("rho" in r for r in out["reasons"])


def test_termination_triggers_when_not_beating_baseline():
    out = termination_checks(**{**_BASE, "switch_net": 0.005})
    assert out["terminated"] is True


def test_termination_triggers_on_ci_crossing_zero():
    out = termination_checks(**{**_BASE, "ci_low": -0.001})
    assert out["terminated"] is True


def test_termination_triggers_below_random_p90():
    out = termination_checks(**{**_BASE, "switch_net": 0.015})
    assert out["terminated"] is True
    assert any("p90" in r for r in out["reasons"])


def test_termination_triggers_on_too_few_blocks():
    out = termination_checks(**{**_BASE, "n_blocks": 29})
    assert out["terminated"] is True


def test_termination_triggers_on_segment_sign_below_three_quarters():
    out = termination_checks(**{**_BASE, "segment_sign_share": 0.5})
    assert out["terminated"] is True