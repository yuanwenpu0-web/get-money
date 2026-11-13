"""H-A 编排层测试：阈值选择隔离、正对照灵敏度、端到端信号/噪声特异性。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ha_common import DAY_MS  # noqa: E402
from ha_market_timing import (  # noqa: E402
    evaluate, perfect_foresight_net, select_threshold,
)
from ha_portfolio import constant_full_returns  # noqa: E402

BLOCKS = np.arange(60, dtype=np.int64) * DAY_MS


def _synthetic(seed: int, strength: float):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=60)
    ret = -strength * factor + 0.005 * rng.normal(size=60)
    return factor, ret


def test_select_threshold_depends_on_train_returns_and_is_deterministic():
    """阈值必须由训练段收益决定且可复现；隔离性由下一个端到端测试保证。"""
    factor, ret = _synthetic(1, 0.05)
    a, _ = select_threshold(factor[:40], ret[:40])
    b, _ = select_threshold(factor[:40], ret[:40] * -1)
    assert a != b  # 训练段收益确实影响选择
    c, _ = select_threshold(factor[:40], ret[:40])
    assert a == c


def test_evaluate_threshold_is_unchanged_when_oos_returns_change():
    factor, ret = _synthetic(2, 0.05)
    base = evaluate(BLOCKS, factor, ret, n_seeds=50, seed=5)
    perturbed = ret.copy()
    perturbed[40:] = perturbed[40:] * -3 + 0.01
    other = evaluate(BLOCKS, factor, perturbed, n_seeds=50, seed=5)
    assert base["threshold"] == other["threshold"]


def test_positive_control_beats_constant_full():
    """完美预见开关必须显著优于恒定满仓，否则量具没有灵敏度。"""
    ret = np.array([0.02, -0.03, 0.01, -0.02, 0.04, -0.01])
    pf = perfect_foresight_net(ret).sum()
    base = constant_full_returns(ret).sum()
    assert pf > base


def test_evaluate_detects_a_genuine_market_timing_signal():
    factor, ret = _synthetic(3, 0.05)
    out = evaluate(BLOCKS, factor, ret, n_seeds=200, seed=11)
    assert out["oos"]["rho"] < 0
    assert out["oos"]["switch_net"] > out["oos"]["baseline_net"]
    assert out["termination"]["terminated"] is False


def test_evaluate_rejects_pure_noise():
    rng = np.random.default_rng(4)
    factor = rng.normal(size=60)
    ret = 0.005 * rng.normal(size=60)
    out = evaluate(BLOCKS, factor, ret, n_seeds=200, seed=12)
    assert out["termination"]["terminated"] is True


def test_evaluate_rejects_insufficient_blocks():
    factor, ret = _synthetic(5, 0.05)
    with pytest.raises(ValueError, match="60"):
        evaluate(BLOCKS[:59], factor[:59], ret[:59], n_seeds=20, seed=1)


def test_evaluate_reports_market_background_and_cost_basis():
    factor, ret = _synthetic(6, 0.05)
    out = evaluate(BLOCKS, factor, ret, n_seeds=50, seed=13)
    assert out["market_background"]["cum_return"] == pytest.approx(
        float(np.prod(1 + ret) - 1), rel=1e-9)
    assert out["cost_model"]["round_trip"] == pytest.approx(0.0012)
    assert out["cost_model"]["basis"] == "notional"