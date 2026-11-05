"""趋势方向评估的口径测试：不重叠采样、收益约定、成本、无前视。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.trend_direction import (
    WARMUP_BARS, DirectionSpec, compute_directions, decision_points,
    evaluate_symbol, extract_direction,
)
from app.research.trend_direction_report import (
    bh_qvalues, cluster_bootstrap, spearman_ic,
)


def test_decision_points_non_overlapping_and_end_at_last():
    n, horizon = 1000, 48
    points = decision_points(n, horizon, 10, warmup=100)
    assert points[-1] == n - 1 - horizon
    assert all(b - a == horizon for a, b in zip(points, points[1:]))
    assert points[0] >= 100
    assert points == sorted(points)


def test_decision_points_empty_when_bars_short():
    assert decision_points(50, 48, 10, warmup=260) == []


def test_extract_direction_sign_diff_regime():
    index = pd.RangeIndex(4)
    sign = DirectionSpec("s", "s", ("v",), "t")
    out = pd.DataFrame({"v": [-2.0, 0.0, 3.0, 0.5]}, index=index)
    assert extract_direction(out, sign).tolist() == [-1.0, 0.0, 1.0, 1.0]

    diff = DirectionSpec("d", "d", ("bull", "bear"), "t", mode="diff")
    out = pd.DataFrame({"bull": [True, False, True, False],
                        "bear": [False, True, False, False]}, index=index)
    assert extract_direction(out, diff).tolist() == [1.0, -1.0, 1.0, 0.0]

    regime = DirectionSpec("r", "r", ("regime",), "t", mode="regime")
    out = pd.DataFrame({"regime": [-1, 0, 1, 2]}, index=index)
    assert extract_direction(out, regime).tolist() == [-1.0, 0.0, 1.0, 0.0]


def test_extract_direction_missing_column_raises():
    with pytest.raises(KeyError):
        extract_direction(pd.DataFrame({"v": [1]}), DirectionSpec("x", "x", ("nope",), "t"))


def _frame(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998, "close": close,
        "volume": 1000.0, "quote_volume": 1e6, "trades": 100,
    }, index=index)


def test_return_and_cost_convention():
    frame = _frame(320)
    horizon = 4
    points = decision_points(len(frame), horizon, 30, warmup=WARMUP_BARS)
    assert points
    direction = pd.Series(1.0, index=frame.index)
    free = evaluate_symbol(frame, {"x": direction}, horizon=horizon, cost=0.0, eval_days=30)
    paid = evaluate_symbol(frame, {"x": direction}, horizon=horizon, cost=0.001, eval_days=30)
    rec = free["indicators"]["x"]
    close = frame["close"].to_numpy()
    open_ = frame["open"].to_numpy()
    expected = np.array([close[t + horizon] / open_[t + 1] - 1 for t in points])
    assert np.allclose(rec["r"], expected.astype("float32"))
    assert np.allclose(rec["net"], rec["gross"])
    assert np.allclose(paid["indicators"]["x"]["net"], rec["gross"] - 0.001)
    assert rec["two_day"]["decision_time"] == str(frame.index[points[-1]])
    assert rec["two_day"]["realized_return"] == pytest.approx(expected[-1], rel=1e-6)
    assert free["two_day_decision_time"] == str(frame.index[points[-1]])


def test_neutral_direction_pays_no_cost():
    frame = _frame(320)
    result = evaluate_symbol(frame, {"x": pd.Series(0.0, index=frame.index)},
                             horizon=4, cost=0.001, eval_days=30)
    assert np.allclose(result["indicators"]["x"]["net"], 0.0)
    assert result["indicators"]["x"]["two_day"]["hit"] is False


def test_no_lookahead_on_truncated_frame():
    """指标必须因果：截断后重新计算，前段方向不能改变。"""
    frame = _frame(600)
    spec = DirectionSpec("ma_regime_bands", "ma_regime_bands", ("bull", "bear"),
                         "trend_direction", mode="diff")
    full, _ = compute_directions(frame, (spec,))
    part, _ = compute_directions(frame.iloc[:420], (spec,))
    assert np.array_equal(full["ma_regime_bands"].to_numpy()[:420],
                          part["ma_regime_bands"].to_numpy())


def test_bh_qvalues_ordered_by_p():
    q = bh_qvalues([0.001, 0.01, 0.04, 0.5])
    assert q[0] < q[1] < q[2] < q[3]
    assert all(0.0 <= value <= 1.0 for value in q)


def test_cluster_bootstrap_constant_series():
    out = cluster_bootstrap([0.01] * 20, draws=200)
    assert out["ci_low"] == pytest.approx(0.01)
    assert out["ci_high"] == pytest.approx(0.01)
    assert out["p_value"] == pytest.approx(1 / 200)


def test_spearman_ic_perfect_and_degenerate():
    d = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert spearman_ic(d, d * 0.01) == pytest.approx(1.0)
    assert spearman_ic(np.zeros(10), np.zeros(10)) is None
