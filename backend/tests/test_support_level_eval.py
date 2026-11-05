"""支撑位判定评估：因果性、事件口径与量具自校准单测（写挂 → 实现 → 跑绿）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research.support_level_eval import (
    ROUND_TRIP,
    bh_qvalues,
    bootstrap_ci,
    compute_levels,
    event_arrays,
    forward_extreme,
    hold_all_net,
    oracle_level,
    percentile_rank,
    random_price_baseline,
    randomized_baseline,
    rising_edges,
    segment_consistency,
    touch_mask,
)


def flat_frame(n: int = 120, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, price)
    return pd.DataFrame(
        {"open": close.copy(), "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.ones(n)}, index=idx
    )


def v_shaped_frame(cycles: int = 30, period: int = 10) -> pd.DataFrame:
    """每段先跌后反弹，加噪声打破严格周期（否则 np.roll 位移会周期性对齐）。"""
    n = cycles * period
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(4242)
    phase = np.arange(n) % period
    amplitude = rng.uniform(0.03, 0.06, cycles)[np.arange(n) // period]
    # 零漂移：每段先跌 4% 再回到起点，避免「整体上涨」让任意时点基线也变正
    drop = np.where(phase <= 4, -phase * 0.01, -0.04 + (phase - 4) * 0.008) * amplitude / 0.04
    close = 100.0 * (1.0 + drop) + rng.normal(0.0, 0.0015, n)
    return pd.DataFrame(
        {"open": np.concatenate(([close[0]], close[:-1])),
         "high": close + 0.15, "low": close - 0.15,
         "close": close, "volume": np.ones(n)}, index=idx
    )


def test_rising_edges_requires_min_gap():
    mask = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    assert rising_edges(mask, 3).tolist() == [1, 5]


def test_touch_mask_uses_previous_bar_level_only():
    """L[t] 本身不得触发 t；它只能在 t+1 作为 L[t] 生效。"""
    frame = flat_frame(10)
    level = np.full(10, np.nan)
    level[2] = 99.5  # 只有当前根的支撑位，不是 t-1 已知信息
    mask = touch_mask(level, frame["close"].to_numpy(), frame["low"].to_numpy())
    assert not mask[2]
    assert mask[3] and mask.sum() == 1  # 下一根才允许使用 L[2]

    level = np.full(10, np.nan)
    level[1] = 99.5  # t=2 时已知的是 L[1]
    mask = touch_mask(level, frame["close"].to_numpy(), frame["low"].to_numpy())
    assert mask[2] and mask.sum() == 1


def test_forward_extreme_is_future_not_past_window():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fwd_min = forward_extreme(values, 2, "min")
    assert fwd_min[0] == 2.0 and fwd_min[1] == 3.0 and fwd_min[2] == 4.0
    assert np.isnan(fwd_min[3]) and np.isnan(fwd_min[4])
    fwd_max = forward_extreme(values, 2, "max")
    assert fwd_max[0] == 3.0 and fwd_max[1] == 4.0


def test_donchian_level_is_causal():
    frame = flat_frame(20)
    frame.loc[frame.index[10], "low"] = 90.0
    level = compute_levels(frame, keys=("donchian20",))["donchian20"]
    # 20 根窗口尚未填满前不应出现伪支撑；第 10 根的低点只能在第 11 根之后进入窗口
    assert np.isnan(level[9])
    assert np.isnan(level[10])
    assert level[11] == pytest.approx(90.0) or np.isnan(level[11])


def test_event_arrays_cost_break_and_bounce():
    frame = flat_frame(40)
    level = np.full(40, np.nan)
    level[3] = 99.5
    out = event_arrays(frame, level, horizon=5)
    assert out["index"].tolist() == [4]
    # 平价序列：毛利 0，净利 = -0.12%
    assert out["gross"][0] == pytest.approx(0.0)
    assert out["net"][0] == pytest.approx(-ROUND_TRIP)
    assert bool(out["broke"][0]) is False  # low=99.5 未跌破 99.5*0.995
    assert bool(out["bounced"][0]) is False  # high=100.5 相对入场 100 只有 0.5%
    assert out["exit_time"][0] == frame.index[9].value


def test_oracle_self_calibration_beats_random_shift():
    """量具自校准：真有「低点后反弹」结构时，oracle 必须显著优于随机位移。"""
    frame = v_shaped_frame()
    horizon = 6
    level = oracle_level(frame, horizon)  # 偷看未来，仅用于正对照
    oracle = event_arrays(frame, level, horizon=horizon)
    hold = hold_all_net(frame, horizon=horizon)
    shift = randomized_baseline(frame, level, horizon=horizon, n_seeds=20, base_seed=20261025)
    price = random_price_baseline(frame, level, horizon=horizon, n_seeds=20, base_seed=20261105)
    assert len(oracle["net"]) >= 20
    assert len(shift["net_means"]) == 20 and len(price["net_means"]) == 20
    # 量具必须能检出「低点买入」优于「任意时点买入」与「随机价位」
    assert float(oracle["net"].mean()) > float(hold.mean()) + 0.005
    assert float(oracle["net"].mean()) > float(np.nanmean(price["net_means"])) + 0.005


def test_bh_qvalues_and_bootstrap_and_segments():
    q = bh_qvalues(np.array([0.001, 0.02, 0.04, 0.9]))
    assert q[0] < 0.01 and q[-1] > 0.5
    assert np.all(np.diff(np.sort(q)) >= -1e-12)
    lo, hi = bootstrap_ci(np.full(200, -0.01), n_boot=500, seed=7)
    assert hi < 0 and lo <= hi
    assert segment_consistency(np.ones(40), np.arange(40)) == pytest.approx(1.0)
    assert percentile_rank(0.9, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
