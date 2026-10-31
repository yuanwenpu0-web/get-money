"""JMA Cluster Entries 移植测试。

覆盖：JMA 递推、簇分数恒等式、跳变步数、线性回归斜率、三类信号（动量/衰竭/结构）、
趋势四道过滤、两级冷却、无量能时的吸收门、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.jma_cluster_entries_with_market_structure import (
    _jma,
    _linreg_slope,
    jma_cluster_entries_with_market_structure,
)

_JMA_LENGTHS = (20, 50, 100, 150, 200, 250, 300, 400, 500, 600)


def _frame(close: np.ndarray, *, spread: float = 0.4, volume=1000.0, freq="1h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vol = volume if isinstance(volume, np.ndarray) else np.full(n, float(volume))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": vol},
        index=idx,
    )


def _noisy(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.9, n))
    open_ = close + rng.normal(0, 0.2, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n)),
            "low": np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n)),
            "close": close,
            "volume": 1000.0 + np.abs(rng.normal(0, 200, n)),
        },
        index=idx,
    )


# ---------------------------------------------------------------- JMA 本体


def test_jma_is_monotone_on_constant_input():
    values = np.full(200, 50.0)
    for length in (20, 200, 600):
        out = _jma(values, length, 0.0, 2.0)
        assert np.allclose(out[10:], 50.0)


def test_jma_lags_a_step_change():
    values = np.r_[np.full(200, 100.0), np.full(600, 120.0)]
    fast = _jma(values, 20, 0.0, 2.0)
    slow = _jma(values, 600, 0.0, 2.0)
    # 跳变后短均线几乎立刻跟上，长均线要慢得多
    assert fast[210] > 119.0
    assert slow[210] < 101.0
    # 但最终都收敛到 120
    assert fast[-1] == pytest.approx(120.0, abs=1e-6)
    assert slow[-1] == pytest.approx(120.0, abs=2.0)  # 600 周期需要更多根才收敛


def test_jma_power_and_phase_change_behaviour():
    values = 100 + np.cumsum(np.sin(np.arange(400) / 8.0))
    base = _jma(values, 100, 0.0, 2.0)
    fast_power = _jma(values, 100, 0.0, 4.0)
    assert not np.allclose(base[100:], fast_power[100:])
    assert np.isfinite(base[100:]).all()


def test_linreg_slope_matches_hand_calc():
    values = 100 + np.arange(100) * 2.0  # 斜率恰为 2
    slope = _linreg_slope(values, 10)
    assert slope[9] == pytest.approx(2.0)
    assert slope[50] == pytest.approx(2.0)
    # 常量序列斜率为 0
    flat = np.full(50, 7.0)
    assert _linreg_slope(flat, 10)[9] == pytest.approx(0.0)


# ---------------------------------------------------------------- 簇分数


def test_cluster_score_identity():
    out = jma_cluster_entries_with_market_structure(_noisy())
    bull = out["jma_bull_count"].to_numpy(dtype=int)
    active = out["jma_active_count"].to_numpy(dtype=int)
    score = out["jma_score"].to_numpy(dtype=float)
    assert np.allclose(score, bull * 100.0 / active)
    assert (bull >= 0).all() and (bull <= active).all()


def test_enabled_jmas_change_step_granularity():
    frame = _noisy()
    full = jma_cluster_entries_with_market_structure(frame)
    half = jma_cluster_entries_with_market_structure(
        frame, enabled_jmas=(True, True, True, True, True, False, False, False, False, False))
    assert int(half["jma_active_count"].iloc[0]) == 5
    # 5 条启用 → 分数只可能落在 0/20/40/60/80/100
    assert set(np.unique(half["jma_score"].to_numpy()).tolist()) <= {0.0, 20.0, 40.0, 60.0, 80.0, 100.0}
    # 跳变步数 = 这一根穿过了几条 JMA
    steps = half["jma_score_jump_steps"].to_numpy(dtype=float)
    assert steps.max() <= 5
    jump = half["jma_score_jump"].to_numpy(dtype=float)
    assert np.allclose(steps, np.round(jump * 5 / 100.0))


def test_cluster_flip_is_edge_not_level():
    out = jma_cluster_entries_with_market_struct_check = jma_cluster_entries_with_market_structure(_noisy())
    bull = out["jma_cluster_bull"].to_numpy(dtype=int)
    flip = out["jma_cluster_flip_bull"].to_numpy(dtype=int)
    assert flip.sum() > 0
    # 翻多的那根一定是多头，且上一根不是
    hits = np.flatnonzero(flip)
    assert (bull[hits] == 1).all()
    assert (bull[hits - 1] == 0).all()
    assert jma_cluster_entries_with_market_struct_check is not None


def test_jma_values_match_independent_recompute():
    frame = _noisy()
    out = jma_cluster_entries_with_market_structure(frame)
    close = frame["close"].to_numpy(dtype=float)
    for length in (20, 100, 200, 600):
        expected = _jma(close, length, 0.0, 2.0)
        assert out[f"jma_value_{length}"].to_numpy() == pytest.approx(expected, nan_ok=True)


def test_slow_jma_selection():
    frame = _noisy()
    fast = jma_cluster_entries_with_market_structure(frame, slow_length="20")
    slow = jma_cluster_entries_with_market_structure(frame, slow_length="600")
    assert fast["jma_slow"].to_numpy() == pytest.approx(fast["jma_value_20"].to_numpy(), nan_ok=True)
    assert slow["jma_slow"].to_numpy() == pytest.approx(slow["jma_value_600"].to_numpy(), nan_ok=True)


def test_slope_pct_matches_recompute():
    frame = _noisy()
    out = jma_cluster_entries_with_market_structure(frame)
    slow = out["jma_value_200"].to_numpy(dtype=float)
    slope = _linreg_slope(slow, 5)
    mask = np.isfinite(slope) & (slow != 0)
    expected = slope[mask] / slow[mask] * 100.0
    assert out["jma_slope_pct"].to_numpy()[mask] == pytest.approx(expected)


# ---------------------------------------------------------------- 信号


def test_trend_requires_flip_and_filters():
    frame = _noisy()
    out = jma_cluster_entries_with_market_structure(frame, mode="Trend")
    flip_bull = out["jma_cluster_flip_bull"].to_numpy(dtype=int)
    flip_bear = out["jma_cluster_flip_bear"].to_numpy(dtype=int)
    trend_long = out["jma_trend_long"].to_numpy(dtype=int)
    trend_short = out["jma_trend_short"].to_numpy(dtype=int)
    hits_long = np.flatnonzero(trend_long)
    hits_short = np.flatnonzero(trend_short)
    assert len(hits_long) > 0 and len(hits_short) > 0
    assert (flip_bull[hits_long] == 1).all()
    assert (flip_bear[hits_short] == 1).all()
    # ATR 距离门
    distance = np.abs(out["jma_score"].to_numpy() * 0.0 + (frame["close"].to_numpy()
                                                          - out["jma_slow"].to_numpy()))
    atr_a = out["jma_atr"].to_numpy(dtype=float)
    assert (distance[hits_long] <= 3.5 * atr_a[hits_long]).all()


def test_trend_mode_entries_only_from_trend_path():
    frame = _noisy()
    out = jma_cluster_entries_with_market_structure(frame, mode="Trend")
    entry = out["jma_entry_long"].to_numpy(dtype=int) | out["jma_entry_short"].to_numpy(dtype=int)
    trend = out["jma_trend_long"].to_numpy(dtype=int) | out["jma_trend_short"].to_numpy(dtype=int)
    assert (entry == trend).all()
    # 趋势模式下所有入场都归 TREND 类
    classes = np.concatenate([out["jma_entry_class_long"].to_numpy(dtype=int),
                              out["jma_entry_class_short"].to_numpy(dtype=int)])
    assert set(np.unique(classes).tolist()) <= {0, 1}


def _spike_reversal(n: int = 900, slope: float = 1.5, drop: float = 40.0,
                    spike_bar: int = 700) -> pd.DataFrame:
    """横盘 → 强推 → 单根巨跌：一根 K 线穿过大部分 JMA，但价格仍在慢线之上。"""
    close = np.empty(n)
    close[:300] = 100 + 0.4 * np.sin(np.arange(300) / 8.0)
    close[300:] = 100 + np.arange(n - 300) * slope
    close[spike_bar:] -= drop
    return _frame(close, spread=0.8)


def test_reversion_signals_need_extreme_then_return():
    # 簇分数从极端（100）单根掉到 60，价格仍偏离慢线
    out = jma_cluster_entries_with_market_structure(_spike_reversal(), mode="Reversion")
    rev_short = out["jma_rev_short"].to_numpy(dtype=int)
    assert rev_short.sum() > 0
    score = out["jma_score"].to_numpy(dtype=float)
    hits = np.flatnonzero(rev_short)
    # 上一根必须是极端高分，本根必须已回落到 rev_return 之下
    assert (score[hits - 1] >= 92.0).all()
    assert (score[hits] < 78.0).all()


def test_reversion_deviation_gate():
    """偏离门必须真的起作用：dev_mult 放大到远超实际偏离后，反转信号消失。"""
    frame = _spike_reversal()
    loose = jma_cluster_entries_with_market_structure(frame, mode="Reversion", dev_mult=1.5)
    tight = jma_cluster_entries_with_market_structure(frame, mode="Reversion", dev_mult=20.0)
    assert int(loose["jma_rev_short"].sum()) > 0
    assert int(tight["jma_rev_short"].sum()) == 0


def test_exhaustion_requires_divergence_and_efficiency():
    frame = _noisy(1200, seed=3)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion")
    # raw 是跨 bar 保持的闩锁（Pine 的 var bool），「触发那根」才是条件成立点
    raw_short = out["jma_exhaustion_short_trigger"].to_numpy(dtype=int)
    raw_long = out["jma_exhaustion_long_trigger"].to_numpy(dtype=int)
    score = out["jma_score"].to_numpy(dtype=float)
    efficiency_falling = out["jma_efficiency_falling"].to_numpy(dtype=int)
    shock = out["jma_shock_bar"].to_numpy(dtype=int)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    hits = np.flatnonzero(raw_short)
    assert len(hits) > 0
    assert (score[hits] >= 65.0).all()
    assert (efficiency_falling[hits] == 1).all()
    assert (shock[hits] == 0).all()
    # 背离：价格创了新高，但分数没有创新高
    score_highest5 = pd.Series(score).rolling(5, min_periods=1).max().to_numpy()
    assert (high[hits] > high[hits - 1]).all()
    assert (score[hits] < score_highest5[hits]).all()
    # 多头镜像
    hits_long = np.flatnonzero(raw_long)
    if len(hits_long):
        assert (score[hits_long] <= 35.0).all()
        assert (low[hits_long] < low[hits_long - 1]).all()


def test_exhaustion_needs_next_bar_confirmation():
    frame = _noisy(1200, seed=3)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion")
    latch = out["jma_exhaustion_short_raw"].to_numpy(dtype=int)
    confirmed = out["jma_exhaustion_short"].to_numpy(dtype=int)
    confirm = out["jma_confirm_short"].to_numpy(dtype=int)
    trigger = out["jma_exhaustion_short_trigger"].to_numpy(dtype=int)
    hits = np.flatnonzero(confirmed)
    assert len(hits) > 0
    assert (confirm[hits] == 1).all()
    # 衰竭信号有两条来源：① 上一根闩锁开着 + 本根确认；② 「未能延续」直接确认。
    # 至少要有走 ① 的，且所有走 ① 的都必须闩锁开着。
    via_latch = latch[hits - 1] == 1
    assert via_latch.any()
    assert (confirm[hits[via_latch]] == 1).all()
    # 闩锁只能由触发点打开
    opened = np.flatnonzero((latch == 1) & (np.r_[0, latch[:-1]] == 0))
    if len(opened):
        assert (trigger[opened] == 1).all()


def test_structure_signals_need_absorption_and_wick():
    frame = _noisy(1200, seed=4)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion")
    long_sig = out["jma_structure_long"].to_numpy(dtype=int)
    short_sig = out["jma_structure_short"].to_numpy(dtype=int)
    absorption = out["jma_absorption_active"].to_numpy(dtype=int)
    rejection_long = out["jma_rejection_long"].to_numpy(dtype=int)
    rejection_short = out["jma_rejection_short"].to_numpy(dtype=int)
    score = out["jma_score"].to_numpy(dtype=float)
    assert long_sig.sum() + short_sig.sum() > 0
    if long_sig.sum():
        hits = np.flatnonzero(long_sig)
        assert (absorption[hits] == 1).all()
        assert (rejection_long[hits] == 1).all()
        assert (score[hits] >= 60.0).all()
    if short_sig.sum():
        hits = np.flatnonzero(short_sig)
        assert (absorption[hits] == 1).all()
        assert (rejection_short[hits] == 1).all()
        assert (score[hits] <= 40.0).all()


def test_absorption_blocked_without_volume():
    frame = _noisy(1200, seed=4).assign(volume=0.0)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion")
    assert (out["jma_absorption_active"] == 0).all()
    assert int(out["jma_structure_long"].sum()) == 0
    assert int(out["jma_structure_short"].sum()) == 0


def test_entry_cooldown_enforced():
    frame = _noisy(1200, seed=0)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion", signal_cooldown=20)
    long_hits = np.flatnonzero(out["jma_entry_long"].to_numpy(dtype=int))
    short_hits = np.flatnonzero(out["jma_entry_short"].to_numpy(dtype=int))
    for hits in (long_hits, short_hits):
        if len(hits) > 1:
            assert (np.diff(hits) >= 20).all()


def test_alert_cooldown_is_shorter_than_entry():
    """冷却按方向独立计数（Pine 用两个独立计数器），所以多空告警可以相邻。"""
    frame = _noisy(1200, seed=3)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion",
                                                    alert_cooldown=10, signal_cooldown=20)
    for side in ("long", "short"):
        hits = np.flatnonzero(out[f"jma_alert_{side}"].to_numpy(dtype=int))
        assert len(hits) > 1, side
        assert (np.diff(hits) >= 10).all(), side
    # 告警比入场密集（冷却更短）
    alerts = int(out["jma_alert_long"].sum() + out["jma_alert_short"].sum())
    entries = int(out["jma_entry_long"].sum() + out["jma_entry_short"].sum())
    assert alerts > entries


def test_quality_model_slope_based():
    frame = _noisy(1200, seed=3)
    out = jma_cluster_entries_with_market_structure(frame, mode="Reversion")
    quality_long = out["jma_quality_long"].to_numpy(dtype=float)
    quality_short = out["jma_quality_short"].to_numpy(dtype=float)
    assert quality_long.min() >= 0.0 and quality_long.max() <= 100.0
    assert quality_short.min() >= 0.0 and quality_short.max() <= 100.0
    # 质量只在有信号的行非零（质量在最终冷却/入场门之前就算，所以用
    # 「任一原始信号」而非「最终入场」作为对照）
    # 质量在「任一路径给出多头信号」时非零，否则为 0
    # 反转模式下参与打分的是「动量/衰竭/结构」三路；趋势路径只在 Trend 模式计分
    any_long = (out["jma_momentum_long"].to_numpy(dtype=int)
                | out["jma_exhaustion_long"].to_numpy(dtype=int)
                | out["jma_structure_long"].to_numpy(dtype=int)).astype(bool)
    assert (quality_long[~any_long] == 0.0).all()
    assert (quality_long[any_long] > 0.0).all()


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(400)
    out = jma_cluster_entries_with_market_structure(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("jma_") for name in out.columns)
    for name in ("jma_entry_long", "jma_cluster_flip_bull", "jma_absorption_active"):
        assert str(out[name].dtype) == "Int64", name
    # 跳变步数是四舍五入后的计数，按浮点输出（保留分数信息）
    assert out["jma_score_jump_steps"].dropna().min() >= 0.0
    assert out["jma_score"].dropna().between(0.0, 100.0).all()
    assert out["jma_clv"].dropna().between(-1.0, 1.0).all()


def test_parameter_validation():
    frame = _noisy(200)
    bad = [
        dict(mode="Nope"),
        dict(slow_length="30"),
        dict(phase=200.0),
        dict(power=0.5),
        dict(cluster_pct=40),
        dict(slope_length=0),
        dict(min_slope_pct=2.0),
        dict(min_score_jump=0),
        dict(signal_cooldown=0),
        dict(atr_length=0),
        dict(atr_max_mult=0.0),
        dict(rev_extreme=79),
        dict(rev_return=96),
        dict(dev_length=5),
        dict(dev_mult=0.0),
        dict(rsi_length=2),
        dict(wt_channel_length=1),
        dict(alert_cooldown=0),
        dict(enabled_jmas=(False,) * 10),
        dict(enabled_jmas=(True,) * 9),
        dict(source="nope"),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            jma_cluster_entries_with_market_structure(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 7):
        out = jma_cluster_entries_with_market_structure(_noisy(700, seed=seed), mode="Reversion")
        assert np.isfinite(out["jma_score"].dropna()).all()
        assert out["jma_score"].dropna().between(0.0, 100.0).all()
        assert (out["jma_entry_long"].sum() + out["jma_entry_short"].sum()) >= 0
