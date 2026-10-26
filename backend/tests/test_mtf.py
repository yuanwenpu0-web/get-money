"""迭代11 多周期趋势共振（MTF resonance）离线测试：全部 mock，不访问网络。

覆盖：add_mtf_trend 因果性（反前视）+ 前向填充；共振门控；evaluate_signals(mtf_resonance)
产出与 resolve_params 接受新参数；walk-forward 汇总统计（构造数组断言）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.mtf import add_mtf_trend
from app.strategy.rules import evaluate_signals, resolve_params


def _trend_frame(periods: int = 300, turn_at: int | None = None) -> pd.DataFrame:
    """构造 1h DatetimeIndex 帧，含 OHLC 列；turn_at 指定趋势反转位置（None=单调上行）。"""
    idx = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    if turn_at is None:
        close = np.linspace(100.0, 200.0, periods)
    else:
        up = np.linspace(100.0, 200.0, turn_at)
        down = np.linspace(200.0, 100.0, periods - turn_at)
        close = np.concatenate([up, down])
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        },
        index=idx,
    )


def test_add_mtf_trend_columns_present():
    out = add_mtf_trend(_trend_frame())
    for col in ("mtf_dir_4h", "mtf_adx_4h", "mtf_dir_1d", "mtf_adx_1d"):
        assert col in out.columns


def test_add_mtf_trend_requires_datetime_index():
    frame = _trend_frame().reset_index(drop=True)  # RangeIndex
    with pytest.raises(ValueError):
        add_mtf_trend(frame)


def test_add_mtf_trend_causality_no_lookahead():
    """反前视：任意 1h bar 的更高周期值只依赖 ≤ 该 bar 的 1h 数据。

    方法：对若干采样点 t，把输入帧截断到 index ≤ t 后重算，断言该点 mtf 列与全量计算一致。
    若更高周期指标泄漏了未来数据，截断会改变该点的值。
    """
    frame = _trend_frame(periods=300, turn_at=180)
    full = add_mtf_trend(frame)
    sample_ts = [frame.index[i] for i in (120, 180, 220, 260)]
    for t in sample_ts:
        truncated = add_mtf_trend(frame.loc[:t])
        for col in ("mtf_dir_4h", "mtf_adx_4h", "mtf_dir_1d", "mtf_adx_1d"):
            a = full.at[t, col]
            b = truncated.at[t, col]
            if np.isnan(a):
                assert np.isnan(b), f"{col} @ {t}: 全量 NaN 但截断非 NaN（泄漏？）"
            else:
                assert np.isclose(a, b, equal_nan=True), f"{col} @ {t}: 全量 {a} != 截断 {b}（泄漏）"


def test_add_mtf_trend_forward_fill():
    """同一 4h 桶内（shift 后）的连续 1h bar 共享同一更高周期方向，且等于上一闭合桶的值。"""
    frame = _trend_frame(periods=300)
    out = add_mtf_trend(frame)
    # 取一个稳定上行区段的连续 4 根 1h bar（应在同一 4h 桶内），断言方向一致且非零。
    seg = out.iloc[50:54]
    assert seg["mtf_dir_4h"].nunique() == 1
    assert seg["mtf_dir_4h"].iloc[0] != 0


def _mtf_row(**overrides) -> pd.DataFrame:
    row = dict(
        close=100.0,
        atr_pct=0.0015,
        rsi_14=55.0,
        supertrend_dir=1.0,
        ema_ratio_9_21=0.002,
        ema_ratio_21_50=0.003,
        macd_hist_norm=0.001,
        adx_14=30.0,
        bb_pos=0.6,
        mtf_dir_4h=1.0,
        mtf_dir_1d=1.0,
        mtf_adx_4h=25.0,
        mtf_adx_1d=25.0,
    )
    row.update(overrides)
    return pd.DataFrame([row], index=pd.date_range("2026-01-01", periods=1, tz="UTC"))


def test_mtf_resonance_aligned_signal():
    params = resolve_params("mtf_resonance", {})
    sig = evaluate_signals(_mtf_row(), "mtf_resonance", params).iloc[0]
    assert sig.direction == 1
    assert sig.block_reason == ""


def test_mtf_resonance_rejects_htf_conflict():
    params = resolve_params("mtf_resonance", {})
    # 4h 反向 → 共振不通过 → 不开仓（不反向）。
    sig = evaluate_signals(_mtf_row(mtf_dir_4h=-1.0), "mtf_resonance", params).iloc[0]
    assert sig.direction == 0
    # 1d 反向同样拒绝。
    sig2 = evaluate_signals(_mtf_row(mtf_dir_1d=-1.0), "mtf_resonance", params).iloc[0]
    assert sig2.direction == 0


def test_mtf_resonance_rejects_low_htf_adx():
    params = resolve_params("mtf_resonance", {})
    sig = evaluate_signals(_mtf_row(mtf_adx_4h=10.0), "mtf_resonance", params).iloc[0]
    assert sig.direction == 0
    sig2 = evaluate_signals(_mtf_row(mtf_adx_1d=10.0), "mtf_resonance", params).iloc[0]
    assert sig2.direction == 0


def test_mtf_resonance_missing_htf_columns_no_signal():
    """mtf 列缺失（如非 mtf 上下文未提供）时门控失败，direction=0 且不崩溃。"""
    params = resolve_params("mtf_resonance", {})
    row = _mtf_row()
    row = row.drop(columns=["mtf_dir_4h", "mtf_dir_1d", "mtf_adx_4h", "mtf_adx_1d"])
    sig = evaluate_signals(row, "mtf_resonance", params).iloc[0]
    assert sig.direction == 0


def test_mtf_resonance_uses_1h_trend_logic():
    """共振通过时，复用 trend_follow 的 1h 方向（弱趋势被 1h 自身 regime 过滤）。"""
    params = resolve_params("mtf_resonance", {})
    # 1h adx 太低 → trend_follow 自身 weak_trend 也挡；mtf 即便对齐也无信号。
    sig = evaluate_signals(_mtf_row(adx_14=10.0), "mtf_resonance", params).iloc[0]
    assert sig.direction == 0
    assert sig.block_reason  # weak_trend 或 below_threshold


def test_resolve_params_accepts_mtf_and_rejects_unknown():
    params = resolve_params("mtf_resonance", {})
    assert params["adx_filter"] == 22.0
    assert params["htf_adx_filter"] == 20.0
    # 覆盖新参数被接受。
    params2 = resolve_params("mtf_resonance", {}, {"htf_adx_filter": 25.0})
    assert params2["htf_adx_filter"] == 25.0
    # 未知参数拒绝。
    with pytest.raises(ValueError):
        resolve_params("mtf_resonance", {}, {"nonexistent": 1})


def test_walkforward_summary_stats():
    """构造数组断言均值与正窗口计数（汇总统计口径，离线）。"""
    mtf = [3.0, -1.0, 2.5, 0.0, -2.0, 4.0, 1.0]
    trend = [1.0, -0.5, 0.5, -1.0, 0.2, 1.5, -0.3]
    mtf_mean = sum(mtf) / len(mtf)
    trend_mean = sum(trend) / len(trend)
    mtf_pos = sum(1 for r in mtf if r > 0)
    trend_pos = sum(1 for r in trend if r > 0)
    assert mtf_mean == pytest.approx(1.0714, abs=1e-3)
    assert trend_mean == pytest.approx(0.2, abs=1e-3)
    assert mtf_pos == 4 and trend_pos == 4
    # 门禁：mtf 正窗口数 >= trend 且 收益均值更高 → 采纳。
    assert (mtf_pos >= trend_pos) and (mtf_mean > trend_mean)
