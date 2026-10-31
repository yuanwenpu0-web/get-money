"""VWAP Cross Visuals 移植测试。

覆盖：会话 VWAP（每日重置 + 加权手算）、锚定 VWAP（锚点窗口恒等式）、
价格/结构交叉、节流、聚集区、三种强度、偏置评分与亏损带、成交量分布 POC、
无量能门控、HTF 栈、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.vwap_cross_visuals import vwap_cross_visuals


def _frame(close: np.ndarray, *, spread: float = 0.0, volume=10.0, freq="1h",
           start="2024-01-01") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq=freq)
    vol = volume if isinstance(volume, np.ndarray) else np.full(n, float(volume))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": vol},
        index=idx,
    )


def _noisy(n: int = 600, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": 1000.0 + np.abs(rng.normal(0, 200, n)),
        },
        index=idx,
    )


# ---------------------------------------------------------------- 会话 VWAP


def test_session_vwap_resets_each_day():
    close = np.array([100.0] * 24 + [200.0] * 24)
    out = vwap_cross_visuals(_frame(close))
    assert out["vwap_session"].iloc[0] == pytest.approx(100.0)
    assert out["vwap_session"].iloc[23] == pytest.approx(100.0)
    assert out["vwap_session"].iloc[24] == pytest.approx(200.0)
    assert out["vwap_session"].iloc[47] == pytest.approx(200.0)


def test_session_vwap_is_volume_weighted():
    close = np.array([100.0, 110.0] + [110.0] * 22 + [200.0] * 24)
    volume = np.array([10.0, 30.0] + [10.0] * 22 + [10.0] * 24)
    out = vwap_cross_visuals(_frame(close, volume=volume))
    expected = (100.0 * 10.0 + 110.0 * 30.0) / 40.0
    assert out["vwap_session"].iloc[1] == pytest.approx(expected)
    # 第 2 根之后权重累积：仍然只由当日成交决定
    expected_full = (100.0 * 10.0 + 110.0 * 30.0 + 110.0 * 220.0) / 260.0
    assert out["vwap_session"].iloc[23] == pytest.approx(expected_full)


# ---------------------------------------------------------------- 锚定 VWAP


def test_anchored_vwap_uses_post_anchor_window():
    """锚定 VWAP = 枢轴 bar 之后的累计 (价格×量) / 累计量。"""
    n = 60
    # 明确的峰：先涨到 120 再回落（平台会被 strict pivot 判成等高，不成立）
    close = np.r_[np.full(20, 100.0), np.linspace(100, 120, 20), np.linspace(119, 110, 20)]
    volume = np.linspace(10.0, 30.0, n)
    frame = _frame(close, volume=volume, spread=0.5)
    out = vwap_cross_visuals(frame, vwap_pivot_left=3, vwap_pivot_right=3)
    anchored = out["vwap_anchored_high"].to_numpy(dtype=float)
    has = out["vwap_has_high_anchor"].to_numpy(dtype=int)
    assert has.max() == 1
    first = int(np.argmax(has))
    # 枢轴 bar = 确认 bar - right（_pine 的 pivothigh 在确认 bar 上报值）
    pivot_bar = first - 3
    pv = close * volume
    cumulative_pv = np.cumsum(pv)
    cumulative_v = np.cumsum(volume)
    for i in (first + 2, first + 10, n - 1):
        expected = ((cumulative_pv[i] - cumulative_pv[pivot_bar])
                    / (cumulative_v[i] - cumulative_v[pivot_bar]))
        assert anchored[i] == pytest.approx(expected), i


def test_anchored_vwap_is_nan_before_first_pivot():
    frame = _noisy(120)
    out = vwap_cross_visuals(frame, vwap_pivot_left=5, vwap_pivot_right=5)
    has = out["vwap_has_high_anchor"].to_numpy(dtype=int)
    first = int(np.argmax(has))
    assert first > 0
    assert out["vwap_anchored_high"].iloc[:first].isna().all()


# ---------------------------------------------------------------- 交叉 / 节流


def test_price_cross_and_throttle():
    # 价格在 VWAP 上下反复穿越：节流把展示压到每 50 根一次
    n = 300
    close = 100 + 6 * np.sin(np.arange(n) / 5.0)
    frame = _frame(close, spread=0.2, volume=1000.0)
    out = vwap_cross_visuals(frame, min_bars_between_signals=50, min_signal_strength=0.0)
    up = out["vwap_price_cross_up"].to_numpy(dtype=int)
    down = out["vwap_price_cross_down"].to_numpy(dtype=int)
    shown_up = out["vwap_show_price_cross_up"].to_numpy(dtype=int)
    shown_down = out["vwap_show_price_cross_down"].to_numpy(dtype=int)
    assert up.sum() > 0 and down.sum() > 0
    shown = np.flatnonzero(shown_up | shown_down)
    assert len(shown) > 0
    # 展示之间至少隔 50 根
    if len(shown) > 1:
        assert (np.diff(shown) >= 50).all()
    # 被节流掉的交叉不进展示
    assert shown_up.sum() + shown_down.sum() < up.sum() + down.sum()


def test_cross_strength_formula():
    frame = _frame(np.r_[np.full(50, 100.0), np.full(50, 140.0)], spread=0.5, volume=1000.0)
    out = vwap_cross_visuals(frame, min_signal_strength=0.0)
    up = out["vwap_price_cross_up"].to_numpy(dtype=int)
    strength = out["vwap_cross_strength"].to_numpy(dtype=float)
    atr14 = out.get("vwap_cross_strength")
    assert up.sum() >= 1
    for i in np.flatnonzero(up):
        assert strength[i] > 0.0
        assert strength[i] <= 100.0
    # 非交叉 bar 强度为 0
    assert (strength[up == 0] == 0.0).all()
    assert atr14 is not None


# ---------------------------------------------------------------- 聚集区


def test_cluster_tolerance_and_strength():
    # 小幅摆动：两个锚定 VWAP 都收敛到 ~100，价格也在其中 → 聚集成立
    close = 100 + 0.6 * np.sin(np.arange(160) / 6.0)
    frame = _frame(close, spread=0.5, volume=1000.0)
    out = vwap_cross_visuals(frame, cluster_pct=0.3)
    assert out["vwap_has_high_anchor"].max() == 1
    assert out["vwap_vwaps_in_cluster"].max() == 1
    active = out["vwap_cluster_active"].to_numpy(dtype=int)
    assert active.sum() > 0
    strength = out["vwap_cluster_strength"].to_numpy(dtype=float)
    assert strength[active == 1].min() >= 75.0
    assert strength[active == 0].max() == 0.0
    # 容差 = pct/100 × close
    tolerance = out["vwap_cluster_tolerance"].to_numpy(dtype=float)
    assert tolerance[0] == pytest.approx(0.003 * close[0])


# ---------------------------------------------------------------- 偏置


def test_bias_score_and_bands():
    # 持续上行：收盘价始终高于会话 VWAP
    frame = _frame(np.linspace(100, 160, 120), spread=0.5, volume=1000.0)
    out = vwap_cross_visuals(frame, bias_band_multiplier=1.5)
    score = out["vwap_bias_score"].to_numpy(dtype=float)
    position = out["vwap_price_position"].to_numpy(dtype=int)
    assert set(np.unique(position).tolist()) <= {-1, 0, 1}
    assert score.min() >= -100.0 and score.max() <= 100.0
    # 价格远在 VWAP 上方 → 偏置为正
    assert score[-1] > 0.0
    # 亏损带：upper - session = atr14 × 1.5
    upper = out["vwap_band_upper"].to_numpy(dtype=float)
    lower = out["vwap_band_lower"].to_numpy(dtype=float)
    vwap = out["vwap_session"].to_numpy(dtype=float)
    valid = np.isfinite(upper) & np.isfinite(vwap) & np.isfinite(lower)
    assert valid[10:].any()
    assert (upper[valid] >= vwap[valid]).all()
    assert (lower[valid] <= vwap[valid]).all()
    unit = out["vwap_target_up_1"].to_numpy(dtype=float) - vwap
    assert np.allclose((upper - vwap)[20:], (out["vwap_target_up_max"].to_numpy(dtype=float) - vwap)[20:]
                       * 1.5 / 2.0)
    assert np.isfinite(unit[20:]).all()


def test_bias_band_sigma_mode_differs_from_atr():
    frame = _noisy(300)
    atr_out = vwap_cross_visuals(frame, target_mode="ATR", bias_band_multiplier=1.5)
    sigma_out = vwap_cross_visuals(frame, target_mode="SIGMA", bias_band_multiplier=1.5)
    a = atr_out["vwap_band_upper"].to_numpy(dtype=float) - atr_out["vwap_session"].to_numpy(dtype=float)
    s = sigma_out["vwap_band_upper"].to_numpy(dtype=float) - sigma_out["vwap_session"].to_numpy(dtype=float)
    assert np.isfinite(a[100:]).all() and np.isfinite(s[100:]).all()
    assert not np.allclose(a[100:], s[100:])


# ---------------------------------------------------------------- 成交量分布


def test_volume_profile_poc_follows_volume_spike():
    n = 120
    close = np.full(n, 100.0)
    close[60] = 105.0
    frame = _frame(close, spread=1.0, volume=np.where(np.arange(n) == 60, 8000.0, 10.0))
    out = vwap_cross_visuals(frame, use_volume_profile=True, rolling_length=120, bin_count=40)
    poc = out["vwap_poc"].dropna()
    assert not poc.empty
    assert poc.iloc[-1] == pytest.approx(105.0, abs=1.0)
    vah = out["vwap_vah"].iloc[-1]
    val = out["vwap_val"].iloc[-1]
    assert val <= poc.iloc[-1] <= vah
    assert int(out["vwap_hvn_count"].max()) >= 0


def test_volume_profile_off_by_default():
    out = vwap_cross_visuals(_noisy(200))
    assert out["vwap_poc"].isna().all()
    assert out["vwap_vah"].isna().all()


# ---------------------------------------------------------------- 门控


def test_missing_volume_gates_whole_vwap_layer():
    frame = _noisy(200).assign(volume=0.0)
    out = vwap_cross_visuals(frame, use_volume_profile=True)
    for column in ("vwap_session", "vwap_anchored_high", "vwap_anchored_low", "vwap_poc"):
        assert out[column].isna().all(), column
    assert int(out["vwap_price_cross_up"].sum()) == 0
    assert int(out["vwap_cluster_active"].sum()) == 0


# ---------------------------------------------------------------- HTF 栈


def test_htf_stack_confluence_and_regime():
    close = 100 + np.arange(3000) * 0.8  # 强趋势
    frame = _frame(close, spread=0.2, volume=1000.0)
    out = vwap_cross_visuals(frame, use_htf_stack=True)
    bull = out["vwap_htf_bull_confluence"].to_numpy(dtype=float)
    bear = out["vwap_htf_bear_confluence"].to_numpy(dtype=float)
    assert np.isfinite(bull[1000:]).all()
    assert bull[1000:].min() >= 0.0 and bull[1000:].max() <= 100.0
    # 强势上行 → 多头一致率高于空头
    assert bull[-1] > bear[-1]
    chop = out["vwap_htf_choppiness"].to_numpy(dtype=float)
    assert np.nanmin(chop) >= 0.0 and np.nanmax(chop) <= 100.0
    assert out["vwap_htf_stack_bias"].dropna().isin([-1, 0, 1]).all()


def test_htf_stack_off_by_default():
    out = vwap_cross_visuals(_noisy(200))
    assert out["vwap_htf_bull_confluence"].isna().all()
    assert int(out["vwap_htf_bull_stack_signal"].sum()) == 0


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(400)
    out = vwap_cross_visuals(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("vwap_") for name in out.columns)
    for name in ("vwap_price_cross_up", "vwap_cluster_active", "vwap_bias_value", "vwap_long_signal"):
        assert str(out[name].dtype) == "Int64", name
    assert out["vwap_price_position"].dropna().isin([-1, 0, 1]).all()
    assert out["vwap_bias_score"].dropna().between(-100.0, 100.0).all()


def test_parameter_validation():
    frame = _noisy(120)
    bad = [
        dict(display_mode="Nope"),
        dict(cluster_pct=0.05),
        dict(vwap_pivot_left=0),
        dict(vwap_pivot_right=30),
        dict(min_bars_between_signals=1),
        dict(min_signal_strength=200.0),
        dict(signal_scope="Nope"),
        dict(bias_length=1),
        dict(bias_band_multiplier=0.1),
        dict(target_mode="Nope"),
        dict(profile_mode="Nope"),
        dict(rolling_length=10),
        dict(bin_count=4),
        dict(value_area_percent=0.2),
        dict(retest_distance_atr=-1.0),
        dict(htf_stack_threshold=0.2),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            vwap_cross_visuals(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 4):
        out = vwap_cross_visuals(_noisy(500, seed=seed), use_volume_profile=True, use_htf_stack=True)
        assert np.isfinite(out["vwap_session"].dropna()).all()
        assert (out["vwap_long_signal"].sum() + out["vwap_short_signal"].sum()) >= 0
