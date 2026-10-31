"""Oscillator Cycle Statistics 移植测试。

覆盖：11 种振荡器口径、量能阻塞、区间映射、周期状态机、进度恒等式、
经验统计的桶/年龄/回溯语义、Wilson 区间、参数校验。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators._pine import cci, mfi, rsi, sma, stoch
from app.catalog_indicators.oscillator_cycle_statistics import (
    _direction_base_rates,
    _SampleStore,
    _statistics,
    _wilson,
    oscillator_cycle_statistics,
)


def _frame(close: np.ndarray, *, spread: float = 0.5, volume=1000.0, freq="1h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vol = volume if isinstance(volume, np.ndarray) else np.full(n, float(volume))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": vol},
        index=idx,
    )


def _noisy(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.9, n))
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


def _sine(n: int = 240, amplitude: float = 8.0, period: float = 9.0) -> pd.DataFrame:
    return _frame(100 + amplitude * np.sin(np.arange(n) / period), spread=0.2)


# ---------------------------------------------------------------- 振荡器口径


def test_rsi_matches_primitive():
    frame = _noisy()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    expected = np.clip(rsi(frame["close"], 14).to_numpy(dtype=float), 0.0, 100.0)
    assert out["ocs_osc"].to_numpy() == pytest.approx(expected, nan_ok=True)


def test_stochastic_matches_primitive():
    frame = _noisy()
    out = oscillator_cycle_statistics(frame, oscillator="Stochastic")
    raw = stoch(frame["close"], frame["high"], frame["low"], 14)
    expected = np.clip(sma(raw, 3).to_numpy(dtype=float), 0.0, 100.0)
    assert out["ocs_osc"].to_numpy() == pytest.approx(expected, nan_ok=True)
    # Williams %R 归一化后与 Fast %K 同源
    williams = oscillator_cycle_statistics(frame, oscillator="Williams %R (normalized = Fast Stoch %K)")
    assert williams["ocs_osc"].to_numpy() == pytest.approx(np.clip(raw.to_numpy(dtype=float), 0.0, 100.0),
                                                           nan_ok=True)


def test_cci_and_mfi_normalisation():
    frame = _noisy()
    cci_out = oscillator_cycle_statistics(frame, oscillator="CCI")
    expected = np.clip(50.0 + cci(frame, 14).to_numpy(dtype=float) * (20.0 / 100.0), 0.0, 100.0)
    assert cci_out["ocs_osc"].to_numpy() == pytest.approx(expected, nan_ok=True)

    mfi_out = oscillator_cycle_statistics(frame, oscillator="MFI (volume)")
    raw = mfi(frame, 14).to_numpy(dtype=float)
    assert mfi_out["ocs_osc"].to_numpy() == pytest.approx(np.where(np.isfinite(raw), raw, 50.0), nan_ok=True)


def test_demarker_matches_reference():
    frame = _noisy()
    out = oscillator_cycle_statistics(frame, oscillator="DeMarker")
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    prev_high = np.r_[np.nan, high[:-1]]
    prev_low = np.r_[np.nan, low[:-1]]
    idx = frame.index
    up = sma(pd.Series(np.where(high > prev_high, high - prev_high, 0.0), index=idx), 14).to_numpy(dtype=float)
    dn = sma(pd.Series(np.where(low < prev_low, prev_low - low, 0.0), index=idx), 14).to_numpy(dtype=float)
    denom = up + dn
    expected = np.where(denom > 0, 100.0 * up / np.where(denom > 0, denom, 1.0), 50.0)
    assert out["ocs_osc"].to_numpy() == pytest.approx(expected, nan_ok=True)


def test_volume_oscillators_go_neutral_without_volume():
    frame = _noisy()
    silent = frame.assign(volume=np.zeros(len(frame)))
    for name in ("MFI (volume)", "EOM (volume)", "Klinger (volume)"):
        out = oscillator_cycle_statistics(silent, oscillator=name)
        assert out["ocs_osc"].unique().tolist() == [50.0]
        assert int(out["ocs_volume_blocked"].iloc[0]) == 1
        assert int(out["ocs_new_cycle"].sum()) == 0  # 中性振荡器永远不开周期
    # 有真实量能时不阻塞
    live = oscillator_cycle_statistics(frame, oscillator="Klinger (volume)")
    assert int(live["ocs_volume_blocked"].iloc[0]) == 0


# ---------------------------------------------------------------- 区间 / 状态


def test_zone_mapping_and_defaults():
    frame = _noisy()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    assert float(out["ocs_overbought"].iloc[0]) == 70.0
    assert float(out["ocs_oversold"].iloc[0]) == 30.0
    assert float(out["ocs_extreme_overbought"].iloc[0]) == 90.0
    assert float(out["ocs_extreme_oversold"].iloc[0]) == 10.0

    eighty = oscillator_cycle_statistics(frame, oscillator="Stochastic")
    assert float(eighty["ocs_overbought"].iloc[0]) == 80.0
    assert float(eighty["ocs_oversold"].iloc[0]) == 20.0

    custom = oscillator_cycle_statistics(
        frame, use_oscillator_default_zones=False, custom_overbought=75.0, custom_oversold=25.0
    )
    assert float(custom["ocs_overbought"].iloc[0]) == 75.0
    assert float(custom["ocs_oversold"].iloc[0]) == 25.0

    # 极值门不弱于 90/10
    tight = oscillator_cycle_statistics(
        frame, use_oscillator_default_zones=False, custom_overbought=95.0, custom_oversold=5.0
    )
    assert float(tight["ocs_extreme_overbought"].iloc[0]) == 95.0
    assert float(tight["ocs_extreme_oversold"].iloc[0]) == 5.0


def test_zone_state_codes():
    frame = _noisy()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    osc = out["ocs_osc"].to_numpy(dtype=float)
    state = out["ocs_zone_state"].to_numpy(dtype=int)
    ob, os_ = 70.0, 30.0
    for value, code in zip(osc, state):
        if not np.isfinite(value):
            continue
        if value <= 10.0:
            assert code == 0
        elif value <= os_:
            assert code == 1
        elif value >= 90.0:
            assert code == 6
        elif value >= ob:
            assert code == 5
        elif value < 45.0:
            assert code == 2
        elif value <= 55.0:
            assert code == 3
        else:
            assert code == 4


# ---------------------------------------------------------------- 状态机


def test_cycle_state_machine_on_sine():
    frame = _sine()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    osc = out["ocs_osc"].to_numpy(dtype=float)
    prev = np.r_[np.nan, osc[:-1]]
    ob, os_ = 70.0, 30.0
    entered_ob = (osc >= ob) & (np.where(np.isfinite(prev), prev, ob - 1.0) < ob)
    entered_os = (osc <= os_) & (np.where(np.isfinite(prev), prev, os_ + 1.0) > os_)

    direction = out["ocs_active_direction"].to_numpy(dtype=int)
    new_cycle = out["ocs_new_cycle"].to_numpy(dtype=int)
    # 交叉上穿 OB → 开启 OB→OS 周期
    assert new_cycle[entered_ob].all()
    assert (direction[entered_ob] == -1).all()
    assert (direction[entered_os] == 1).all()
    # 成功必然发生在对侧极值
    success = out["ocs_success"].to_numpy(dtype=int) == 1
    assert success.any()
    assert ((osc[success] <= os_) | (osc[success] >= ob)).all()


def test_progress_matches_definition():
    frame = _sine()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    osc = out["ocs_osc"].to_numpy(dtype=float)
    direction = out["ocs_active_direction"].to_numpy(dtype=int)
    progress = out["ocs_progress"].to_numpy(dtype=float)
    span = 70.0 - 30.0
    expected = np.where(
        direction == -1,
        np.clip((70.0 - osc) / span, 0.0, 1.0),
        np.where(direction == 1, np.clip((osc - 30.0) / span, 0.0, 1.0), 0.0),
    )
    assert progress == pytest.approx(expected, nan_ok=True)


def test_max_progress_and_samples_grow():
    frame = _sine()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    max_progress = out["ocs_max_progress"].to_numpy(dtype=float)
    assert (max_progress >= 0.0).all()
    assert (max_progress <= 1.0 + 1e-12).all()
    assert int(out["ocs_sample_count"].iloc[-1]) >= 5
    # 每完成一个周期样本数单调不减
    counts = out["ocs_sample_count"].to_numpy(dtype=int)
    assert (np.diff(counts) >= 0).all()


def test_armed_only_after_leaving_origin_zone():
    frame = _sine()
    out = oscillator_cycle_statistics(frame, oscillator="RSI")
    osc = out["ocs_osc"].to_numpy(dtype=float)
    armed = out["ocs_armed"].to_numpy(dtype=int)
    direction = out["ocs_active_direction"].to_numpy(dtype=int)
    fresh = np.zeros(len(out), dtype=bool)
    fresh[1:] = (direction[1:] != 0) & (direction[:-1] != direction[1:])
    # 周期第一根一定未武装
    assert (armed[fresh] == 0).all()
    # 一旦武装，在同一周期内不会回退（除非周期换了）
    for i in range(1, len(out)):
        if direction[i] != direction[i - 1] or direction[i] == 0:
            continue
        if armed[i - 1] == 1:
            assert armed[i] == 1


# ---------------------------------------------------------------- 统计语义


def _seeded_store() -> _SampleStore:
    store = _SampleStore(capacity=100)
    rows = [
        (-1, 1, 0, 0, 20, 100, 0.50, 2.0, 1.0, np.array([1, 2, 3, 4, 5, -1, -1, -1, -1, -1])),
        (-1, 0, 1, 0, 30, 110, 0.95, 1.0, 3.0, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, -1])),
        (-1, 1, 0, 0, 15, 120, 0.30, 1.5, 0.5, np.array([1, 2, 3, -1, -1, -1, -1, -1, -1, -1])),
        (1, 1, 0, 0, 10, 130, 1.00, 1.2, 0.4, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
    ]
    for row in rows:
        store.push(row)
    return store


def test_statistics_bucket_and_age_filters():
    store = _seeded_store()
    plain = _statistics(store, 140, -1, 1000, False, 0, 0.0, -1, True)
    assert (plain["success"], plain["total"]) == (2, 3)
    assert plain["prob"] == pytest.approx(200.0 / 3.0)

    # 进度 >= 0.5 的只有两个样本（0.50 成功 / 0.95 失败）
    bucketed = _statistics(store, 140, -1, 1000, False, 0, 0.50, -1, True)
    assert (bucketed["cond_success"], bucketed["cond_total"]) == (1, 2)
    assert bucketed["cond_prob"] == pytest.approx(50.0)

    # 年龄门 25：只剩 duration 30 的那个（first_reach[4]=5 <= 25），且它是失败的
    aged = _statistics(store, 140, -1, 1000, False, 0, 0.50, 25, True)
    assert (aged["cond_success"], aged["cond_total"]) == (0, 1)

    # 回溯窗
    assert _statistics(store, 140, -1, 20, False, 0, 0.0, -1, False)["total"] == 1
    assert _statistics(store, 140, -1, 45, False, 0, 0.0, -1, False)["total"] == 3
    # 状态过滤
    assert _statistics(store, 140, -1, 1000, True, 0, 0.0, -1, False)["total"] == 3
    assert _statistics(store, 140, -1, 1000, True, 1, 0.0, -1, False)["total"] == 0
    # 方向基准
    assert _direction_base_rates(store, 140, 1000) == pytest.approx((200.0 / 3.0, 3, 100.0, 1))


def test_statistics_medians_split_by_outcome():
    store = _seeded_store()
    stats = _statistics(store, 140, -1, 1000, False, 0, 0.0, -1, True)
    # 成功样本 duration 20 / 15 → 中位数 17.5，MFE 2.0/1.5 → 1.75，MAE 1.0/0.5 → 0.75
    assert stats["median_duration"] == pytest.approx(17.5)
    assert stats["median_mfe"] == pytest.approx(1.75)
    assert stats["median_mae"] == pytest.approx(0.75)
    # 失败样本 MFE 1.0 / MAE 3.0
    assert stats["median_failed_mfe"] == pytest.approx(1.0)
    assert stats["median_failed_mae"] == pytest.approx(3.0)


def test_wilson_interval_matches_hand_calc():
    lower, upper = _wilson(2, 3)
    z, n, p = 1.96, 3.0, 2.0 / 3.0
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    margin = z / (1.0 + z2 / n) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    assert lower == pytest.approx(100.0 * max(0.0, center - margin))
    assert upper == pytest.approx(100.0 * min(1.0, center + margin))
    # 空样本
    assert math.isnan(_wilson(0, 0)[0])


def test_store_capacity_drops_oldest():
    store = _SampleStore(capacity=3)
    for k in range(5):
        store.push((-1, 1, 0, 0, k, 10 + k, 1.0, 1.0, 1.0, np.zeros(10, dtype=np.int64)))
    arrays = store.arrays()
    assert arrays[0].size == 3
    assert arrays[5].tolist() == [12, 13, 14]


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _noisy(200)
    out = oscillator_cycle_statistics(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("ocs_") for name in out.columns)
    for name in ("ocs_zone_state", "ocs_regime", "ocs_new_cycle", "ocs_success", "ocs_sample_count"):
        assert str(out[name].dtype) == "Int64", name
    assert out["ocs_osc"].dropna().between(0.0, 100.0).all()
    assert out["ocs_cond_prob"].dropna().between(0.0, 100.0).all()


def test_parameter_validation():
    frame = _noisy(120)
    bad = [
        dict(oscillator="Nope"),
        dict(primary_length=1),
        dict(smooth_k=0),
        dict(wt_channel_length=1),
        dict(wt_average_length=1),
        dict(cci_reference=1.0),
        dict(klinger_fast_length=60, klinger_slow_length=55),
        dict(custom_overbought=100.0),
        dict(custom_oversold=50.0),
        dict(max_cycle_bars=1),
        dict(recent_lookback=10),
        dict(medium_lookback=50),
        dict(long_lookback=100),
        dict(max_stored_samples=10),
        dict(min_conditional_samples=1),
        dict(regime_ema_length=10),
        dict(regime_slope_bars=0),
        dict(regime_atr_length=1),
        dict(regime_slope_threshold=9.0),
        dict(calib_checkpoint=100.0),
        dict(opportunity_edge_threshold=1.0),
        dict(opportunity_zone_min=90, opportunity_zone_max=10),
        dict(opportunity_max_ci_width=1.0),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            oscillator_cycle_statistics(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 7):
        out = oscillator_cycle_statistics(_noisy(400, seed=seed))
        assert np.isfinite(out["ocs_osc"].dropna()).all()
        assert out["ocs_stage"].dropna().isin([0, 1, 2, 3, 4, 5]).all()
        assert out["ocs_label_state"].dropna().isin([0, 1, 2, 3, 4, 5, 6]).all()
        assert int(out["ocs_active_direction"].abs().max()) <= 1
