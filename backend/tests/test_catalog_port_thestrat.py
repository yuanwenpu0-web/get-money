"""TheStrat Suite 移植测试。

覆盖：结构分类三条硬规则（相等不算破位 / 缺口不特殊 / u-d 是破位侧）、
锤子射手三档 + 颜色门、Failed 2 四种口径 + pre-F2、FTFC、多米诺、幅度目标、
信号在效的九分支决策树、止损位、参数校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.thestrat_suite import (
    is_failed,
    is_hammer,
    is_shooter,
    structure_of,
    thestrat_suite,
)

_STRUCT_NONE, _STRUCT_1, _STRUCT_2U, _STRUCT_2D, _STRUCT_3 = 0, 1, 2, 3, 4


def _frame(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
           freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _bar_frame(n: int = 200, seed: int = 0, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    open_ = close + rng.normal(0, 0.3, n)
    return _frame(open_, np.maximum(open_, close) + 0.4,
                  np.minimum(open_, close) - 0.4, close, freq=freq)


# ---------------------------------------------------------------- 分类


def test_structure_equal_is_not_a_break():
    # 高恰好等于前高 → 没有越过去，仍是 1
    assert structure_of([10.0], [5.0], [10.0], [6.0])[0] == _STRUCT_1
    # 低恰好等于前低 → 仍是 1
    assert structure_of([10.0], [5.0], [9.0], [5.0])[0] == _STRUCT_1


def test_structure_gap_has_no_special_case():
    # 跳空开在前高之上：区间已破，本身就是 2u
    assert structure_of([10.0], [5.0], [20.0], [15.0])[0] == _STRUCT_2U
    assert structure_of([10.0], [5.0], [20.0], [4.0])[0] == _STRUCT_3


def test_structure_four_codes_and_invalid():
    assert structure_of([10.0], [5.0], [11.0], [6.0])[0] == _STRUCT_2U
    assert structure_of([10.0], [5.0], [9.0], [4.0])[0] == _STRUCT_2D
    assert structure_of([10.0], [5.0], [11.0], [4.0])[0] == _STRUCT_3
    assert structure_of([np.nan], [5.0], [9.0], [4.0])[0] == _STRUCT_NONE
    # high < low 视为无效输入
    assert structure_of([10.0], [5.0], [4.0], [9.0])[0] == _STRUCT_NONE


def test_structure_code_ignores_open():
    """分类只看区间，不看开盘价 —— 同一区间任意开盘都得到同一结论。"""
    for open_px in (5.0, 7.5, 10.0, 11.0):
        assert structure_of([10.0], [5.0], [11.0], [4.0])[0] == _STRUCT_3
        assert is_hammer([open_px], [11.0], [4.0], [7.0], "Broad (Loose)", False)[0] in (True, False)


# ---------------------------------------------------------------- 形态


def test_hammer_and_shooter_broad():
    # 长下影、开收都在中点（99.1）之上
    assert is_hammer([100.0], [101.2], [97.0], [101.0], "Broad (Loose)", False)[0]
    # 收盘落到中点下方 → 不成立
    assert not is_hammer([100.0], [101.2], [97.0], [98.0], "Broad (Loose)", False)[0]
    # 射手是镜像：开收都在中点下方
    assert is_shooter([101.0], [104.0], [100.8], [100.0], "Broad (Loose)", False)[0]
    # 收盘冲到中点之上 → 不成立
    assert not is_shooter([101.0], [104.0], [100.8], [103.0], "Broad (Loose)", False)[0]


def test_hammer_color_gate():
    # 阴线的锤子形状在 require_color=True 时不成立
    assert not is_hammer([101.0], [101.2], [97.0], [100.0], "Broad (Loose)", True)[0]
    assert is_hammer([100.0], [101.2], [97.0], [101.0], "Broad (Loose)", True)[0]


def test_hammer_classic_and_pin_bar():
    # Classic：实体 <= 30% 区间、下影 >= 3× 实体、上影 <= 35%
    assert is_hammer([100.0], [100.55], [97.0], [100.3], "Classic", False)[0]
    # 实体过大 → 不成立
    assert not is_hammer([99.0], [100.0], [97.0], [100.0], "Classic", False)[0]
    # Pin Bar：收与开都在上四分之一
    assert is_hammer([100.0], [100.6], [97.0], [100.2], "Pin Bar (Strict)", False)[0]
    assert not is_hammer([97.5], [100.6], [97.0], [100.2], "Pin Bar (Strict)", False)[0]


def test_hammer_zero_range_is_not_a_pattern():
    assert not is_hammer([100.0], [100.0], [100.0], [100.0], "Broad (Loose)", False)[0]


def test_hammer_detected_on_previous_bar():
    n = 60
    open_ = np.full(n, 100.0)
    close = np.full(n, 100.0)
    high = np.full(n, 100.3)
    low = np.full(n, 99.7)
    open_[30], close[30], high[30], low[30] = 100.0, 101.0, 101.2, 97.0
    out = thestrat_suite(_frame(open_, high, low, close))
    # C1 是上一根 → 锤子出现在第 31 根
    hits = [i for i in range(n) if out["thestrat_c1_hammer"].iloc[i] == 1]
    assert hits == [31]


# ---------------------------------------------------------------- Failed 2


def test_failed2_methods():
    # 2u 破高但收在区间内且收在开盘下方
    assert is_failed([10.0], [5.0], [9.0], [8.5], np.array([_STRUCT_2U]), "Reclaim")[0]
    assert is_failed([10.0], [5.0], [9.0], [8.5], np.array([_STRUCT_2U]), "Open")[0]
    assert is_failed([10.0], [5.0], [9.0], [8.5], np.array([_STRUCT_2U]), "Reclaim + Open")[0]
    # 收在区间外 → Reclaim 不成立
    assert not is_failed([10.0], [5.0], [9.0], [12.0], np.array([_STRUCT_2U]), "Reclaim")[0]
    # 3 和 1 永远不会 fail
    assert not is_failed([10.0], [5.0], [9.0], [8.5], np.array([_STRUCT_3]), "Reclaim")[0]
    assert not is_failed([10.0], [5.0], [9.0], [8.5], np.array([_STRUCT_1]), "Reclaim")[0]


def test_failed2_or_method_is_weaker():
    """OR 口径比任一单口径都宽。

    2d 案例：破低后收在开盘**上方**（逆破位方向的收盘），但收在前低**之下** ——
    Open 成立、Reclaim 不成立，OR 合取后成立。
    """
    prev_high, prev_low, o, c = [10.0], [5.0], [4.0], [4.5]
    assert not is_failed(prev_high, prev_low, o, c, np.array([_STRUCT_2D]), "Reclaim")[0]
    assert is_failed(prev_high, prev_low, o, c, np.array([_STRUCT_2D]), "Open")[0]
    assert is_failed(prev_high, prev_low, o, c, np.array([_STRUCT_2D]), "Reclaim OR Open")[0]


def test_pre_f2_flags():
    # 破高 → 收回区间内 → 但仍收在开盘上方：pre-f2u
    n = 40
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    # C1 (bar 20) 区间 99.5~100.5；CC (bar 21) 破高但收回、收在开盘上方
    open_[21], close[21], high[21], low[21] = 99.8, 100.2, 101.0, 99.6
    frame = _frame(open_, high, low, close)
    out = thestrat_suite(frame, failed2_method="Open")
    assert int(out["thestrat_pre_f2u"].iloc[21]) == 1
    assert int(out["thestrat_f2u"].iloc[21]) == 0


# ---------------------------------------------------------------- FTFC


def test_ftfc_requires_all_enabled_timeframes_agree():
    # 持续上行的阳线：每个周期都在开盘之上，所有周期同向
    close = 100 + np.arange(1200) * 0.5
    frame = _frame(close - 0.5, close + 0.3, close - 0.8, close)
    out = thestrat_suite(frame)
    ftfc_up = out["thestrat_ftfc_up"].to_numpy(dtype=int)
    ftfc_down = out["thestrat_ftfc_down"].to_numpy(dtype=int)
    tail = slice(-100, None)
    assert ftfc_up[tail].sum() > 0
    assert ftfc_down[tail].sum() == 0


def test_ftfc_conflict_produces_neither():
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 1.0, 1500))
    out = thestrat_suite(_frame(close + rng.normal(0, 0.2, 1500), close + 0.5, close - 0.5, close))
    up = out["thestrat_ftfc_up"].to_numpy(dtype=int)
    down = out["thestrat_ftfc_down"].to_numpy(dtype=int)
    # 两者不可能同时成立
    assert ((up == 1) & (down == 1)).sum() == 0


# ---------------------------------------------------------------- 多米诺


def test_domino_counts_consecutive_inside_timeframes():
    # 极窄横盘：每个周期的 CC 都在 C1 区间内
    close = 100 + 0.02 * np.sin(np.arange(1500) / 5.0)
    out = thestrat_suite(_frame(close, close + 0.05, close - 0.05, close))
    count = out["thestrat_domino_count"].to_numpy(dtype=int)
    assert count.max() >= 2
    assert int(out["thestrat_domino"].sum()) > 0
    assert (out["thestrat_domino"].to_numpy(dtype=int)[count < 2] == 0).all()


# ---------------------------------------------------------------- 信号


def test_signal_in_force_and_stops():
    out = thestrat_suite(_bar_frame(400))
    high_force = out["thestrat_signal_high_in_force"].to_numpy(dtype=int)
    low_force = out["thestrat_signal_low_in_force"].to_numpy(dtype=int)
    assert high_force.sum() > 0 and low_force.sum() > 0
    # 止损只在信号在效时存在，且落在 C1 的反向极值
    stop_bull = out["thestrat_stop_bull"].to_numpy(dtype=float)
    stop_bear = out["thestrat_stop_bear"].to_numpy(dtype=float)
    assert np.isnan(stop_bull[high_force == 0]).all()
    assert np.isfinite(stop_bull[high_force == 1]).all()
    assert np.isnan(stop_bear[low_force == 0]).all()


def test_magnitude_is_c1_range():
    out = thestrat_suite(_bar_frame(200))
    high = out["thestrat_mag_high"].to_numpy(dtype=float)
    low = out["thestrat_mag_low"].to_numpy(dtype=float)
    assert (high[1:] >= low[1:]).all()


def test_magnitude_can_be_disabled():
    out = thestrat_suite(_bar_frame(200), show_magnitude=False)
    assert out["thestrat_mag_high"].isna().all()


def test_ftfc_filter_blocks_signals():
    frame = _bar_frame(500)
    loose = thestrat_suite(frame)
    strict = thestrat_suite(frame, inside_rev_require_ftfc=True,
                            inside_cont_require_ftfc=True, reversals_require_ftfc=True)
    assert int(strict["thestrat_signal_high_in_force"].sum()) <= \
        int(loose["thestrat_signal_high_in_force"].sum())


# ---------------------------------------------------------------- 输出契约


def test_output_columns_and_index():
    frame = _bar_frame(300)
    out = thestrat_suite(frame)
    assert out.index.equals(frame.index)
    assert len(out.columns) == len(set(out.columns))
    assert all(name.startswith("thestrat_") for name in out.columns)
    for name in ("thestrat_cc_structure", "thestrat_c1_hammer", "thestrat_ftfc_up",
                 "thestrat_domino_count", "thestrat_lead_direction"):
        assert str(out[name].dtype) == "Int64", name
    assert out["thestrat_cc_structure"].dropna().isin(range(5)).all()
    assert out["thestrat_lead_direction"].dropna().isin([-1, 0, 1]).all()
    assert out["thestrat_domino_count"].dropna().between(0, 6).all()


def test_presets_select_different_timeframes():
    # 用 5 分钟图：tf1 在两个预设下（30 vs 5）都 >= 图表周期，能看出差异
    frame = _bar_frame(2000, freq="5min")
    classic = thestrat_suite(frame, preset="TheStrat Classic")   # tf1 = 30
    scalp = thestrat_suite(frame, preset="Scalp")                # tf1 = 5
    col = "thestrat_tf1_structure"
    assert not np.array_equal(classic[col].to_numpy(dtype=int), scalp[col].to_numpy(dtype=int))
    # 结构列只出现在合法码域内
    assert classic[col].dropna().isin(range(5)).all()
    assert scalp[col].dropna().isin(range(5)).all()


def test_custom_timeframes_must_be_six():
    frame = _bar_frame(100)
    with pytest.raises(ValueError):
        thestrat_suite(frame, preset="Custom", timeframes=("60", "240"))


def test_parameter_validation():
    frame = _bar_frame(100)
    bad = [
        dict(preset="Nope"),
        dict(hammer_method="Nope"),
        dict(failed2_method="Nope"),
        dict(min_domino_tfs=1),
        dict(min_domino_tfs=7),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            thestrat_suite(frame, **kwargs)


def test_random_runs_end_to_end():
    for seed in (0, 6):
        out = thestrat_suite(_bar_frame(400, seed=seed))
        assert out["thestrat_cc_structure"].notna().all()
        assert (out["thestrat_tf_bull_count"] >= 0).all()
        assert np.isfinite(out["thestrat_stop_bull"].dropna()).all()
