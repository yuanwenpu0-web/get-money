"""P12 探针纯函数的回归测试：支撑报告里三条关键方法学论断。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_p12_common import (  # noqa: E402
    HOUR_MS, bh_fdr, block_bootstrap_ci, build_labels, ic_series,
)

SYMBOLS = [f"S{i}" for i in range(40)]
GRID = np.arange(6, dtype=np.int64) * HOUR_MS


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=GRID, columns=SYMBOLS)


def _valid() -> pd.DataFrame:
    return pd.DataFrame(True, index=GRID, columns=SYMBOLS)


def test_rank_ic_is_invariant_to_per_period_constant_shift():
    """报告 §4.2：横截面 rank IC 天然剔除市场共同平移。"""
    rng = np.random.default_rng(7)
    factor = _frame(rng.normal(size=(len(GRID), len(SYMBOLS))))
    label = _frame(rng.normal(size=(len(GRID), len(SYMBOLS))))
    shift = np.arange(len(GRID))[:, None] * 0.01  # 每期一个不同常数
    base, _ = ic_series(factor, label, _valid())
    shifted, _ = ic_series(factor, label + shift, _valid())
    np.testing.assert_allclose(base, shifted, atol=1e-12)


def test_ic_series_requires_min_cross_section():
    factor = _frame(np.ones((len(GRID), len(SYMBOLS))))
    label = _frame(np.ones((len(GRID), len(SYMBOLS))))
    small = pd.DataFrame(True, index=GRID, columns=SYMBOLS)
    small.iloc[:, 8:] = False
    ics, counts = ic_series(factor, label, small)
    assert np.all(np.isnan(ics))
    assert np.all(counts == 8)


def test_bh_fdr_is_monotone_and_bounded():
    q = bh_fdr([0.001, 0.02, 0.5, 1.0])
    assert q == sorted(q)
    assert all(0.0 <= v <= 1.0 for v in q)
    assert all(round(v, 6) >= round(p, 6) for v, p in zip(q, [0.001, 0.02, 0.5, 1.0]))


def test_block_bootstrap_ci_on_constant_series():
    out = block_bootstrap_ci(np.full(120, 0.03), seed=1)
    assert out["ci_low"] == pytest.approx(0.03)
    assert out["ci_high"] == pytest.approx(0.03)
    assert out["method"] == "block24"


def test_build_labels_masks_windows_with_missing_bars():
    """反向 rolling：t..t+H 任一根缺失则该格标签为 NaN，不插值、不前视。"""
    rows = [{"symbol": s, "open_time": int(t), "close": 100.0 + i}
            for t in GRID for i, s in enumerate(("A", "B", "C"))]
    candles = pd.DataFrame([r for r in rows
                            if not (r["symbol"] == "A" and r["open_time"] == 3 * HOUR_MS)])
    fwd, excess, complete, _ = build_labels(candles, GRID, 2,
                                            columns=pd.Index(["A", "B", "C"]))
    assert np.isfinite(fwd.loc[0, "A"])
    assert np.isnan(fwd.loc[HOUR_MS, "A"])       # 窗口 1,2,3 含缺口
    assert np.isnan(fwd.loc[3 * HOUR_MS, "A"])   # 窗口 3,4,5 起点即缺失
    assert bool(complete.loc[0, "B"]) is True  # B 在 t=0 窗口完整
    assert excess.shape == fwd.shape
