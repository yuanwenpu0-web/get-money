"""P13 / H-A 市场级择时量具的纯函数测试。

对应假设卡 research/hypotheses/H-A-market-leverage-timing.md 的预登记口径：
等权市场收益（不是 P12 的中位）、非重叠 24h 块、40/20 切分、≥60 天硬门禁。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ha_common import (  # noqa: E402
    DAY_MS, HOUR_MS, MIN_CROSS_SECTION, block_readiness, data_gate, market_factor,
    market_forward_return, nonoverlap_block_times, oi_zscore, split_train_oos,
)

GRID = np.arange(5, dtype=np.int64) * HOUR_MS


def _px() -> pd.DataFrame:
    """A/B 平盘、C 在 t=2 涨 30%：等权均值 0.10，中位 0.0，两者必须可区分。"""
    return pd.DataFrame(
        {"A": [100.0] * 5, "B": [100.0] * 5, "C": [100.0, 100.0, 130.0, 130.0, 130.0]},
        index=GRID,
    )


def test_data_gate_blocks_below_sixty_days():
    start = 0
    end = int(36.5 * DAY_MS)
    out = data_gate(start, end)
    assert out["ready"] is False
    assert out["span_days"] == pytest.approx(36.5)
    assert out["days_remaining"] == pytest.approx(23.5)


def test_data_gate_eta_is_start_plus_sixty_days():
    start = 1_700_000_000_000
    out = data_gate(start, start + int(36.5 * DAY_MS))
    assert out["eta_ms"] == start + 60 * DAY_MS


def test_data_gate_passes_at_sixty_days():
    out = data_gate(0, 60 * DAY_MS)
    assert out["ready"] is True
    assert out["days_remaining"] == pytest.approx(0.0)
    assert out["oos_days"] == 20


def test_oi_zscore_uses_only_past_window():
    """因果性：删掉 t 之后的全部数据，t 处的 z 必须逐位不变。"""
    rng = np.random.default_rng(11)
    oi = pd.DataFrame(rng.lognormal(mean=10, sigma=1, size=(300, 4)),
                      index=np.arange(300, dtype=np.int64) * HOUR_MS,
                      columns=list("ABCD"))
    full = oi_zscore(oi)
    cut = oi_zscore(oi.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], cut, check_exact=False, atol=1e-12)


def test_oi_zscore_requires_min_periods():
    oi = pd.DataFrame(np.ones((200, 2)),
                      index=np.arange(200, dtype=np.int64) * HOUR_MS, columns=list("AB"))
    oi.iloc[:, 0] = np.arange(200, dtype=float)
    z = oi_zscore(oi)
    assert z["A"].iloc[:71].isna().all()
    assert np.isfinite(z["A"].iloc[71:]).all()


def test_market_factor_is_cross_sectional_mean():
    z = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=GRID[:3], columns=list("ABCD"))
    out = market_factor(z, min_cross_section=4)
    np.testing.assert_allclose(out.to_numpy(), [2.5, 2.5, 2.5])


def test_market_factor_nan_when_cross_section_too_thin():
    z = pd.DataFrame(np.ones((3, 40)), index=GRID[:3], columns=[f"S{i}" for i in range(40)])
    z.iloc[:, MIN_CROSS_SECTION - 1:] = np.nan  # 只剩 29 个有效币，低于门槛
    out = market_factor(z)
    assert out.isna().all()


def test_market_forward_return_is_equal_weight_not_median():
    out = market_forward_return(_px(), horizon=2)
    assert out.loc[0] == pytest.approx(0.10)
    assert out.loc[0] != pytest.approx(float(_px().shift(-2).div(_px()).sub(1).loc[0].median()))


def test_market_forward_return_masks_incomplete_windows():
    px = _px()
    px.loc[HOUR_MS, "A"] = np.nan  # t=0..2 窗口对 A 不完整
    out = market_forward_return(px, horizon=2)
    assert out.loc[0] == pytest.approx(0.15)  # 只剩 B(0.0) 与 C(0.30)


def test_market_factor_ignores_symbols_not_yet_listed():
    """point-in-time：未来才上市的币不得影响早期时点的市场因子。"""
    px = _px()
    base = market_forward_return(px, horizon=2)
    px["D"] = [np.nan, np.nan, np.nan, 50.0, 80.0]
    assert market_forward_return(px, horizon=2).loc[0] == pytest.approx(base.loc[0])


def test_nonoverlap_blocks_spaced_exactly_horizon():
    blocks = nonoverlap_block_times(0, 10 * DAY_MS, horizon=24)
    assert np.all(np.diff(blocks) == 24 * HOUR_MS)
    assert blocks[0] == 0


def test_nonoverlap_blocks_exclude_incomplete_label_tail():
    blocks = nonoverlap_block_times(0, 4 * HOUR_MS, horizon=2)
    np.testing.assert_array_equal(blocks, np.array([0, 2 * HOUR_MS], dtype=np.int64))


def test_split_train_oos_uses_last_sixty_blocks():
    blocks = np.arange(70, dtype=np.int64) * DAY_MS
    train, oos = split_train_oos(blocks)
    np.testing.assert_array_equal(train, blocks[10:50])
    np.testing.assert_array_equal(oos, blocks[50:70])


def test_split_train_oos_rejects_insufficient_blocks():
    with pytest.raises(ValueError, match="60"):
        split_train_oos(np.arange(59, dtype=np.int64) * DAY_MS)


def test_block_readiness_rejects_when_warmup_eats_into_window():
    """72h z-score 预热使前 3 块因子为 NaN；62 块时尾部 60 块仍含 1 个 NaN。"""
    blocks = np.arange(62, dtype=np.int64) * DAY_MS
    f = np.ones(62)
    f[:3] = np.nan
    out = block_readiness(blocks, f, np.ones(62))
    assert out["ready"] is False
    assert out["window_invalid_blocks"] == 1
    assert out["extra_days_lower_bound"] >= 1


def test_block_readiness_passes_once_warmup_is_outside_window():
    blocks = np.arange(63, dtype=np.int64) * DAY_MS
    f = np.ones(63)
    f[:3] = np.nan
    out = block_readiness(blocks, f, np.ones(63))
    assert out["ready"] is True
    assert out["window_finite_blocks"] == 60


def test_block_readiness_rejects_interior_gap_and_does_not_drop_it():
    """窗口内部的缺口不能靠等更多数据补掉，必须显式报不可用而不是静默丢块。"""
    blocks = np.arange(70, dtype=np.int64) * DAY_MS
    f = np.ones(70)
    f[-10] = np.nan
    out = block_readiness(blocks, f, np.ones(70))
    assert out["ready"] is False
    assert out["window_invalid_blocks"] == 1
    assert out["interior_gap"] is True


def test_block_readiness_reports_shortfall_when_too_few_blocks():
    blocks = np.arange(35, dtype=np.int64) * DAY_MS
    out = block_readiness(blocks, np.ones(35), np.ones(35))
    assert out["ready"] is False
    assert out["blocks_short"] == 25
    assert out["total_blocks"] == 35


def test_block_readiness_shortfall_includes_warmup_prefix():
    """35 块 + 3 块预热：需要 63 块才够 60 个可用，下界必须是 28 而不是 25。"""
    blocks = np.arange(35, dtype=np.int64) * DAY_MS
    f = np.ones(35)
    f[:3] = np.nan
    out = block_readiness(blocks, f, np.ones(35))
    assert out["ready"] is False
    assert out["extra_days_lower_bound"] == 28
    assert out["warmup_prefix_blocks"] == 3