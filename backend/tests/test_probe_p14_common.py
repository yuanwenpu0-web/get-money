"""P14 逐笔顺序探针的量具自校准单测（不联网、不连 DB）。"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from probe_p14_common import (  # noqa: E402
    HOUR_MS, MIN_TRADES_PER_HOUR, _flip_dev, _one_hour, _run_dev,
    hour_rows, mean_xs_corr, to_frames, variance_decomp,
)
import probe_p12_common as p12  # noqa: E402
from probe_p14_robust import (  # noqa: E402
    decision_summary, label_overlap, nonoverlapping_blocks, partial_xs_ic,
    two_way_decomp,
)


def _clustered_dirs(n: int = 1200, block: int = 20, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=n // block + 1)
    return np.repeat(signs, block)[:n]


def test_flip_dev_separates_independent_from_clustered():
    rng = np.random.default_rng(11)
    indep = rng.choice([-1, 1], size=4000)
    assert abs(_flip_dev(indep, float((indep == 1).mean()))) < 0.10
    clustered = _clustered_dirs()
    assert _flip_dev(clustered, float((clustered == 1).mean())) < -0.8


def test_run_dev_separates_independent_from_long_runs():
    rng = np.random.default_rng(12)
    indep = rng.choice([-1, 1], size=4000)
    assert abs(_run_dev(indep)) < 0.10
    assert _run_dev(_clustered_dirs()) > 1.0


def test_row_permutation_kills_order_but_preserves_taker_imb():
    """整行置换只打乱先后：顺序特征归零，K 线级 taker 汇总严格不变。"""
    rng = np.random.default_rng(3)
    n = 1500
    d = _clustered_dirs(n=n, block=25, seed=8)
    price = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    quote = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    out = _one_hour(0, "TESTUSDT", price, quote, d, rng)
    assert out["seq_flip_dev"] < -0.5
    assert abs(out["shuf_seq_flip_dev"]) < 0.15
    assert abs(out["shuf_big_run_dev"]) < 0.30
    assert abs(out["shuf_taker_imb"] - out["taker_imb"]) < 1e-12
    assert np.isfinite(out["impact_rev"])


def test_hour_rows_buckets_by_utc_hour_and_skips_thin_hours():
    base = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    lines = ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
             "transact_time,is_buyer_maker"]
    for hour in range(2):
        for j in range(160):
            ts = base + hour * HOUR_MS + j * 1000
            side = "false" if (j // 20) % 2 == 0 else "true"
            lines.append(f"{hour * 200 + j},100.5,10.0,{j},{j},{ts},{side}")
    for j in range(MIN_TRADES_PER_HOUR - 1):  # 稀疏小时应被剔除
        ts = base + 2 * HOUR_MS + j * 1000
        lines.append(f"9000,100.5,10.0,{j},{j},{ts},false")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("TESTUSDT-aggTrades-2026-11-10.csv", "\n".join(lines))
    rows, stats = hour_rows(buf.getvalue(), "TESTUSDT", "2026-11-10", 42)
    assert len(rows) == 2
    assert stats["skipped_thin_hours"] == 1
    assert stats["source_rows"] == 320 + MIN_TRADES_PER_HOUR - 1
    assert [r["hour"] for r in rows] == [base, base + HOUR_MS]
    assert all(r["n_trades"] == 160 for r in rows)


def test_to_frames_reindexes_to_grid_and_universe():
    grid = np.arange(0, 3 * HOUR_MS, HOUR_MS)
    rec = [{"rows": [{"hour": int(grid[1]), "symbol": "AUSDT",
                      "n_trades": 200, "quote_total": 1000.0,
                      **{c: 0.5 for c in
                         ("seq_flip_dev", "big_run_dev", "impact_rev",
                          "taker_imb", "shuf_seq_flip_dev",
                          "shuf_big_run_dev", "shuf_taker_imb")}}]}]
    frames = to_frames(rec, grid, ["AUSDT", "BUSDT"])
    assert list(frames["seq_flip_dev"].index) == list(grid)
    assert list(frames["seq_flip_dev"].columns) == ["AUSDT", "BUSDT"]
    assert np.isnan(frames["seq_flip_dev"].iloc[0, 0])
    assert np.isnan(frames["seq_flip_dev"].iloc[1, 1])


def test_variance_decomposition_time_common_vs_cross_section():
    rng = np.random.default_rng(9)
    level = rng.normal(size=80)
    common = pd.DataFrame(np.tile(level[:, None], (1, 6))
                          + rng.normal(0, 1e-6, (80, 6)))
    assert variance_decomp(common)["time_common_share"] > 0.99
    offsets = np.linspace(-2, 2, 6)
    by_symbol = pd.DataFrame(offsets[None, :] + rng.normal(0, 0.2, (80, 1)))
    assert variance_decomp(by_symbol)["cross_section_share"] > 0.90


def test_mean_xs_corr_detects_collinear_feature():
    rng = np.random.default_rng(4)
    a = pd.DataFrame(rng.normal(size=(60, 8)))
    b = a * 3.0 + 7.0
    assert mean_xs_corr(a, b, 6)["mean_rho"] > 0.99
    assert mean_xs_corr(a, pd.DataFrame(rng.normal(size=(60, 8))), 6)[
        "periods"] == 60


def test_two_way_decomp_identifies_symbol_fixed_effect():
    rng = np.random.default_rng(21)
    offsets = np.linspace(-3, 3, 8)
    frame = pd.DataFrame(offsets[None, :] + rng.normal(0, 0.01, (100, 8)))
    out = two_way_decomp(frame)
    assert out["symbol_fixed_effect_share"] > 0.95
    assert out["time_effect_share"] < 0.05


def test_two_way_decomp_identifies_time_common_effect():
    rng = np.random.default_rng(22)
    level = rng.normal(size=100)
    frame = pd.DataFrame(level[:, None] + rng.normal(0, 0.01, (100, 8)))
    out = two_way_decomp(frame)
    assert out["time_effect_share"] > 0.95
    assert out["symbol_fixed_effect_share"] < 0.05


def test_partial_xs_ic_removes_liquidity_driven_correlation(monkeypatch):
    """因子与标签都由成交笔数驱动时，偏相关应回落到 0 附近。"""
    monkeypatch.setattr(p12, "MIN_CROSS_SECTION", 2)
    rng = np.random.default_rng(23)
    base = np.tile(np.linspace(2, 12, 6)[None, :], (60, 1))
    liq = base + rng.normal(0, 0.3, (60, 6))
    # 因子/标签各自带独立大噪声，保证 rank 与控制量不完全共线（残差不退化）
    factor = pd.DataFrame(base + rng.normal(0, 1.5, (60, 6)))
    label = pd.DataFrame(base + rng.normal(0, 1.5, (60, 6)))
    control = pd.DataFrame(np.log(np.clip(liq, 0.1, None)))
    valid = pd.DataFrame(True, index=factor.index, columns=factor.columns)
    out = partial_xs_ic(factor, label, valid, control)
    assert out["periods"] >= 30
    assert out["degenerate_periods"] == 60 - out["periods"]
    assert abs(out["mean_partial_ic"]) < 0.30


def test_nonoverlapping_blocks_takes_every_24th_finite_point():
    ics = np.full(100, np.nan)
    finite = np.arange(100)
    ics[finite] = -0.01 * (finite + 1)
    out = nonoverlapping_blocks(ics)
    assert out["n_blocks"] == 5
    assert out["values"][1] == round(float(ics[24]), 4)
    assert out["negative_share"] == 1.0


def test_label_overlap_detects_smoothed_forward_labels():
    rng = np.random.default_rng(24)
    walk = pd.DataFrame(np.cumsum(rng.normal(0, 0.01, (80, 3)), axis=0))
    noise = pd.DataFrame(rng.normal(size=(80, 3)))
    assert label_overlap(walk)["mean_lag1_autocorr"] > 0.90
    assert abs(label_overlap(noise)["mean_lag1_autocorr"]) < 0.30


def test_decision_summary_serializable_and_blocks_scale_up():
    """回归：np.sign 比较曾产生 numpy.bool_，导致最终 JSON 写盘失败。"""
    import json
    rob = {
        "two_way_variance_decomposition": {
            "seq_flip_dev": {"symbol_fixed_effect_share": 0.795}},
        "summary": {"original_mean_ic": -0.06608,
                    "after_symbol_demean_mean_ic": 0.11024},
        "liquidity_partial_ic": {
            "ci": {"ci_low": -0.1306, "ci_high": 0.00174}},
        "up_down_regime_split": {"down_mean_ic": -0.14576,
                                 "up_mean_ic": 0.05322},
        "nonoverlapping_block_ic": {"negative_share": 0.6154},
        "h24_label_overlap": {"mean_lag1_autocorr": 0.9368},
    }
    out = decision_summary(rob, True)
    assert out["gate1_formal_pass"] is True
    assert out["n_flags_triggered"] >= 3
    assert out["action"] == "DO_NOT_SCALE_UP"
    assert all(type(v) is bool for v in out["robustness_flags"].values())
    json.dumps(out)  # 不得抛 TypeError
