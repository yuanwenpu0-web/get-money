"""P16 固定规模 PIT 币池量具单测（不联网、不连 DB）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from probe_p16_fixed_pool import (  # noqa: E402
    eligible_ranking, forward_width, jaccard, make_periods, posthoc_diagnostic,
    rank_percentile, summarize, symbol_allowed,
)

H = 3_600_000


def test_symbol_allowed_filters_blacklist_and_leveraged_tokens():
    # EXCLUDED 是线上研究池既定黑名单，含 BTCUSDT/ETHUSDT/稳定币与法币对
    assert symbol_allowed("SOLUSDT") is True
    assert symbol_allowed("ADAUSDT") is True
    assert symbol_allowed("BTCUSDT") is False
    assert symbol_allowed("ETHUSDT") is False
    assert symbol_allowed("BTCBUSD") is False
    assert symbol_allowed("BTCUPUSDT") is False
    assert symbol_allowed("BTCBULLUSDT") is False
    assert symbol_allowed("ETHDOWNUSDT") is False


def _window():
    w = np.full((168, 4), np.nan)
    w[:, 0] = 10.0                 # A: 168 bars, volume > 0
    w[:100, 1] = 20.0             # B: only 100 bars -> too few
    w[:, 2] = 0.0                 # C: full bars but zero volume
    w[0, 3] = 30.0                # D: one stale bar
    return w


def test_eligible_ranking_needs_full_window_positive_volume_and_fresh_bar():
    t = 168 * H
    out = eligible_ranking(_window(), ["AUSDT", "BUSDT", "CUSDT", "DUSDT"], t, 0)
    assert [s for s, _ in out] == ["AUSDT"]
    assert out[0][1] == 168 * 10.0


def test_eligible_ranking_sorts_volume_desc_then_symbol_asc():
    w = np.full((168, 3), np.nan)
    w[:, 0] = 1.0
    w[:, 1] = 2.0
    w[:, 2] = 2.0
    t = 168 * H
    out = eligible_ranking(w, ["ZUSDT", "AUSDT", "BUSDT"], t, 0)
    assert [s for s, _ in out] == ["AUSDT", "BUSDT", "ZUSDT"]


def test_forward_width_requires_all_24_bars():
    fwd = np.full((24, 3), np.nan)
    fwd[:, 0] = 1.0
    fwd[:23, 1] = 1.0
    idx = {"AUSDT": 0, "BUSDT": 1, "CUSDT": 2}
    assert forward_width(["AUSDT"], fwd, idx) == 1
    assert forward_width(["AUSDT", "BUSDT", "CUSDT"], fwd, idx) == 1
    assert forward_width([], fwd, idx) == 0


def test_jaccard_and_rank_percentile():
    assert jaccard(["A", "B", "C"], ["B", "C", "D"]) == 0.5
    assert jaccard(["A"], ["A"]) == 1.0
    assert jaccard([], []) == 1.0
    assert rank_percentile(0, 100) == 1.0
    assert rank_percentile(99, 100) == 0.0
    assert rank_percentile(None, 100) is None
    assert rank_percentile(0, 1) is None


def test_make_periods_respects_lookback_and_forward_window():
    grid = np.arange(0, 200 * 24 * H, H, dtype=np.int64)
    periods = make_periods(grid, step_days=7)
    assert periods[0] >= 168 * H
    assert all((p % 86_400_000) == 0 for p in periods)
    assert all(p + 24 * H <= int(grid[-1]) + H for p in periods)
    assert periods[1] - periods[0] == 7 * 86_400_000


def _records(stable: bool = True):
    out = []
    for i in range(10):
        members = {n: [f"S{j:03d}USDT" for j in range(n)] for n in (10, 20, 30, 50, 100)}
        if not stable:
            members = {n: [f"T{j:03d}USDT" for j in range(n)] if i % 2
                       else [f"S{j:03d}USDT" for j in range(n)]
                       for n in (10, 20, 30, 50, 100)}
        out.append({"t": i * 7 * 86_400_000, "eligible_count": 500,
                    "members": members,
                    "width": {n: n for n in (10, 20, 30, 50, 100)},
                    "ranks": {f"S{j:03d}USDT": j for j in range(100)}})
    return out


def test_summarize_gate_flags_are_native_bools_and_serializable():
    summary = summarize(_records())
    assert summary["30"]["gate"]["passed"] is True
    assert summary["100"]["gate"]["C3"] is True
    assert summary["10"]["gate"]["C3"] is True  # N<30 按预注册豁免
    assert summary["10"]["supports_cross_section_30"] is False
    assert all(type(v) is bool for n in summary.values()
               for v in n["gate"].values())
    json.dumps(summary)


def test_summarize_detects_full_pool_turnover():
    summary = summarize(_records(stable=False))
    assert summary["30"]["jaccard_median"] == 0.0
    assert summary["30"]["gate"]["C2"] is False
    assert summary["30"]["gate"]["passed"] is False


def test_posthoc_diagnostic_reports_not_eligible_share():
    records = _records()
    for r in records[:5]:
        r["ranks"] = {}
    diag = posthoc_diagnostic(records, sizes=(10,))
    assert diag["10"]["early_not_eligible_share"] == 1.0
    assert diag["10"]["all_periods_not_eligible_share_max"] == 1.0
    json.dumps(diag)
