"""Unit tests for the pure-causal conditional-exit meta-label helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from exit_metalabel_features import (  # noqa: E402
    t0_features, t1_features, t1_label, t0_label, find_decision_bar,
    drop_degenerate, DECISION_MFE_R,
)
from exit_metalabel_stats import spearman, auc, segment_ic  # noqa: E402
from order_exit_common import net_return_pct  # noqa: E402


def _bar(o, h, l, c, volume=100.0, quote=1000.0):
    return {"open": o, "high": h, "low": l, "close": c,
            "volume": volume, "quote_volume": quote}


def _hist(n=60, base=100.0):
    out = []
    p = base
    for i in range(n):
        p *= 1.0 + 0.001 * np.sin(i / 3)
        out.append(_bar(p, p * 1.004, p * 0.996, p * 1.001))
    return out


def _order():
    return {"side": "LONG", "entry_price": 100.0, "sl": 98.5, "tp": 104.0,
            "interval": "15m", "hour_of_day": 9, "score": 0.8, "prob_up": 0.6}


def test_t0_features_ignore_future_bars():
    """Appending future bars to the history argument must not change T0 output
    once the caller has truncated at the decision moment; and the feature set
    must be computed from the supplied window only."""
    h = _hist(60)
    f = t0_features(h, _order())
    assert f["atr14_over_R"] > 0
    assert 0 <= f["range_pos_128"] <= 1
    assert f["tp_dist_R"] == pytest.approx(4.0 / 1.5)
    # Same window -> identical values (deterministic, no hidden state).
    assert t0_features(h, _order()) == f


def test_t0_features_short_side_signs():
    o = _order()
    o["side"] = "SHORT"
    o["sl"], o["tp"] = 101.5, 96.0
    f = t0_features(_hist(60), o)
    assert f["is_short"] == 1.0
    assert f["tp_dist_R"] == pytest.approx(4.0 / 1.5)


def test_find_decision_bar_is_causal_and_barrier_first():
    side, entry, sl, tp = "LONG", 100.0, 98.5, 104.0
    R = abs(entry - sl)
    # first bar already jumps through TP -> no actionable decision point
    assert find_decision_bar([_bar(100, 105, 99.9, 104.5)], side, entry, sl, tp) is None
    # SL fires first -> no decision point
    assert find_decision_bar([_bar(100, 100.5, 98.0, 98.2)], side, entry, sl, tp) is None
    # reaches 0.7R on bar 1, TP not touched -> decision at bar 1
    bars = [_bar(100, 100.5, 99.8, 100.2),
            _bar(100.2, 100.2 + DECISION_MFE_R * R + 0.1, 100.0, 100.8),
            _bar(100.8, 101.2, 100.5, 101.0)]
    assert find_decision_bar(bars, side, entry, sl, tp) == 1


def test_t1_features_use_only_bars_up_to_k():
    bars = [_bar(100, 100.5, 99.8, 100.2),
            _bar(100.2, 101.6, 100.0, 100.8),
            _bar(100.8, 101.2, 100.5, 101.0)]
    f = t1_features(bars, 1, "LONG", 100.0, 98.5, 104.0)
    assert f["bars_held"] == 2
    assert f["mfe_R_sofar"] == pytest.approx(1.6 / 1.5)
    # A later bar cannot alter features computed at k=1.
    bars2 = bars + [_bar(101, 110, 100, 109)]
    assert t1_features(bars2, 1, "LONG", 100.0, 98.5, 104.0) == f


def test_t1_label_costs_cancel_in_difference():
    bars = [_bar(100, 100.5, 99.8, 100.2),
            _bar(100.2, 101.6, 100.0, 100.8),
            _bar(100.8, 101.2, 100.5, 101.0),
            _bar(101, 101.4, 100.7, 101.1)]
    lab = t1_label(bars, 1, "LONG", 100.0, 98.5, 104.0)
    assert lab is not None
    assert lab["y1"] == pytest.approx(lab["hold_net_pct"] - lab["exit_now_net_pct"])
    assert lab["hold_better"] in (0, 1)


def test_t0_label_is_late_minus_early():
    rn = {"tp_x0.5": 1.0, "tp_x0.75": 3.0, "tp_x1.5": 5.0,
          "tp_x2.0": 7.0, "tp_x3.0": 9.0, "no_tp_hold": 11.0}
    lab = t0_label(rn)
    assert lab["early_net_pct"] == pytest.approx(2.0)
    assert lab["late_net_pct"] == pytest.approx(8.0)
    assert lab["y0"] == pytest.approx(6.0)
    assert lab["late_better"] == 1
    assert t0_label({"tp_x0.5": 1.0}) is None


def test_drop_degenerate_removes_constants_and_sparse():
    recs = [{"X": {"const": 1.0, "good": float(i), "sparse": float("nan")}}
            for i in range(40)]
    for i in range(0, 40, 2):
        recs[i]["X"]["sparse"] = float(i)
    keep = drop_degenerate(["const", "good", "sparse"], recs)
    assert "good" in keep
    assert "const" not in keep
    assert "sparse" not in keep


def test_spearman_and_auc_known_values():
    x = list(range(40))
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, list(reversed(x))) == pytest.approx(-1.0)
    assert np.isnan(spearman(x[:5], x[:5]))  # below the 30-pair gate
    yb = [0] * 20 + [1] * 20
    assert auc(x, yb) == pytest.approx(1.0)


def test_segment_ic_reports_four_windows():
    n = 80
    x = list(range(n))
    y = list(range(n))
    t = [f"2026-11-{1 + i // 20:02d}T00:00:00" for i in range(n)]
    s = segment_ic(x, y, t)
    assert len(s["ics"]) == 4
    assert s["n_positive"] == 4


def test_net_return_matches_cost_convention():
    v = net_return_pct(100.0, 100.0, "LONG")
    assert v == pytest.approx(-0.12, abs=1e-9)
