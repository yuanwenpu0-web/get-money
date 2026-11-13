"""P17 / H-E 选币实验判定量具单测（不联网、不连 DB）。

判据来自 research/hypotheses/H-E-supertrend-pit-pool.md §4 的 C1–C5，
**事前预登记**。这里锁死「与」关系与词表，防止看到结果后放宽门槛。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research.pit_pool_stats import arm_names, bh_fdr, jaccard, judge  # noqa: E402


def _ci(trades=500, mean=0.5, lo=0.1, hi=0.9, share=0.99, verdict="positive"):
    return {"trades": trades, "mean_return_pct": mean, "ci_low_pct": lo,
            "ci_high_pct": hi, "positive_share": share, "verdict": verdict}


def _arm(ci=None, folds=(0.4, 0.5, 0.6, 0.3, 0.2), pct=0.99):
    ci = _ci() if ci is None else ci
    means = [m for m in folds if m is not None]
    return {"pooled": ci, "fold_means": means,
            "positive_folds": sum(1 for m in means if m > 0),
            "percentile_rank": pct}


def test_bh_fdr_matches_hand_computed_values():
    q = bh_fdr({"a": 0.001, "b": 0.4, "c": 0.6})
    assert q["a"] == 0.003 and q["b"] == 0.6 and q["c"] == 0.6


def test_bh_fdr_monotone_step_up_all_equal():
    q = bh_fdr({f"p{i}": p for i, p in enumerate([0.01, 0.02, 0.03, 0.04, 0.05])})
    assert all(abs(v - 0.05) < 1e-9 for v in q.values())


def test_bh_fdr_empty_and_single():
    assert bh_fdr({}) == {}
    assert bh_fdr({"only": 0.04}) == {"only": 0.04}


def test_arm_names_cover_ten_arms():
    names = arm_names()
    assert len(names) == 10
    assert names[0] == "all"
    assert "pit_liq_20" in names and "pit_vol_30" in names and "posthoc_10" in names


def test_judge_pass_requires_all_five_checks():
    summary = {name: _arm() for name in arm_names()}
    out = judge(summary)
    assert out["arms"]["pit_liq_20"]["verdict"] == "pass"
    assert all(out["arms"]["pit_liq_20"]["checks"].values())
    assert out["overall"] == "pass"


def test_judge_rejects_when_random_percentile_below_threshold():
    summary = {name: _arm() for name in arm_names()}
    summary["pit_liq_20"] = _arm(pct=0.90)
    out = judge(summary)
    assert out["arms"]["pit_liq_20"]["checks"]["C4_beats_random"] is False
    assert out["arms"]["pit_liq_20"]["verdict"] != "pass"


def test_judge_rejects_when_fold_signs_below_four_of_five():
    summary = {name: _arm() for name in arm_names()}
    summary["pit_vol_10"] = _arm(folds=(0.4, 0.5, 0.6, -0.3, -0.2))
    out = judge(summary)
    assert out["arms"]["pit_vol_10"]["checks"]["C3_fold_signs"] is False
    assert out["arms"]["pit_vol_10"]["verdict"] != "pass"


def test_judge_marks_small_sample_insufficient_even_with_positive_ci():
    summary = {name: _arm() for name in arm_names()}
    summary["pit_liq_10"] = _arm(ci=_ci(trades=12, lo=0.8, hi=3.0, verdict="insufficient"))
    out = judge(summary)
    assert out["arms"]["pit_liq_10"]["verdict"] == "insufficient"
    assert out["arms"]["pit_liq_10"]["checks"]["C2_min_trades"] is False


def test_judge_negative_when_ci_upper_below_zero():
    summary = {name: _arm() for name in arm_names()}
    for name in arm_names():
        summary[name] = _arm(ci=_ci(mean=-0.4, lo=-0.6, hi=-0.2, share=0.0, verdict="negative"),
                             folds=(-0.3, -0.4, -0.5, -0.2, -0.6), pct=0.01)
    out = judge(summary)
    assert out["arms"]["pit_liq_20"]["verdict"] == "negative"
    assert out["overall"] == "negative"


def test_judge_fdr_can_veto_an_arm_that_passes_c1_to_c4():
    # 6 个 PIT 臂里只有一个 p 很小，其余 p 大 → 该臂 q 仍可能 <0.10；
    # 构造全部 p 偏大的情形，验证 C5 能单独否决
    summary = {name: _arm() for name in arm_names()}
    for name in arm_names():
        if name.startswith("pit_"):
            summary[name] = _arm(ci=_ci(share=0.90))  # p = 0.10
    out = judge(summary)
    for name in arm_names():
        if name.startswith("pit_"):
            assert out["arms"][name]["checks"]["C5_fdr"] is False
            assert out["arms"][name]["verdict"] != "pass"
    assert out["overall"] != "pass"


def test_judge_posthoc_and_all_are_diagnostic_not_gated():
    summary = {name: _arm() for name in arm_names()}
    out = judge(summary)
    assert out["arms"]["posthoc_10"]["diagnostic_only"] is True
    assert out["arms"]["all"]["diagnostic_only"] is True
    assert out["arms"]["pit_liq_10"]["diagnostic_only"] is False


def test_judge_overall_inconclusive_when_ci_crosses_zero():
    summary = {name: _arm() for name in arm_names()}
    for name in arm_names():
        summary[name] = _arm(ci=_ci(mean=0.05, lo=-0.2, hi=0.3, share=0.6, verdict="inconclusive"))
    out = judge(summary)
    assert out["overall"] == "inconclusive"


def test_jaccard_identical_disjoint_and_partial():
    assert jaccard(["A", "B"], ["B", "A"]) == 1.0
    assert jaccard(["A", "B"], ["C", "D"]) == 0.0
    assert jaccard(["A", "B"], ["B", "C"]) == round(1 / 3, 4)


def test_jaccard_two_empty_pools_are_stable_not_undefined():
    # 两期都凑不出成员时，「换血率」无意义；约定为 1.0 并在报告里标注期数
    assert jaccard([], []) == 1.0
    assert jaccard(["A"], []) == 0.0


def test_jaccard_ignores_duplicates_and_order():
    assert jaccard(["A", "A", "B"], ["B", "A"]) == 1.0