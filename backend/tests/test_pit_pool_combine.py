"""P17 / H-E 双臂合并报告单测（不联网、不连 DB）。

合并只允许做「汇总与并列」，**不允许把 1h 与 15m 的成交池化成一个 CI**：
两臂样本期不同（365 天 vs 180 天），池化就是 H3 禁止的混比。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from pit_pool_combine import combine, combine_verdict, render_combined  # noqa: E402


def _payload(interval, overall, mean=-0.2, posthoc=1.5, evaluated=426):
    return {
        "name": "supertrend-pit-pool", "strategy": "supertrend_current",
        "hypothesis_card": "research/hypotheses/H-E-supertrend-pit-pool.md",
        "generated_at": "2026-11-12T23:30:00+00:00",
        "interval": interval, "days": 365 if interval == "1h" else 180, "folds": 5,
        "window": {"start": "2025-11-12T00:00:00+00:00", "end": "2026-11-12T00:00:00+00:00"},
        "universe": {"requested": evaluated, "evaluated": evaluated, "skipped": []},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
                  "basis": "notional", "cash_per_symbol": 10000.0},
        "parameters": {"supertrend_factor": 2.5, "target_r": 3.0},
        "protocol": {"sizes": [10, 20, 30], "seeds": 10},
        "fold_meta": [], "elapsed_seconds": 80.0, "processes": 20,
        "summary": {
            "pit_liq_10": {"pooled": {"trades": 2040, "mean_return_pct": mean,
                                      "ci_low_pct": mean - 0.15, "ci_high_pct": mean + 0.15},
                           "positive_folds": 0, "fold_means": [mean] * 5,
                           "percentile_rank": 0.0,
                           "folds": [{"members": ["A", "B"]}] * 5,
                           "side_split": {"long": {"trades": 1, "mean_return_pct": mean,
                                                   "ci_low_pct": -1, "ci_high_pct": 1,
                                                   "verdict": "negative"},
                                          "short": {"trades": 1, "mean_return_pct": 0.1,
                                                    "ci_low_pct": -1, "ci_high_pct": 1,
                                                    "verdict": "inconclusive"}}},
            "posthoc_10": {"pooled": {"trades": 1246, "mean_return_pct": posthoc,
                                      "ci_low_pct": posthoc - 0.2, "ci_high_pct": posthoc + 0.2},
                           "positive_folds": 5, "fold_means": [posthoc] * 5,
                           "percentile_rank": 1.0,
                           "folds": [{"members": ["A", "B"]}] * 5,
                           "side_split": {"long": {"trades": 1}, "short": {"trades": 1}}},
        },
        "verdicts": {"overall": overall, "n_folds": 5, "fdr_family": ["pit_liq_10"],
                     "arms": {"pit_liq_10": {"verdict": overall, "q": 1.0},
                              "posthoc_10": {"verdict": "inconclusive", "q": None}}},
    }


def test_combine_verdict_all_negative_is_negative():
    assert combine_verdict(["negative", "negative"]) == "negative"


def test_combine_verdict_any_pass_escalates_to_pass():
    assert combine_verdict(["negative", "pass"]) == "pass"


def test_combine_verdict_mixed_negative_and_inconclusive_is_inconclusive():
    assert combine_verdict(["negative", "inconclusive"]) == "inconclusive"


def test_combine_verdict_all_insufficient_is_insufficient():
    assert combine_verdict(["insufficient", "no_trades"]) == "insufficient"
    assert combine_verdict(["no_trades", "no_trades"]) == "no_trades"


def test_combine_keeps_intervals_separate_and_never_pools_their_cis():
    out = combine({"1h": _payload("1h", "negative"), "15m": _payload("15m", "negative")})
    assert out["verdict"] == "negative"
    assert set(out["intervals"]) == {"1h", "15m"}
    # 顶层不得出现把两周期成交混在一起的池化 CI
    assert "pooled" not in out
    assert out["intervals"]["1h"]["summary"]["pit_liq_10"]["pooled"]["trades"] == 2040


def test_combine_declares_limitations_and_enable_decision():
    out = combine({"1h": _payload("1h", "negative"), "15m": _payload("15m", "negative")})
    assert isinstance(out["limitations"], list) and out["limitations"]
    assert out["default_enabled"] is False
    assert "enabled=0" in out["binding_state"]


def test_combine_extracts_the_leakage_gap_per_interval():
    out = combine({"1h": _payload("1h", "negative", mean=-0.2, posthoc=1.5)})
    row = out["leakage"][0]
    assert row["interval"] == "1h" and row["n"] == 10
    assert row["gap_pp"] == 1.7


def test_render_combined_has_required_sections_and_no_banned_phrases():
    out = combine({"1h": _payload("1h", "negative"), "15m": _payload("15m", "negative")})
    text = render_combined(out)
    for word in ("窗口", "币池", "fee_rate", "名义本金", "CI", "随机基线", "局限", "结论",
                 "default_enabled", "negative"):
        assert word in text, word
    for banned in ("显著盈利", "稳定盈利", "已验证有效", "基本有效", "微正"):
        assert banned not in text, banned


def test_render_combined_states_the_preregistered_termination():
    out = combine({"1h": _payload("1h", "negative"), "15m": _payload("15m", "negative")})
    text = render_combined(out)
    assert "不得换" in text
    assert "H-F-funding-multi-venue.md" in text


def test_render_combined_lists_arms_in_canonical_order_not_insertion_order():
    payload = _payload("1h", "negative")
    base = payload["summary"]["pit_liq_10"]
    # 故意按 N 分组打乱插入序：liq_10, vol_10, liq_20, vol_20
    payload["summary"] = {"pit_liq_10": base, "pit_vol_10": base,
                          "pit_liq_20": base, "pit_vol_20": base}
    payload["verdicts"]["arms"] = {k: {"verdict": "negative", "q": 1.0}
                                   for k in payload["summary"]}
    payload["verdicts"]["fdr_family"] = list(payload["summary"])
    text = render_combined(combine({"1h": payload}))
    assert text.index("| pit_liq_20 ") < text.index("| pit_vol_10 ")


def test_render_combined_separates_present_and_qualified_counts():
    payload = _payload("1h", "negative")
    payload["universe"].update({"requested": 535, "qualified": 462, "evaluated": 460})
    text = render_combined(combine({"1h": payload}))
    for number in ("535", "462", "460"):
        assert number in text


def test_combine_surfaces_interval_specific_pool_shortfall():
    # 分报告已声明「N=30 只凑出 26 币」，合并报告是权威交付件，不能把这条边界丢掉
    payload = _payload("15m", "negative")
    payload["limitations"] = [
        "无关条目",
        "**池规模凑不满**：实测 `pit_liq_30` 最少 26/30。",
    ]
    out = combine({"15m": payload})
    hit = [x for x in out["limitations"] if "凑不满" in x]
    assert len(hit) == 1
    assert hit[0].startswith("15m")
    assert "26/30" in hit[0]
    assert "凑不满" in render_combined(out)


def test_combine_without_shortfall_keeps_limitations_unchanged():
    out = combine({"1h": _payload("1h", "negative")})
    assert not [x for x in out["limitations"] if "凑不满" in x]