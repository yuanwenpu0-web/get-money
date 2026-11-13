"""P17 / H-E 成对报告渲染单测（不联网、不连 DB）。

报告是判定的一部分：必须把「事后选币偏差有多大」和「哪些币被跳过」写在纸面上，
否则读者只看到 10 个臂的收益数字，看不到泄漏与数据边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research.pit_pool_report import limitations, render  # noqa: E402
from app.research.pit_pool_stats import arm_names, judge  # noqa: E402


def _ci(trades=400, mean=-0.2, lo=-0.3, hi=-0.1, share=0.02):
    return {"trades": trades, "mean_return_pct": mean, "ci_low_pct": lo,
            "ci_high_pct": hi, "positive_share": share, "verdict": "negative"}


def _arm(members=("AAAUSDT", "BBBUSDT"), mean=-0.2):
    return {
        "pooled": _ci(mean=mean, lo=mean - 0.1, hi=mean + 0.1),
        "side_split": {"long": _ci(trades=200, mean=mean), "short": _ci(trades=200, mean=mean)},
        "folds": [
            {"index": 0, "start": "2025-11-12", "end": "2026-01-24",
             "members": list(members), "requested": len(members), "trades": 200, "mean": mean},
            {"index": 1, "start": "2026-01-24", "end": "2026-04-07",
             "members": list(members), "requested": len(members), "trades": 200, "mean": mean},
        ],
        "fold_means": [mean, mean],
        "positive_folds": 0,
        "account_mean_pct": -3.0,
        "random_mean_pct": -1.0,
        "random_p95_pct": 0.5,
        "percentile_rank": 0.1,
        "symbols_unique": list(members),
    }


def _payload():
    summary = {name: _arm() for name in arm_names()}
    summary["posthoc_10"] = _arm(mean=5.0)
    summary["posthoc_20"] = _arm(mean=4.0)
    summary["posthoc_30"] = _arm(mean=3.0)
    return {
        "name": "supertrend-pit-pool", "strategy": "supertrend_current",
        "hypothesis_card": "research/hypotheses/H-E-supertrend-pit-pool.md",
        "generated_at": "2026-11-12T23:00:00+00:00",
        "interval": "1h", "days": 365, "folds": 2,
        "window": {"start": "2025-11-12T00:00:00+00:00", "end": "2026-11-12T00:00:00+00:00"},
        "universe": {"kind": "coverage_filtered", "requested": 461, "evaluated": 459,
                     "coverage_threshold": 0.95, "expected_bars": 8760,
                     "excluded": ["BTCUSDT"],
                     "skipped": ["ZZZUSDT: 无满足长度的连续K线段"]},
        "parameters": {"supertrend_factor": 2.5, "supertrend_atr": 14, "target_r": 3.0,
                       "stop_atr_mult": 1.5},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "cash_per_symbol": 10000.0,
                  "basis": "notional", "round_trip": 0.0012},
        "protocol": {"warmup_bars": 600, "min_fold_bars": 200, "min_segment_bars": 800,
                     "sizes": [10, 20, 30], "seeds": 10, "base_seed": 20261025,
                     "min_pooled_trades": 30, "ranking_window_hours": 168,
                     "ranking_interval": "1h", "arms": arm_names()},
        "fold_meta": [{"index": 0, "start": 0, "end": 1, "start_iso": "2025-11-12",
                       "end_iso": "2026-01-24", "eligible_count": 500},
                      {"index": 1, "start": 1, "end": 2, "start_iso": "2026-01-24",
                       "end_iso": "2026-04-07", "eligible_count": 505}],
        "processes": 12, "elapsed_seconds": 120.0,
        "summary": summary, "verdicts": judge(summary),
    }


def test_render_states_the_overall_verdict():
    text = render(_payload())
    assert "negative" in text
    assert "H-E-supertrend-pit-pool.md" in text


def test_render_lists_every_arm_with_its_verdict():
    payload = _payload()
    text = render(payload)
    for name in arm_names():
        assert name in text
        assert payload["verdicts"]["arms"][name]["verdict"] in text


def test_render_quantifies_posthoc_leakage_gap_in_pp_per_trade():
    # posthoc_10 = +5.0%/笔，pit_liq_10 = -0.2%/笔 → 泄漏 5.2pp/笔
    text = render(_payload())
    assert "5.2" in text
    assert "泄漏" in text


def test_render_reports_jaccard_overlap_of_adjacent_folds():
    text = render(_payload())
    assert "Jaccard" in text
    assert "1.000" in text  # 两期成员完全相同


def test_render_lists_skipped_symbols_and_coverage():
    text = render(_payload())
    assert "ZZZUSDT" in text
    assert "459" in text and "461" in text


def test_render_declares_read_only_scope_and_frozen_binding():
    text = render(_payload())
    assert "不下真实订单" in text
    assert "enabled=0" in text


def test_render_shows_side_split_verdicts():
    text = render(_payload())
    assert "做多" in text and "做空" in text


def test_render_tolerates_a_summary_missing_arms():
    payload = _payload()
    payload["summary"] = {"all": _arm()}
    payload["verdicts"] = judge(payload["summary"])
    text = render(payload)
    assert "| all " in text
    assert "未产出" in text


def test_render_marks_diagnostic_arms_as_not_gated():
    text = render(_payload())
    assert "诊断" in text


def test_render_shows_curated_frozen_params_not_the_whole_dict():
    text = render(_payload())
    assert "supertrend_factor=2.5" in text
    assert "target_r=3.0" in text
    assert "cl_pending_expire_bars" not in text  # 完整参数留在 JSON，不糊在报告里


def test_render_arm_count_matches_the_arms_actually_produced():
    payload = _payload()
    payload["summary"] = {"all": _arm()}
    payload["verdicts"] = judge(payload["summary"])
    assert "## 4. 1 臂结果" in render(payload)


def test_render_warns_when_pool_size_reaches_the_whole_universe():
    # 15m 只有 30 币达标时，N=30 的「缩小池」等于不缩小，必须显式声明退化
    payload = _payload()
    payload["universe"]["evaluated"] = 30
    text = render(payload)
    assert "退化" in text
    assert "30" in text


def test_render_no_degenerate_warning_when_universe_is_large():
    payload = _payload()
    payload["universe"]["evaluated"] = 426
    assert "退化" not in render(payload)


def test_limitations_are_exposed_as_data_and_all_rendered():
    payload = _payload()
    items = limitations(payload)
    assert isinstance(items, list) and len(items) >= 5
    assert all(isinstance(x, str) and x for x in items)
    text = render(payload)
    for item in items:
        assert item in text


def test_render_states_enable_decision_and_binding_state():
    text = render(_payload())
    assert "default_enabled=False" in text
    assert "enabled=0" in text


def test_render_separates_present_qualified_and_evaluated_counts():
    # H8：必须让人看出「窗口内有多少币」与「覆盖率筛掉多少」，不能只报达标数
    payload = _payload()
    payload["universe"].update({"requested": 535, "qualified": 462, "evaluated": 460})
    text = render(payload)
    for number in ("535", "462", "460"):
        assert number in text


def test_limitations_flag_pool_smaller_than_requested_size():
    # 15m 实测：N=30 只凑出 26 币（1h 排名候选不足），「固定规模」没做到就必须说出来
    payload = _payload()
    for fold in payload["summary"]["pit_liq_30"]["folds"]:
        fold["members"] = fold["members"][:1]
        fold["requested"] = 30
    items = limitations(payload)
    hit = [x for x in items if "凑不满" in x]
    assert len(hit) == 1
    assert "pit_liq_30" in hit[0]
    assert "1/30" in hit[0]


def test_limitations_no_shortfall_item_when_every_fold_is_full():
    payload = _payload()
    assert not [x for x in limitations(payload) if "凑不满" in x]