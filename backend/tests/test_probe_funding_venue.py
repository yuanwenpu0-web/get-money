"""H-F Gate 1a 判据装配与成对报告（假设卡 §7 预登记的测试入口）。"""

import json

import pytest

from app.research.funding_venue_render import render, write_reports
from app.research.funding_venue_report import assemble

BANNED = ("显著盈利", "稳定盈利", "已验证有效", "基本有效", "微正")


def _bonus(total, part):
    return {"total": total, "non_hedgeable": part,
            "share": part / total if total > 0 else None,
            "n_selected": 10, "n_non_hedgeable": 6}


def _raw(n=30, full_rate=0.003, hedge_rate=0.0005, ic=0.6, bonus=None):
    bonus = bonus if bonus is not None else [_bonus(0.02, 0.015)] * n
    per_k = {}
    for k in (10, 20):
        per_k[str(k)] = {
            "full_returns": [full_rate] * n,
            "hedgeable_returns": [hedge_rate] * n,
            "bonus": bonus,
            "ic_non_hedgeable": [ic] * n,
            "ic_full": [ic] * n,
            "ic_hedgeable": [ic] * n,
            "cross_counts_non_hedgeable": [156] * n,
            "cross_counts_full": [507] * n,
            "cross_counts_hedgeable": [351] * n,
            "missing_forward": 0,
            "n_symbol_periods": 10 * n,
            "random_baseline": {
                "k": k, "n_seeds": 20, "seed": 20261112,
                "samples_pct": [10.0] * 20, "mean_pct": 10.0,
                "p05_pct": 10.0, "median_pct": 10.0, "p95_pct": 10.0},
        }
    return {
        "data": {"rows": 1085206, "symbols": 507, "start_ms": 0,
                 "end_ms": 416 * 86_400_000, "days": 416.0,
                 "rate_types": ["Regular"],
                 "events_per_symbol": {"min": 107, "median": 2498, "max": 3957}},
        "spot_snapshot": {"source": "https://api.binance.com/api/v3/exchangeInfo",
                          "fetched_at": "2026-11-12T23:40:00+00:00",
                          "symbols_total": 3742, "trading": 1372,
                          "distinct_trading_bases": 519, "from_cache": False},
        "universe": {"hedgeable": [f"H{i}USDT" for i in range(351)],
                     "non_hedgeable": [f"N{i}USDT" for i in range(156)],
                     "historical_reference": {"hedgeable": 348, "total": 507,
                                              "as_of": "2026-10-27"}},
        "periods": {"count": n, "first_ms": 7 * 86_400_000,
                    "last_ms": (7 + 3 * (n - 1)) * 86_400_000},
        "per_k": per_k,
    }


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


def test_assemble_d3_is_full_minus_hedgeable_annualized_gross():
    out = assemble(_raw())
    arm = out["arms"]["10"]
    assert arm["full"]["annual_gross_pct"] == pytest.approx(
        (1.003 ** (365 / 3) - 1) * 100)
    assert arm["d3_increment_pp"] == pytest.approx(
        arm["full"]["annual_gross_pct"] - arm["hedgeable_only"]["annual_gross_pct"])
    assert arm["d3_increment_pp"] > 25


def test_assemble_pools_bonus_share_instead_of_averaging_ratios():
    bonus = [_bonus(0.02, 0.015), _bonus(0.01, 0.004)] * 15
    arm = assemble(_raw(bonus=bonus))["arms"]["10"]
    assert arm["bonus"]["pooled_share"] == pytest.approx(0.019 / 0.03, abs=1e-4)
    assert arm["bonus"]["pooled_share"] != pytest.approx(0.575, abs=1e-3)


def test_assemble_requires_both_k_to_pass():
    raw = _raw()
    raw["per_k"]["20"]["ic_non_hedgeable"] = [0.1] * 30
    out = assemble(raw)
    assert out["gates"]["10"]["pass"] is True
    assert out["gates"]["20"]["pass"] is False
    assert out["verdict"] == "negative"


def test_assemble_pass_when_every_gate_met_at_both_k():
    out = assemble(_raw())
    assert out["verdict"] == "pass"
    assert all(out["gates"][k]["pass"] for k in ("10", "20"))


def test_assemble_insufficient_when_too_few_periods():
    out = assemble(_raw(n=5))
    assert out["verdict"] == "insufficient"


def test_assemble_emits_no_net_return_field_anywhere():
    out = assemble(_raw())
    assert not [k for k in _walk_keys(out) if "net" in k.lower()]


def test_assemble_records_gross_only_protocol_and_frozen_thresholds():
    out = assemble(_raw())
    assert out["protocol"]["costs_applied"] is False
    assert out["protocol"]["hold_days"] == 3
    assert out["protocol"]["lookback_days"] == 7
    assert out["protocol"]["ks"] == [10, 20]
    assert out["thresholds"]["D1_min_share"] == 0.60
    assert out["thresholds"]["D3_min_annual_pp"] == 25.0
    assert out["default_enabled"] is False


def test_assemble_counts_non_hedgeable_universe_and_spot_drift():
    out = assemble(_raw())
    assert out["universe"]["hedgeable_count"] == 351
    assert out["universe"]["non_hedgeable_count"] == 156
    assert out["universe"]["spot_drift_vs_2026_10_27"] == 3


def test_render_states_verdict_both_k_and_gross_only_boundary():
    text = render(assemble(_raw()))
    assert "`pass`" in text
    assert "K=10" in text and "K=20" in text
    assert "不报净收益" in text
    assert "幸存者偏差" in text
    assert "348" in text and "351" in text


def test_render_states_negative_verdict_when_gate_fails():
    raw = _raw()
    raw["per_k"]["20"]["bonus"] = [_bonus(0.02, 0.002)] * 30
    assert "`negative`" in render(assemble(raw))


def test_render_contains_no_banned_phrases():
    text = render(assemble(_raw()))
    assert not [w for w in BANNED if w in text]


def test_render_shows_missing_forward_and_skipped_ic_periods():
    raw = _raw()
    raw["per_k"]["10"]["missing_forward"] = 7
    raw["per_k"]["10"]["ic_non_hedgeable"] = [0.6] * 25 + [None] * 5
    text = render(assemble(raw))
    assert "缺前瞻 7" in text
    assert "跳过 5" in text


def test_write_reports_produces_parseable_pair(tmp_path):
    payload = assemble(_raw())
    prefix = tmp_path / "probe-funding-venue"
    md_path, json_path = write_reports(payload, str(prefix))
    assert md_path.exists() and json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["name"] == payload["name"]
    assert md_path.read_text(encoding="utf-8").startswith("# ")


def test_assemble_rounds_ic_readings_for_report():
    arm = assemble(_raw(ic=0.123456789))["arms"]["10"]
    assert arm["ic_non_hedgeable"]["mean"] == 0.1235
    assert arm["ic_full"]["mean"] == 0.1235


def test_assemble_reports_cross_section_for_every_universe():
    arm = assemble(_raw())["arms"]["10"]
    assert arm["ic_non_hedgeable"]["min_cross_section"] == 156
    assert arm["ic_full"]["min_cross_section"] == 507
    assert arm["ic_hedgeable"]["min_cross_section"] == 351


def test_render_warns_gross_is_unexecutable_counterfactual():
    text = render(assemble(_raw()))
    assert "反事实" in text
    assert "不是收益预期" in text


def test_render_compares_to_historical_single_venue_gross():
    text = render(assemble(_raw()))
    assert "55.6" in text and "10.8" in text
    assert "不可直接比较" in text


def test_render_gate_summary_is_readable_not_python_dict():
    text = render(assemble(_raw()))
    assert "True" not in text
    assert "D1 过" in text


def test_assemble_attaches_block_bootstrap_ci_to_each_arm():
    arm = assemble(_raw())["arms"]["10"]
    ci = arm["full"]["ci"]
    assert ci["method"] == "moving_block8"
    assert ci["n_boot"] == 2000 and ci["seed"] == 20261112
    assert ci["ci_low_pct"] == pytest.approx(0.3, abs=1e-4)
    assert ci["ci_high_pct"] == pytest.approx(0.3, abs=1e-4)


def test_assemble_attaches_paired_ci_to_d3_increment():
    arm = assemble(_raw())["arms"]["10"]
    assert arm["d3_ci"]["ci_low_pp"] > 25
    assert arm["d3_ci"]["n_pairs"] == 30


def test_assemble_attaches_ci_to_bonus_share_and_subset_ic():
    arm = assemble(_raw())["arms"]["10"]
    assert arm["bonus"]["share_ci"]["ci_low"] == pytest.approx(0.75)
    assert arm["ic_non_hedgeable"]["ci"]["ci_low"] == pytest.approx(0.6)


def test_assemble_reports_random_baseline_percentile():
    arm = assemble(_raw())["arms"]["10"]
    baseline = arm["random_baseline"]
    assert baseline["percentile_of_strategy"] == 1.0
    assert baseline["n_seeds"] == 20


def test_assemble_percentile_is_none_not_nan_when_baseline_empty():
    raw = _raw()
    for k in ("10", "20"):
        raw["per_k"][k]["random_baseline"]["samples_pct"] = [None] * 20
        raw["per_k"][k]["random_baseline"]["mean_pct"] = None
    payload = assemble(raw)
    assert payload["arms"]["10"]["random_baseline"]["percentile_of_strategy"] is None
    json.dumps(payload, allow_nan=False)


def test_assemble_emits_costs_window_and_sample_unit_fields():
    payload = assemble(_raw())
    assert payload["costs"]["applied"] is False
    assert payload["costs"]["fee_rate"] == 0.0004
    assert payload["costs"]["slippage"] == 0.0002
    assert payload["window"]["days"] == 416.0
    assert payload["window"]["periods"] == 30
    assert payload["trades"]["simulated_fills"] is False
    assert payload["trades"]["n_symbol_periods"] == 300


def test_render_shows_confidence_intervals_and_random_baseline():
    text = render(assemble(_raw()))
    assert "95% CI" in text
    assert "随机基线" in text
    assert "分位" in text


def test_render_does_not_claim_significance_from_a_point_estimate():
    text = render(assemble(_raw()))
    assert "移动块" in text
    assert "口径略有差异" in text


def test_assemble_rounds_d3_ci_bounds():
    raw = _raw()
    raw["per_k"]["10"]["full_returns"] = [0.003 + 0.0001 * i for i in range(30)]
    ci = assemble(raw)["arms"]["10"]["d3_ci"]
    for field in ("ci_low_pp", "ci_high_pp", "mean_pp"):
        assert ci[field] == round(ci[field], 4)


def test_render_flags_arms_whose_period_mean_ci_crosses_zero():
    raw = _raw()
    raw["per_k"]["10"]["hedgeable_returns"] = [0.004, -0.004] * 15
    text = render(assemble(raw))
    assert "跨 0" in text
    assert "仅可对冲 TopK（K=10）" in text


def test_render_no_cross_zero_flag_when_every_ci_is_positive():
    assert "跨 0" not in render(assemble(_raw()))