import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import fit_pool_review as review_tool
from fit_pool_review import review_interval


def test_frozen_protocol_edit_is_rejected_before_review(tmp_path):
    plan = tmp_path / "plan.json"
    original = {"costs": {"round_trip": 0.0012}}
    plan.write_text(json.dumps(original), encoding="utf-8")
    plan.with_suffix(".sha256").write_text(hashlib.sha256(plan.read_bytes()).hexdigest(), encoding="ascii")
    assert review_tool.load_frozen_plan(plan) == original
    plan.write_text(json.dumps({"costs": {"round_trip": 0.0001}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen protocol"):
        review_tool.load_frozen_plan(plan)


def test_both_original_labels_are_preserved_even_if_they_differ():
    candidate = row(scope="major")
    candidate["verdict"] = "likely_noise"
    result = review_interval([row(), candidate])[0]
    assert result["original_verdict_base"] == "likely_noise"
    assert result["original_verdict_cell"] == "candidate"


def row(scope="all", direction=1, net=0.2, horizon=24, benchmark="unconditional"):
    return {"key": "indicator|signal", "scope": scope, "direction": direction,
            "horizon_bars": horizon, "benchmark": benchmark,
            "mean_net_pct": net, "verdict": "candidate", "verdict_cell": "candidate"}


def test_nonpositive_overall_net_downgrades_pool_without_mutating_evidence():
    rows = [row(net=-0.12), row(scope="major", net=0.8)]
    snapshot = copy.deepcopy(rows)
    result = review_interval(rows)[0]
    assert result["direction_gate_status"] == "FAIL"
    assert result["review_verdict"] == "exploratory"
    assert result["reason"] == "pooled_nonpositive_net"
    assert result["pooled_mean_net_pct"] == -0.12
    assert rows == snapshot


@pytest.mark.parametrize("changes", [
    {"direction": -1}, {"horizon": 96}, {"benchmark": "cross_section"}])
def test_other_direction_horizon_or_benchmark_cannot_supply_overall_support(changes):
    rows = [row(**changes), row(scope="major")]
    result = review_interval(rows)[0]
    assert result["direction_gate_status"] == "BLOCKED"
    assert result["reason"] == "matching_pooled_row_missing"


def test_short_direction_uses_already_direction_aligned_net_without_second_flip():
    result = review_interval([row(direction=-1, net=0.3), row(scope="major", direction=-1)])[0]
    assert result["direction_gate_status"] == "PASS"
    assert result["review_verdict"] == "candidate"


@pytest.mark.parametrize("net", [None, float("nan"), float("inf"), -float("inf"), 0])
def test_missing_or_nonfinite_overall_net_cannot_pass(net):
    result = review_interval([row(net=net), row(scope="major")])[0]
    assert result["review_verdict"] == "exploratory"
    assert result["direction_gate_status"] in ("BLOCKED", "FAIL")


def test_last_opposite_direction_cannot_overwrite_matching_overall_row():
    rows = [row(direction=1, net=-0.2), row(direction=-1, net=0.2), row(scope="major", direction=1)]
    result = review_interval(rows)[0]
    assert result["pooled_mean_net_pct"] == -0.2
    assert result["review_verdict"] == "exploratory"


def test_non_candidates_are_not_promoted_and_conflicting_pooled_duplicates_fail():
    noncandidate = row(scope="major")
    noncandidate["verdict_cell"] = "no_edge"
    assert review_interval([row(), noncandidate]) == []
    with pytest.raises(ValueError, match="duplicate"):
        review_interval([row(net=0.2), row(net=-0.2), row(scope="major")])
