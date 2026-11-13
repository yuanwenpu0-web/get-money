"""台账写入的测试：幂等、换 sha 必须换编号、记录里的数字必须与产物同源。

台账是这套研究流程里唯一「写了就代表本轮结论」的地方，重复写或把两套口径挂在同一个
`experiment_id` 上，后面所有引用它的文档都会说不清指的是哪一次。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import fit_matrix_report as fmr  # noqa: E402

PER_IV = {
    "symbols": 119,
    "verdict_counts_all": {"candidate": 37, "no_edge": 400},
    "redundancy_pairs": 31,
    "redundancy_members": 28,
    "redundancy_clusters": 11,
}


def make_s(**over) -> dict:
    s = {
        "scope": "A", "sha": "b323c0912533842e", "intervals": ["1h"],
        "per_interval": {"1h": dict(PER_IV)},
        "config": {"n_boot": 2000, "q_alpha": 0.10, "blocks_for_candidate": 15,
                   "round_trip": 0.0012, "redundancy_r_min": 0.8},
        "horizons_bars": [6, 24, 96], "benchmarks": ["unconditional", "cross_section"],
        "directions_sha": "2ad8c54c75753576",
        "columns_declared": 885, "columns_evaluated": 737, "columns_degenerate_excluded": 148,
        "columns_promotable": 165, "columns_candidate": 59, "pool_specific_columns": 31,
        "indicators_promotable": 75,
        "pool_specific_by_scope": {"liq_q2": 12}, "promotable_by_interval": {"1h": 165},
        "rows_evaluated_total": 48000,
        "window_days": {"1h": 90}, "pool_sizes": {"liq_q1": 20},
        "liquidity_months": 3, "tier_stability_median": 0.83,
        "provenance_rule": "native 满密度优先", "bars_total_by_interval": {"1h": 257040},
    }
    s.update(over)
    return s


EVENTS = {"elapsed_s": 1210.4, "intervals": {"1h": {"computed": 119, "cached": 0,
                                                    "seconds": 1210.4}}}


def test_record_numbers_come_from_the_payload():
    rec = fmr.ledger_record(make_s(), EVENTS, "P17-001", "36 passed", "")
    assert rec["parent_experiment_id"] == "P17-001"
    assert rec["metrics"]["columns_promotable"] == 165
    assert rec["metrics"]["indicators_promotable"] == 75
    assert rec["metrics"]["verdict_counts_all_by_interval"]["1h"]["candidate"] == 37
    assert rec["parameters"]["scope_sha"] == "b323c0912533842e"
    assert rec["parameters"]["blocks_for_candidate"] == 15
    assert rec["runtime_cost"]["events_total_seconds"] == 1210.4
    assert rec["artifacts"][0] == "docs/indicator-fit-matrix.md"
    # 台账行必须是一行合法 JSON（append 时不校验，坏了要等到下次读才发现）
    assert json.loads(json.dumps(rec, ensure_ascii=False))["experiment_id"] == fmr.LEDGER_EXPERIMENT_ID


def test_suffix_deliverables_are_named_in_the_record():
    rec = fmr.ledger_record(make_s(), EVENTS, "P17-001", "v", "-dev")
    assert rec["artifacts"][0] == "docs/indicator-fit-matrix-dev.md"


def test_append_is_idempotent_for_the_same_sha(tmp_path: Path):
    path = tmp_path / "experiments.jsonl"
    rec = fmr.ledger_record(make_s(), EVENTS, "P17-001", "v", "")
    assert "append" in fmr.append_ledger(rec, path)
    assert "幂等跳过" in fmr.append_ledger(rec, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_same_id_with_a_different_sha_is_refused(tmp_path: Path):
    path = tmp_path / "experiments.jsonl"
    fmr.append_ledger(fmr.ledger_record(make_s(), EVENTS, "P17-001", "v", ""), path)
    changed = fmr.ledger_record(make_s(sha="deadbeefdeadbeef"), EVENTS, "P17-001", "v", "")
    try:
        fmr.append_ledger(changed, path)
    except SystemExit as exc:
        assert "换 experiment_id" in str(exc)
    else:
        raise AssertionError("同编号换 sha 必须报错，不然台账会出现一号两口径")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_dirty_label_counts_paths_without_naming_blame(monkeypatch):
    """`git_state` 只能陈述「脏、几个路径」，不能替本轮结论断言是哪些文件没提交。"""
    def fake(cmd, **kw):
        out = "abc1234\n" if cmd[-1] == "HEAD" else " M a.py\n M b.py\n?? c.py\n"
        return type("R", (), {"stdout": out})()

    monkeypatch.setattr(fmr.subprocess, "run", fake)
    state = fmr.git_state()
    assert state.startswith("abc1234") and "3 个路径" in state
    assert "fit pipeline" not in state

    def clean(cmd, **kw):
        return type("R", (), {"stdout": "abc1234\n" if cmd[-1] == "HEAD" else ""})()

    monkeypatch.setattr(fmr.subprocess, "run", clean)
    assert fmr.git_state() == "abc1234"


def test_last_id_reads_the_tail(tmp_path: Path):
    path = tmp_path / "experiments.jsonl"
    path.write_text('{"experiment_id": "P1-001"}\nnot json\n{"experiment_id": "P2-003"}\n',
                    encoding="utf-8")
    assert fmr.last_ledger_id(path) == "P2-003"
    assert fmr.last_ledger_id(tmp_path / "missing.jsonl") is None
