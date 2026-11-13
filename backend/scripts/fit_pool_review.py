"""Frozen-artifact overall-direction review; no DB, new returns, or parameter search."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.research.fit_pool_gate import review_interval

DOCS = ROOT / "docs"
SOURCE = DOCS / "indicator-fit-stats-A.json"
MATRIX = DOCS / "indicator-fit-matrix.json"
PLAN = DOCS / "indicator-fit-pool-review-plan.json"
PREFIX = DOCS / "indicator-fit-pool-review"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_plan(path):
    fingerprint = path.with_suffix(".sha256")
    if not fingerprint.exists() or fingerprint.read_text(encoding="ascii").strip() != digest(path):
        raise SystemExit("frozen protocol fingerprint missing or changed; review aborted")
    return json.loads(path.read_text(encoding="utf-8"))


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in (SOURCE, MATRIX,
            ROOT / "backend/app/research/fit_pool_gate.py", Path(__file__))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="freeze protocol before seeing review counts")
    parser.add_argument("--ledger", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if PLAN.exists() or PLAN.with_suffix(".sha256").exists():
            raise SystemExit("plan exists; do not overwrite a frozen review protocol")
        if args.ledger:
            raise SystemExit("freeze is not an executed review and cannot enter result ledger")
        plan = {
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment_id": "P18-005", "parent_experiment_id": "P18-002",
            "operation": "necessary-overall-direction-gate-only",
            "source_hashes": sources(),
            "comparison": "same interval/key/direction/horizon/benchmark; overall scope=all",
            "required": "matching overall direction-aligned mean_net_pct must be finite and strictly > 0",
            "missing_policy": "BLOCKED; preserve as exploratory, do not borrow another coordinate",
            "failure_policy": "nonpositive overall mean -> FAIL this gate and exploratory review label",
            "pass_policy": "retain raw candidate label only; still exploratory inference pending other gates",
            "universe": "original A stats rows, unchanged; no new symbol selection or DB access",
            "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012},
            "parameter_trials": 0, "recompute_statistics": False, "new_returns": False,
            "remaining_blockers": ["max-T", "corrected omnibus interaction/multiplicity", "PIT membership",
                                   "overall significance/stability requirements", "independent OOS/history"],
        }
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        PLAN.with_suffix(".sha256").write_text(digest(PLAN) + "\n", encoding="ascii")
        print(f"frozen {PLAN.name}; no review counts calculated")
        return
    if not PLAN.exists():
        raise SystemExit("freeze the protocol first with --freeze")
    plan = load_frozen_plan(PLAN)
    if plan["source_hashes"] != sources():
        raise SystemExit("frozen inputs/code changed; do not overwrite the plan or silently rerun")
    stats = json.loads(SOURCE.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if stats["sha"] != matrix["sha"]:
        raise SystemExit("source scopes differ")
    if stats["config"].get("round_trip") != plan["costs"]["round_trip"]:
        raise SystemExit("original cost model differs from frozen protocol")
    reviewed, by_interval = [], {}
    for interval, block in stats["intervals"].items():
        items = review_interval(block["rows"])
        for row in items:
            row["interval"] = interval
        reviewed += items
        by_interval[interval] = dict(Counter(r["direction_gate_status"] for r in items))
    totals = dict(Counter(r["direction_gate_status"] for r in reviewed))
    downgraded = [r for r in reviewed if r["direction_gate_status"] != "PASS"]
    identifiers = {(r["key"], r["interval"], r["scope"], r["direction"],
                    r["horizon_bars"], r["benchmark"]): r for r in reviewed}
    selected = []
    for row in matrix["pool_specific_cells"]:
        key = (row["key"], row["interval"], row["scope"], row["direction"],
               row["horizon_bars"], row["benchmark"])
        if key not in identifiers:
            raise AssertionError(f"selected original cell missing from raw source: {key}")
        selected.append(identifiers[key])
    selected_counts = dict(Counter(r["direction_gate_status"] for r in selected))
    verification = {}
    for name in ("fit-pool-review-focused-pytest.log", "fit-pool-review-full-pytest.log"):
        log = ROOT / "output" / name
        if log.exists():
            verification[f"output/{name}"] = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": "P18-005", "parent_experiment_id": "P18-002", "scope_sha": stats["sha"],
        "stage": "overall-direction necessary gate correction; no new edge test",
        "decision": "BLOCKED_OTHER_GATES", "inference_status": "EXPLORATORY_ONLY",
        "verdict": "inconclusive", "gate_execution": "PASS", "downstream_promotions_allowed": False,
        "protocol_sha256": digest(PLAN), "source_hashes": plan["source_hashes"],
        "costs": plan["costs"], "costs_applied": "in original source; no new cost calculation",
        "parameter_trials": 0, "db_access": "NONE", "backtest": "NOT_RUN", "trades": None,
        "ci": "original intervals preserved; no recomputation", "random_baseline": "NOT_RUN_NO_NEW_EDGE_TEST",
        "window": "original A 90d selection sample; not OOS",
        "universe": {"intervals": list(stats["intervals"]), "selection": "unchanged original rows"},
        "metrics": {"raw_pool_candidate_rows": len(reviewed), "gate_counts": totals,
                    "downgraded_rows": len(downgraded), "by_interval": by_interval,
                    "original_selected_pool_specific_columns": len(selected),
                    "selected_direction_gate_counts": selected_counts,
                    "selected_downgraded_columns": sum(r["direction_gate_status"] != "PASS" for r in selected)},
        "reviewed_rows": reviewed, "downgraded_rows": downgraded,
        "original_selected_cells_review": selected,
        "remaining_blockers": plan["remaining_blockers"], "verification": verification,
        "limitations": ["仅补必要的整体净超额正号门，不等同于整体显著性/稳定性通过。",
                        "只收紧原candidate标签，不重新筛指标/周期/池/h，不重算原统计；按原产物已舍入净值判断，0或-0保守不通过。", 
                        "保留原A轮统计和台账；通过方向门的行仍不能交易化或声称独立OOS。",
                        "原池成员时点、交互和多重比较问题未在本轮修复。"],
        "default_enabled_changed": False, "live_orders": False,
    }
    lines = ["# P18 池级候选整体方向门复核", "", f"生成：{result['generated_at']}。", "",
             "结论：必要方向门已执行；完整研究仍为BLOCKED_OTHER_GATES，收益verdict为inconclusive（未新评估）。", "",
             "## 冻结口径", "", "协议在查看修订计数前冻结，协议指纹与原始输入/代码哈希见同名json。",
             "按同周期、同列、同方向、同h、同基准匹配all池；只要求方向化整体净超额有限且严格大于0。",
             "缺配对或非有限值记BLOCKED，整体非正记FAIL；两者的复核标签均降为exploratory。",
             "通过该必要门仍不能说明整体统计显著或满足全部预登记。", "",
             "窗口是原A轮90天筛选样本；币池、方向、成本、参数均不重选，数据库访问0、参数试验0。",
             "成本沿用原统计名义本金往返0.12%（fee_rate=0.0004/slippage=0.0002，每边），不新增净收益。", "",
             "## 结果", "", f"- 原池级candidate行：{len(reviewed)}；方向门计数：{totals}。",
             f"- 降级行：{len(downgraded)}。原55列口径下逐列选中格的配对复核：{selected_counts}；降级{result['metrics']['selected_downgraded_columns']}列。",
             "- 原先按整体跨周期最强格看到的6个反例，不能代替本轮同坐标配对的计数。", "",
             "| 周期 | PASS | FAIL | BLOCKED |", "|---|---:|---:|---:|"]
    for iv, counts in by_interval.items():
        lines.append(f"| {iv} | {counts.get('PASS',0)} | {counts.get('FAIL',0)} | {counts.get('BLOCKED',0)} |")
    lines += ["", "## 原池特有选中格的降级明细", "",
              "| 列 | 周期 | 池 | 方向 | h | 基准 | 池净% | 匹配整体净% | 原标签 | 复核标签 |", "|---|---|---|---:|---:|---|---:|---:|---|---|"]
    for row in selected:
        if row["direction_gate_status"] != "PASS":
            lines.append(f"| `{row['key'].replace('|','.')}` | {row['interval']} | {row['scope']} | {row['direction']} | {row['horizon_bars']} | {row['benchmark']} | {row['cell_mean_net_pct']} | {row['pooled_mean_net_pct']} | {row['original_verdict']} | {row['review_verdict']} |")
    lines += ["", "## 局限与启用决策", ""] + [f"- {x}" for x in result["limitations"]]
    lines += ["", "未跑交易回测、未新增置信区间CI或随机基线；default_enabled/绑定不改，未启用paper/live。", "",
              "## 验证", ""] + [f"- `{name}`：{value}" for name, value in verification.items()]
    lines += ["", "复现：`.venv/Scripts/python.exe backend/scripts/fit_pool_review.py`；禁止覆写已冻结协议。", ""]
    PREFIX.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    PREFIX.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    if args.ledger:
        from fit_matrix_report import append_ledger, git_state
        record = {"timestamp": result["generated_at"], "experiment_id": "P18-005", "parent_experiment_id": "P18-002",
                  "strategy_id": None, "code_commit": git_state(), "stage": result["stage"],
                  "decision": result["decision"], "parameters": {"scope_sha": stats["sha"],
                  "protocol_sha256": result["protocol_sha256"], "parameter_trials": 0},
                  "metrics": result["metrics"], "verification": verification,
                  "next_action": "repair remaining multiplicity, PIT and independent OOS gates",
                  "artifacts": ["docs/indicator-fit-pool-review.md", "docs/indicator-fit-pool-review.json",
                                "docs/indicator-fit-pool-review-plan.json"]}
        print(append_ledger(record, ROOT / "research/experiments.jsonl"))
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
