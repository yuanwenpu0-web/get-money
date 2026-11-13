from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine

DOCS = ROOT / "docs"
ISSUES = [
    {"id": "max_t", "status": "NOT_IMPLEMENTED",
     "finding": "预登记的 Westfall–Young max-T 未实施，原 candidate 仅是当前实现的标签，不能声称完整通过预登记。",
     "evidence": ["docs/indicator-evaluation-plan.md:150", "backend/app/research/fit_stats.py:382"]},
    {"id": "interaction", "status": "METHOD_DEVIATION",
     "finding": "交互 p 是池内外逐币聚合均值的 bootstrap 比较，未做预登记的先整体交互再分格，也未校正交互 p 的多重比较。",
     "evidence": ["docs/indicator-evaluation-plan.md:194", "backend/app/research/fit_stats.py:312"]},
    {"id": "membership", "status": "POSTHOC_GROUPS",
     "finding": "Q 池按评估窗内月成交额事后固定分档，不是事件时点可得的 PIT 成员；major/alt 统计时仍读取可变数据库池。",
     "evidence": ["backend/scripts/fit_prep.py:316", "backend/scripts/indicator_fit_matrix.py:90"]},
    {"id": "pooled_direction", "status": "MISSING_GATE",
     "finding": "子集 candidate 未检查同列整体方向；整体无效的子集按预登记只能是复核线索。反例清单保留原始标签，不事后改统计数字。",
     "evidence": ["docs/indicator-evaluation-plan.md:206", "backend/app/research/fit_stats.py:430"]},
    {"id": "interval_attribution", "status": "CONFOUNDED",
     "finding": "跨所有池选最强格混合了币种构成、周期和实际持有时间；最强格分布不能单独证明周期适配。",
     "evidence": ["backend/scripts/fit_matrix_report.py:119", "backend/scripts/indicator_fit_matrix.py:77"]},
    {"id": "log_warning", "status": "UNATTRIBUTED",
     "finding": "log 除零告警未定位，inf 比较可能为 True，NaN != 0 也为 True；零抽取失败不能证明信号无污染。",
     "evidence": ["output/fit-events-A.log", "backend/app/research/fit_events.py:233"]},
]


def history_gate(first_ms: int | None, last_ms: int | None) -> dict:
    if first_ms is None or last_ms is None:
        return {"status": "BLOCKED", "reason": "无历史数据", "required_first_ms": None}
    required = pd.Timestamp(last_ms, unit="ms", tz="UTC") - pd.DateOffset(months=15)
    return {
        "status": "BLOCKED" if first_ms > required.value // 1_000_000
        else "NEEDS_SYMBOL_QUALITY_CHECK",
        "required_first_ms": int(required.value // 1_000_000),
        "span_days": (last_ms - first_ms) / 86_400_000,
        "rule": "12 calendar months train + 3 calendar months validation; global span is only an upper bound",
    }


def pool_counterexamples(cells: list[dict]) -> list[dict]:
    return [{"key": r["key"], "interval": r["interval"], "scope": r["scope"],
             "pooled_mean_net_pct": r["pooled_mean_net_pct"],
             "raw_label": r.get("verdict_cell") or r.get("verdict"),
             "audit_status": "EXPLORATORY_ONLY",
             "comparison": "report selected overall best cell, not matched horizon/benchmark"}
            for r in cells if r.get("pooled_mean_net_pct") is not None
            and r["pooled_mean_net_pct"] <= 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", action="store_true")
    args = parser.parse_args()
    inputs = {name: DOCS / name for name in (
        "indicator-fit-matrix.json", "indicator-eval-scope.json", "fit-events-A.json")}
    matrix = json.loads(inputs["indicator-fit-matrix.json"].read_text(encoding="utf-8"))
    rows = []
    engine = make_engine(read_only=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '45s'"))
            read_only = conn.execute(text("SHOW transaction_read_only")).scalar()
            if read_only != "on":
                raise RuntimeError("research connection must be read-only")
            for iv in matrix["intervals"]:
                row = dict(conn.execute(text(
                    "SELECT min(open_time) first_ms, max(open_time) last_ms, "
                    "count(*) bars, count(distinct symbol) symbols "
                    "FROM klines WHERE interval=:iv"), {"iv": iv}).mappings().one())
                row.update(interval=iv, **history_gate(row["first_ms"], row["last_ms"]))
                rows.append(row)
    finally:
        engine.dispose()
    blocked = any(r["status"] == "BLOCKED" for r in rows)
    verification = {}
    for name in ("fit-readiness-focused-pytest.log", "fit-readiness-full-pytest.log",
                 "fit-readiness-isolated-pytest.log"):
        log = ROOT / "output" / name
        if log.exists():
            verification[f"output/{name}"] = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": "P18-002", "parent_experiment_id": "P18-001",
        "verification": verification,
        "stage": "S6/S7 readiness and S5 interpretation audit",
        "decision": "BLOCKED", "db_access": "read_only", "transaction_read_only": read_only,
        "scope_sha": matrix["sha"],
        "source_hashes": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in inputs.items()},
        "source_metrics": {k: matrix[k] for k in (
            "columns_evaluated", "columns_candidate", "pool_specific_columns", "rows_evaluated_total")},
        "history": rows, "history_requirement_blocked": blocked,
        "issues": ISSUES, "pool_counterexamples": pool_counterexamples(matrix["pool_specific_cells"]),
        "cost_model": {"fee_side": 0.0004, "slippage_side": 0.0002, "round_trip": 0.0012},
        "backtest_status": "NOT_RUN", "parameter_trials": 0,
        "oos_status": "NOT_AVAILABLE_FROM_SELECTED_90D_WINDOW",
        "risk_gate": "BLOCKED", "paper_status": "NOT_STARTED", "live_orders": False,
        "next_actions": [
            "先定位告警到指标/币种/时间，复核候选是否因非有限值触发。",
            "冻结方法修订及新 scope；补 max-T、先整体交互的校正、PIT 分档和整体方向门。",
            "按每个训练折重新选择指标/周期/池/持有期；不得把90天全窗冠军倒灌早期折。",
            "补足预登记历史并逐币查密度，或登记新的探索性短窗方案，不能将短窗结果冒充原协议通过。",
            "冻结候选后收集未参与选择的新数据；同窗真实仓位回测只能标样本内交易化诊断。",
        ],
    }
    diagnosis = matrix.get("log_diagnosis") or {}
    source = ROOT / "backend/app/catalog_indicators/trend_persistence_score.py"
    if (diagnosis.get("diagnostic_status") == "PASS"
            and diagnosis.get("scope_sha") == matrix["sha"]
            and diagnosis.get("source_sha256") == hashlib.sha256(source.read_bytes()).hexdigest()):
        payload["issues"] = [dict(issue) for issue in ISSUES]
        log_issue = next(issue for issue in payload["issues"] if issue["id"] == "log_warning")
        log_issue.update(status="RESOLVED_FOR_TPS_5M",
                         finding="已定位RIVERUSDT的TPS未选分支log(0)，修复后80币完整输出与旧函数相同，选定缓存统计不变；不解除其余阻塞。",
                         evidence=["docs/indicator-fit-log-diagnosis.md"])
        payload["next_actions"][0] = "log告警工程复核已完成；继续补齐其余统计方法与PIT门槛。"
        payload["log_diagnosis"] = "docs/indicator-fit-log-diagnosis.json"
    lines = ["# P18 后续验证就绪审计", "", f"生成：{payload['generated_at']}；结论：**BLOCKED**。",
             "", "审计与数据边界核验已执行；回测 NOT_RUN，参数试验 0，paper/live 未启动。",
             "原始统计与台账不改写；原 candidate 和池特有数量是实现标签，不能当完整预登记发现。",
             "", "## 数据实查", "", "预登记要求 12 个自然月训练 + 3 个自然月验证。以下是全库覆盖上界，非逐币质量通过。",
             "", "| 周期 | 币数 | bar数 | 全库跨度（天） | 历史门 |", "|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['interval']} | {r['symbols']} | {r['bars']} | {r['span_days']:.2f} | {r['status']} |")
    lines += ["", "90 天窗口已用于选择列、周期、池和 h，不能重新切分后称为这些候选的独立 OOS。",
              "", "## 方法审计", ""]
    for issue in payload["issues"]:
        lines += [f"- **{issue['id']} / {issue['status']}**：{issue['finding']} 证据："
                  + "、".join(f"`{e}`" for e in issue["evidence"]) + "。"]
    lines += ["", "## 整体非正的池特有标签反例", "",
              "以下比的是报告已选出的整体最强格；不是同 h/基准配对，也不是完整方向门复算。",
              "", "| 列 | 周期 | 池 | 报告整体净超额% | 审计解释 |", "|---|---|---|---:|---|"]
    for r in payload["pool_counterexamples"]:
        lines.append(f"| `{r['key']}` | {r['interval']} | {r['scope']} | {r['pooled_mean_net_pct']} | EXPLORATORY_ONLY |")
    lines += ["", "## 后续动作", ""] + [f"{i}. {a}" for i, a in enumerate(payload["next_actions"], 1)]
    lines += ["", "成本固定往返 0.12%；本审计没有新增净收益数字。", "", "## 验证", ""]
    lines += [f"- `{name}`：{result}" for name, result in verification.items()]
    lines += ["", "全量失败中3项是代理格式问题，局部清理NO_PROXY后通过；其余是行情尾部偏移及交易入口缺少halt_status。",
              "这些失败位于本次未修改的执行路径，不宣称全量测试通过。",
              "", "复现：`.venv/Scripts/python.exe backend/scripts/fit_readiness.py`（只读数据库）。", ""]
    (DOCS / "indicator-fit-readiness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (DOCS / "indicator-fit-readiness.md").write_text("\n".join(lines), encoding="utf-8")
    if args.ledger:
        from fit_matrix_report import append_ledger, git_state
        record = {
            "timestamp": payload["generated_at"], "experiment_id": "P18-002",
            "parent_experiment_id": "P18-001", "strategy_id": None,
            "code_commit": git_state(), "stage": payload["stage"],
            "decision": "BLOCKED_VALIDATION", "supersedes_interpretation_of": "P18-001",
            "dataset_version": "local PostgreSQL klines read_only; global coverage upper bound",
            "interval": matrix["intervals"], "parameters": {"scope_sha": matrix["sha"],
                "parameter_trials": 0, "grid_search_performed": False},
            "fee_model": "0.0004 per side", "slippage_model": "0.0002 per side",
            "funding_model": "NOT_MODELLED; no backtest executed",
            "metrics": {"source_labels": payload["source_metrics"],
                        "nonpositive_pooled_counterexamples": len(payload["pool_counterexamples"])},
            "failure_reason": "history and methodological gates unmet; not a negative-return verdict",
            "next_action": payload["next_actions"], "verification": verification,
            "artifacts": ["docs/indicator-fit-readiness.md", "docs/indicator-fit-readiness.json"],
        }
        print(append_ledger(record, ROOT / "research" / "experiments.jsonl"))
    print(f"readiness={payload['decision']}; counterexamples={len(payload['pool_counterexamples'])}")
    for row in rows:
        print(f"{row['interval']}: {row['span_days']:.2f}d {row['status']}")


if __name__ == "__main__":
    main()
