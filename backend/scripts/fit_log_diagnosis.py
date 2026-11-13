"""P18 engineering-only replay: local read-only DB, fixed cached windows, no edge verdict."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.catalog_indicators._pine import highest, lowest, sum_
from app.catalog_indicators.trend_persistence_score import trend_persistence_score
from app.db import make_engine
from app.market.data_service import MarketDataService
from app.research.fit_events import FIELDS, bool_column_positions, forward_returns, rising_edges

CACHE = ROOT / "output/cache/fit/b323c0912533842e/5m"
PREFIX = ROOT / "docs/indicator-fit-log-diagnosis"


def legacy_function():
    """Reconstruct the exact pre-fix expression for paired engineering replay only."""
    source = inspect.getsource(trend_persistence_score)
    start = source.index("    usable_fdi =")
    end = source.index("    fdi_trend_score =", start)
    old = ("    fdi_raw = pd.Series(\n"
           "        np.where((range_hl > 0.0) & (path_length > 0.0),\n"
           "                 np.log(path_length / range_hl) / np.log(length) + 1.0, 1.5),\n"
           "        index=frame.index)\n")
    source = source[:start] + old + source[end:]
    namespace = dict(trend_persistence_score.__globals__)
    exec(compile(source, "<pre-fix-trend-persistence-score>", "exec"), namespace)
    return namespace["trend_persistence_score"], hashlib.sha256(source.encode()).hexdigest()


def audit_symbol(svc, path, legacy):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["__meta__"][0])
        stored = {key: archive[f"a{i}"].copy() for i, key in enumerate(meta["keys"])
                  if key.startswith("trend_persistence_score|")}
    frame = svc.load_frame(meta["symbol"], "5m", limit=0,
                           start_ms=meta["first_ms"], end_ms=meta["last_ms"] + 300_000 - 1)
    if len(frame) != meta["bars"]:
        raise ValueError(f"{meta['symbol']}: frame bars no longer match original cache")
    path_length = sum_(frame["close"].diff().abs(), 34)
    range_hl = highest(frame["high"], 34) - lowest(frame["low"], 34)
    ratio = path_length / range_hl
    zero = ratio.eq(0)
    with warnings.catch_warnings(record=True) as old_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        before = legacy(frame)
    with warnings.catch_warnings(record=True) as new_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        result = trend_persistence_score(frame)
    pd.testing.assert_frame_equal(result, before, check_exact=True)
    if new_warnings:
        raise AssertionError(f"{meta['symbol']}: warnings after fix: {new_warnings}")
    baseline_check = {"all_output_columns_exact": True,
                      "rows": len(result), "columns": len(result.columns)}
    checks = 0
    for position, column in bool_column_positions(result):
        key = f"trend_persistence_score|{column}"
        if key not in stored:
            raise AssertionError(f"missing original cache column {key}")
        mask = result.iloc[:, position].to_numpy(dtype=bool)
        for hi, horizon in enumerate(meta["horizons"]):
            returns = forward_returns(frame["open"].to_numpy(float), frame["close"].to_numpy(float), horizon)
            events = rising_edges(mask, horizon)
            events = events[np.isfinite(returns[events])]
            values = returns[events]
            excess = values - meta["bench"][str(horizon)]
            observed = {"n": len(events), "hit_bars": int(mask.sum()),
                        "sum_r": float(values.sum()), "sum_sq": float(np.square(excess).sum())}
            for field, value in observed.items():
                cached = float(stored[key][hi, FIELDS.index(field)])
                if field in ("n", "hit_bars"):
                    if value != cached:
                        raise AssertionError(f"{meta['symbol']} {key} h{horizon} {field}: {value} != {cached}")
                elif not np.isclose(value, cached, rtol=1e-5, atol=1e-6):
                    raise AssertionError(f"{meta['symbol']} {key} h{horizon} {field}: {value} != {cached}")
                checks += 1
    times = frame.index[zero]
    return {"symbol": meta["symbol"], "bars": len(frame),
            "first_ms": meta["first_ms"], "last_ms": meta["last_ms"],
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "zero_log_inputs": int(zero.sum()),
            "times_ms": [int(t.value // 1_000_000) for t in times],
            "times_utc": [t.isoformat() for t in times],
            "legacy_warnings": [str(w.message) for w in old_warnings],
            "post_fix_warnings": [], "fdi_exact": True,
            "cache_checks": checks, "baseline_check": baseline_check,
            "zero_input_details": [{"time_ms": int(t.value // 1_000_000),
                                    "path_length": float(path_length.loc[t]),
                                    "range_hl": float(range_hl.loc[t]),
                                    "fdi_after": float(result.loc[t, "fractal_dimension"])} for t in times]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-status", choices=["NOT_VERIFIED", "BLOCKED_APPROVAL"], default="NOT_VERIFIED")
    parser.add_argument("--ledger", action="store_true")
    args = parser.parse_args()
    verification = {}
    for name in ("fit-log-focused-pytest.log", "fit-log-full-pytest.log"):
        log = ROOT / "output" / name
        if log.exists():
            verification[f"output/{name}"] = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    rows = []
    legacy, baseline_sha = legacy_function()
    engine = make_engine(read_only=True)
    try:
        with sessionmaker(bind=engine)() as db:
            if db.execute(text("SHOW transaction_read_only")).scalar() != "on":
                raise RuntimeError("read-only transaction required")
            svc = MarketDataService(db)
            for path in sorted(CACHE.glob("*.npz")):
                if path.name == "csbench.npz":
                    continue
                rows.append(audit_symbol(svc, path, legacy))
                if len(rows) % 10 == 0:
                    print(f"checked {len(rows)} symbols", flush=True)
    finally:
        engine.dispose()
    affected = [r for r in rows if r["zero_log_inputs"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": "P18-003", "parent_experiment_id": "P18-002",
        "stage": "工程诊断，非 edge 判定", "diagnostic_status": "PASS",
        "verification": verification, "backend_reload_status": args.restart_status,
        "backend_reload_verified": False,
        "verdict": "inconclusive", "edge_verdict": "NOT_EVALUATED",
        "window": {"interval": "5m", "source": "per-symbol original cache first_ms/last_ms", "days": 90},
        "universe": {"symbols": [r["symbol"] for r in rows], "count": len(rows), "skipped": []},
        "costs": {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
                  "basis": "notional", "costs_applied": False},
        "ci": {"status": "NOT_APPLICABLE_ENGINEERING_DIAGNOSIS"},
        "random_baseline": {"status": "NOT_RUN_NO_EDGE_CLAIM"}, "trades": None,
        "source_sha256": hashlib.sha256((ROOT / "backend/app/catalog_indicators/trend_persistence_score.py").read_bytes()).hexdigest(),
        "baseline_function_sha256": baseline_sha,
        "baseline_method": "reconstructed original function; exact old FDI expression, other source unchanged",
        "scope_sha": "b323c0912533842e",
        "metrics": {"symbols_checked": len(rows), "affected_symbols": len(affected),
                    "zero_log_input_bars": sum(r["zero_log_inputs"] for r in rows),
                    "cache_field_comparisons": sum(r["cache_checks"] for r in rows),
                    "post_fix_warnings": 0, "cached_event_fields_unchanged": True},
        "root_cause": "np.where eagerly evaluated log(0) in unused FDI branch; positive range with zero close path",
        "fix": "Mask path and range before division/log; preserve original 1.5 fallback and valid-path formula",
        "affected": affected, "per_symbol": rows,
        "backtest": "NOT_RUN", "default_enabled_changed": False, "live_orders": False,
        "limitations": ["仅复核trend_persistence_score在原5m/80币窗口的告警与输出，不证明其余指标无数值问题。",
                        "缓存比较涵盖n/hit_bars/sum_r/sum_sq，未重算截面或分段统计；浮点缓存使用rtol1e-5/atol1e-6。",
                        "80币完整输出与重建旧函数逐值精确相等；原始数据库未保存不可变OHLCV快照。", 
                        "告警关闭不解除max-T、PIT分档、整体方向与独立OOS等审计阻塞。"],
    }
    lines = ["# P18 5m log 告警复核（工程诊断，非 edge 判定）", "",
             f"生成：{report['generated_at']}；工程诊断 PASS，策略收益判定 inconclusive（未评估）。", "",
             "## 窗口与币池", "", f"原5m缓存窗口逐币重放，90天口径，币数 {len(rows)}，skip 0。",
             "每币实际首尾时间、bar数及缓存哈希见同名JSON的per_symbol；严格只读数据库。", "",
             "## 根因与修复", "", "trend_persistence_score 的34根收盘路径长度为0、高低区间非零时，",
             "np.where提前计算了本应丢弃的log(0)。修复在log之前屏蔽不可用输入，保留FDI=1.5回退和原有效分支公式。", "",
             "| 币种 | UTC时间 | 路径长度 | 高低区间 | 修复后FDI |", "|---|---|---:|---:|---:|"]
    for r in affected:
        for item, utc in zip(r["zero_input_details"], r["times_utc"], strict=True):
            lines.append(f"| {r['symbol']} | {utc} | {item['path_length']} | {item['range_hl']} | {item['fdi_after']} |")
    metrics = report["metrics"]
    river = next(r for r in rows if r["symbol"] == "RIVERUSDT")["baseline_check"]
    lines += ["", "## 配对复核", "",
              f"- 完整重放 {len(rows)} 币，修复后 RuntimeWarning 0；与原缓存核对 {metrics['cache_field_comparisons']} 项计数/收益充分统计，全部一致。",
              f"- 80币均与重建旧函数的完整输出逐值精确一致；其中RIVERUSDT为{river['rows']}行 × {river['columns']}列，包括7个布尔列及全部数值/状态输出。", 
              "- 旧公式可重现同类pandas告警，屏蔽后的FDI在全部80币上与旧公式结果逐值精确相同。", "",
              "## 成本、CI与随机基线", "", "固定成本参照为名义本金往返0.12%（fee_rate=0.0004、slippage=0.0002，每边）。",
              "本轮没有交易回测，不产生净收益、置信区间CI或随机基线数字；这些项目为NOT_RUN/NOT_APPLICABLE。", "",
              "## 局限与启用决策", ""] + [f"- {s}" for s in report["limitations"]]
    lines += ["", "default_enabled与绑定均未修改，未开启paper/live，未下单。", "",
              "## 验证与后端加载状态", ""]
    lines += [f"- `{name}`：{value}" for name, value in verification.items()]
    lines += [f"- 后端重载状态：{args.restart_status}；未验证常驻后端加载新代码。"]
    if args.restart_status == "BLOCKED_APPROVAL":
        lines += ["- 自动审批拒绝停止/重启已有QuantDesk-GetMoney任务；须用户明确授权该重启，未改用其他方式终止进程。"]
    lines += ["", "复现：`.venv/Scripts/python.exe backend/scripts/fit_log_diagnosis.py`；依赖原聚合缓存和本地行情库，旧函数由脚本重建。", ""]
    PREFIX.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    PREFIX.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    if args.ledger:
        from fit_matrix_report import append_ledger, git_state
        record = {
            "timestamp": report["generated_at"], "experiment_id": "P18-003",
            "parent_experiment_id": "P18-002", "strategy_id": None,
            "code_commit": git_state(), "stage": report["stage"],
            "decision": "PASS_ENGINEERING_ONLY", "edge_verdict": "NOT_EVALUATED",
            "dataset_version": "original scope-A 5m cache boundaries, complete final candle; local DB read_only",
            "symbols": report["universe"]["symbols"], "interval": ["5m"],
            "parameters": {"scope_sha": report["scope_sha"], "parameter_trials": 0,
                           "formula_changed": False, "invalid_branch_evaluation_removed": True},
            "fee_model": "0.0004 per side reference only", "slippage_model": "0.0002 per side reference only",
            "funding_model": "NOT_MODELLED; no trades simulated",
            "metrics": report["metrics"], "verification": verification,
            "backend_reload_status": args.restart_status,
            "next_action": "resolve remaining P18-002 statistical/data gates; backend reload needs explicit authorization",
            "artifacts": ["docs/indicator-fit-log-diagnosis.md", "docs/indicator-fit-log-diagnosis.json"],
        }
        print(append_ledger(record, ROOT / "research/experiments.jsonl"), flush=True)
    print(json.dumps(report["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
