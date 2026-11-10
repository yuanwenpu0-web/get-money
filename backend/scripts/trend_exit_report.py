"""趋势反转入场 × 卖出信号：Gate 1 探针报告 + Gate 4/5 主报告生成。

只读 output/trend_exit_study.json，产出两对 docs/*.{md,json}。判定词严格取
strategy-backtest-playbook 第 3 节词表；不做参数选择、不写生产代码。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from trend_exit_common import PRICE_SIGNALS, PURGE_MS  # noqa: E402
from trend_exit_markdown import main_md, probe_md  # noqa: E402
from trend_exit_stats import (  # noqa: E402
    MIN_POOLED_TRADES, gate1_probe, paired_stats,
    percentile_rank, segment_means,
)


def _p_normal(v: np.ndarray) -> float:
    if len(v) < 2:
        return 1.0
    se = float(v.std(ddof=1)) / math.sqrt(len(v))
    z = abs(float(v.mean()) / se) if se > 0 else 0.0
    return float(math.erfc(z / math.sqrt(2)))


def _split(records):
    ts = np.array([r["time_ms"] for r in records], dtype=float)
    lo, hi = float(ts.min()), float(ts.max())
    b1, b2 = lo + 0.60 * (hi - lo), lo + 0.75 * (hi - lo)
    clean = [(r, t) for r, t in zip(records, ts)
             if abs(t - b1) > PURGE_MS and abs(t - b2) > PURGE_MS]
    parts = {"train": [r for r, t in clean if t < b1],
             "valid": [r for r, t in clean if b1 <= t < b2],
             "test": [r for r, t in clean if t >= b2]}
    return parts, {"train_end_ms": int(b1), "valid_end_ms": int(b2),
                   "purge_ms": PURGE_MS, "purged_records": len(records) - len(clean),
                   "sizes": {k: len(v) for k, v in parts.items()}}


def _diffs(records, name):
    return np.array([r["rule_net"][name] - r["baseline_net"] for r in records], dtype=float)


def _signal_table(d, parts, rng):
    signals = d["signals"]
    rows, pvals = [], []
    for i, name in enumerate(signals):
        diffs_all = _diffs(d["records"], name)
        row = {"signal": name, "class": "price" if name in PRICE_SIGNALS else "nonprice",
               "hit_rate": round(float(np.mean([r["trigger_j"][name] is not None
                                               for r in d["records"]])), 4),
               "all": paired_stats(diffs_all),
               "segments": segment_means(diffs_all, [r["time_ms"] for r in d["records"]]),
               "random_early_mean_pct": round(float(np.mean(d["random_early"][name])), 6),
               "random_early_p95_pct": round(float(np.percentile(d["random_early"][name], 95)), 6),
               "random_percentile": round(percentile_rank(float(diffs_all.mean()),
                                                          d["random_early"][name]), 4),
               "splits": {k: paired_stats(_diffs(v, name), n_boot=600)
                          for k, v in parts.items()}}
        pvals.append(_p_normal(diffs_all))
        row["p_normal"] = round(float(pvals[i]), 8)
        rows.append(row)
    from order_exit_common import bh_fdr  # noqa: PLC0415  局部导入避免循环
    for row, q in zip(rows, bh_fdr(pvals)):
        row["fdr_q"] = round(float(q), 6)
        seg = row["segments"]
        # 分段同号必须与全样本配对差同号，不能只数绝对值多数
        same_seg = (seg["n_positive"] if row["all"]["mean_pct"] > 0 else seg["n_negative"])
        row["three_gates"] = {
            "ci_low_gt_0": row["all"]["ci_low_pct"] is not None and row["all"]["ci_low_pct"] > 0,
            "fdr_q_below_0.10": q < 0.10,
            "segments_same_sign_ge_3of4": same_seg >= 3,
        }
        row["three_gates_pass"] = all(row["three_gates"].values())
        row["note"] = ("likely_noise" if row["all"]["ci_low_pct"] is not None
                       and row["all"]["ci_low_pct"] > 0 and q >= 0.10 else "")
    return rows


def _verdict(champ, rows, parts):
    by = {r["signal"]: r for r in rows}
    row = by[champ]
    test = row["splits"]["test"]
    train = row["splits"]["train"]
    if test["n"] < MIN_POOLED_TRADES:
        return "insufficient", "测试段成交笔数低于池化门槛"
    if (test["verdict"] == "pass" and row["fdr_q"] < 0.10
            and row["three_gates"]["segments_same_sign_ge_3of4"]
            and row["random_percentile"] > 0.95 and train["mean_pct"] > 0):
        return "pass", "训练/测试一致、三关与随机基线全过"
    if test["verdict"] == "negative":
        return "negative", "测试段配对差 CI 上界为负"
    return "inconclusive", "测试段配对差 CI 跨 0，或 FDR / 分段 / 随机基线未过关"



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="output/trend_exit_study.json")
    ap.add_argument("--prefix", default="docs/trend-reversal-exit-signals")
    args = ap.parse_args()
    d = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = sorted(d["records"], key=lambda r: r["time_ms"])
    parts, split_meta = _split(records)
    rng = np.random.default_rng(20261110)

    probe_recs = [{"features": r["decision"]["features"], "fwd_pct": r["decision"]["fwd_pct"],
                   "symbol": r["symbol"], "time_ms": r["time_ms"]}
                  for r in parts["train"] if r.get("decision")]
    probe = gate1_probe(probe_recs, d["signals"], rng)
    rows = _signal_table(d, parts, rng)
    champ = max(rows, key=lambda r: r["splits"]["train"]["mean_pct"])["signal"]
    verdict, reason = _verdict(champ, rows, parts)
    baseline = paired_stats([r["baseline_net"] for r in records])
    rand_dir = d["random_dir"]

    stamp = datetime.now(timezone.utc).isoformat()
    main_json = {
        "strategy": d["strategy"], "generated_at": stamp, "interval": d["interval"],
        "days": d["days"], "window": d["window"], "universe": d["universe"],
        "parameters": d["parameters"], "costs": d["costs"], "split": split_meta,
        "min_pooled_trades": MIN_POOLED_TRADES,
        "random_baseline_seeds": d["config"]["seeds"],
        "config": d["config"], "data": d["data"], "signals": d["signals"],
        "trades": d["n_records"],
        "baseline": {"mean_return_pct": baseline["mean_pct"],
                     "ci_low_pct": baseline["ci_low_pct"],
                     "ci_high_pct": baseline["ci_high_pct"],
                     "verdict": baseline["verdict"], "n": baseline["n"],
                     "random_direction_distribution": rand_dir,
                     "random_direction_percentile": round(
                         percentile_rank(baseline["mean_pct"], rand_dir), 4)},
        "ci": {"baseline": [baseline["ci_low_pct"], baseline["ci_high_pct"]],
               "champion_test": [rows[[r["signal"] for r in rows].index(champ)]
                                 ["splits"]["test"]["ci_low_pct"],
                                 rows[[r["signal"] for r in rows].index(champ)]
                                 ["splits"]["test"]["ci_high_pct"]]},
        "groups": rows, "champion": champ, "gate1": probe,
        "gate1_sell_direction_pass_count": probe["sell_direction_pass_count"],
        "verdict": verdict, "verdict_reason": reason, "default_enabled": False,
        "baseline_verdict": baseline["verdict"],
        "baseline_reason_counts": dict(Counter(r["baseline_reason"] for r in records)),
        "skip_reason_counts": dict(Counter(s["reason"] for s in d["universe"]["skipped"])),
        "limitations": [
            "未模拟资金费率、盘口深度、强平、分批建仓",
            "固定阈值与固定决策点，未做网格寻优；价格类信号命中率偏高",
            "sentiment 仅 23 天、funding 末端止于 2026-10-27，oi_crowd 判 insufficient",
            "半年窗口依赖 native 1h，15m 深度不足",
            "名义本金口径，不得与 margin 口径混比",
        ],
    }
    probe_json = {
        "strategy": d["strategy"], "generated_at": stamp, "interval": d["interval"],
        "days": d["days"], "window": d["window"], "universe": d["universe"],
        "costs": d["costs"], "split": split_meta, "trades": len(probe_recs),
        "min_pooled_trades": MIN_POOLED_TRADES, "probe": probe,
        "verdict": probe["verdict"], "default_enabled": False,
        "ci": {"decision_ic": [r["ic"] for r in probe["rows"] if r["ic"] is not None]},
        "limitations": ["只用训练段；决策点固定第 8 根", "sentiment 覆盖率低",
                        "名义本金口径，不得与 margin 口径混比"],
    }
    prefix = Path(args.prefix)
    probe_prefix = prefix.parent / "probe-trend-exit-signals"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    text_main = main_md(d, probe, parts, split_meta, rows, champ, verdict, reason,
                        baseline, rand_dir)
    text_probe = probe_md(d, probe, parts, split_meta, probe_prefix)
    Path(str(prefix) + ".md").write_text(text_main, encoding="utf-8")
    Path(str(prefix) + ".json").write_text(
        json.dumps(main_json, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    Path(str(probe_prefix) + ".md").write_text(text_probe, encoding="utf-8")
    Path(str(probe_prefix) + ".json").write_text(
        json.dumps(probe_json, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[report] verdict={verdict} champion={champ} "
          f"gate1_pass={probe['probe_pass_count']}/{len(probe['rows'])} "
          f"three_gates_pass={sum(1 for r in rows if r['three_gates_pass'])}/{len(rows)}")
    print(f"[report] → {prefix}.{{md,json}} + {probe_prefix}.{{md,json}}")


if __name__ == "__main__":
    main()
