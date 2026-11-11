#!/usr/bin/env python3
"""P12 只读探针：OI 拥挤度的横截面去 beta rank IC（Gate 1）。

P9 的 oi_value_usd 逐币时序 IC=-0.0746/84% 同号，是横截面 alpha 还是市场共同
beta？每个 1h 时点按因子给币排序，与「未来 H 根收益减去该时点全市场中位收益」
的排序算 Spearman IC；demean 后剩下的才是选币能力。事前注册主判据仅一个：
oi_value_z7d × H=24 × 去 beta 标签。门槛（H1）：|IC 均值|>0.02、block bootstrap
95% CI 不跨 0、同号期 >=60%、逐币同号币 >=60%、置换经验 p<0.05。其余因子/H
为探索性读数，不参与 gate。只读 DB；IC 是毛价格关联，未扣 0.12% 双边成本，
不是净收益 edge 判定，过 gate 仅授权后续受控 A/B。不改 spec / 绑定 / 参数。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from probe_p12_common import (  # noqa: E402
    FACTOR_COLS, HORIZONS, HOUR_MS, MIN_CROSS_SECTION, PRIMARY_FACTOR,
    PRIMARY_HORIZON, bh_fdr, block_bootstrap_ci, build_factors, build_labels,
    common_factor_diagnostic, fetch_frames, ic_series, permutation_null,
    permutation_p, per_symbol_ic, summarize,
)

COSTS = {"fee_rate": 0.0004, "slippage": 0.0002, "round_trip": 0.0012,
         "basis": "notional", "applied_to_ic": False}


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def run(args) -> dict:
    sent, candles, bounds = fetch_frames()
    grid = np.arange(bounds["sentiment_start_ms"],
                     bounds["sentiment_end_ms"] + HOUR_MS, HOUR_MS)
    factors = build_factors(sent, grid)
    cols = factors[PRIMARY_FACTOR].columns
    labels = {h: build_labels(candles, grid, h, cols) for h in HORIZONS}
    fwd, excess, valid, px = labels[PRIMARY_HORIZON]

    ics, counts = ic_series(factors[PRIMARY_FACTOR], excess, valid)
    primary = summarize(ics, "future_24h_return_minus_cross_sectional_median")
    primary["ci"] = block_bootstrap_ci(ics, seed=args.seed)
    primary["permutation_null"] = permutation_null(
        factors[PRIMARY_FACTOR], excess, valid, args.permutations, args.seed)
    primary["per_symbol_ic"] = per_symbol_ic(factors[PRIMARY_FACTOR], excess)
    raw_ics, _ = ic_series(factors[PRIMARY_FACTOR], fwd, valid)
    primary["raw_label_comparison"] = summarize(raw_ics, "future_24h_return_raw")
    look, _ = ic_series(excess, excess, valid)
    primary["positive_control"] = summarize(look, "lookahead_factor_equals_label")

    ci, pm = primary["ci"], primary["per_symbol_ic"]
    gate = {
        "abs_mean_ic_gt_0.02": abs(primary["mean_ic"] or 0) > 0.02,
        "ci_excludes_zero": bool(ci["ci_low"] is not None
                                 and (ci["ci_low"] > 0 or ci["ci_high"] < 0)),
        "same_sign_period_share_ge_0.60": bool(primary["same_sign_share"] is not None
                                                and primary["same_sign_share"] >= 0.60),
        "per_symbol_same_sign_ge_0.60": bool(pm["same_sign_share"] is not None
                                              and pm["same_sign_share"] >= 0.60),
        "permutation_p_lt_0.05": primary["permutation_null"]["empirical_p_two_sided"] < 0.05,
    }
    gate["passed"] = all(v for k, v in gate.items() if k != "passed")

    exploratory, pvals = [], []
    for name in FACTOR_COLS:
        for horizon in HORIZONS:
            if name == PRIMARY_FACTOR and horizon == PRIMARY_HORIZON:
                continue
            _, ex, ok, _ = labels[horizon]
            series, _ = ic_series(factors[name], ex, ok)
            row = summarize(series, f"H{horizon}_excess")
            row.update({"factor": name, "horizon": horizon,
                        "ci": block_bootstrap_ci(series, seed=args.seed),
                        "empirical_p": permutation_p(
                            factors[name], ex, ok, 100, args.seed + len(pvals) * 7919)})
            pvals.append(row["empirical_p"])
            exploratory.append(row)
    for row, q in zip(exploratory, bh_fdr(pvals)):
        row["fdr_q"] = q

    market = {"median_symbols_per_period": int(np.median(counts[counts >= MIN_CROSS_SECTION]))
              if np.any(counts >= MIN_CROSS_SECTION) else 0,
              "periods_below_min_cross_section": int((counts < MIN_CROSS_SECTION).sum())}
    btc = px.get("BTCUSDT")
    if btc is not None and len(btc.dropna()) > PRIMARY_HORIZON:
        b = btc.dropna()
        market["btc_return_pct_window"] = round(float(b.iloc[-1] / b.iloc[0] - 1) * 100, 3)

    return {
        "topic": "P12 OI crowding cross-sectional IC probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "backend/scripts/probe_p12_oi_cross_section.py",
        "stage": "Gate 1 IC probe", "strategy": None, "interval": "1h",
        "primary": {"factor": PRIMARY_FACTOR, "horizon": PRIMARY_HORIZON,
                    "label": "cross-sectionally demeaned forward return", **primary},
        "gate": gate, "exploratory": exploratory,
        "exploratory_multiple_testing": {
            "method": "Benjamini-Hochberg", "n_tests": len(exploratory),
            "q_threshold": 0.10,
            "n_q_below_threshold": sum(1 for r in exploratory if r["fdr_q"] < 0.10),
        },
        "common_factor_diagnostic": common_factor_diagnostic(
            factors[PRIMARY_FACTOR], fwd),
        "rank_ic_invariant_note": (
            "Spearman rank IC 对逐期标签减常数不变，故 raw 与 demean 标签 IC 完全相同；"
            "这不是缺陷，而是说明横截面 IC 本身已剔除市场共同平移，"
            "逐币时序 IC 与横截面 IC 的差异来自聚合维度（时间 vs 券）而非标签 demean"),
        "costs": COSTS,
        "trades": 0,
        "trades_note": "探针阶段无撮合、无成交；IC 不是净收益判定",
        "window": {"start": iso(bounds["sentiment_start_ms"]),
                   "end": iso(bounds["sentiment_end_ms"]),
                   "days": round((bounds["sentiment_end_ms"]
                                  - bounds["sentiment_start_ms"]) / 86400000, 2)},
        "universe": {"pool": "sentiment ∩ native 1h",
                     "symbols_in_sentiment": int(sent["symbol"].nunique()),
                     "symbols_in_native_1h": int(candles["symbol"].nunique()),
                     "skipped": [],
                     "coverage_note": "逐时点有效币数见 market；不足 30 币的时点跳过"},
        "data_freshness": {"sentiment_end": iso(bounds["sentiment_end_ms"]),
                           "native_1h_end": iso(bounds["native_1h_end_ms"]),
                           "sentiment_lag_days_vs_1h": round(
                               (bounds["native_1h_end_ms"]
                                - bounds["sentiment_end_ms"]) / 86400000, 2)},
        "data_pipeline_status": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "kline_last_bar_utc": {k: iso(v) for k, v in
                                   sorted(bounds["kline_ends_ms"].items())},
            "funding_last_bar_utc": iso(bounds["funding_end_ms"]),
            "sentiment_last_bar_utc": iso(bounds["sentiment_end_ms"]),
            "p9_ab_requirement_days": 60,
            "p9_available_days": round((bounds["sentiment_end_ms"]
                                        - bounds["sentiment_start_ms"]) / 86400000, 2),
            "note": "各周期尾部以本字段实时核表为准；调度是否覆盖需另查自动化／"
                    "APScheduler，不在探针内推断",
        },
        "next_hypotheses": [
            {"id": "H-A", "name": "市场级杠杆择时",
             "falsifiable": "若全市场平均 OI z 对全市场未来收益有择时价值，则非重叠样本"
                            "的样本外净收益 CI 下界 > 0 且跑赢多种子随机基线；否则终止",
             "data_prerequisite": "恢复 sentiment 采集并补回 10-28 之后缺口，累积 >=60 天",
             "status": "BLOCKED_BY_DATA", "h2_compliant": True},
            {"id": "H-B", "name": "逐笔成交顺序（非 K 线汇总）",
             "falsifiable": "若逐笔方向游程 / 大单簇携带信息，则 IC 探针 |IC|>0.02 "
                            "且同号币 >=60%；否则终止",
             "data_prerequisite": "新建 aggTrades 采集（data.binance.vision 日归档可回填）",
             "status": "BLOCKED_BY_NEW_SOURCE", "h2_compliant": True},
            {"id": "H-C", "name": "盘口挂撤单 / 多所现货腿",
             "falsifiable": "若挂撤单失衡或跨所价差可预测，则先过 IC 探针；否则终止",
             "data_prerequisite": "盘口需从现在开始录快照；多所需接 Gate/MEXC 数据",
             "status": "BLOCKED_BY_NEW_SOURCE", "h2_compliant": True},
        ],
        "market": market, "ci": primary["ci"],
        "verdict": "inconclusive", "default_enabled": False,
        "verdict_scope": "探针级：IC 的 95% CI 跨 0；本轮无撮合、无净收益 CI、"
                         "无多种子随机基线，不构成 H4 的 edge 判定",
        "limitations": [
            "sentiment 仅 21.83 天且停在 2026-10-28，native 1h 停在 2026-11-04；"
            "单一样本期、BTC -5.691% 的下跌市，跨市场状态稳健性无法验证",
            "oi_value_z7d 以 168 根 rolling、min_periods=72 近似 7 天 z-score，"
            "与 P9 的完整 7 天口径存在预热期偏离",
            "标签为 close[t+H]/close[t]-1 的固定持有收益，不是 P9 的 triple-barrier "
            "止盈标签，逐币 IC 数值不可与 P9 直接相减",
            "IC 是毛价格关联，未扣 0.12% 双边成本、资金费率、盘口深度、滑点与强平",
            "置换零分布按各期独立打乱，未保留跨期自相关；block bootstrap 保留了自相关，"
            "两者结论不一致时以更保守的 CI 为准",
            "横截面 IC 只测券维度排序，不能排除 OI 在时间维度上存在择时价值",
            "探索性组合未用于 gate，已做 BH-FDR，仍不得事后挑选取参",
        ],
        "seed": args.seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P12 横截面 OI 拥挤度 IC 探针（只读）")
    ap.add_argument("--out", default="docs/probe-p12-oi-cross-section.json")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"输出已存在，拒绝覆盖：{out}（如需重跑加 --force）", file=sys.stderr)
        return 2
    result = run(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    p = result["primary"]
    print(json.dumps({"out": str(out), "window_days": result["window"]["days"],
                      "periods": p["periods"], "mean_ic": p["mean_ic"],
                      "ci": p["ci"], "same_sign_share": p["same_sign_share"],
                      "per_symbol": p["per_symbol_ic"],
                      "permutation_null": p["permutation_null"],
                      "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
