"""H-F Gate 1a 判据装配（渲染见 funding_venue_render.py）。

判据来自 `research/hypotheses/H-F-funding-multi-venue.md` §5，事前冻结：
D1 奖金占比 ≥60%、D2 不可对冲子集 rank IC ≥0.50 且正期占比 ≥90%、
D3 加回该子集的毛年化增量 ≥25pp。**两个 K 都必须通过**（保守 AND，防事后挑 K）。

本模块只产出毛超额，不产出任何净收益字段。
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.research.funding_venue_stats import (
    BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED, D1_MIN_SHARE, D2_MIN_IC,
    D2_MIN_POSITIVE_SHARE, D3_MIN_ANNUAL_PP, HOLD_DAYS, KS, LOOKBACK_DAYS,
    MIN_CROSS_SECTION, annualize_gross, block_bootstrap_ci, gate_checks,
    paired_increment_ci, pooled_share_ci, summarize_ic,
)
from app.research.lab_stats import percentile_rank

NAME = "funding-venue-gate1a"
CARD = "research/hypotheses/H-F-funding-multi-venue.md"
MIN_PERIODS = 20
SCOPE = "资金费率多所路径可行性（Gate 1a 毛超额），非 edge 判定；不含任何净收益。"
HISTORICAL_GROSS = {"as_of": "2026-10-27", "k": 10, "unfiltered_annual_pct": 55.6,
                    "hedgeable_annual_pct": 10.8,
                    "source": "docs/roadmap-p11-funding-harvest.md"}


def _pct(value):
    return None if value is None else round(value * 100, 4)


def _round4(value):
    return None if value is None else round(value, 4)


def _ci_pct(ci):
    return {"ci_low_pct": _pct(ci["ci_low"]), "ci_high_pct": _pct(ci["ci_high"]),
            "mean_pct": _pct(ci["mean"]), "n": ci["n"], "block": ci["block"],
            "n_boot": ci["n_boot"], "seed": ci["seed"], "method": ci["method"]}


def _arm_summary(returns, missing_forward):
    values = [r for r in returns if r is not None]
    return {
        "annual_gross_pct": _pct(annualize_gross(returns, hold_days=HOLD_DAYS)),
        "mean_period_pct": _pct(float(np.mean(values))) if values else None,
        "positive_period_share": (round(sum(1 for v in values if v > 0) / len(values), 4)
                                  if values else None),
        "n_periods": len(values),
        "n_skipped_periods": len(returns) - len(values),
        "missing_forward": int(missing_forward),
        "ci": _ci_pct(block_bootstrap_ci(returns)),
    }


def _bonus_summary(bonus):
    total = float(sum(b["total"] for b in bonus))
    part = float(sum(b["non_hedgeable"] for b in bonus))
    shares = [b["share"] for b in bonus if b["share"] is not None]
    counts = [b["n_non_hedgeable"] for b in bonus]
    ci = pooled_share_ci(bonus)
    return {
        "pooled_share": round(part / total, 4) if total > 0 else None,
        "pooled_total": round(total, 6),
        "pooled_non_hedgeable": round(part, 6),
        "period_median_share": round(float(np.median(shares)), 4) if shares else None,
        "periods_nonpositive_total": len(bonus) - len(shares),
        "mean_non_hedgeable_in_topk": round(float(np.mean(counts)), 2) if counts else None,
        "share_ci": {"ci_low": _round4(ci["ci_low"]), "ci_high": _round4(ci["ci_high"]),
                     "n": ci["n"], "block": ci["block"], "n_boot": ci["n_boot"],
                     "seed": ci["seed"]},
    }


def _ic_summary(ics, counts=None):
    out = summarize_ic(ics)
    for field in ("mean", "positive_share"):
        if out[field] is not None:
            out[field] = round(out[field], 4)
    valid = [c for c in (counts or []) if c]
    out["min_cross_section"] = min(valid) if valid else 0
    out["median_cross_section"] = int(np.median(valid)) if valid else 0
    ci = block_bootstrap_ci(ics)
    out["ci"] = {"ci_low": _round4(ci["ci_low"]), "ci_high": _round4(ci["ci_high"]),
                 "n": ci["n"], "block": ci["block"], "n_boot": ci["n_boot"],
                 "seed": ci["seed"], "method": ci["method"]}
    return out


def _baseline_summary(baseline, strategy_annual_pct):
    out = dict(baseline)
    samples = [v for v in baseline.get("samples_pct", []) if v is not None]
    out["percentile_of_strategy"] = (
        round(percentile_rank(strategy_annual_pct, samples), 4)
        if samples and strategy_annual_pct is not None else None)
    return out


def _limitations(payload):
    uni, data = payload["universe"], payload["data"]
    items = [
        "**全宇宙毛年化不是收益预期，是不可执行的反事实**：它把不可对冲币按「能在第二家所买到"
        "现货腿」计入，而那个所的费率、深度、滑点全未知。单所口径的同类反事实（净 +41.8%）"
        "已终判不可执行。D3 增量只说明奖金量级值得去问他所数据，不构成任何收益承诺。",
        "**只报毛超额，不报净收益**：他所现货费率、深度、滑点均未采集，任何净数字都是编造。"
        "H3 全局成本 fee_rate=0.0004 / slippage=0.0002 仅作参照记录，未参与计算。",
        "**幸存者偏差**：币池为采集日在架币种，已下架币费率史缺失，方向上高估收益。",
        f"**现货对可得性是今日快照**：{uni['hedgeable_count']}/{data['symbols']} 对照 "
        f"{uni['historical_reference']['as_of']} 的 {uni['historical_reference']['hedgeable']}"
        f"/{uni['historical_reference']['total']}，差 {uni['spot_drift_vs_2026_10_27']} 币"
        "来自现货上架漂移；历史各期实际可对冲性未落库，无法逐期还原。",
        "**1000X 前缀换算**：永续 `1000BONKUSDT` 按现货 `BONKUSDT` 记为可对冲；"
        "若现货同名不同币会误判，逐币清单已落 JSON 供人工抽查。",
        "**毛收益不含价格损益**：假设现货腿完全对冲 delta；未计保证金占用、双腿清算风险、"
        "出入金与对冲腿再平衡的运营摩擦。",
        "**结算周期逐币不同**：前瞻费率和按实际 `funding_time` 事件求和"
        "（4h 币 18 次/3 天、8h 币 9 次），不假设统一 8h。",
        "**Gate 1a 通过 ≠ H-F1 成立**：只说明值得去问他所数据；Gate 1b（E1–E6）仍 "
        "`BLOCKED`，须用户单独授权采集他所历史 funding 与现货深度。",
        f"**CI 口径**：移动块 bootstrap（block={BOOT_BLOCK} 期≈{BOOT_BLOCK * HOLD_DAYS} 天、"
        f"n_boot={BOOT_DRAWS}、seed={BOOT_SEED}），因为费率强持续、相邻期高度自相关，"
        "iid bootstrap 会低估方差。D3 的 CI 对**配对期收益均值**年化后作差，"
        "点估计用逐期复利，两者**口径略有差异**；CI 只用于判断增量是否稳健为正。",
        "**随机基线是随机选币、不是随机方向**：本策略不预测价格方向，H6 的方向翻转基线"
        "在此无意义。基线同期同 K 从 trailing>0 的合格币里随机挑，用于分离"
        "「排序有用」与「随便拿正费率币就有钱」。",
    ]
    if payload["periods"]["count"] < MIN_PERIODS:
        items.append(f"**期数不足**：{payload['periods']['count']} 期 < {MIN_PERIODS} 期下限，"
                     "判据读数不足以支撑可行性结论。")
    return items


def assemble(raw: dict) -> dict:
    arms, gates = {}, {}
    for k in KS:
        part = raw["per_k"][str(k)]
        full = _arm_summary(part["full_returns"], part["missing_forward"])
        hedge = _arm_summary(part["hedgeable_returns"], part["missing_forward"])
        bonus = _bonus_summary(part["bonus"])
        ic_nh = _ic_summary(part["ic_non_hedgeable"], part["cross_counts_non_hedgeable"])
        d3 = (round(full["annual_gross_pct"] - hedge["annual_gross_pct"], 4)
              if full["annual_gross_pct"] is not None
              and hedge["annual_gross_pct"] is not None else None)
        gate = gate_checks(bonus["pooled_share"], ic_nh, d3)
        d3_ci = paired_increment_ci(part["full_returns"], part["hedgeable_returns"])
        for field in ("ci_low_pp", "ci_high_pp", "mean_pp"):
            d3_ci[field] = _round4(d3_ci[field])
        arms[str(k)] = {
            "k": k, "full": full, "hedgeable_only": hedge, "d3_increment_pp": d3,
            "d3_ci": d3_ci,
            "bonus": bonus, "ic_non_hedgeable": ic_nh,
            "ic_full": _ic_summary(part["ic_full"], part["cross_counts_full"]),
            "ic_hedgeable": _ic_summary(part["ic_hedgeable"],
                                        part["cross_counts_hedgeable"]),
            "random_baseline": _baseline_summary(part["random_baseline"],
                                                 full["annual_gross_pct"]),
            "gate": gate,
        }
        gates[str(k)] = gate
    count = raw["periods"]["count"]
    verdict = ("insufficient" if count < MIN_PERIODS
               else "pass" if all(g["pass"] for g in gates.values()) else "negative")
    uni = raw["universe"]
    universe = {
        "hedgeable_count": len(uni["hedgeable"]),
        "non_hedgeable_count": len(uni["non_hedgeable"]),
        "hedgeable_symbols": uni["hedgeable"],
        "non_hedgeable_symbols": uni["non_hedgeable"],
        "historical_reference": uni["historical_reference"],
        "spot_drift_vs_2026_10_27": (len(uni["hedgeable"])
                                     - uni["historical_reference"]["hedgeable"]),
    }
    payload = {
        "name": NAME, "hypothesis_card": CARD, "verdict": verdict,
        "verdict_scope": SCOPE, "default_enabled": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": raw.get("code_commit", "unknown"),
        "protocol": {
            "lookback_days": LOOKBACK_DAYS, "hold_days": HOLD_DAYS, "ks": list(KS),
            "selection": "过去 7 天均费率 > 0 降序取 TopK（只用 trailing，因果）",
            "forward": "[T, T+3d) 内实际结算事件的 funding_rate 之和",
            "weighting": "等权，收益口径为名义本金上的毛费率",
            "min_cross_section": MIN_CROSS_SECTION, "min_periods": MIN_PERIODS,
            "costs_applied": False, "fee_rate_reference": 0.0004,
            "slippage_reference": 0.0002,
            "gate_rule": "两个 K 都必须通过（保守 AND，防止事后挑 K）",
        },
        "thresholds": {"D1_min_share": D1_MIN_SHARE, "D2_min_ic": D2_MIN_IC,
                       "D2_min_positive_share": D2_MIN_POSITIVE_SHARE,
                       "D3_min_annual_pp": D3_MIN_ANNUAL_PP},
        "data": raw["data"], "spot_snapshot": raw["spot_snapshot"],
        "universe": universe, "periods": raw["periods"],
        "historical_single_venue": HISTORICAL_GROSS,
        "window": {"start_ms": raw["data"]["start_ms"],
                   "end_ms": raw["data"]["end_ms"], "days": raw["data"]["days"],
                   "periods": count, "rebalance_days": HOLD_DAYS,
                   "lookback_days": LOOKBACK_DAYS},
        "costs": {"applied": False, "fee_rate": 0.0004, "slippage": 0.0002,
                  "round_trip_pct": 0.12,
                  "reason": "他所现货费率/深度/滑点未知，Gate 1a 只报毛超额；"
                            "此处仅记录 H3 全局参照常量，未参与任何计算。"},
        "trades": {"simulated_fills": False, "unit": "币·期",
                   "n_periods": count,
                   "n_symbol_periods": raw["per_k"][str(KS[0])]["n_symbol_periods"],
                   "note": "本探针不模拟成交，无笔级样本；统计单位是重平衡期与选中币·期。"},
        "arms": arms, "gates": gates,
    }
    payload["limitations"] = _limitations(payload)
    return payload
