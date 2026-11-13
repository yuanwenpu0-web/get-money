"""P17 / H-E 选币实验的判定量具 —— 纯函数，不碰 DB。

判据在 `research/hypotheses/H-E-supertrend-pit-pool.md` §4 **事前预登记**：
C1 CI 下界>0 ∧ C2 笔数≥30 ∧ C3 正 fold≥4/5 ∧ C4 随机分位>0.95 ∧ C5 BH-FDR q<0.10。
五关是「与」关系，任一不过只能写 inconclusive/negative/insufficient，
不得写「微正／有苗头」。本模块把这条纪律固化成代码，避免事后放宽门槛。

`all` 与 `posthoc_*` 是**诊断臂**：前者是不选币的基线，后者是按当期已实现收益
选币的作弊对照，用来量化事后选币偏差。它们不参与 gate，也不参与 FDR 家族。
"""
from __future__ import annotations

import math

SIZES = (10, 20, 30)
PIT_KINDS = ("liq", "vol")
MIN_TRADES = 30
FOLD_SIGN_SHARE = 0.8      # 5 folds → 至少 4 期为正
RANDOM_PERCENTILE = 0.95
FDR_Q = 0.10


def arm_names(sizes: tuple = SIZES) -> list[str]:
    names = ["all"]
    names += [f"pit_{kind}_{n}" for kind in PIT_KINDS for n in sizes]
    names += [f"posthoc_{n}" for n in sizes]
    return names


def jaccard(left, right) -> float:
    """相邻两期成员的 Jaccard 重合度：1.0 = 完全没换血，0.0 = 整批换掉。

    两期都为空时约定 1.0（没有成员就没有换血），调用方须同时报出期数，
    否则「稳定性满分」会掩盖「根本凑不出池」这个更严重的问题。
    """
    a, b = set(left), set(right)
    union = a | b
    return round(len(a & b) / len(union), 4) if union else 1.0


def bh_fdr(pvals: dict[str, float]) -> dict[str, float]:
    """Benjamini–Hochberg step-up，返回每个假设的 q 值。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    total = len(items)
    out: dict[str, float] = {}
    running = 1.0
    for index in range(total - 1, -1, -1):
        name, p = items[index]
        running = min(running, float(p) * total / (index + 1))
        out[name] = round(min(1.0, running), 6)
    return out


def _checks(summary: dict, q: float | None, n_folds: int, min_trades: int) -> dict:
    ci = summary.get("pooled") or {}
    folds = summary.get("fold_means") or []
    pct = summary.get("percentile_rank")
    need = math.ceil(FOLD_SIGN_SHARE * n_folds) if n_folds else 1
    return {
        "C1_ci_low_positive": bool(ci.get("ci_low_pct", 0.0) > 0),
        "C2_min_trades": bool(ci.get("trades", 0) >= min_trades),
        "C3_fold_signs": bool(summary.get("positive_folds", 0) >= need),
        "C4_beats_random": bool(pct is not None and pct > RANDOM_PERCENTILE),
        "C5_fdr": bool(q is not None and q < FDR_Q),
    }


def _verdict(ci: dict, checks: dict, min_trades: int) -> str:
    if not ci.get("trades"):
        return "no_trades"
    if ci["trades"] < min_trades:
        return "insufficient"
    if all(checks.values()):
        return "pass"
    if ci.get("ci_high_pct", 0.0) < 0:
        return "negative"
    return "inconclusive"


def judge(summary: dict, sizes: tuple = SIZES, min_trades: int = MIN_TRADES) -> dict:
    """对 10 个臂逐一判定，并给出课题级 overall。"""
    names = [name for name in arm_names(sizes) if name in summary]
    pit = [name for name in names if name.startswith("pit_")]
    n_folds = max((len(summary[name].get("fold_means") or []) for name in names), default=5)
    # bootstrap 单边 p 值：重采样均值 ≤ 0 的比例
    pvals = {name: 1.0 - float((summary[name].get("pooled") or {}).get("positive_share", 0.0))
             for name in pit}
    qvals = bh_fdr(pvals)

    arms: dict[str, dict] = {}
    for name in names:
        diagnostic = not name.startswith("pit_")
        q = None if diagnostic else qvals.get(name)
        checks = _checks(summary[name], q, n_folds, min_trades)
        ci = summary[name].get("pooled") or {}
        arms[name] = {
            "verdict": _verdict(ci, checks, min_trades),
            "checks": checks,
            "q": q,
            "p": pvals.get(name),
            "diagnostic_only": diagnostic,
        }

    verdicts = [arms[name]["verdict"] for name in pit]
    if "pass" in verdicts:
        overall = "pass"
    elif verdicts and all(v == "no_trades" for v in verdicts):
        overall = "no_trades"
    elif verdicts and all(v in ("insufficient", "no_trades") for v in verdicts):
        overall = "insufficient"
    elif verdicts and all(v == "negative" for v in verdicts):
        overall = "negative"
    else:
        overall = "inconclusive"
    return {"overall": overall, "n_folds": n_folds, "fdr_family": pit, "arms": arms}