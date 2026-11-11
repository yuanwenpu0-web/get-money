#!/usr/bin/env python3
"""P14 稳健性证伪诊断：区分「可交易动态信号」与「静态风格暴露／标签重叠假象」。

Gate 1 的 H1 判据形式上通过后必须做这一组检验，否则单一段普跌行情里的
稳定横截面 IC 很容易被误读成 alpha。全部离线计算，不下载、不连 DB。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import probe_p12_common as p12  # noqa: E402

BLOCK = 24


def _nanmean_axis(x: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(x).sum(axis=axis)
        return np.divide(np.nansum(x, axis=axis), np.where(cnt > 0, cnt, np.nan),
                         out=np.full(np.nansum(x, axis=axis).shape, np.nan),
                         where=cnt > 0)


def two_way_decomp(frame) -> dict:
    """x_ij ≈ grand + 时间效应 a_i + 币固定效应 b_j + 残差（ALS, 容忍缺失）。"""
    x = frame.to_numpy(float)
    mask = np.isfinite(x)
    g = float(np.nanmean(x)) if mask.any() else 0.0
    a = _nanmean_axis(np.where(mask, x - g, np.nan), 1)
    a = np.nan_to_num(a, nan=0.0)
    b = np.zeros(x.shape[1])
    for _ in range(30):
        b = np.nan_to_num(_nanmean_axis(
            np.where(mask, x - g - a[:, None], np.nan), 0), nan=0.0)
        a = np.nan_to_num(_nanmean_axis(
            np.where(mask, x - g - b[None, :], np.nan), 1), nan=0.0)
    resid = x - g - a[:, None] - b[None, :]
    total = float(np.nanvar(x))
    vt, vs, vr = float(np.var(a)), float(np.var(b)), float(np.nanvar(resid))
    return {
        "time_effect_variance": round(vt, 6),
        "symbol_fixed_effect_variance": round(vs, 6),
        "residual_variance": round(vr, 6),
        "time_effect_share": round(vt / total, 4) if total > 0 else None,
        "symbol_fixed_effect_share": round(vs / total, 4) if total > 0 else None,
        "residual_share": round(vr / total, 4) if total > 0 else None,
        "interpretation": "symbol_fixed_effect_share 高 => 因子几乎是每币恒定属性，"
                          "横截面 IC 可能只是静态风格在单一段行情里的表现",
    }


def partial_xs_ic(factor, label, valid, control) -> dict:
    """每期对 rank(因子)/rank(标签) 先剔除 rank(log 成交笔数) 的线性成分再相关。"""
    f, l, c, v = (x.to_numpy(float) for x in (factor, label, control, valid))
    min_cs = p12.MIN_CROSS_SECTION
    vals, degenerate = [], 0
    for i in range(len(f)):
        m = np.isfinite(f[i]) & np.isfinite(l[i]) & np.isfinite(c[i]) & v[i].astype(bool)
        if m.sum() < min_cs:
            continue
        rf, rl, rc = rankdata(f[i][m]), rankdata(l[i][m]), rankdata(c[i][m])
        rf, rl, rc = rf - rf.mean(), rl - rl.mean(), rc - rc.mean()
        den = float(rc @ rc)
        if den <= 0:
            degenerate += 1
            continue
        rf -= (rf @ rc / den) * rc
        rl -= (rl @ rc / den) * rc
        d = float(np.sqrt((rf @ rf) * (rl @ rl)))
        if d > 0 and np.isfinite(d):
            vals.append(float(rf @ rl / d))
        else:
            degenerate += 1  # 因子与控制量完全共线，残差退化
    if not vals:
        return {"periods": 0, "mean_partial_ic": None,
                "degenerate_periods": degenerate}
    return {"degenerate_periods": degenerate, "periods": len(vals),
            "mean_partial_ic": round(float(np.mean(vals)), 5),
            "control": "rank(log n_trades) per period",
            "ci": p12.block_bootstrap_ci(np.asarray(vals), block=BLOCK)}


def nonoverlapping_blocks(ics: np.ndarray) -> dict:
    """每 24 期只取一个时点，把 H=24 重叠标签降到近似独立。"""
    idx = np.flatnonzero(np.isfinite(ics))
    picked = ics[idx[::BLOCK]]
    if not len(picked):
        return {"n_blocks": 0}
    return {"n_blocks": int(len(picked)),
            "mean_ic": round(float(picked.mean()), 5),
            "median_ic": round(float(np.median(picked)), 5),
            "negative_share": round(float((picked < 0).mean()), 4),
            "values": [round(float(v), 4) for v in picked],
            "note": "块数远低于 20，点估计不稳定；用于看符号是否由个别时段驱动"}


def regime_split(ics, fwd) -> dict:
    """按非重叠日的全市场未来 24h 收益中位数分「跌市/涨市」两段看 IC 符号。"""
    market = fwd.median(axis=1).to_numpy(float)
    idx = np.flatnonzero(np.isfinite(ics) & np.isfinite(market))[::BLOCK]
    down, up = ics[idx[market[idx] <= 0]], ics[idx[market[idx] > 0]]
    return {"down_days": int(len(down)), "up_days": int(len(up)),
            "down_mean_ic": round(float(down.mean()), 5) if len(down) else None,
            "up_mean_ic": round(float(up.mean()), 5) if len(up) else None,
            "note": "普跌样本里涨日极少；若两段符号不一致则不支持稳定方向性"}


def label_overlap(fwd) -> dict:
    """量化 H=24 标签的相邻期自相关，判断 block24 CI 是否仍可能过窄。"""
    x = fwd.to_numpy(float)
    corrs = []
    for j in range(x.shape[1]):
        a, b = x[:-1, j], x[1:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            continue
        u, v = a[m] - np.nanmean(a[m]), b[m] - np.nanmean(b[m])
        d = float(np.sqrt((u @ u) * (v @ v)))
        if d > 0:
            corrs.append(float(u @ v / d))
    return {"mean_lag1_autocorr": round(float(np.mean(corrs)), 5) if corrs else None,
            "symbols": len(corrs),
            "note": "H=24 重叠标签 lag1 相关很高时，block24 移动块仍可能低估方差；"
                    "H=48 的 IC 单调放大尤其要按此折价"}


def decision_summary(rob: dict, gate_all: bool) -> dict:
    """把稳健性诊断折算成明确的「是否扩量」决策，避免 gate_passed 被单独误读。"""
    two = rob["two_way_variance_decomposition"]["seq_flip_dev"]
    orig = rob["summary"]["original_mean_ic"]
    dyn = rob["summary"]["after_symbol_demean_mean_ic"]
    part = rob["liquidity_partial_ic"]
    regime = rob["up_down_regime_split"]
    nb = rob["nonoverlapping_block_ic"]
    share = nb.get("negative_share")
    raw_flags = {
        "symbol_fixed_effect_share_gt_0.5":
            (two["symbol_fixed_effect_share"] or 0) > 0.5,
        "ic_sign_flips_after_symbol_demean":
            dyn is not None and np.sign(orig) != np.sign(dyn),
        "liquidity_partial_ci_crosses_zero":
            (part["ci"]["ci_low"] is not None
             and part["ci"]["ci_low"] <= 0 <= part["ci"]["ci_high"]),
        "ic_sign_flips_between_down_and_up_days":
            (regime["down_mean_ic"] is not None and regime["up_mean_ic"] is not None
             and np.sign(regime["down_mean_ic"]) != np.sign(regime["up_mean_ic"])),
        "h24_label_lag1_autocorr_gt_0.9":
            (rob["h24_label_overlap"]["mean_lag1_autocorr"] or 0) > 0.9,
        "nonoverlapping_block_sign_consistency_lt_0.75":
            share is not None and 0.25 < share < 0.75,
    }
    # np.sign 比较产生 numpy.bool_，json 无法序列化；统一转成原生 bool
    flags = {k: bool(v) for k, v in raw_flags.items()}
    n = int(sum(flags.values()))
    return {
        "gate1_formal_pass": bool(gate_all),
        "robustness_flags": flags,
        "n_flags_triggered": int(n),
        "action": "DO_NOT_SCALE_UP" if n >= 3 else "REVIEW",
        "interpretation": (
            "Gate 1 的 H1 形式门槛通过，但稳健性诊断显示 IC 由每币静态属性、"
            "流动性代理与单一普跌市场状态驱动；去币均值后符号翻转；"
            "H=24 标签重叠使 block24 CI 偏窄。据此不授权 Gate 2 实现、"
            "不启动 500 币×60 天（数百 GB）全量归档采集、不启用任何绑定。"),
        "stop_loss_note": (
            "逐笔 aggTrades 方向切换率／大单游程的『静态横截面水平』"
            "不是可交易选币 alpha；不得靠换 H、换分位数或扩窗补救。"),
    }


def robustness(frames, labels, grid) -> dict:
    """主入口：对主判据做全部证伪性诊断，返回可序列化 dict。"""
    fwd, excess, valid, _ = labels[24]
    factor = frames["seq_flip_dev"]
    ics, _ = p12.ic_series(factor, excess, valid)
    dyn = factor.sub(factor.mean(axis=0), axis=1)
    dyn_ics, _ = p12.ic_series(dyn, excess, valid)
    control = np.log(frames["n_trades"].clip(lower=1))
    out = {
        "two_way_variance_decomposition": {
            f: two_way_decomp(frames[f])
            for f in ("seq_flip_dev", "big_run_dev", "taker_imb")},
        "symbol_fixed_effect_removed": {
            "definition": "每币减去全窗口均值后重新算横截面 rank IC",
            **p12.summarize(dyn_ics, "H24_excess_on_demeaned_by_symbol_factor"),
            "ci": p12.block_bootstrap_ci(dyn_ics, seed=20261111),
            "interpretation": "若 IC 大幅衰减，则原 IC 主要来自币的静态属性差异"},
        "liquidity_partial_ic": partial_xs_ic(factor, excess, valid, control),
        "nonoverlapping_block_ic": nonoverlapping_blocks(ics),
        "up_down_regime_split": regime_split(ics, fwd),
        "h24_label_overlap": label_overlap(fwd),
    }
    decays = out["symbol_fixed_effect_removed"]["mean_ic"]
    orig = float(np.nanmean(ics))
    out["summary"] = {
        "original_mean_ic": round(orig, 5),
        "after_symbol_demean_mean_ic": decays,
        "abs_ic_retention_after_demean":
            round(abs(decays / orig), 4) if decays is not None and orig else None,
        "partial_ic_after_liquidity_control":
            out["liquidity_partial_ic"]["mean_partial_ic"],
        "nonoverlapping_negative_share":
            out["nonoverlapping_block_ic"].get("negative_share"),
        "caution": "以上均为证伪性诊断；任何一项显示 IC 由静态属性或重叠标签驱动，"
                   "都不得据此授权 Gate 2 实现或全量采集",
    }
    return out
