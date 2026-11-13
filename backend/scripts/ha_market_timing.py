"""P13 / H-A 市场级杠杆周期择时：编排、阈值隔离、对照组与硬数据门禁。

工程预置，非 edge 判定。预登记见 research/hypotheses/H-A-market-leverage-timing.md。
CLI 默认只做就绪核查：覆盖不足 60 天拒绝计算任何读数；即使达标也需显式 --evaluate
且按假设卡 §9 由人工重新登记授权，自动化不得自行启动。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ha_common import (  # noqa: E402
    DAY_MS, HOUR_MS, HORIZON, OOS_DAYS, TRAIN_DAYS, block_readiness, data_gate,
    market_factor, market_forward_return, nonoverlap_block_times, oi_zscore,
    split_train_oos,
)
from ha_portfolio import (  # noqa: E402
    FEE_RATE, ROUND_TRIP, SLIPPAGE, constant_full_returns, fit_markov,
    matched_random_switches, paired_bootstrap_ci, percentile_rank, spearman,
    switch_from_signal, switch_net_returns, termination_checks,
    time_segment_sign_share,
)

THRESHOLD_QUANTILES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SEED = 20261112
N_SEEDS = 200
COST_MODEL = {"fee_rate": FEE_RATE, "slippage": SLIPPAGE,
              "round_trip": ROUND_TRIP, "basis": "notional"}


def select_threshold(train_factor: np.ndarray, train_ret: np.ndarray,
                     fee: float = FEE_RATE,
                     slippage: float = SLIPPAGE) -> tuple[float, dict]:
    """只在训练段按预登记分位档位选阈值；函数签名不接收样本外数据。"""
    f, r = np.asarray(train_factor, float), np.asarray(train_ret, float)
    ok = np.isfinite(f) & np.isfinite(r)
    f, r = f[ok], r[ok]
    if len(f) < 3:
        raise ValueError("train segment has fewer than 3 usable blocks")
    cands = {}
    for q in THRESHOLD_QUANTILES:
        thr = float(np.quantile(f, q))
        on = switch_from_signal(f, thr)
        cands[f"q{int(q * 100)}"] = {
            "threshold": thr,
            "net": float(switch_net_returns(r, on, fee, slippage).sum()),
            "exposure": float(on.mean())}
    best = max(cands, key=lambda k: cands[k]["net"])
    return cands[best]["threshold"], {"selected": best, "candidates": cands}


def perfect_foresight_net(block_ret: np.ndarray, fee: float = FEE_RATE,
                          slippage: float = SLIPPAGE) -> np.ndarray:
    """正对照：偷看未来方向的完美开关，用来验证量具确有灵敏度。"""
    r = np.asarray(block_ret, float)
    return switch_net_returns(r, r > 0, fee, slippage)


def evaluate(block_ts: np.ndarray, factor: np.ndarray, block_ret: np.ndarray, *,
             n_seeds: int = N_SEEDS, seed: int = SEED, fee: float = FEE_RATE,
             slippage: float = SLIPPAGE, train_days: int = TRAIN_DAYS,
             oos_days: int = OOS_DAYS) -> dict:
    ts = np.asarray(block_ts, dtype=np.int64)
    f, r = np.asarray(factor, float), np.asarray(block_ret, float)
    if not (len(f) == len(r) == len(ts)):
        raise ValueError("block_ts / factor / block_ret length mismatch")
    train_ts, oos_ts = split_train_oos(ts, train_days, oos_days)
    start = len(ts) - (train_days + oos_days)
    itr = slice(start, start + len(train_ts))
    io = slice(start + len(train_ts), len(ts))

    thr, sel = select_threshold(f[itr], r[itr], fee, slippage)
    r_oos = r[io]
    on_oos = switch_from_signal(f[io], thr)
    sw = switch_net_returns(r_oos, on_oos, fee, slippage)
    base = constant_full_returns(r_oos, fee, slippage)
    diff = sw - base
    ci = paired_bootstrap_ci(diff, seed=seed)
    draws = matched_random_switches(on_oos, n_seeds=n_seeds, seed=seed)
    rand = np.array([switch_net_returns(r_oos, d, fee, slippage).sum() for d in draws])
    sw_net, base_net = float(sw.sum()), float(base.sum())
    p90 = float(np.percentile(rand, 90))
    train_rho, oos_rho = spearman(f[itr], r[itr]), spearman(f[io], r_oos)
    seg = time_segment_sign_share(diff)
    # 卡 §2 的「非重叠块数≥30」按整个评估窗（训练+样本外）计；
    # 样本外单独只有 20 块，其 rho 的 CI 天然偏宽，属预登记局限。
    term = termination_checks(
        train_rho=train_rho, oos_rho=oos_rho, switch_net=sw_net,
        baseline_net=base_net, ci_low=ci["ci_low"], random_p90=p90,
        n_blocks=len(train_ts) + len(oos_ts), segment_sign_share=seg)
    window = r[start:]
    return {
        "n_blocks_total": int(len(train_ts) + len(oos_ts)),
        "train_blocks": int(len(train_ts)), "oos_blocks": int(len(oos_ts)),
        "threshold": thr, "threshold_selection": sel,
        "train": {"rho": train_rho,
                  "net": float(switch_net_returns(
                      r[itr], switch_from_signal(f[itr], thr), fee, slippage).sum())},
        "oos": {"rho": oos_rho, "switch_net": sw_net, "baseline_net": base_net,
                "diff_mean": float(diff.mean()), "ci_block1": ci,
                "ci_block4": paired_bootstrap_ci(diff, seed=seed, block=4),
                "segment_sign_share": seg, "exposure": float(on_oos.mean()),
                "switches": int(np.sum(on_oos[1:] != on_oos[:-1])),
                "persistence_stay_on_off": list(fit_markov(on_oos))},
        "random_baseline": {"n_seeds": int(n_seeds), "seed": seed,
                            "mean": float(rand.mean()), "p50": float(np.percentile(rand, 50)),
                            "p90": p90, "max": float(rand.max()),
                            "signal_percentile": percentile_rank(sw_net, rand),
                            "null": "matched run-length permutation (exposure and turnover identical)"},
        "positive_control": {"perfect_foresight_net": float(
            perfect_foresight_net(r_oos, fee, slippage).sum()),
            "beats_baseline": bool(perfect_foresight_net(r_oos, fee, slippage).sum() > base_net)},
        "market_background": {"cum_return": float(np.prod(1 + window) - 1),
                              "up_block_share": float((window > 0).mean())},
        "cost_model": dict(COST_MODEL),
        "termination": term,
    }


def fetch_frames():
    """只读 DB；延迟 import 以便纯函数单测不依赖数据库。"""
    from sqlalchemy import text
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
    from app.db import make_engine
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        lo, hi = conn.execute(text("SELECT min(ts), max(ts) FROM sentiment")).one()
        sent = pd.read_sql_query(text("SELECT symbol, ts, oi_value FROM sentiment"), conn)
        px = pd.read_sql_query(text(
            "SELECT symbol, open_time, close FROM klines WHERE interval='1h' "
            "AND open_time BETWEEN :a AND :b"), conn,
            params={"a": int(lo) - 2 * HOUR_MS, "b": int(hi) + (HORIZON + 2) * HOUR_MS})
    return sent, px, int(lo), int(hi)


def build_market_series(sent: pd.DataFrame, px: pd.DataFrame,
                        lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oi = sent.pivot(index="ts", columns="symbol", values="oi_value").sort_index()
    factor = market_factor(oi_zscore(oi.replace(0.0, np.nan)))
    close = px.pivot(index="open_time", columns="symbol", values="close").sort_index()
    label = market_forward_return(close, HORIZON)
    day_start = ((lo + DAY_MS - 1) // DAY_MS) * DAY_MS  # 墙钟对齐 UTC 日界
    blocks = nonoverlap_block_times(day_start, hi, HORIZON)
    return (blocks, factor.reindex(blocks).to_numpy(float),
            label.reindex(blocks).to_numpy(float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H-A 市场级杠杆择时量具（工程预置）")
    ap.add_argument("--evaluate", action="store_true",
                    help="显式授权后才计算读数；须先按假设卡 §9 人工重新登记")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    sent, px, lo, hi = fetch_frames()
    gate = data_gate(lo, hi)
    blocks, f, r = build_market_series(sent, px, lo, hi)
    ready_blocks = block_readiness(blocks, f, r)
    payload = {
        "hypothesis_card": "research/hypotheses/H-A-market-leverage-timing.md",
        "stage": "Gate 0 readiness / engineering pre-build, NOT an edge verdict",
        "sentiment_start_ms": lo, "sentiment_end_ms": hi,
        "sentiment_symbols": int(sent["symbol"].nunique()),
        "data_gate": gate, "block_readiness": ready_blocks,
        "cost_model": dict(COST_MODEL),
        "default_enabled": False, "trades": 0, "metrics": None,
    }
    if not (gate["ready"] and ready_blocks["ready"]):
        payload["status"] = "BLOCKED_BY_DATA"
        payload["next_action"] = (
            f"keep daily 08:30 collection running; span short by "
            f"{gate['days_remaining']:.2f}d and evaluation window short by "
            f"{ready_blocks['extra_days_lower_bound']} usable block(s); "
            f"then re-register the hypothesis card by hand")
    elif not args.evaluate:
        payload["status"] = "READY_AWAITING_HUMAN_AUTHORIZATION"
        payload["next_action"] = "data gate met; human must re-register and pass --evaluate"
    else:
        payload["metrics"] = evaluate(blocks, f, r, n_seeds=args.seeds, seed=args.seed)
        payload["status"] = "EVALUATED"
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())