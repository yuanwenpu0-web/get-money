"""指标组合、滚动验证与随机基线。

组合逻辑只做两件事的显式对照：

- `and`：两个信号同一根 bar 同向才入场（过滤型组合，频率必然下降）
- `or` ：任一信号同向即入场（覆盖型组合，频率上升但容易被弱信号稀释）
- `vote`：top-N 中至少 k 个同向（多数票，介于两者之间）

所有组合共用同一个 rsi50_cross 出场与同一套撮合参数，保证与单指标结果可比。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lab_experiment import Dataset, evaluate, sign, strategy_trades, split_trades
from .lab_signals import Signals
from .lab_sim import SimConfig, Trade, simulate, to_ms
from .lab_stats import bh_qvalues, percentile_rank, summarize


def signal_cache(ds: Dataset, cands) -> dict[str, dict[str, Signals]]:
    return {cand.key: {s: sign(cand, f) for s, f in ds.frames.items()} for cand in cands}


def combine(a: Signals, b: Signals, rule: str) -> Signals:
    if rule == "and":
        return Signals(a.long & b.long, a.short & b.short)
    if rule == "or":
        return Signals(a.long | b.long, a.short | b.short)
    raise ValueError(f"未知组合规则: {rule}")


def vote(cache: dict[str, dict[str, Signals]], keys: list[str], need: int) -> dict[str, Signals]:
    out: dict[str, Signals] = {}
    for symbol in cache[keys[0]]:
        long_votes = np.zeros(len(cache[keys[0]][symbol].long), dtype=np.int8)
        short_votes = np.zeros_like(long_votes)
        for key in keys:
            long_votes += cache[key][symbol].long.astype(np.int8)
            short_votes += cache[key][symbol].short.astype(np.int8)
        out[symbol] = Signals(long_votes >= need, short_votes >= need)
    return out


def _trades_from(ds: Dataset, signals: dict[str, Signals], cfg: SimConfig) -> list[Trade]:
    out: list[Trade] = []
    for symbol, frame in ds.frames.items():
        out.extend(simulate(symbol, frame, signals[symbol], ds.exits[symbol], cfg))
    return out


def _row(ds: Dataset, label: str, rule: str, components: list[str], signals: dict[str, Signals], cfg: SimConfig) -> dict:
    trades = _trades_from(ds, signals, cfg)
    train, test, purged = split_trades(trades, ds.split_ts)
    return {
        "key": label,
        "label": label,
        "rule": rule,
        "components": components,
        "train": summarize(train, cfg.stake, len(ds.frames)),
        "test": summarize(test, cfg.stake, len(ds.frames)),
        "all": summarize(trades, cfg.stake, len(ds.frames)),
        "purged": purged,
    }


def run_combos(ds: Dataset, tops: list, cfg: SimConfig, vote_size: int = 3) -> list[dict]:
    """对训练集排名前 K 的候选两两组合；另加一组多数票。tops 为 Candidate 列表。"""
    cands = tops
    cache = signal_cache(ds, cands)
    keys = [c.key for c in cands]
    rows: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for rule in ("and", "or"):
                signals = {
                    s: combine(cache[keys[i]][s], cache[keys[j]][s], rule) for s in ds.frames
                }
                label = f"{cands[i].label} {rule.upper()} {cands[j].label}"
                rows.append(_row(ds, label, rule, [keys[i], keys[j]], signals, cfg))
    if len(keys) >= vote_size:
        signals = vote(cache, keys[:vote_size], need=2)
        label = "VOTE(2/3): " + " | ".join(cands[k].label for k in range(vote_size))
        rows.append(_row(ds, label, "vote", keys[:vote_size], signals, cfg))
    return rows


def apply_global_fdr(single_rows: list[dict], combo_rows: list[dict], min_train_trades: int = 30) -> None:
    """把单指标与组合放进同一个多重比较集合里重新校正 q 值。

    两阶段各自校正会低估选择成本：组合是「从训练集 top-K 里挑出来的」，
    这一步已经用掉自由度，所以合并校正才是保守口径。
    """
    merged = single_rows + combo_rows
    qvalues = bh_qvalues(np.array([row["train"]["p_value"] for row in merged]))
    for row, q in zip(merged, qvalues):
        row["q_value"] = round(float(q), 5)
        train, test = row["train"], row["test"]
        if train["trades"] < min_train_trades:
            verdict = "insufficient"
        elif train["total_pnl"] <= 0:
            verdict = "no_edge"
        elif q >= 0.10:
            verdict = "likely_noise"
        elif test["total_pnl"] <= 0:
            verdict = "unstable_oos"
        else:
            verdict = "candidate"
        row["verdict"] = verdict


def walk_forward(ds: Dataset, tops: list, cfg: SimConfig, folds: int = 3, min_trades: int = 20) -> dict:
    """锚定式滚动：每折用「起点 → b_k」选参，在 (b_k, b_k+1] 上判定。"""
    lo = min(int(to_ms(f.index)[0]) for f in ds.frames.values())
    hi = max(int(to_ms(f.index)[-1]) for f in ds.frames.values())
    bounds = [int(lo + (hi - lo) * frac) for frac in (0.4, 0.6, 0.8, 1.0)][: folds + 1]
    cached = {c.key: strategy_trades(ds, c, cfg) for c in tops}
    fold_rows = []
    for k in range(len(bounds) - 1):
        train_end, test_end = bounds[k], bounds[k + 1]
        best, best_pnl = None, -1e18
        for cand in tops:
            trades = cached[cand.key]
            train = [t for t in trades if t.exit_ts < train_end]
            if len(train) < min_trades:
                continue
            stats = summarize(train, cfg.stake, len(ds.frames))
            if stats["total_pnl"] > best_pnl:
                best, best_pnl = (cand, stats), stats["total_pnl"]
        if best is None:
            continue
        cand, train_stats = best
        trades = cached[cand.key]
        test = [t for t in trades if t.entry_ts >= train_end and t.exit_ts <= test_end]
        fold_rows.append(
            {
                "fold": k + 1,
                "train_end": str(pd.to_datetime(train_end, unit="ms", utc=True)),
                "test_end": str(pd.to_datetime(test_end, unit="ms", utc=True)),
                "selected": cand.label,
                "train": train_stats,
                "test": summarize(test, cfg.stake, len(ds.frames)),
            }
        )
    return {"folds": fold_rows}


def random_baseline(ds: Dataset, cand, cfg: SimConfig, draws: int = 60, seed: int = 20261101) -> dict:
    """同频率随机入场基线：保留每个币的多空入场次数，位置随机。"""
    rng = np.random.default_rng(seed)
    signals = {s: sign(cand, f) for s, f in ds.frames.items()}
    totals: list[float] = []
    for _ in range(draws):
        pnl = 0.0
        for symbol, frame in ds.frames.items():
            sig = signals[symbol]
            n = len(frame)
            long_mask = np.zeros(n, dtype=bool)
            short_mask = np.zeros(n, dtype=bool)
            for mask, target in ((long_mask, int(sig.long.sum())), (short_mask, int(sig.short.sum()))):
                if target <= 0:
                    continue
                picks = rng.choice(n, size=min(target, n), replace=False)
                mask[picks] = True
            for trade in simulate(symbol, frame, Signals(long_mask, short_mask), ds.exits[symbol], cfg):
                pnl += trade.pnl
        totals.append(pnl)
    actual = summarize(strategy_trades(ds, cand, cfg), cfg.stake, len(ds.frames))["total_pnl"]
    return {
        "draws": draws,
        "seed": seed,
        "actual_total_pnl": round(actual, 4),
        "baseline_mean": round(float(np.mean(totals)), 4),
        "baseline_p95": round(float(np.percentile(totals, 95)), 4),
        "percentile": round(percentile_rank(actual, totals), 4),
        "p_value": round(float(np.mean(np.asarray(totals) >= actual)), 4),
    }
