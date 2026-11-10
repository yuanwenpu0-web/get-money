"""Shared pure-causal helpers for the Kronos order-exit probe and evaluation.

No DB, no model imports: everything here operates on already-observed bars or
on forecast paths anchored at the order's entry price.
"""
from __future__ import annotations

import numpy as np

STEP_MS = {"15m": 900_000, "1h": 3_600_000}
HORIZON_BARS = {"15m": 96, "1h": 24}  # both equal 24 wall-clock hours
TP_REASONS = {"take_profit", "exchange_tp"}


def barrier(highs, lows, side, tp, sl):
    """Triple-barrier outcome, same-bar SL first (project convention)."""
    for hi, lo in zip(highs, lows):
        if side == "LONG":
            if lo <= sl:
                return 0
            if hi >= tp:
                return 1
        else:
            if hi >= sl:
                return 0
            if lo <= tp:
                return 1
    return -1  # unresolved inside the horizon


def _mfe_and_end(highs, lows, last_price, side, entry, tp):
    denom = (tp - entry) if side == "LONG" else (entry - tp)
    if denom <= 0 or not highs:
        return 0.0, 0.0
    if side == "LONG":
        return (max(highs) - entry) / denom, (last_price - entry) / denom
    return (entry - min(lows)) / denom, (entry - last_price) / denom


def replay_real(bars, side, entry, tp, sl):
    """Label + continuous features from the REAL future bars (absolute prices)."""
    highs, lows = [float(b["high"]) for b in bars], [float(b["low"]) for b in bars]
    label = barrier(highs, lows, side, tp, sl)
    mfe, end = _mfe_and_end(highs, lows, float(bars[-1]["close"]), side, entry, tp)
    return {"label": label, "mfe_ratio": float(mfe), "end_ratio": float(end)}


def path_features(pred, ref_close, side, entry, tp, sl):
    """Score ONE generated OHLC path, using only its shape (not its level)."""
    denom = (tp - entry) if side == "LONG" else (entry - tp)
    prev, price, violations = float(ref_close), float(entry), 0
    if denom <= 0 or prev <= 0:
        return None
    highs, lows, hit = [], [], -1
    for o, h, l, c in pred[["open", "high", "low", "close"]].itertuples(index=False, name=None):
        o, h, l, c = map(float, (o, h, l, c))
        hi, lo = max(o, h, c), min(o, l, c)
        violations += int(h < max(o, c) or l > min(o, c))
        hi_p, lo_p = price * hi / prev, price * lo / prev
        highs.append(hi_p)
        lows.append(lo_p)
        b = barrier([hi_p], [lo_p], side, tp, sl)
        if b >= 0:
            hit = b
            break
        price *= c / prev
        prev = c
    mfe, end = _mfe_and_end(highs, lows, price, side, entry, tp)
    return {"hit": int(hit == 1), "resolved": int(hit >= 0),
            "mfe_ratio": float(mfe), "end_ratio": float(end),
            "violations": violations}


def rw_features(ratios, side, entry, tp, sl, horizon, rng, n_paths=120):
    """Random-walk control: bootstrap past bar shapes into future paths.

    Captures everything available from 'TP/SL distance + realized volatility'
    without any learned forecast. If Kronos cannot beat this, it adds nothing.
    """
    denom = (tp - entry) if side == "LONG" else (entry - tp)
    if denom <= 0 or not ratios:
        return {"p_tp": 0.0, "mfe_ratio": 0.0, "end_ratio": 0.0}
    arr = np.asarray(ratios, dtype=float)
    n = len(arr)
    hits, mfes, ends = [], [], []
    for _ in range(n_paths):
        idx = rng.integers(0, n, size=horizon)
        price, hit = float(entry), -1
        highs, lows = [], []
        for j in idx:
            hi_r, lo_r, c_r = arr[j]
            hi_p, lo_p = price * hi_r, price * lo_r
            highs.append(hi_p)
            lows.append(lo_p)
            b = barrier([hi_p], [lo_p], side, tp, sl)
            if b >= 0:
                hit = b
                break
            price *= c_r
        mfe, end = _mfe_and_end(highs, lows, price, side, entry, tp)
        hits.append(int(hit == 1))
        mfes.append(mfe)
        ends.append(end)
    return {"p_tp": float(np.mean(hits)), "mfe_ratio": float(np.mean(mfes)),
            "end_ratio": float(np.mean(ends))}


def net_return_pct(rec, fee_rate=0.0004, slippage=0.0002):
    """Notional-basis net return per order; exact fee, double-sided slippage."""
    entry, close = rec["entry_price"], rec["close_price"]
    gross = (close - entry) / entry if rec["side"] == "LONG" else (entry - close) / entry
    cost = fee_rate * (1 + close / entry) + 2 * slippage
    return (gross - cost) * 100.0


def auc_score(y, s):
    """Mann-Whitney AUC with tie handling; returns nan on degenerate labels."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def bootstrap_ci(values, n_boot=2000, seed=20261110, confidence=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()
