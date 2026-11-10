"""Pure-causal replay helpers for the trend-reversal order-exit diagnosis.

No DB access, no production imports. All rules share the SAME future bar
sequence and the SAME notional-basis cost model, so paired differences isolate
the exit rule and nothing else.
"""
from __future__ import annotations

import numpy as np

FEE_RATE = 0.0004
SLIPPAGE = 0.0002
TP_REASONS = {"take_profit", "exchange_tp"}
SL_REASONS = {"stop_loss", "exchange_sl"}
REVERSE_REASONS = {"signal_reverse"}
MARKET_EXIT_REASONS = TP_REASONS | SL_REASONS | REVERSE_REASONS
ADMIN_EXIT_REASONS = {"manual", "env_switch_archived", "research_hold", ""}


def net_return_pct(entry: float, exit_price: float, side: str) -> float:
    """Notional-basis net return, exact fee + double-sided slippage (H3)."""
    if entry <= 0 or exit_price <= 0:
        return float("nan")
    gross = ((exit_price - entry) / entry if side == "LONG"
             else (entry - exit_price) / entry)
    cost = FEE_RATE * (1 + exit_price / entry) + 2 * SLIPPAGE
    return (gross - cost) * 100.0


def atr(hist: list[dict], n: int = 14) -> float:
    """Average true range over the last n HISTORICAL bars only (causal)."""
    rows = hist[-(n + 1):]
    if len(rows) < 2:
        return 0.0
    trs = []
    for prev, cur in zip(rows, rows[1:]):
        h, l, pc = float(cur["high"]), float(cur["low"]), float(prev["close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-n:]))


def run_rule(bars, side, entry, sl, rule):
    """Replay ONE exit rule over the future bars.

    Project convention: same-bar SL first, gap fills at the open, exit price is
    the barrier price (not the bar close). Returns (exit_price, idx, reason).
    """
    is_long = side == "LONG"
    peak = float(entry)
    last_close = float(entry)
    for i, b in enumerate(bars):
        o, h, l, c = (float(b[k]) for k in ("open", "high", "low", "close"))
        fav, adv = (h, l) if is_long else (l, h)
        last_close = c
        # Gap: the open is already through a barrier. SL takes precedence.
        if (o <= sl if is_long else o >= sl):
            return float(o), i, "sl_gap"
        if rule["tp"] is not None and (o >= rule["tp"] if is_long else o <= rule["tp"]):
            return float(o), i, "tp_gap"
        # Same-bar SL first (project convention).
        if (adv <= sl if is_long else adv >= sl):
            return float(sl), i, "sl"
        if rule["tp"] is not None and (fav >= rule["tp"] if is_long else fav <= rule["tp"]):
            return float(rule["tp"]), i, "tp"
        # Update the running favorable peak using THIS bar's extreme only.
        peak = max(peak, fav) if is_long else min(peak, fav)
        gain = (peak - entry) if is_long else (entry - peak)
        if rule["kind"] == "trailing" and gain >= rule["activate_R"] * rule["R"]:
            trail_price = (peak - rule["trail_R"] * rule["R"] if is_long
                           else peak + rule["trail_R"] * rule["R"])
            if l <= trail_price <= h:
                return float(trail_price), i, "trail"
        if rule["kind"] == "time" and i + 1 >= rule["max_bars"]:
            return float(c), i, "time"
    return float(last_close), len(bars) - 1, "horizon"


def build_rules(entry, tp, sl, atr_value, horizon, side):
    """Finite NEGATIVE-CONTROL rule family. No champion selection (H2 / Gate 0)."""
    sign = 1.0 if tp > entry else -1.0
    R = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rules = [{"name": "baseline", "kind": "barrier", "tp": tp, "sl": sl,
              "R": R, "activate_R": 0, "trail_R": 0, "max_bars": 0}]
    for scale in (0.5, 0.75, 1.5, 2.0, 3.0):
        rules.append({"name": f"tp_x{scale}", "kind": "barrier",
                      "tp": entry + sign * tp_dist * scale, "sl": sl, "R": R,
                      "activate_R": 0, "trail_R": 0, "max_bars": 0})
    for scale in (1.5, 2.0):
        rules.append({"name": f"sl_x{scale}", "kind": "barrier", "tp": tp,
                      "sl": entry + (sl - entry) * scale, "R": R,
                      "activate_R": 0, "trail_R": 0, "max_bars": 0})
    if atr_value > 0:
        for k in (1.5, 2.5, 4.0):
            rules.append({"name": f"atr_tp_x{k}", "kind": "barrier",
                          "tp": entry + sign * k * atr_value, "sl": sl, "R": R,
                          "activate_R": 0, "trail_R": 0, "max_bars": 0})
    for trail in (0.5, 1.0, 2.0):
        rules.append({"name": f"trail_{trail}R", "kind": "trailing", "tp": None,
                      "sl": sl, "R": R, "activate_R": 1.0, "trail_R": trail,
                      "max_bars": 0})
    for frac in (0.25, 0.5, 1.0):
        rules.append({"name": f"hold_{frac}H", "kind": "time", "tp": None,
                      "sl": sl, "R": R, "activate_R": 0, "trail_R": 0,
                      "max_bars": max(1, int(round(horizon * frac)))})
    rules.append({"name": "no_tp_hold", "kind": "time", "tp": None, "sl": sl,
                  "R": R, "activate_R": 0, "trail_R": 0, "max_bars": 10**9})
    return rules


def diagnostics(bars, side, entry, sl, tp, exit_price, exit_bar_idx, real_close,
                real_hold_bars, post_cap=96):
    """Descriptive 'too early / too late' quantities. Oracle is LOOKAHEAD only.

    The look-forward window is BOUNDED at (holding bars + post_cap). Without the
    cap, a 72h window behind a 3h holding period inflates MFE and post-exit run
    by an order of magnitude and makes every TP look 'too early'.
    """
    is_long = side == "LONG"
    R = abs(entry - sl)
    tp_dist = abs(tp - entry)
    if R <= 0 or tp_dist <= 0 or not bars:
        return {}
    window = bars[:min(len(bars), real_hold_bars + post_cap)]
    highs = [float(b["high"]) for b in window]
    lows = [float(b["low"]) for b in window]
    mfe = (max(highs) - entry if is_long else entry - min(lows)) / R
    mae = (entry - min(lows) if is_long else max(highs) - entry) / R
    best_price = max(highs) if is_long else min(lows)
    oracle_net_pct = net_return_pct(entry, best_price, side)
    hold = window[:max(1, min(len(window), real_hold_bars))]
    hold_mfe = ((max(float(b["high"]) for b in hold) - entry) if is_long
                else (entry - min(float(b["low"]) for b in hold))) / R
    post = window[max(1, min(len(window), real_hold_bars)):]
    post_run = ((max([float(b["high"]) for b in post] + [real_close]) - real_close)
                if is_long
                else (real_close - min([float(b["low"]) for b in post] + [real_close]))) / R
    realized = ((real_close - entry) if is_long else (entry - real_close)) / R
    return {
        "mfe_R": float(mfe), "mae_R": float(mae),
        "oracle_net_pct": float(oracle_net_pct),
        "hold_mfe_R": float(hold_mfe), "post_exit_run_R": float(post_run),
        "giveback_R": float(hold_mfe - realized),
        "capture_ratio": float(realized / hold_mfe) if hold_mfe > 1e-9 else float("nan"),
        "tp_dist_R": float(tp_dist / R),
        "window_bars": len(window), "post_bars_used": len(post),
    }


def bootstrap_ci(values, n_boot=2000, seed=20261110, confidence=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - confidence) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


def verdict(ci, n, min_trades=30):
    lo, hi = ci
    if n < min_trades:
        return "insufficient"
    if lo > 0:
        return "pass"
    if hi < 0:
        return "negative"
    return "inconclusive"
