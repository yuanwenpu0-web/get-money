from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import FEATURES
from ..ml.dataset import build_dataset
from ..ml.trainer import _build_estimator
from ..state import all_state

try:
    import vectorbt as vbt

    HAS_VBT = True
except Exception:  # pragma: no cover
    vbt = None
    HAS_VBT = False


BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
}


def _apply_cooldown(direction: pd.Series, min_bars: int) -> pd.Series:
    if min_bars <= 0:
        return direction
    values = direction.to_numpy()
    result = np.zeros(len(values), dtype=int)
    current = 0
    last_change = -10**9
    for index, desired in enumerate(values):
        if desired != current and (index - last_change) >= min_bars:
            current = int(desired)
            last_change = index
        result[index] = current
    return pd.Series(result, index=direction.index)


def _direction_series(data: pd.DataFrame, strategy: str, prob_up: pd.Series | None, threshold: float, min_hold: int = 30) -> tuple[pd.Series, pd.Series]:
    st = data["supertrend_dir"].fillna(0)
    ema = np.sign(data["ema_ratio_9_21"].fillna(0))
    macd = np.sign(data["macd_hist_norm"].fillna(0))

    if strategy == "mean_reversion":
        rsi = data["rsi_14"].fillna(50)
        low, high = 30.0, 70.0
        signed = pd.Series(0.0, index=data.index)
        signed = signed.where(rsi >= low, (low - rsi) / low * 0.9)
        signed = signed.where(rsi <= high, -(rsi - high) / (100 - high) * 0.9)
    elif strategy == "trend_follow":
        adx = data["adx_14"].fillna(0)
        strength = np.where(adx >= 18, 1.0, 0.5)
        signed = (0.5 * st + 0.3 * ema + 0.2 * macd) * strength
    else:
        rules = 0.5 * st + 0.3 * ema + 0.2 * macd
        if prob_up is not None:
            ml = (prob_up - 0.5) * 2
            signed = 0.6 * ml + 0.4 * rules
        else:
            signed = rules

    score = 0.5 + 0.5 * signed.abs()
    direction = pd.Series(0, index=data.index)
    direction[(signed > 0) & (score >= threshold)] = 1
    direction[(signed < 0) & (score >= threshold)] = -1
    return _apply_cooldown(direction, min_hold), score


def _size_series(score: pd.Series, state: dict) -> pd.Series:
    base = float(state.get("base_position_pct", 0.08))
    peak = float(state.get("max_position_pct", 0.25))
    heavy = float(state.get("heavy_score", 0.75))
    min_score = float(state.get("min_score", 0.55))
    size = pd.Series(0.0, index=score.index)
    mid = score[(score > min_score) & (score < heavy)]
    size.loc[mid.index] = base + (mid - min_score) / max(heavy - min_score, 1e-6) * (peak * 0.7 - base)
    size.loc[score >= heavy] = peak
    return size.clip(0.01, 1.0)


def _extract_trades(portfolio) -> list[dict]:
    trades = []
    try:
        frame = portfolio.trades.records_readable
    except Exception:
        return trades
    if frame is None or frame.empty:
        return trades
    for _, row in frame.iterrows():
        entry_time = row.get("Entry Timestamp")
        exit_time = row.get("Exit Timestamp")
        fees = float(row.get("Entry Fees", 0) or 0) + float(row.get("Exit Fees", 0) or 0)
        trades.append(
            {
                "entry_time": str(entry_time),
                "exit_time": str(exit_time) if pd.notna(exit_time) else "",
                "side": str(row.get("Direction", "")),
                "entry_price": round(float(row.get("Avg Entry Price", 0) or 0), 4),
                "exit_price": round(float(row.get("Avg Exit Price", 0) or 0), 4),
                "size": round(float(row.get("Size", 0) or 0), 6),
                "pnl": round(float(row.get("PnL", 0) or 0), 4),
                "return_pct": round(float(row.get("Return", 0) or 0) * 100, 3),
                "fees": round(fees, 4),
                "status": str(row.get("Status", "")),
            }
        )
    return trades


def _metrics_from_equity(equity: pd.Series, trades: list[dict], initial: float, interval: str) -> dict:
    equity = equity.dropna()
    if equity.empty:
        return {}
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    max_dd = float(drawdown.min())
    sharpe = 0.0
    try:
        daily = equity.resample("1D").last().pct_change().dropna()
        if len(daily) > 2 and daily.std() > 0:
            sharpe = float(daily.mean() / daily.std() * math.sqrt(365))
    except Exception:
        sharpe = 0.0
    pnl = [t["pnl"] for t in trades]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "initial_cash": round(initial, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round((float(equity.iloc[-1]) / initial - 1) * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit else 0.0),
        "total_fees": round(sum(t["fees"] for t in trades), 2),
        "bars": int(len(equity)),
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
    }


def _downsample(equity: pd.Series, limit: int = 1500) -> list[dict]:
    step = max(1, len(equity) // limit)
    sampled = equity.iloc[::step]
    return [{"time": str(index), "equity": round(float(value), 2)} for index, value in sampled.items()]


def _run_vectorbt(data: pd.DataFrame, direction: pd.Series, score: pd.Series, state: dict, initial: float, fee: float, slippage: float, interval: str, sl_mult: float, tp_mult: float):
    close = data["close"]
    long_entries = (direction == 1) & (direction.shift(1).fillna(0) != 1)
    long_exits = (direction != 1) & (direction.shift(1).fillna(0) == 1)
    short_entries = (direction == -1) & (direction.shift(1).fillna(0) != -1)
    short_exits = (direction != -1) & (direction.shift(1).fillna(0) == -1)

    atr_pct = data["atr_pct"].fillna(0.002)
    min_sl = float(state.get("min_sl_pct", 0.0025))
    min_tp = float(state.get("min_tp_pct", 0.005))
    sl_stop = (atr_pct * sl_mult).clip(min_sl, 0.2)
    tp_stop = (atr_pct * tp_mult).clip(min_tp, 0.5)
    size = _size_series(score, state)
    use_change = bool(state.get("backtest_trailing", False))

    long_pf = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        direction="longonly",
        init_cash=initial,
        size=size,
        size_type="percent",
        fees=fee,
        slippage=slippage,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        **({"sl_trail": True, "high": data["high"], "low": data["low"]} if use_change else {}),
        freq="1min" if interval == "1m" else interval,
    )
    short_pf = vbt.Portfolio.from_signals(
        close=close,
        entries=short_entries,
        exits=short_exits,
        direction="shortonly",
        init_cash=initial,
        size=size,
        size_type="percent",
        fees=fee,
        slippage=slippage,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        **({"sl_trail": True, "high": data["high"], "low": data["low"]} if use_change else {}),
        freq="1min" if interval == "1m" else interval,
    )
    equity = long_pf.value() + short_pf.value() - initial
    trades = _extract_trades(long_pf) + _extract_trades(short_pf)
    return equity, trades, "vectorbt"


def _run_fallback(data: pd.DataFrame, direction: pd.Series, initial: float, fee: float, slippage: float, interval: str, sl_mult: float, tp_mult: float, position_pct: float):
    """vectorbt 不可用时的顺序撮合回测，规则与模拟盘一致。"""
    cash = initial
    equity_values = []
    trades = []
    position = None

    for row, price in zip(data.itertuples(), data["close"]):
        signal_dir = int(direction.loc[row.Index])
        high, low = float(row.high), float(row.low)
        atr = float(row.atr_pct) * price

        if position is not None:
            side, entry, qty, margin, sl, tp, entry_fee = position
            exit_price = None
            reason = ""
            if side == 1:
                if low <= sl:
                    exit_price, reason = sl, "stop_loss"
                elif high >= tp:
                    exit_price, reason = tp, "take_profit"
            else:
                if high >= sl:
                    exit_price, reason = sl, "stop_loss"
                elif low <= tp:
                    exit_price, reason = tp, "take_profit"
            if exit_price is None and signal_dir != side:
                exit_price, reason = price, "signal"

            if exit_price is not None:
                fill = exit_price * (1 - slippage) if side == 1 else exit_price * (1 + slippage)
                gross = (fill - entry) * qty * side
                exit_fee = fill * qty * fee
                cash += margin + gross - exit_fee
                trades.append(
                    {
                        "entry_time": "",
                        "exit_time": str(row.Index),
                        "side": "long" if side == 1 else "short",
                        "entry_price": round(entry, 4),
                        "exit_price": round(fill, 4),
                        "size": round(qty, 6),
                        "pnl": round(gross - entry_fee - exit_fee, 4),
                        "return_pct": round((gross - entry_fee - exit_fee) / margin * 100, 3),
                        "fees": round(entry_fee + exit_fee, 4),
                        "status": reason,
                    }
                )
                position = None

        if position is None and signal_dir != 0:
            margin = cash * position_pct
            entry = price * (1 + slippage) if signal_dir == 1 else price * (1 - slippage)
            lev = max(1, int(0.015 / max(float(row.atr_pct), 1e-4)))
            lev = min(lev, 10)
            qty = margin * lev / entry
            entry_fee = margin * lev * fee
            if cash > margin + entry_fee:
                cash -= margin + entry_fee
                sl = entry - atr * sl_mult if signal_dir == 1 else entry + atr * sl_mult
                tp = entry + atr * tp_mult if signal_dir == 1 else entry - atr * tp_mult
                position = (signal_dir, entry, qty, margin, sl, tp, entry_fee)

        if position is not None:
            side, entry, qty, margin, sl, tp, entry_fee = position
            unrealized = (price - entry) * qty * side
            equity_values.append(cash + margin + unrealized)
        else:
            equity_values.append(cash)

    if position is not None:
        side, entry, qty, margin, sl, tp, entry_fee = position
        price = float(data["close"].iloc[-1])
        fill = price * (1 - slippage) if side == 1 else price * (1 + slippage)
        gross = (fill - entry) * qty * side
        exit_fee = fill * qty * fee
        cash += margin + gross - exit_fee
        trades.append(
            {
                "entry_time": "",
                "exit_time": str(data.index[-1]),
                "side": "long" if side == 1 else "short",
                "entry_price": round(entry, 4),
                "exit_price": round(fill, 4),
                "size": round(qty, 6),
                "pnl": round(gross - entry_fee - exit_fee, 4),
                "return_pct": round((gross - entry_fee - exit_fee) / margin * 100, 3),
                "fees": round(entry_fee + exit_fee, 4),
                "status": "close_at_end",
            }
        )
        equity_values[-1] = cash

    equity = pd.Series(equity_values, index=data.index)
    return equity, trades, "fallback"


def run_backtest(
    db: Session,
    symbol: str,
    interval: str,
    strategy: str,
    algo: str = "lightgbm",
    days: int = 30,
    initial_cash: float | None = None,
    fee: float | None = None,
    slippage: float | None = None,
    train_ratio: float = 0.75,
    threshold: float | None = None,
) -> dict:
    state = all_state(db)
    initial_cash = float(initial_cash if initial_cash is not None else state.get("initial_capital", 10_000))
    fee = float(fee if fee is not None else state.get("fee_rate", 0.0004))
    slippage = float(slippage if slippage is not None else state.get("slippage", 0.0002))
    threshold = float(threshold if threshold is not None else state.get("min_score", 0.55))

    data = build_dataset(db, symbol, interval, horizon=int(state.get("train_horizon", 5)), days=days)
    if len(data) < 800:
        raise ValueError(f"回测样本不足（{len(data)} 行），请先拉取至少 7 天 1 分钟K线")

    split = int(len(data) * train_ratio)
    train, test = data.iloc[:split], data.iloc[split:]
    if len(test) < 200:
        raise ValueError("测试集不足 200 根K线，请增加回测天数或降低训练占比")

    prob_up = None
    algo_used = "rule"
    model_metrics = {}
    if algo in ("lightgbm", "xgboost"):
        estimator = _build_estimator(algo)
        estimator.fit(train[FEATURES], train["label"])
        prob_up = pd.Series(estimator.predict_proba(test[FEATURES])[:, 1], index=test.index)
        algo_used = algo

    direction, score = _direction_series(test, strategy, prob_up, threshold, int(state.get("min_hold_bars", 30)))
    direction = direction.astype(int)
    direction.iloc[:2] = 0

    sl_mult = float(state.get("sl_atr_mult", 1.5))
    tp_mult = float(state.get("tp_atr_mult", 2.5))
    if HAS_VBT:
        try:
            equity, trades, engine = _run_vectorbt(test, direction, score, state, initial_cash, fee, slippage, interval, sl_mult, tp_mult)
        except Exception as exc:
            equity, trades, engine = _run_fallback(
                test, direction, initial_cash, fee, slippage, interval, sl_mult, tp_mult,
                float(state.get("base_position_pct", 0.08)),
            )
            model_metrics["vectorbt_error"] = str(exc)[:200]
    else:
        equity, trades, engine = _run_fallback(
            test, direction, initial_cash, fee, slippage, interval, sl_mult, tp_mult,
            float(state.get("base_position_pct", 0.08)),
        )

    metrics = _metrics_from_equity(equity, trades, initial_cash, interval)
    metrics.update(
        {
            "engine": engine,
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy,
            "algo": algo_used,
            "fee_rate": fee,
            "slippage": slippage,
            "threshold": threshold,
            "buy_hold_return_pct": round((float(test["close"].iloc[-1]) / float(test["close"].iloc[0]) - 1) * 100, 2),
            "signal_bars": int((direction != 0).sum()),
            "long_bars": int((direction == 1).sum()),
            "short_bars": int((direction == -1).sum()),
            **model_metrics,
        }
    )
    return {
        "metrics": metrics,
        "equity_curve": _downsample(equity),
        "trades": trades[-500:],
    }
