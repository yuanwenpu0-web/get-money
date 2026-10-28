"""Persistent, local-only paper accounts for converted Pine strategies.

All confirmed candles are retained and replayed on updates, so strategy state
survives process restarts. This module never submits exchange orders. Its file
lock is process-local; run a single service process per account.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

from . import compile_pine
from .backtest import run_backtest

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_active_runners: dict[str, object] = {}
_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")
_SETTINGS = {"initial_equity", "quantity", "fee_rate", "slippage", "interval_ms"}


def _lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("version") != 1:
        raise ValueError("unsupported paper account format")
    return state


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".paper-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_account(path: str | Path, source: str, settings: dict | None = None) -> dict:
    """Create an account once. Existing accounts are never overwritten."""
    path = Path(path)
    settings = dict(settings or {})
    if set(settings) - _SETTINGS:
        raise ValueError("unknown paper account setting")
    if settings.get("interval_ms", 60_000) is None:
        raise ValueError("paper accounts require a fixed candle interval")
    compile_pine(source)
    sample = pd.DataFrame([dict(open_time=0, open=1., high=1., low=1., close=1., volume=0.)])
    no_signals = pd.DataFrame({key: [False] for key in ("long_entry", "short_entry", "long_close", "short_close")})
    result = run_backtest(sample, no_signals, **settings)
    result["equity_curve"] = []
    result["pending_signals"] = None
    state = {"version": 1, "source": source, "settings": settings, "candles": [], "result": result}
    with _lock(path):
        if path.exists():
            raise FileExistsError(f"paper account already exists: {path}")
        _save(path, state)
    return state


def _claim_runner(path: Path, token: object) -> None:
    with _lock(path):
        key = str(path.resolve())
        if key in _active_runners:
            raise RuntimeError("paper account already has a running worker")
        _active_runners[key] = token


def _release_runner(path: Path, token: object) -> None:
    with _lock(path):
        key = str(path.resolve())
        if _active_runners.get(key) is token:
            del _active_runners[key]


def bind_market(path: str | Path, symbol: str) -> dict:
    """Bind an empty account to one supported futures market permanently."""
    if symbol not in ("BTCUSDT", "ETHUSDT"):
        raise ValueError("unsupported paper market")
    path = Path(path)
    with _lock(path):
        state = _read(path)
        existing = state.get("market_symbol")
        if existing == symbol:
            return state
        if existing or state["candles"]:
            raise ValueError("only an empty, unbound account can be bound")
        state["market_symbol"] = symbol
        _save(path, state)
        return state


def read_account(path: str | Path) -> dict:
    path = Path(path)
    with _lock(path):
        return _read(path)


def append_candles(path: str | Path, candles: pd.DataFrame, *, _runner_token: object = None) -> dict:
    """Append confirmed candles; repeated identical candles are idempotent.

    Changed or out-of-order historical bars are rejected; failed updates do
    not change the saved account. Bars whose interval has not elapsed according
    to the local system clock are rejected; callers must supply confirmed data.
    """
    path = Path(path)
    if not set(_COLUMNS).issubset(candles.columns) or candles.empty:
        raise ValueError("confirmed candles must have all OHLCV columns")
    with _lock(path):
        active = _active_runners.get(str(path.resolve()))
        if active is not None and active is not _runner_token:
            raise RuntimeError("paper account has a running worker")
        state = _read(path)
        if state.get("market_symbol") and _runner_token is None:
            raise RuntimeError("market-bound accounts accept only runner candles")
        existing = state["candles"]
        incoming = candles.loc[:, _COLUMNS].to_dict(orient="records")
        interval = state["settings"].get("interval_ms", 60_000)
        now_ms = time.time() * 1000
        if any(not isinstance(row["open_time"], (int, float)) or row["open_time"] + interval > now_ms for row in incoming):
            raise ValueError("unconfirmed candle: candle interval has not elapsed")
        previous = {row["open_time"]: row for row in existing}
        additions = []
        last = existing[-1]["open_time"] if existing else None
        for row in incoming:
            timestamp = row["open_time"]
            if timestamp in previous:
                if row != previous[timestamp]:
                    raise ValueError("previously processed candle changed")
                continue
            if last is not None and timestamp <= last:
                raise ValueError("new candles must follow the saved history")
            additions.append(row)
            previous[timestamp] = row
            last = timestamp
        if not additions:
            return state
        combined = pd.DataFrame(existing + additions, columns=_COLUMNS)
        signals = compile_pine(state["source"]).evaluate(combined)
        result = run_backtest(combined, signals, **state["settings"])
        state["candles"] = combined.to_dict(orient="records")
        state["result"] = result
        _save(path, state)
        return state
