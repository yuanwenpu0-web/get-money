"""Explicit, local-only polling runner for Binance USD-M 1-minute candles.

A next-open fill is replayed only once that next candle has closed; this is
not a real-time execution simulator. Runner state is never auto-started.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .paper import _claim_runner, _lock, _release_runner, append_candles, bind_market, read_account

_INTERVAL = 60_000
_registry_lock = threading.Lock()
_running: set[str] = set()


class BinancePublicCandles:
    """Read-only client: no authentication or order endpoints."""
    base_url = "https://fapi.binance.com"

    def _get(self, path: str, params: dict | None = None):
        query = "?" + urllib.parse.urlencode(params) if params else ""
        with urllib.request.urlopen(self.base_url + path + query, timeout=10) as response:
            return json.load(response)

    def server_time(self) -> int:
        return int(self._get("/fapi/v1/time")["serverTime"])

    def klines(self, symbol: str, start_ms: int, limit: int):
        return self._get("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": limit})


class PaperRunner:
    """An explicitly started polling worker; one worker per account per process."""
    def __init__(self, path: str | Path, symbol: str, *, client=None, poll_seconds: float = 5, max_backfill: int = 10_000):
        if symbol not in ("BTCUSDT", "ETHUSDT"):
            raise ValueError("unsupported paper market")
        if poll_seconds <= 0 or max_backfill <= 0:
            raise ValueError("poll interval and max_backfill must be positive")
        self.path = Path(path).resolve()
        self.symbol = symbol
        self.client = client or BinancePublicCandles()
        self.poll_seconds = poll_seconds
        self.max_backfill = max_backfill
        self._event = threading.Event()
        self._thread = None
        self._state_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._token = object()
        self._status = "stopped"
        self._error = None

    def status(self) -> dict:
        with self._state_lock:
            return {"status": self._status, "error": self._error, "symbol": self.symbol}

    def poll_once(self) -> int:
        """Append all closed candles since the account's last bar, or raise."""
        with self._poll_lock:
            return self._poll_once()

    def _poll_once(self) -> int:
        state = read_account(self.path)
        if state.get("market_symbol") != self.symbol:
            raise ValueError("paper account market does not match runner")
        if state["settings"].get("interval_ms", _INTERVAL) != _INTERVAL:
            raise ValueError("runner requires a 1-minute account")
        now = self.client.server_time()
        latest = (now // _INTERVAL - 1) * _INTERVAL
        expected = state["candles"][-1]["open_time"] + _INTERVAL if state["candles"] else latest
        if expected > latest:
            return 0
        count = (latest - expected) // _INTERVAL + 1
        if count > self.max_backfill:
            raise ValueError("candle backlog exceeds maximum backfill")
        rows = []
        while expected <= latest:
            page = self.client.klines(self.symbol, expected, min(1000, count - len(rows)))
            if not isinstance(page, list) or not page:
                raise ValueError("unable to recover missing candles")
            before = expected
            for row in page:
                if len(row) < 7:
                    raise ValueError("invalid Binance candle")
                timestamp = int(row[0])
                if timestamp > latest:
                    break
                if timestamp != expected or int(row[6]) != timestamp + _INTERVAL - 1:
                    raise ValueError("unable to recover missing or invalid candles")
                rows.append(dict(open_time=timestamp, open=float(row[1]), high=float(row[2]),
                                 low=float(row[3]), close=float(row[4]), volume=float(row[5])))
                expected += _INTERVAL
            if expected == before:
                raise ValueError("unable to recover missing candles")
        append_candles(self.path, pd.DataFrame(rows), _runner_token=self._token)
        return len(rows)

    def start(self) -> None:
        """Start once; no network request or order is placed synchronously."""
        with _registry_lock:
            if str(self.path) in _running:
                raise RuntimeError("paper account already has a running worker")
            with _lock(self.path):
                state = bind_market(self.path, self.symbol)
                if state["settings"].get("interval_ms", _INTERVAL) != _INTERVAL:
                    raise ValueError("runner requires a 1-minute account")
                _claim_runner(self.path, self._token)
            _running.add(str(self.path))
            self._event.clear()
            with self._state_lock:
                self._status, self._error = "running", None
            self._thread = threading.Thread(target=self._run, name="paper-candle-poller", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                _running.discard(str(self.path))
                _release_runner(self.path, self._token)
                with self._state_lock:
                    self._status = "stopped"
                raise

    def _run(self) -> None:
        error = None
        try:
            while not self._event.is_set():
                self.poll_once()
                if self._event.wait(self.poll_seconds):
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with _registry_lock:
                _running.discard(str(self.path))
                _release_runner(self.path, self._token)
                with self._state_lock:
                    self._status = "error" if error else "stopped"
                    self._error = error

    def stop(self, timeout: float = 15) -> bool:
        """Request shutdown; false means an in-flight request has not finished."""
        self._event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)
        return not thread or not thread.is_alive()
