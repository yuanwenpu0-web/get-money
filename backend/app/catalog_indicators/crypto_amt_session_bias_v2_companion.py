"""Crypto AMT Session Bias v2 Companion — session levels, sweeps and reclaim bias.

Source: pine-crypto-collection/latifangren_crypto_amt/indicator/crypto_amt_session_bias_v2_companion.pine

The script tracks three exchange sessions (Asia, London, New York). While a
session runs it accumulates open/high/low/mid plus the opening range (the first
`opening_range_minutes` of the session); when a session ends its levels become
the "previous session" target that later bars sweep and reclaim against. The
bias state flips to -1 after a high sweep is reclaimed back below the previous
session high and to +1 after a low sweep reclaims above the previous session
low.

Sessions are evaluated on **UTC bar-open hours** (the Pine default session
timezone is Etc/UTC): the session strings "HHMM-HHMM" are parsed to UTC minutes
of day and a bar belongs to a session when its open time falls inside, with
wrap-around support for ranges crossing midnight. Session boundaries are
therefore exact only for bar opens inside the session window.

Dropped: the dashboard table, plot/colour plumbing, the background tint, bar
colouring, the markers and the alert plumbing (including the
`confirmAlertsOnClose` bar-state gate, which only affects real-time alerts).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ._pine import boolean, integer, ohlcv

_SESSION_RE = re.compile(r"^(\d{2})(\d{2})-(\d{2})(\d{2})$")

_SESSIONS = (("Asia", "asia"), ("London", "london"), ("New York", "new_york"))


def _session_minutes(name: str, value: str) -> tuple[int, int]:
    match = _SESSION_RE.match(value)
    if not match:
        raise ValueError(f"参数 {name} 必须是 HHMM-HHMM 格式的时段字符串")
    start = int(match.group(1)) * 60 + int(match.group(2))
    end = int(match.group(3)) * 60 + int(match.group(4))
    if int(match.group(1)) > 23 or int(match.group(3)) > 23 or int(match.group(2)) > 59 or int(match.group(4)) > 59:
        raise ValueError(f"参数 {name} 超出允许范围")
    return start, end


def _in_session(minute_of_day: pd.Series, start: int, end: int) -> pd.Series:
    if start == end:
        return pd.Series(False, index=minute_of_day.index)
    if start < end:
        return (minute_of_day >= start) & (minute_of_day < end)
    return (minute_of_day >= start) | (minute_of_day < end)


def crypto_amt_session_bias_v2_companion(
        frame: pd.DataFrame, *,
        show_asia: bool = True,
        asia_session: str = "0000-0800",
        show_london: bool = True,
        london_session: str = "0700-1600",
        show_new_york: bool = True,
        new_york_session: str = "1300-2200",
        opening_range_minutes: int = 60,
        allow_same_bar_reclaim: bool = True) -> pd.DataFrame:
    boolean("show_asia", show_asia)
    _session_minutes("asia_session", asia_session)
    boolean("show_london", show_london)
    _session_minutes("london_session", london_session)
    boolean("show_new_york", show_new_york)
    _session_minutes("new_york_session", new_york_session)
    integer("opening_range_minutes", opening_range_minutes, 1, 240)
    boolean("allow_same_bar_reclaim", allow_same_bar_reclaim)
    open_, high, low, close = ohlcv(frame, "open", "high", "low", "close")

    index = frame.index
    minute_of_day = (index.hour * 60 + index.minute).to_series(index=index)
    windows = {
        "asia": (show_asia, *_session_minutes("asia_session", asia_session)),
        "london": (show_london, *_session_minutes("london_session", london_session)),
        "new_york": (show_new_york, *_session_minutes("new_york_session", new_york_session)),
    }
    inside = {name: _in_session(minute_of_day, start, end) if enabled
              else pd.Series(False, index=index)
              for name, (enabled, start, end) in windows.items()}

    opens = open_.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    times = index.astype("int64").to_numpy()  # ns since epoch

    # Per-session running state.
    state = {name: {"open": np.nan, "high": np.nan, "low": np.nan, "start": None,
                    "or_high": np.nan, "or_low": np.nan} for _, name in _SESSIONS}
    previous = {"name": "None", "open": np.nan, "high": np.nan, "low": np.nan}
    current_name = "None"
    high_swept = low_swept = high_reclaimed = low_reclaimed = False
    high_sweep_bar = low_sweep_bar = -1
    bias_state = 0

    columns = {
        "in_asia": np.zeros(len(index), dtype=bool),
        "in_london": np.zeros(len(index), dtype=bool),
        "in_new_york": np.zeros(len(index), dtype=bool),
        "active_session": np.empty(len(index), dtype=object),
        "session_open": np.full(len(index), np.nan),
        "session_high": np.full(len(index), np.nan),
        "session_low": np.full(len(index), np.nan),
        "session_mid": np.full(len(index), np.nan),
        "opening_range_high": np.full(len(index), np.nan),
        "opening_range_low": np.full(len(index), np.nan),
        "opening_range_mid": np.full(len(index), np.nan),
        "previous_session": np.empty(len(index), dtype=object),
        "previous_session_open": np.full(len(index), np.nan),
        "previous_session_high": np.full(len(index), np.nan),
        "previous_session_low": np.full(len(index), np.nan),
        "previous_session_mid": np.full(len(index), np.nan),
        "sweep_high": np.zeros(len(index), dtype=bool),
        "sweep_low": np.zeros(len(index), dtype=bool),
        "reclaim_high": np.zeros(len(index), dtype=bool),
        "reclaim_low": np.zeros(len(index), dtype=bool),
        "bias_flip": np.zeros(len(index), dtype=bool),
        "bias_state": np.zeros(len(index), dtype="int64"),
    }

    or_span = opening_range_minutes * 60 * 1_000_000_000
    for bar in range(len(index)):
        was_in = {name: bool(inside[name].iloc[bar - 1]) if bar else False
                  for _, name in _SESSIONS}
        is_in = {name: bool(inside[name].iloc[bar]) for _, name in _SESSIONS}

        # 1. Session accumulation.
        for label, name in _SESSIONS:
            book = state[name]
            if is_in[name] and not was_in[name]:
                book.update({"open": opens[bar], "high": highs[bar], "low": lows[bar],
                             "start": times[bar], "or_high": highs[bar], "or_low": lows[bar]})
            elif is_in[name]:
                book["high"] = highs[bar] if np.isnan(book["high"]) else max(book["high"], highs[bar])
                book["low"] = lows[bar] if np.isnan(book["low"]) else min(book["low"], lows[bar])
                if book["start"] is not None and times[bar] < book["start"] + or_span:
                    book["or_high"] = (highs[bar] if np.isnan(book["or_high"])
                                       else max(book["or_high"], highs[bar]))
                    book["or_low"] = (lows[bar] if np.isnan(book["or_low"])
                                      else min(book["or_low"], lows[bar]))

        # 2. On session end the finished levels become the previous target.
        previous_changed = False
        for label, name in _SESSIONS:
            if was_in[name] and not is_in[name]:
                book = state[name]
                if not np.isnan(book["high"]) and not np.isnan(book["low"]):
                    previous = {"name": label, "open": book["open"],
                                "high": book["high"], "low": book["low"]}
                    previous_changed = True

        # 3. Active session selection (New York > London > Asia).
        if is_in["new_york"]:
            active = "new_york"
        elif is_in["london"]:
            active = "london"
        elif is_in["asia"]:
            active = "asia"
        else:
            active = None

        # 4. State reset on session change or new previous target.
        if (active or "None") != current_name or previous_changed:
            high_swept = low_swept = high_reclaimed = low_reclaimed = False
            high_sweep_bar = low_sweep_bar = -1
            bias_state = 0
            current_name = active or "None"

        in_active = active is not None
        book = state[active] if active else None
        has_previous = not np.isnan(previous["high"]) and not np.isnan(previous["low"])

        # 5. Sweeps of the previous session levels.
        sweep_high = in_active and has_previous and not high_swept and highs[bar] > previous["high"]
        sweep_low = in_active and has_previous and not low_swept and lows[bar] < previous["low"]
        if sweep_high:
            high_swept, high_sweep_bar = True, bar
        if sweep_low:
            low_swept, low_sweep_bar = True, bar

        # 6. Reclaims back inside the previous levels.
        high_ready = (high_swept and not high_reclaimed
                      and (allow_same_bar_reclaim or bar > high_sweep_bar))
        low_ready = (low_swept and not low_reclaimed
                     and (allow_same_bar_reclaim or bar > low_sweep_bar))
        reclaim_high = in_active and high_ready and closes[bar] < previous["high"]
        reclaim_low = in_active and low_ready and closes[bar] > previous["low"]
        bias_flip = False
        if reclaim_high:
            high_reclaimed = True
            bias_flip = bias_state != -1
            bias_state = -1
        if reclaim_low:
            low_reclaimed = True
            bias_flip = bias_state != 1
            bias_state = 1

        columns["in_asia"][bar] = is_in["asia"]
        columns["in_london"][bar] = is_in["london"]
        columns["in_new_york"][bar] = is_in["new_york"]
        columns["active_session"][bar] = {"asia": "Asia", "london": "London",
                                          "new_york": "New York"}.get(active, "None")
        if book is not None:
            columns["session_open"][bar] = book["open"]
            columns["session_high"][bar] = book["high"]
            columns["session_low"][bar] = book["low"]
            columns["session_mid"][bar] = (book["high"] + book["low"]) * 0.5
            columns["opening_range_high"][bar] = book["or_high"]
            columns["opening_range_low"][bar] = book["or_low"]
            columns["opening_range_mid"][bar] = (book["or_high"] + book["or_low"]) * 0.5
        columns["previous_session"][bar] = previous["name"]
        columns["previous_session_open"][bar] = previous["open"]
        columns["previous_session_high"][bar] = previous["high"]
        columns["previous_session_low"][bar] = previous["low"]
        columns["previous_session_mid"][bar] = (previous["high"] + previous["low"]) * 0.5 \
            if not np.isnan(previous["high"]) else np.nan
        columns["sweep_high"][bar] = sweep_high
        columns["sweep_low"][bar] = sweep_low
        columns["reclaim_high"][bar] = reclaim_high
        columns["reclaim_low"][bar] = reclaim_low
        columns["bias_flip"][bar] = bias_flip
        columns["bias_state"][bar] = bias_state

    return pd.DataFrame(columns, index=index).astype({"bias_state": "Int64"})
