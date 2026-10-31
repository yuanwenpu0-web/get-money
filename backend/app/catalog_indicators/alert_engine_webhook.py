"""Alert Engine — webhook-ready EMA-cross signal layer.

Source: pine-crypto-collection/davlacombe_alert/alert-engine-webhook.pine

Author: davlacombe (folder `davlacombe_alert`). The upstream script carries no
licence header, so only the formulas and the payload layout it publishes are
reproduced here. Nothing is sent anywhere: the port never opens a connection and
returns the payload as a data column instead of calling `alert()`.

Two layers are exposed. `raw_bull` / `raw_bear` are the bare `ta.crossover` /
`ta.crossunder` of the two EMAs, `bull_signal` / `bear_signal` additionally pass
the bar gate and the once-per-bar guard (`firedOnBar` upstream, so the same bar
can never fire twice). `message` holds the payload of the bar that fired and an
empty string elsewhere, rendered as JSON or as the plain-text variant selected by
`message_format`.

Discarded presentation: the two EMA `plot`s and the `plotshape` triangles.

Input must carry a numeric `close` column on a bar-open index; no other series is
read and no request is made.

`confirm_only` mirrors `input.bool("Fire only on closed bars")`. Upstream gates
on `barstate.isconfirmed`, which is always true for a closed historical bar, and
this port only ever replays closed bars, so the gate never blocks a bar here; the
input is kept for parity and still validated.

`symbol`, `exchange` and `timeframe` are the `syminfo.ticker`, `syminfo.prefix`
and `timeframe.period` fields of the payload. The port cannot infer them from a
bare OHLCV frame, so they are optional strings defaulting to "".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ._pine import boolean, choice, crossover, crossunder, ema, flags, integer, ohlcv


def _text(name: str, value) -> str:
    if not isinstance(value, str):
        raise ValueError(f"参数 {name} 必须为字符串")
    return value


def _price_text(value: float) -> str:
    """`str.tostring(close, format.mintick)` without knowing the symbol tick.

    The trailing zeros of a fixed tick precision are trimmed, so 11.0 renders as
    "11" and 103.55 stays "103.55".
    """
    if not np.isfinite(value):
        return ""
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def _stamps(index: pd.Index) -> list[str]:
    """`str.format_time(time, "yyyy-MM-dd'T'HH:mm:ss", "UTC")` per bar."""
    if not isinstance(index, pd.DatetimeIndex):
        return [""] * len(index)
    utc = index.tz_convert("UTC") if index.tz is not None else index
    return utc.strftime("%Y-%m-%dT%H:%M:%S").tolist()


def alert_engine_webhook(frame: pd.DataFrame, *, fast_len: int = 21, slow_len: int = 50,
                         message_format: str = "JSON", strategy_id: str = "ema-cross",
                         confirm_only: bool = True, symbol: str = "", exchange: str = "",
                         timeframe: str = "") -> pd.DataFrame:
    integer("fast_len", fast_len, 1)
    integer("slow_len", slow_len, 1)
    choice("message_format", message_format, ("JSON", "Plain text"))
    _text("strategy_id", strategy_id)
    boolean("confirm_only", confirm_only)
    _text("symbol", symbol)
    _text("exchange", exchange)
    _text("timeframe", timeframe)
    (close,) = ohlcv(frame, "close")

    fast, slow = ema(close, fast_len), ema(close, slow_len)
    raw_bull, raw_bear = crossover(fast, slow), crossunder(fast, slow)

    # `gate = confirmOnly ? barstate.isconfirmed : true`; isconfirmed is true on
    # every closed bar, and this port only sees closed bars.
    bull, bear = flags(raw_bull), flags(raw_bear)

    count = len(frame)
    fire_bull = np.zeros(count, dtype=bool)
    fire_bear = np.zeros(count, dtype=bool)
    bull_flags, bear_flags = bull.to_numpy(), bear.to_numpy()
    fired_on_bar = -1
    for position in range(count):
        if not (bull_flags[position] or bear_flags[position]):
            continue
        if fired_on_bar == position:  # var int firedOnBar duplicate guard
            continue
        if bull_flags[position]:
            fire_bull[position] = True
        else:
            fire_bear[position] = True
        fired_on_bar = position

    stamps = _stamps(frame.index)
    prices = [_price_text(value) for value in close.to_numpy(dtype=float)]
    messages = [""] * count
    for position in range(count):
        if not (fire_bull[position] or fire_bear[position]):
            continue
        side = "long" if fire_bull[position] else "short"
        price, stamp = prices[position], stamps[position]
        if message_format == "JSON":
            messages[position] = ("{"
                                  f'"strategy":{json.dumps(strategy_id)},'
                                  f'"symbol":{json.dumps(symbol)},'
                                  f'"exchange":{json.dumps(exchange)},'
                                  f'"tf":{json.dumps(timeframe)},'
                                  f'"side":"{side}",'
                                  f'"price":{price},'
                                  f'"time":{json.dumps(stamp + "Z")}'
                                  "}")
        else:
            messages[position] = f"{symbol} {timeframe} - {side} @ {price} ({stamp} UTC)"

    return pd.DataFrame({
        "ema_fast": fast,
        "ema_slow": slow,
        "raw_bull": raw_bull,
        "raw_bear": raw_bear,
        "bull_signal": pd.Series(fire_bull, index=frame.index),
        "bear_signal": pd.Series(fire_bear, index=frame.index),
        "message": pd.Series(messages, index=frame.index, dtype=object),
    }, index=frame.index)
