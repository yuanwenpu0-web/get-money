"""Futures Context Module — term structure, ΔOI/price read and volume state.

Source: pine-collection/indicators/composite/futures_context_module/futures_context_module.pine
(WavesUnchained, MIT)

The Pine script pulls two futures legs with `request.security(..., lookahead_off)`
on the chart timeframe; here both legs (and the front contract's open interest and
volume) must be supplied as columns of the frame, one row per bar, so the values
are already the non-repainting ones:

- `front_close` / `next_close` — front-month and next-month settlement/last prices,
  quote currency. `spread = front_close - next_close`: positive is backwardation,
  negative contango.
- `front_open_interest` — open interest of the front contract, in contracts.
- `front_volume` — traded volume of the front contract, in contracts.

The spread is tracked by fast/slow EMAs; its slope and level classify the curve
trend. ΔOI against `oi_lookback` bars combined with the front price change gives
the classic positioning read (new longs / new shorts / short covering / long
liquidation / flat). Volume is compared against a 20-bar SMA.

Port notes:
- The `input.symbol` legs, the `syminfo.volumetype` tick detection and the
  daily-or-higher timeframe gate cannot be reproduced offline; "not real (tick)"
  degenerates to "no volume" when the volume column never exceeds 0.
- All the Pine status strings are kept verbatim, including "n/a" rows.

Dropped: the symbol inputs, the context table, its position option and the
"NO DATA FEED" status machinery.
"""
from __future__ import annotations

import pandas as pd

from ._pine import ema, integer, nz, ohlcv, sma


def futures_context_module(frame: pd.DataFrame, *, oi_lookback: int = 1,
                           term_ema_fast: int = 5, term_ema_slow: int = 20) -> pd.DataFrame:
    integer("oi_lookback", oi_lookback, 1, 20)
    integer("term_ema_fast", term_ema_fast, 2)
    integer("term_ema_slow", term_ema_slow, 5)
    front_close, next_close, front_oi, front_volume = ohlcv(
        frame, "front_close", "next_close", "front_open_interest", "front_volume")

    spread = front_close - next_close
    # A missing leg makes the spread na for that bar; Pine's ta.ema prints na there
    # too (the pandas EMA underneath would carry the last value forward instead).
    defined = spread.notna()
    spread_ema_fast = ema(spread, term_ema_fast).where(defined)
    spread_ema_slow = ema(spread, term_ema_slow).where(defined)
    spread_slope = (spread_ema_fast - nz(spread_ema_fast.shift(1))
                    ).where(spread_ema_fast.notna())

    term_structure = pd.Series("Flat", index=frame.index, dtype=object)
    term_structure[spread > 0] = "Backwardation"
    term_structure[spread < 0] = "Contango"
    term_structure[spread.isna()] = "n/a"

    curve_trend = pd.Series("flat", index=frame.index, dtype=object)
    rising = (spread_ema_fast > spread_ema_slow) & (spread_slope > 0)
    falling = (spread_ema_fast < spread_ema_slow) & (spread_slope < 0)
    curve_trend[rising] = "rising"
    curve_trend[falling] = "falling"
    undefined = spread_ema_fast.isna() | spread_ema_slow.isna() | spread_slope.isna()
    curve_trend[undefined] = "n/a"

    previous_oi = front_oi.shift(oi_lookback)
    front_price_change = front_close - front_close.shift(oi_lookback)
    oi_change = front_oi - previous_oi
    oi_change_pct = (oi_change / previous_oi * 100.0).where(previous_oi.ne(0))

    oi_price_state = pd.Series("OI flat", index=frame.index, dtype=object)
    oi_price_state[(oi_change > 0) & (front_price_change > 0)] = \
        "OI rising + Price rising -> new longs"
    oi_price_state[(oi_change > 0) & (front_price_change < 0)] = \
        "OI rising + Price falling -> new shorts"
    oi_price_state[(oi_change < 0) & (front_price_change > 0)] = \
        "OI falling + Price rising -> short covering"
    oi_price_state[(oi_change < 0) & (front_price_change < 0)] = \
        "OI falling + Price falling -> long liquidation"
    oi_price_state[oi_change.isna() | front_price_change.isna()] = "n/a"

    front_vol_avg = sma(front_volume, 20)
    volume_state = pd.Series("n/a", index=frame.index, dtype=object)
    if bool(front_volume.gt(0).any()):
        volume_state[front_vol_avg.notna()] = "below avg"
        volume_state[front_volume > front_vol_avg] = "above avg"
    else:
        volume_state[:] = "no volume"

    return pd.DataFrame({
        "spread": spread,
        "spread_ema_fast": spread_ema_fast,
        "spread_ema_slow": spread_ema_slow,
        "spread_slope": spread_slope,
        "term_structure": term_structure,
        "curve_trend": curve_trend,
        "front_price_change": front_price_change,
        "oi_change": oi_change,
        "oi_change_pct": oi_change_pct,
        "oi_price_state": oi_price_state,
        "front_vol_avg": front_vol_avg,
        "volume_state": volume_state,
    }, index=frame.index)
