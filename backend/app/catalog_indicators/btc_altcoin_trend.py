"""BTC/Altcoin Trend — per-timeframe BTC vs dominance trend and altcoin behaviour.

Source: pine-crypto-collection/lucas_btc_altcoins/script.pine
(Indicador de Tendências do BTC e Altcoins, author Lucas Pinheiro; no explicit
license header in the source file)

For each requested timeframe the script classifies two external series — BTC
and BTC dominance — as "up" / "down" / "lateral" and combines them into the
classic 1x/2x altcoin-behaviour matrix (BTC up + dominance up = alts 1x up and
BTC 2x up; BTC down + dominance up = alts 2x down; BTC up + dominance down =
alts 2x up; both down = alts 1x down and BTC 2x down).

Two trend modes:
* ``"1"`` — direction of the timeframe's own closes: the previous completed
  period's close against the one before it (the lookahead-free reading of the
  upstream ``request.security(..., close, close[1], lookahead_on)``).
* ``"2"`` — the timeframe's EMA1/EMA2 (EMA2 falls back through EMA3/EMA4/SMA)
  against that timeframe's close: both below close = up, both above = down,
  otherwise lateral.

Required input columns: ``close`` (the chart symbol, used by mode "2" on the
chart timeframe), ``btc_close`` and ``dominance_close`` aligned to the same
index. Timeframes coarser than the base bar spacing aggregate through
``_tf.htf_frame``/``calendar_frame`` and publish the previous completed period
via ``lift_series``/``lift_previous_calendar``; timeframes finer than the base
spacing are reported as ``"-"`` because no data exists below the base
resolution.

Dropped: the dashboard table, all colours, line/label drawing and the
exchange-symbol plumbing (data arrives as columns instead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import choice, ema, integer, ohlcv, sma
from ._tf import aligned_index, base_step_minutes, calendar_frame, htf_frame


def _lift_labels(labels: pd.Series, utc: pd.DatetimeIndex, rule: str) -> pd.Series:
    """Broadcast HTF labels onto base bars using the previous completed period."""
    minutes = int(pd.Timedelta(rule).total_seconds() // 60)
    period = utc.floor(f"{minutes}min")
    return pd.Series(labels.reindex(period - pd.Timedelta(minutes=minutes)).to_numpy(),
                     index=utc, dtype=object)


def _lift_calendar_labels(labels: pd.Series, utc: pd.DatetimeIndex, rule: str) -> pd.Series:
    period = utc.tz_localize(None).to_period(rule)
    previous = pd.PeriodIndex(period - 1, freq=rule)
    starts = pd.DatetimeIndex([p.start_time for p in previous]).tz_localize("UTC")
    return pd.Series(labels.reindex(starts).to_numpy(), index=utc, dtype=object)

_TIMEFRAMES = (("1M", "M"), ("1W", "W"), ("1D", "D"), ("4h", "240min"),
               ("1h", "60min"), ("15m", "15min"), ("5m", "5min"), ("1m", "1min"))


def _single_series_frame(series: pd.Series) -> pd.DataFrame:
    """Wrap one price series into the minimal OHLC shape the aggregators need."""
    return pd.DataFrame({"open": series, "high": series, "low": series,
                         "close": series}, index=series.index)


def _classify(series: pd.Series, trend_type: str, ema_lengths: tuple[int, int, int, int],
              ma_length: int) -> pd.Series:
    """Up/down/lateral labels on an HTF close series."""
    if trend_type == "1":
        previous = series.shift(1)
        older = series.shift(2)
        return pd.Series(np.where(previous > older, "up",
                                  np.where(previous < older, "down", "lateral")),
                         index=series.index)
    ref = series  # ta.ema(close, 1) == close
    ema1 = ema(series, ema_lengths[0])
    ema2 = ema(series, ema_lengths[1])
    ema3 = ema(series, ema_lengths[2])
    ema4 = ema(series, ema_lengths[3])
    ma5 = sma(series, ma_length)
    # Fallback chain: SMA -> EMA4 -> EMA3 -> EMA2 -> EMA1 -> 0.
    trend_ema2 = ma5.fillna(ema4).fillna(ema3).fillna(ema2).fillna(ema1).fillna(0.0)
    both_below = (ema1 < ref) & (trend_ema2 < ref)
    both_above = (ema1 > ref) & (trend_ema2 > ref)
    return pd.Series(np.where(both_below, "up",
                              np.where(both_above, "down", "lateral")),
                     index=series.index)


def _behaviour(btc_trend: str, dominance_trend: str) -> tuple[str, str]:
    if btc_trend == "up" and dominance_trend == "up":
        return "up_2x", "up_1x"
    if btc_trend == "down" and dominance_trend == "up":
        return "down_1x", "down_2x"
    if btc_trend == "up" and dominance_trend == "down":
        return "up_1x", "up_2x"
    if btc_trend == "down" and dominance_trend == "down":
        return "down_2x", "down_1x"
    return "-", "-"


def btc_altcoin_trend(
        frame: pd.DataFrame, *, trend_type: str = "2", ema1: int = 20, ema2: int = 50,
        ema3: int = 100, ema4: int = 200, ma1: int = 200,
        timeframes: tuple[str, ...] = ("1M", "1W", "1D", "4h", "1h", "15m", "5m", "1m"),
        chart_trend: bool = True) -> pd.DataFrame:
    choice("trend_type", trend_type, ("1", "2"))
    integer("ema1", ema1, 1)
    integer("ema2", ema2, 1)
    integer("ema3", ema3, 1)
    integer("ema4", ema4, 1)
    integer("ma1", ma1, 1)
    ohlcv(frame, "close", "btc_close", "dominance_close")

    utc = aligned_index(frame, 1)
    base = base_step_minutes(frame)
    lengths = (ema1, ema2, ema3, ema4)

    columns: dict[str, pd.Series] = {}
    for label in timeframes:
        rule = next((r for key, r in _TIMEFRAMES if key == label), None)
        if rule is None:
            raise ValueError(f"未知周期标签: {label}")
        minutes = (int(rule[:-3]) if rule.endswith("min")
                   else {"M": 40320, "W": 10080, "D": 1440}[rule])
        if rule in ("M", "W", "D") and base > 60:
            # Day/week/month aggregation is meaningless off an intraday feed
            # this coarse; report the Pine "no data" marker instead.
            placeholder = pd.Series("-", index=frame.index, dtype=object)
            columns[f"btc_trend_{label}"] = placeholder
            columns[f"dominance_trend_{label}"] = placeholder.copy()
            columns[f"btc_behavior_{label}"] = placeholder.copy()
            columns[f"alt_behavior_{label}"] = placeholder.copy()
            continue
        if minutes < base:
            placeholder = pd.Series("-", index=frame.index, dtype=object)
            columns[f"btc_trend_{label}"] = placeholder
            columns[f"dominance_trend_{label}"] = placeholder.copy()
            columns[f"btc_behavior_{label}"] = placeholder.copy()
            columns[f"alt_behavior_{label}"] = placeholder.copy()
            continue

        if rule in ("M", "W", "D"):
            btc_htf = calendar_frame(_single_series_frame(frame["btc_close"]), rule)["close"]
            dom_htf = calendar_frame(_single_series_frame(frame["dominance_close"]),
                                     rule)["close"]
            btc_lifted = _lift_calendar_labels(
                _classify(btc_htf, trend_type, lengths, ma1), utc, rule)
            dom_lifted = _lift_calendar_labels(
                _classify(dom_htf, trend_type, lengths, ma1), utc, rule)
        else:
            btc_htf = htf_frame(_single_series_frame(frame["btc_close"]), minutes)["close"]
            dom_htf = htf_frame(_single_series_frame(frame["dominance_close"]),
                                minutes)["close"]
            btc_lifted = _lift_labels(_classify(btc_htf, trend_type, lengths, ma1),
                                      utc, rule)
            dom_lifted = _lift_labels(_classify(dom_htf, trend_type, lengths, ma1),
                                      utc, rule)
        btc_label = btc_lifted.fillna("-").astype(object)
        dom_label = dom_lifted.fillna("-").astype(object)
        columns[f"btc_trend_{label}"] = btc_label
        columns[f"dominance_trend_{label}"] = dom_label
        pairs = [_behaviour(str(b), str(d)) for b, d in zip(btc_label, dom_label)]
        columns[f"btc_behavior_{label}"] = pd.Series([p[0] for p in pairs], index=utc,
                                                     dtype=object)
        columns[f"alt_behavior_{label}"] = pd.Series([p[1] for p in pairs], index=utc,
                                                     dtype=object)

    output = pd.DataFrame(columns, index=frame.index)

    # Chart-timeframe trend in mode "2" uses the base frame's own EMAs.
    if chart_trend and trend_type == "2":
        close = frame["close"].astype(float)
        ema1_series = ema(close, ema1)
        trend_ema2 = (sma(close, ma1).fillna(ema(close, ema4)).fillna(ema(close, ema3))
                      .fillna(ema(close, ema2)).fillna(ema1_series).fillna(0.0))
        both_below = (ema1_series < close) & (trend_ema2 < close)
        both_above = (ema1_series > close) & (trend_ema2 > close)
        output["btc_trend_chart"] = pd.Series(
            np.where(both_below, "up", np.where(both_above, "down", "lateral")),
            index=frame.index, dtype=object)
    return output
