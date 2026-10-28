# Anchored Session VWAP — SD Bands + Volume Signals

A production-grade Pine Script v5 indicator for TradingView. Session-anchored VWAP with standard deviation bands, volume-confirmed signals, dedup logic, and a status dashboard. Includes a companion strategy file for backtesting.

Designed primarily for US futures (NQ, ES) on intraday timeframes, but works on any instrument with reliable volume data.

---

## What's in this repo

| File | What it is |
|---|---|
| `SessionVWAP_SDBands.pine` | The main indicator — overlay on the chart with bands, signals, dashboard |
| `SessionVWAP_Strategy.pine` | Companion strategy version for TradingView's built-in backtester |
| `PUBLISHING_GUIDE.md` | Notes on TradingView publication (description, tags, etc.) |
| `screenshots/` | Example screenshots from real charts |

---

## Features

- **Session-anchored VWAP** — resets at session open (default 09:30–16:00 ET / US RTH), accumulates throughout the day
- **Three SD band levels** (1, 2, 3 SD) with toggle visibility per band
- **Filled zones** between adjacent bands for clear visual hierarchy
- **Color-coded VWAP line** — green when price holds above VWAP with above-average volume (bullish), red when below with selling pressure (bearish), default color when neutral / chop
- **Status dashboard** in chart corner — VWAP value, current state, distance from VWAP in SD units, volume regime, last signal age
- **Volume confirmation filter** — signals fire only when current bar volume exceeds the prior-bar SMA by a configurable multiplier
- **Dedup window** — suppress repeat signals in the same direction for N bars (default 10)
- **First-bar-of-session signal guard** — no false triggers from stale prior-session band values
- **Configurable session window + timezone** — RTH for futures, custom for crypto / forex
- **Buy/Sell signal markers** when price touches bands with volume + dedup confirmation
- **Built-in alerts** for signals and VWAP crosses (close-of-bar confirmed)
- **Companion strategy file** for backtest validation with three exit modes

---

## How it works

### The math

VWAP is a volume-weighted average:

```
VWAP = Σ(price × volume) / Σ(volume), accumulated from the session anchor.
```

The SD bands use volume-weighted standard deviation:

```
Variance = E[price²] − (E[price])²
```

…where `E[·]` denotes a volume-weighted expectation, not a simple mean.

Why volume-weighted SD instead of regular SD? Because price moves on high volume carry more meaning than equal-magnitude moves on low volume — high-volume bars effectively "vote louder" in the distribution. The resulting bands reflect where actual trading activity has clustered, not just where price has oscillated. On thin-tape sessions you'll often see narrower bands than a simple-stdev version would draw, because the formula correctly down-weights the low-volume noise.

The implementation wraps TradingView's built-in `ta.vwap()` with a custom session anchor — so the math is consistent with the platform's standard VWAP indicator. If you put the standard TradingView VWAP next to this one with matching settings, the lines overlap exactly. No surprises, no proprietary tweaks.

---

## Three trading plays this indicator supports

### 1. Mean reversion (the classic)

Wait for price to tag the 2 SD band with volume confirmation and dedup active. The expectation: price reverts back toward VWAP.

- **Stop-loss:** beyond the 3 SD band
- **First target:** the 1 SD band
- **Second target:** VWAP itself

Works best on ranging sessions — check the daily chart first to confirm the day isn't strongly trending.

### 2. Trend following with VWAP as dynamic support / resistance

When price holds above VWAP for an extended stretch and pulls back to the line (or to the 1 SD band on the trend side) without breaking through, that's a continuation entry. The bands act as soft S/R in the direction of the trend. Skip mean-reversion signals against the trend on these days — the volume filter helps but won't catch every trend-day false signal.

### 3. Volume divergence

When price pushes hard into a band but volume is below the filter threshold (signal does NOT fire), that's information too — the move lacks conviction. Watching the chart for "near-misses" on the 2 SD band where volume is light can flag exhaustion moves, particularly on the second or third tag in a session.

---

## Parameter tuning by instrument

The defaults (1.0 / 2.0 / 3.0 SD multipliers, 1.2× volume) are calibrated for NQ on 5-minute bars during US RTH. Other instruments and timeframes will want adjustments.

| Instrument | Timeframe | SD multipliers | Volume multiplier |
|---|---|---|---|
| NQ / ES | 5m (RTH) | 1.0 / 2.0 / 3.0 | 1.2× |
| ES / MES | 1m (RTH) | 1.5 / 2.5 / 3.5 | 1.5× |
| BTC / ETH | 15m (24/7) | 2.0 / 3.0 / 4.0 | 1.1× |
| EURUSD / forex | 1H | 1.0 / 2.0 / 3.0 | OFF (TV tick-volume unreliable for forex) |

These are starting points, not gospel. The right way to tune is to load the indicator on your instrument and timeframe, eyeball whether 2 SD touches happen at sensible reversal zones, and adjust the multiplier until they do.

---

## Installation

### On TradingView

1. Open any chart on TradingView
2. Go to **Pine Editor** (bottom panel)
3. Click the dropdown next to the script name → **Create new** → **Indicator**
4. Delete the default template
5. Copy the contents of `SessionVWAP_SDBands.pine` into the editor
6. Save (Ctrl+S), give it a name
7. Click **Add to chart**

For the strategy version, repeat the steps but choose **Strategy** instead of **Indicator** in step 3, then paste `SessionVWAP_Strategy.pine`.

---

## Notes

- The indicator is non-repainting on confirmed bars (signals are guarded by `barstate.isconfirmed`). VWAP itself updates intrabar as new volume arrives, which is standard `ta.vwap()` behavior — the calculated value at bar close is final.
- The volume filter uses `ta.sma(volume[1], lookback)` — meaning the prior bar's SMA, not including the current bar — to avoid look-ahead bias on the current bar's own volume.
- This is an indicator, not a strategy. Signals are educational markers — they do not constitute trade recommendations. Always combine with your own analysis, market context, and risk management.

---

## Screenshot
![Main view](screenshots-main-chart.png)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Serghey Dimitroglo** — automation and trading systems developer.

- Upwork: https://www.upwork.com/freelancers/sergheyd
- GitHub: https://github.com/serghey-d

If you find a bug or have a feature request, open an issue on this repo.
