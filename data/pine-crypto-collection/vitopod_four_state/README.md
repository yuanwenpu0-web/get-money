# Bullish Lines — Four-State Classification

A free Pine Script for TradingView. Companion to the book *Reading the Lines* by Simone Pozzi (2026).

This script implements the four-state classification vocabulary the book teaches: it computes four smoothed moving averages on the HL2 midpoint of each daily candle and labels the chart, every day, as **Bullish**, **Neutral-Yellow**, **Neutral-Blue**, or **Bearish**.

It is open-source. The methods are not the moat.

---

## What this script is

- Four SMMA lines on the HL2 midpoint, configurable lengths.
- A four-state classification (Bullish / Neutral-Yellow / Neutral-Blue / Bearish) rendered as line colour, optional background tint, and optional fill between the fast and slow lines.
- Alert conditions on each state transition.
- MIT licence. Use it freely. Fork it. Modify it.

## What this script is not

- **Not the Bullish Lines platform.** The platform calibrates the SMMA lengths per coin, publishes a daily classification across a tracked universe of major crypto assets, and stamps every reading to the Bitcoin blockchain. None of that lives in this script. If you want the calibrated lengths and the daily reading without doing the work yourself, the platform is at [bullishlines.com](https://bullishlines.com/today).
- **Not financial advice.** This is a non-personalised classification tool. The classification describes what the chart is currently doing in the four-state vocabulary. It does not tell you what to do with your money. Past results do not predict future performance.
- **Not a calibrated indicator.** The classification algorithm in this script is exactly the platform's. What is *not* here is the calibration: the per-coin SMMA lengths the platform finds by walk-forward optimisation. The script ships with the four length inputs defaulted to zero so that you have to choose your own.

## Install on TradingView

1. Open any chart on [TradingView](https://www.tradingview.com).
2. Open the **Pine Editor** at the bottom of the page.
3. Paste the contents of [`bullish-lines-four-state.pine`](./bullish-lines-four-state.pine).
4. Click **Add to chart**.

When you first add it, the script shows a gray placeholder and a label asking you to configure the four SMMA lengths. Once you set them, the indicator draws four SMMA lines colour-coded by the current state.

## Configure the SMMA lengths

The script's four SMMA length inputs default to **0**. Until you set all four to positive integers (≥ 2), the script will show a gray placeholder on the chart and a label asking you to fill the inputs in. This is deliberate: the platform's calibrated values are not bundled with this script, and you should choose your own.

Some public starting points you can try (none of these are the platform's calibrated lengths):

- **Conventional textbook stack**: 20 / 50 / 100 / 200
- **Tighter (more reactive, more noise)**: 8 / 13 / 21 / 34
- **Looser (less reactive, less noise)**: 50 / 100 / 150 / 250

The Bullish Lines platform uses different, per-coin-calibrated lengths derived from walk-forward optimisation on each asset's historical data. Those lengths are not published in this script (or anywhere else in the public material) because finding good lengths is part of the work the platform does for its subscribers. The book's Chapter 6 explains the calibration method and why the discipline of having chosen the lengths honestly matters more than the specific values.

## Reproducing the platform's classification

The state-classification logic in this script is the **exact same algorithm** the platform uses. The formula is documented in the header of the script and in the source-of-truth file (`src/lib/indicator.ts` in the bullishlines.com codebase):

```
mixed = (v1 < m1) != (v1 < v2) OR (m2 < v2) != (v1 < v2)

Bullish        = !mixed AND v1 >  v2
Bearish        = !mixed AND v1 <= v2
Neutral-Yellow =  mixed AND v1 >  v2
Neutral-Blue   =  mixed AND v1 <= v2
```

If you set the four SMMA lengths to the platform's exact values and leave the source on **HL2**, this script reproduces the platform's daily classification on any chart with sufficient history (roughly 5× the slowest length, e.g. ~150 daily bars).

**Numerical note**: this script uses Pine's standard Wilder smoothing (`ta.rma`), which seeds at bar (length − 1) with the SMA of the first *length* bars. The platform seeds at bar 0 with the same SMA value and then applies the recursive formula every bar. The two converge with sufficient history; classifications can differ by one bar at marginal flip transitions where the lines are nearly touching.

## The four states

| State | Colour | What it means |
|-------|--------|---------------|
| **Bullish** | warm yellow | Price is above the slowest line and the four lines are stacked from fast (top) to slow (bottom). The chart is in an established up-trend. |
| **Neutral-Yellow** | paler amber | Transitional. The chart was Bearish recently and is warming. The lines are tightening or beginning to rearrange. |
| **Neutral-Blue** | sky blue | Transitional. The chart was Bullish recently and is cooling. The lines are tightening or beginning to rearrange in the other direction. |
| **Bearish** | saturated blue | Price is below the slowest line and the four lines are stacked from fast (bottom) to slow (top). The chart is in an established down-trend. |

The colour vocabulary is not arbitrary. Green and red belong to the candles (rising bars, falling bars), and that association should stay where it is. Yellow and blue carry the established colour grammar of this class of trend-line system, so anyone who has spent time with moving-average classification work will already read the gradient correctly. The full rationale is in Chapter 3 of the book.

## Alerts

Four `alertcondition` entries fire on each state transition. Use TradingView's alert panel to subscribe to whichever flips you care about.

## Honest limitations

This is a four-state classification on a single chart, computed in your browser. It does not give you:

- Cross-asset confluence (how many coins in the universe are in each state).
- The macro layer (17 macro indicators across four groups, classified into a regime backdrop).
- Per-asset historical statistics (win rate, expectancy, time-in-state distributions).
- The cryptographic record (daily manifests stamped to the Bitcoin blockchain).
- Per-asset calibrated SMMA lengths.

If you want any of those, the platform is where they live. The book explains why each of them matters.

## License

MIT licence. The full notice is in the header of the Pine Script.

## Links

- 📘 The book: *Reading the Lines* by Simone Pozzi
- 🌐 The platform: [bullishlines.com](https://bullishlines.com)
- 🔍 Verify any historical classification: `bullishlines.com/proof/verify/YYYY-MM-DD`
- 📜 Methodology: `bullishlines.com/proof/methodology`

---

*Past results do not predict future performance. This is a non-personalised market-data and algorithmic-classification tool. It is not financial advice.*
