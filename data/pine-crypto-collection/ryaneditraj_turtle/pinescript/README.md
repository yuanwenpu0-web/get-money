# Pine scripts

## Nifty Mean Reversion + Pivots + Options Levels

**File:** [`nifty_mean_reversion_pivots.pine`](./nifty_mean_reversion_pivots.pine)  
**Strategy name:** `Nifty MR + Pivots + Opt Levels`

Intraday mean-reversion for Nifty: fade VWAP/BB + RSI extremes near daily pivots, with HTF trend filter, reversal candles, RVOL, and **manual** max-pain / put-wall / call-wall / gamma-flip levels (paste daily from Sensibull/Opstra — TV has no NSE Greeks). Prefer **Nifty futures** for real volume; 5m/15m chart; NSE session defaults IST.

---

# Turtle / Dennis Donchian Breakout (Pine v5)

Paste-ready TradingView strategy implementing classic Richard Dennis / Turtle Traders rules with disciplined risk controls.

**File:** [`turtle_dennis_breakout.pine`](./turtle_dennis_breakout.pine)  
**Strategy name in TradingView:** `Turtle Dennis Breakout (S1/S2)`

## Thesis

Trend-following breakout system: enter when price clears a Donchian channel, size each unit so volatility (ATR / “N”) defines risk, hard-stop at **2N**, optionally pyramid into winners, and exit on a shorter opposite channel and/or ATR trail. The edge is in capturing fat-tailed trends — not predicting tops/bottoms.

## Recommended markets & timeframes

| Prefer | Avoid |
|--------|--------|
| Liquid futures (ES, NQ, CL, GC), FX majors, BTC/ETH | Illiquid alts, thin stocks |
| Daily chart (primary); 4H on majors | Ultra-low TFs (noise destroys expectancy) |

## Suggested defaults

| Parameter | Default | Notes |
|-----------|---------|--------|
| System 1 entry / exit | 20 / 10 | Classic fast Turtle |
| System 2 entry / exit | 55 / 20 | Classic slow Turtle |
| Hard stop | 2N | Always attached via `strategy.exit` |
| Risk % / unit | 1% | Sized to stop; qty clamped by max notional % |
| Max units | 4 | Add every 0.5N (same entry ID) |
| Max heat | 6% | Caps simultaneous unit risk |
| Max notional % / unit | 25% | Prevents insane qty on low-ATR symbols |
| Trend MA | EMA 200 | Long above / short below (ON) |
| ADX filter | Off | Optional chop filter; off so Daily liquid names trade |
| S1 skip-if-last-won | On | Classic System 1 only; S2 always allowed |
| Exit mode | Donchian | Primary exit = opposite channel via `strategy.close` |
| Commission | 0.03% | Adjust to your venue |
| Slippage | 2 ticks | Set in strategy() |

**v2 exit fix:** An earlier multi-entry-ID build could leave stops unattached and fail to close on Donchian, producing a single multi-decade hold. This version uses one `"Long"` / `"Short"` ID, a always-on 2N stop, and explicit Donchian `strategy.close`.

## How to backtest fairly

1. Open the `.pine` file → copy all → TradingView Pine Editor → Add to chart.
2. Set commission and slippage to match your broker/exchange.
3. Use enough history to include bull, bear, and chop regimes.
4. Do **not** optimize entry lengths per symbol until you have a walk-forward plan.
5. Judge **expectancy, max DD, profit factor, and exposure** — not just net profit.
6. Prefer `process_orders_on_close` behavior (already set) so signals use confirmed bars.

## Honest caveats

- Turtle-style systems **bleed in ranges**; optional ADX/ATR filters reduce but do not eliminate that.
- Past performance ≠ future results. Nothing here guarantees profitability.
- Pyramiding magnifies both home runs and adverse excursions — keep `maxUnits` / `maxHeatPct` conservative until you trust the stats.
- Crypto/FX point values and TV’s qty model differ from futures contract dollars; verify unit size on your symbol before going live.
- Always paper-trade or use tiny size first.
- Expect **many** round-trips on Daily SPY/ES/BTC over 2000–2026 (dozens+), not one open trade — if you still see a decades-long hold, re-paste the script and reset the chart.

## How to re-test (TradingView)

1. Copy `turtle_dennis_breakout.pine` → Pine Editor → Save → Add to chart.
2. Symbol: **SPY**, **ES1!**, or **BTCUSDT**; timeframe **Daily**.
3. Leave defaults (EMA200 ON, ADX OFF, Exit=Donchian).
4. Strategy Tester → List of Trades: you should see repeated entries/exits; Donchian exits should close within weeks/months of a failed trend, not decades.
5. Optional: set Exit mode to **Both** for Donchian + ATR trail ratchet.

## Why this vs a simple ATR breakout

- Separate **S1 / S2** with classic skip-if-won on S1 only  
- **Volatility-normalized** sizing from equity + ATR, with notional clamp  
- **2N hard stop** always on + Donchian / ATR-trail exits  
- **Pyramiding** on one entry ID with 0.5N adds and stop ratchet  
- **Heat cap**, trend MA, optional ADX / ATR percentile / session filters  
- No look-ahead on channels (`high[1]` / `low[1]`)
