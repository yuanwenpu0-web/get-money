# Crypto Intermarket Correlation (CIC)

**Crypto Intermarket Correlation (CIC)** is a TradingView indicator that visualizes how closely BTC and ETH are moving with a benchmark market (for example `CME_MINI:NQ1!`, `SPY`, `DXY`, or `TVC:GOLD`).

Use it to quickly assess whether crypto is behaving like a high-beta extension of the benchmark, drifting into a transitional regime, or structurally decoupling.

## What it shows

- Rolling Pearson correlation for `BTC` vs the benchmark
- Rolling Pearson correlation for `ETH` vs the benchmark
- A regime label (Sync / Loose / Decouple, computed from the average of BTC and ETH correlation, with optional smoothing) and optional background highlighting

## How it works

### 1. Correlation mode

CIC supports two correlation inputs:

- **Returns (recommended):** correlation of log returns (co-movement of price changes)
- **Price:** correlation of closing prices (trend co-direction)

### 2. Regime classification

The indicator classifies the **average** of BTC and ETH correlation into three regimes (thresholds are configurable):

| Average correlation | Regime   |
| --- | --- |
| `>= 0.50` | Sync |
| `0.25 – 0.50` | Loose |
| `< 0.25` | Decouple |

By default, the regime uses automatic smoothing based on the chart timeframe:

- Weekly and above: `1`
- Daily to `30m`: `3`
- `5m` to below `30m`: `5`
- Below `5m`: `8`
- Non-standard charts: `3`

You can switch to manual smoothing and set the number of bars directly.

If enabled, the background is:

- Blue for **Sync**
- Gray for **Decouple**
- Unshaded for **Loose** (neutral transition)

## Benchmarks

Any TradingView symbol can be used as the benchmark. Common examples:

- `CME_MINI:NQ1!` (Nasdaq 100 futures)
- `SPY` (S&P 500 ETF)
- `TVC:GOLD` or `COMEX:GC1!` (gold)
- `DXY` (US Dollar Index)

## Interpretation (practical guide)

- **Sync:** crypto is largely moving with the benchmark; macro risk conditions are dominating.
- **Loose:** correlation is weakening; transitions are common and leadership can rotate.
- **Decouple:** crypto is moving more independently; crypto-native drivers are more likely.

## Notes and limitations

- Not a signal generator; it does not predict direction.
- Correlation describes co-movement, not causation.
- Uses `request.security(..., lookahead_off)` (non-repainting) and guards log returns against invalid values.

## License

Mozilla Public License 2.0
