# Volume Flow Anomaly · AnomIQ Preview

A TradingView Pine Script indicator built from AnomIQ's real-time microstructure scanner logic. It surfaces anomalous volume events using rolling z-scores, trend cleanliness, and relative price impact. These match the metrics [AnomIQ](https://anomiq.io/?utm_source=github&utm_medium=readme&utm_campaign=vfa_indicator) computes from tick-level exchange data.

> **Preview limitations.** The buy/sell split uses bar-range geometry rather than tick-level taker classification. The full platform uses real taker data, scans all Binance and Coinbase pairs simultaneously, and delivers alerts via Telegram and NTFY.

---

## How It Works

Every signal calculates against a coin's own rolling volume history. There are no fixed thresholds. A 2σ event on BTC and a 2σ event on a small-cap altcoin represent the same statistical rarity for that coin. The filter works across any market cap.

### Three core metrics

**Vol Z**: rolling z-score of window volume against the lookback period. Values above 2.0 mean current volume is statistically significant for this coin.

**Trend Cleanliness**: directional consistency of price bars within the window. High values mean bars stack in one direction. High volume with low cleanliness signals absorption.

**Relative Price Impact (RPI)**: volume per unit of price movement. High volume with low RPI means a large passive order is absorbing aggressive flow. A directional move typically follows within 2-5 bars.

---

## Signal Lifecycle

Trade the phases, not the individual signals.

```
Yellow dot  →  Absorption diamond  →  Triangle
Setup       →  Entry               →  Exit warning
```

### Phase 1: Setup
**🟡 Yellow dot**: Vol Z is building above the early threshold but Trend Cleanliness sits below 35%, so price has no committed direction yet. Fires 1-3 bars before the main signal. Watch, do not enter.

### Phase 2: Entry
**🔶 Absorption diamond**: Vol Z is above threshold, window return is below 0.3%, RPI is low. A large passive order is absorbing aggressive flow. This is the leading signal in the sequence.

- **Teal diamond**: buy side absorbing (bullish lean)
- **Red diamond**: sell side absorbing (bearish lean)

### Phase 3: Exit Warning
**🟢 Teal triangle**: Upside move confirmed with anomalous volume. Vol Z has peaked and will mean-revert. Trail your stop or watch for a pullback entry.

**🔴 Red triangle**: same conditions, downside move.

Triangles confirm a move had volume behind it. By the time one fires, Vol Z is already mean-reverting. Enter on the absorption diamond, not the triangle.

### Extreme Volume
**🟠 Orange dot**: Vol Z above 3.5σ with no confirmed direction or absorption pattern. Classify manually.

---

## Settings

| Parameter | Default | Description |
|---|---|---|
| Window Length | 5 | Bars to aggregate. Use 5 on 1m to match AnomIQ's 5m timeframe. |
| Historical Lookback | 100 | Bars used for z-score baseline. Higher = slower to adapt, more stable. |
| Vol Z Threshold | 2.0 | Minimum z-score to trigger a confirmed signal. |
| Early Warning Vol Z | 1.3 | Z-score that triggers the setup dot. Lower = earlier but noisier. |
| Trend Cleanliness Min% | 50 | Minimum directional consistency required for surge signals. |
| Min Window Return % | 0.3 | Minimum price move required to classify as a surge. |

Start on a 1m chart with Window=5, Lookback=100. This approximates AnomIQ's 5m timeframe scanner.

---

## Alerts

Four alert conditions are available under `Add Alert → Condition`:

| Alert | Fires when |
|---|---|
| AnomIQ · Early Warning | Volume building without committed direction |
| AnomIQ · Upside Surge Confirmed | Upside move confirmed with anomalous volume |
| AnomIQ · Downside Surge Confirmed | Downside move confirmed with anomalous volume |
| AnomIQ · Absorption | High volume absorbed without price impact |

---

## Limitations

**Buy/sell split is approximated.** The script estimates buy and sell volume using bar-range geometry: `buy_fraction = (close - low) / (high - low)`. On wide-spread candles and during gaps this diverges from real taker classification. Vol Z and Trend Cleanliness are fully accurate.

**Single-pair only.** This indicator runs on one chart at a time. AnomIQ scans all Binance and Coinbase pairs simultaneously and surfaces every anomaly in real time.

---

## Full Platform

The [AnomIQ platform](https://anomiq.io/?utm_source=github&utm_medium=readme&utm_campaign=vfa_indicator) runs the same logic across every Binance and Coinbase pair at once:

- Tick-level taker classification from exchange WebSocket feeds
- Custom scanner builder with z-score, buy/sell ratio, volume, and taker imbalance filters across 5m / 15m / 60m timeframes
- Real-time alerts via web dashboard, Telegram, and NTFY
- Signals calculate relative to each coin's own history. The same filter works for BTC and a small-cap altcoin

AnomIQ has a free tier. Real-time results and Telegram/NTFY push alerts require a paid subscription.

---

## Installation

1. Open TradingView and go to the Pine Script editor (bottom panel)
2. Paste the contents of `volume-flow-anomaly.pine`
3. Click **Add to chart**
4. Set chart timeframe to **1m** for default settings

Or search for **"Volume Flow Anomaly AnomIQ"** in TradingView's public indicator library.

---

## License

MIT. Use and modify without restriction.
