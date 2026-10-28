# TradingView Pine Script Indicators

A personal collection of Pine Script indicators and strategies built on TradingView since 2022. This repository serves as a structured archive so the scripts are version-controlled, searchable, and easy to revisit or share.

## Overview

The collection spans **34 scripts** across **8 categories**, ranging from classic oscillator enhancements to multi-timeframe dashboards and a k-Nearest Neighbors machine learning indicator. Scripts are written in Pine Script v5 or v6 and are ready to paste directly into the TradingView Pine Editor.

**Categories at a glance:**

- **Bollinger Bands** — 7 scripts including a multi-type MA basis, a Fibonacci extension variant, a Nadaraya-Watson smoothed version, and a full strategy with RSI and SMA trend filters
- **Channel** — Linear regression channel with Fibonacci levels, and an ATR-based multi-timeframe support/resistance drawer
- **Machine Learning** — A k-NN momentum index that classifies trend direction using WMA/RSI crossover history
- **Moving Average** — ROC-based strategies, MACD expressed as a percentage, multi-timeframe MA overlays, and an MA crossover state table
- **Relative Strength Index** — Six RSI variants: multi-timeframe oscillator, signal/alert overlays, candlestick rendering, a 3-layer stacked view, and a full automated strategy
- **Renko** — A Renko emulator that works on standard charts (ATR and Traditional box sizing), plus a simplified ATR Renko with reversal signals
- **Stochastic** — A single-period stochastic and a 3-period stacked layout for multi-speed momentum reading
- **Volume** — Volume-colored candles, a statistical volume table, buy/sell volume breakdown, and a log₁₀ volume Bollinger Band
- **Other** — Utility scripts for BTC arbitrage premium tracking, DCA backtesting, live PnL display, USD/KRW rate, and a watermark overlay

## How to Use

1. Browse to the script you want under its category folder.
2. Copy the full file contents.
3. In TradingView, open the Pine Editor (bottom panel) and paste.
4. Click **Add to chart**.

Most scripts expose grouped input settings in the indicator settings panel. Scripts named `*-strategy.pine` are backtestable strategies — open the **Strategy Tester** tab after adding them to a chart.

---

## Bollinger Bands (`bollinger-bands/`)

| File | Name | Description |
|------|------|-------------|
| `01-bollinger-bands.pine` | Bollinger Bands | Multi-timeframe BB with selectable MA type (SMA/EMA/WMA/HMA/RMA/VWMA/ALMA) |
| `02-bollinger-bands-strategy.pine` | Bollinger Bands Strategy | BB strategy with multi-level standard deviation bands |
| `03-rsi-bb-oscillator.pine` | RSI + BB Oscillator | RSI centered at zero combined with Bollinger Band signals |
| `04-rsi-bb-signal-alert.pine` | RSI + BB Signal Alert | RSI + BB oscillator with signal markers and alert conditions |
| `05-fibonacci-bollinger-bands.pine` | Fibonacci Bollinger Bands | BB extended with Fibonacci internal and external retracement levels |
| `06-ema-bb-signal.pine` | Bollinger Bands And EMA | Nadaraya-Watson smoothed multi-band BB with EMA signal (based on FluxCharts) |
| `07-bb-rsi-sma-trend-filter.pine` | Bollinger + RSI + SMA Trend Filter | Strategy combining BB, RSI, EMA trend filter, and volume filter |

## Channel (`channel/`)

| File | Name | Description |
|------|------|-------------|
| `01-regression-trend-with-fibonacci.pine` | Regression Trend with Fibonacci | Linear regression channel overlaid with Fibonacci extension levels |
| `02-atr-channel-multi-timeframe.pine` | Multi-Timeframe Support & Resistance | ATR-based support/resistance lines drawn for 6 timeframes (1m–4h) |

## Machine Learning (`machine-learning/`)

| File | Name | Description |
|------|------|-------------|
| `01-machine-learning-momentum-index.pine` | Machine Learning Momentum Index (MLMI) | k-Nearest Neighbors momentum predictor using WMA/RSI crossovers (by Zeiierman) |

## Moving Average (`moving-average/`)

| File | Name | Description |
|------|------|-------------|
| `01-ma-roc-strategy.pine` | MA ROC Strategy | Strategy based on rate-of-change of 5 configurable moving averages |
| `02-ma-rate-of-change.pine` | MA Rate of Change | Single MA ROC oscillator with configurable type, length, and multiplier |
| `03-moving-average-keltner-channel.pine` | Moving Average & Keltner Channel | Up to 4 MA lines overlaid with a Keltner Channel |
| `04-macd-percent.pine` | MACD % | MACD expressed as percentage of the slow MA instead of absolute value |
| `05-multi-timeframe-ma.pine` | Multi Timeframe MA | Multiple MAs each fetched from an independently configurable timeframe |
| `06-ma-crossover-table.pine` | MA Crossover Table | Table showing MA crossover state across 4 timeframes and 3 MA lengths |

## Other (`other/`)

| File | Name | Description |
|------|------|-------------|
| `01-btc-arbitrage.pine` | BTC Arbitrage | Displays CME futures premium and Korean Upbit kimchi premium vs Binance |
| `02-dca-simulator.pine` | DCA Simulator | Backtests a simple dollar-cost-averaging strategy over a configurable date range |
| `03-my-position-and-pnl.pine` | My Position and PnL | Tracks open EUR, JPY, and BTC positions and shows combined unrealized PnL |
| `04-usdkrw-table.pine` | USDKRW Table | Displays the current USD/KRW exchange rate in a chart table |
| `05-watermark.pine` | Watermark | Adds a customizable watermark with title, subtitle, symbol, and timeframe |

## Relative Strength Index (`relative-strength-index/`)

| File | Name | Description |
|------|------|-------------|
| `01-rsi-oscillator.pine` | RSI Oscillator | RSI plotted across 4 configurable timeframes simultaneously |
| `02-rsi-signal.pine` | RSI Signal | RSI with buy/sell signal markers on the chart |
| `03-rsi-signal-alert.pine` | RSI Signal Alert | RSI signals with alert conditions and trend-following/reversal mode toggle |
| `04-rsi-strategy.pine` | RSI Strategy | Automated long/short strategy driven by multi-timeframe RSI signals |
| `05-candlestick-rsi.pine` | Candlestick RSI | Renders RSI values as OHLC candlesticks for clearer pattern reading |
| `06-rsi-3-layers.pine` | RSI 3 | Three RSI periods (e.g. 14/28/56) stacked in vertically separated lanes |

## Renko (`renko/`)

| File | Name | Description |
|------|------|-------------|
| `01-renko-emulator.pine` | Renko Emulator | Emulates Renko bricks (ATR or Traditional) on a standard chart (forked from JustUncleL) |
| `02-atr-renko.pine` | ATR Renko | ATR-based Renko with BUY/SELL labels on brick direction reversals |

## Stochastic (`stochastic/`)

| File | Name | Description |
|------|------|-------------|
| `01-stochastic.pine` | Stochastic | Single stochastic oscillator (%K/%D) with overbought/oversold fill |
| `02-stochastic-3-layers.pine` | Stochastic 3 | Three stochastic periods (5-3-3, 10-6-6, 20-12-12) stacked in separate lanes |

## Volume (`volume/`)

| File | Name | Description |
|------|------|-------------|
| `01-candle-color-by-volume.pine` | Candle Color by Volume | Colors candles in 6 shades of green/red based on relative volume vs EMA |
| `02-trading-volume-table.pine` | Trading Volume Table | Table showing volume statistics with standard deviation levels |
| `03-volume-details.pine` | Volume Details | Shows buy/sell volume split and comparison against 30-day/30-bar averages |
| `04-volume-log10-bb.pine` | Log Volume + Bollinger Band | log₁₀-transformed volume with Bollinger Band overlay for anomaly detection |
