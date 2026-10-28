# RSI Stoch + Volume + ATR Trailing (TradingView Strategy)

> Long-only, volatility-aware trend-following strategy for TradingView, combining **EMA trend**, **volume confirmation**, **RSI + Stochastic entries**, and an **ATR-based trailing stop**.

![Pine Script](https://img.shields.io/badge/Pine_Script-v6-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Built for trending, volatile markets such as crypto, indices, and FX.

---

## Table of Contents

- [Overview](#overview)
- [Strategy Logic](#strategy-logic)
  - [Trend Filter](#trend-filter)
  - [Volume Filter](#volume-filter)
  - [Entry Conditions](#entry-conditions)
  - [Exit Conditions / Risk Management](#exit-conditions--risk-management)
- [Strategy Math (Formulas)](#strategy-math-formulas)
  - [Exponential Moving Average (EMA)](#exponential-moving-average-ema)
  - [Relative Strength Index (RSI)](#relative-strength-index-rsi)
  - [Stochastic Oscillator (%K)](#stochastic-oscillator-k)
  - [Average True Range (ATR)](#average-true-range-atr)
  - [ATR-Based Trailing Stop](#atr-based-trailing-stop)
- [Inputs](#inputs)
- [How to Use](#how-to-use)
- [Backtesting Notes](#backtesting-notes)
- [Repository Structure](#repository-structure)
- [Roadmap / Ideas](#roadmap--ideas)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This TradingView strategy looks for **long entries in bullish conditions** using a combination of:

- **Trend direction** from a long-term EMA  
- **Volume confirmation** to avoid thin/liquidity traps  
- **RSI + Stochastic** to time entries into momentum  
- **ATR-based trailing stop** to adapt exits to volatility  

Implemented in **Pine Script v6** as a `strategy` script.

The design goal is to:

- Participate in strong trends  
- Filter out low-quality moves (no trend, low volume)  
- Let winners run with a volatility-adjusted trailing stop  

---

## Strategy Logic

### Trend Filter

- Controlled by the `Use 200 EMA Trend Filter?` input.
- Uses an EMA of length `emaLen` (default: 200).
- Long entries are allowed only when:

```text
close > emaTrend
```

If the trend filter is disabled, this condition is ignored and entries are allowed regardless of EMA.

---

### Volume Filter

- Controlled by the `Require Volume > Avg?` input.
- Computes a simple moving average of volume over `volLen` bars (default: 20).
- Volume confirmation requires:

```text
volume > volAvg
```

When disabled, the strategy will not check for volume confirmation.

---

### Entry Conditions

A **buy signal** is generated when **all** of the following conditions hold:

1. **Trend condition**  
   - Either the trend filter is off, or:
   - `close > emaTrend`

2. **Volume confirmation**  
   - Either the volume filter is off, or:
   - `volume > volAvg`

3. **RSI filter**  
   - RSI is below a configurable level:
   - `rsi < rsiBuyLevel` (default: 55.0)

4. **Stochastic trigger**  
   - The smoothed %K line crosses above `stochBuyLevel` (default: 60.0):
   - `ta.crossover(k, stochBuyLevel)`

If there is **no existing position** and all conditions are true, the strategy enters a **long**:

```pine
strategy.entry("Long", strategy.long)
```

---

### Exit Conditions / Risk Management

Exits are fully controlled through an **ATR-based trailing stop**:

- ATR length: `atrLen` (default: 14)  
- Multiplier: `atrMultiplier` (default: 3.0)  

For an open long position:

1. Compute a **candidate stop**:
   ```text
   potentialStop = close - atr * atrMultiplier
   ```

2. **Ratchet behavior**:
   - The trailing stop only moves **upwards** (never down).
   - New stop = `max(previous_trailStop, potentialStop)`.

3. When price trades at or below `trailStop`, the position is closed via:
   ```pine
   strategy.exit("Trailing Stop", from_entry="Long", stop=trailStop)
   ```

This approach locks in more profit as the trend continues, while adapting to market volatility.

---

## Strategy Math (Formulas)

This section provides the underlying formulas for the indicators used by the strategy.

### Exponential Moving Average (EMA)

For a series of closes \( C_t \) and period \( N \):

- Smoothing factor:

  \[
  lpha = rac{2}{N + 1}
  \]

- Recursive definition:

  \[
  	ext{EMA}_t = lpha C_t + (1 - lpha)\,	ext{EMA}_{t-1}
  \]

The strategy uses an EMA of length \( N = 	ext{emaLen} \) as the **trend filter**.

---

### Relative Strength Index (RSI)

For period \( N \):

1. Up-moves and down-moves:
   \[
   U_t = \max(C_t - C_{t-1}, 0), \quad
   D_t = \max(C_{t-1} - C_t, 0)
   \]

2. Smoothed averages (Wilder’s smoothing / EMA-style):
   \[
   	ext{AvgGain}_t = 	ext{smooth}(U_t, N), \quad
   	ext{AvgLoss}_t = 	ext{smooth}(D_t, N)
   \]

3. Relative strength:
   \[
   RS_t = rac{	ext{AvgGain}_t}{	ext{AvgLoss}_t}
   \]

4. RSI:
   \[
   	ext{RSI}_t = 100 - rac{100}{1 + RS_t}
   \]

The strategy only allows entries when:

\[
	ext{RSI}_t < 	ext{rsiBuyLevel}
\]

---

### Stochastic Oscillator (%K)

For period \( N = 	ext{stochLen} \):

- Highest high:
  \[
  H^{(N)}_t = \max(H_{t-N+1}, \dots, H_t)
  \]

- Lowest low:
  \[
  L^{(N)}_t = \min(L_{t-N+1}, \dots, L_t)
  \]

Raw %K:

\[
\%K_{	ext{raw},t} = 100 \cdot rac{C_t - L^{(N)}_t}{H^{(N)}_t - L^{(N)}_t}
\]

Smoothed %K (what the script uses as \( k_t \)):

\[
k_t = 	ext{SMA}ig(\%K_{	ext{raw}},\, 	ext{smoothK}ig)
\]

The entry trigger uses a crossover condition:

\[
	ext{crossover}(k_t, 	ext{stochBuyLevel})
\]

i.e. \( k_t \) crosses upward through the threshold level.

---

### Average True Range (ATR)

True Range (TR) for each bar \( t \):

\[
	ext{TR}_t = \max\left(
H_t - L_t,\,
|H_t - C_{t-1}|,\,
|L_t - C_{t-1}|

ight)
\]

ATR over period \( N = 	ext{atrLen} \) is usually an EMA of TR:

\[
	ext{ATR}_t = 	ext{EMA}_N(	ext{TR}_t)
\]

ATR measures volatility in price units.

---

### ATR-Based Trailing Stop

For a long position, the **candidate stop** is:

\[
	ext{StopCandidate}_t = C_t - 	ext{ATR}_t \cdot M
\]

where \( M = 	ext{atrMultiplier} \) (e.g. \( M = 3.0 \)).

The trailing stop is defined as a **ratchet**:

\[
	ext{TrailStop}_t =
egin{cases}
	ext{StopCandidate}_t, & 	ext{if there is no previous stop} \
\max(	ext{TrailStop}_{t-1}, 	ext{StopCandidate}_t), & 	ext{otherwise}
\end{cases}
\]

Exit condition:

\[
	ext{Exit when } C_t \leq 	ext{TrailStop}_t
\]

---

## Inputs

| Name                         | Type  | Default | Description                                              |
|------------------------------|-------|---------|----------------------------------------------------------|
| Use 200 EMA Trend Filter?    | bool  | `true`  | Restrict longs to when price is above EMA.               |
| Trend Filter Length          | int   | `200`   | EMA length for trend filter.                             |
| Require Volume > Avg?        | bool  | `true`  | Enforce volume > moving average.                         |
| Volume Avg Length            | int   | `20`    | Length of volume SMA.                                    |
| RSI Length                   | int   | `14`    | RSI period.                                              |
| Stoch Length                 | int   | `14`    | Stochastic lookback period.                              |
| Stoch K Smoothing            | int   | `3`     | Smoothing period for %K.                                 |
| RSI Buy Below (Relaxed)      | float | `55.0`  | RSI must be below this to allow entries.                 |
| Stoch Buy Below              | float | `60.0`  | Level for %K crossover trigger.                          |
| ATR Length                   | int   | `14`    | ATR lookback period.                                     |
| ATR Trailing Multiplier      | float | `3.0`   | Multiplier for ATR trailing stop.                        |
| Start Year                   | int   | `2011`  | Start year for backtest date filter.                     |
| End Year                     | int   | `2069`  | End year for backtest date filter.                       |

---

## How to Use

1. Open **TradingView**.
2. Open the **Pine Editor** panel.
3. Copy the contents of `src/rsi_stoch_vol_atr_trailing.pine`.
4. Paste into a new Pine Script in the editor.
5. Click **Save**, then **Add to chart**.
6. Configure the inputs to match:
   - Instrument (crypto / FX / indices / stocks).
   - Timeframe (e.g. 1h, 4h, 1D).
   - Risk preference (ATR multiplier, filters on/off).

If you maintain an `examples/` folder with screenshots, you can reference them, for example:

```markdown
## Example Chart

BTCUSDT, 1D timeframe:

![BTCUSDT Daily Example](examples/btcusdt_daily_example.png)
```

---

## Backtesting Notes

When backtesting:

- Test across multiple symbols and timeframes.
- Pay attention to:
  - Net profit and equity curve shape.
  - Maximum drawdown.
  - Win rate and average R per trade.
  - Trade frequency (too many vs. too few trades).
- Experiment with:
  - Different EMA lengths (e.g. 100, 200, 300).
  - ATR multipliers (e.g. 2.0–4.0).
  - RSI and Stoch thresholds to control how aggressive entries are.

---

## Repository Structure

```text
.
├─ src/
│  └─ rsi_stoch_vol_atr_trailing.pine
├─ examples/
│  ├─ btcusdt_daily_example.png
│  └─ ethusdt_4h_example.png
├─ .gitignore
├─ LICENSE
├─ README.md
└─ CHANGELOG.md
```

- `src/` – Pine Script source code.  
- `examples/` – Optional chart screenshots / usage examples.  
- `CHANGELOG.md` – Version history and changes over time.  

---

## Roadmap / Ideas

- Add short-side logic for downtrends.
- Provide multiple trend filters (multi-EMA, Supertrend, etc.).
- Time-of-day / session filters for intraday trading.
- ATR-based position sizing (fixed % of equity per trade).
- Optional logging/export of trades for external analysis.

---

## Contributing

Contributions, ideas, and critiques are welcome:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes with clear, focused commits.
4. Open a pull request describing:
   - The motivation for the change.
   - Any impact on behavior or backtest results.

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
