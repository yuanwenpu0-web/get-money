# Ethereum-ETH--Strategy
Backtestable ETH/USDT Trading Strategy using Bollinger Bands &amp; EMA in Pine Script v6 with train/test date segmentation.

# BB EMA ETH/USDT 4h Strategy (Pine Script v6)

This repository contains a TradingView Pine Script v6 strategy for **Ethereum (ETH/USDT)** on the **4-hour timeframe**.  
It combines **Bollinger Bands (BB)** and an **Exponential Moving Average (EMA)** to generate long signals and manage exits.  
The script also includes **train/test date segmentation**, making it easier to validate performance on historical data.

---

## ✨ Features
- **Indicators Used**:
  - Bollinger Bands (configurable length & multiplier)
  - Exponential Moving Average (configurable length)
- **Signals**:
  - Long entry when price crosses above the upper Bollinger Band and is above EMA
  - Exit when price crosses under the Bollinger Band middle line or EMA
- **Train/Test Split**:
  - Select a time segment (Training, Testing, or All) for your backtest
  - Adjustable start & end dates for each segment
- **Strategy Parameters**:
  - Initial capital: 1000 USDT (can be adjusted)
  - Position size: 100% of equity per trade (can be adjusted)

---

## 🛠 How to Use
1. Open [TradingView](https://www.tradingview.com/).
2. Go to **Pine Editor** and paste the contents of `BB_EMA_ETHUSDT_4h.pine`.
3. Save and add it to your chart.
4. Select the **4H or 1min timeframe** and **ETH/USDT** pair.
5. Adjust the input parameters (EMA length, BB length, multiplier, train/test dates) as needed.
6. Choose the segment to backtest: `Training`, `Testing`, or `All`.

---

## 📈 Strategy Logic
- **Long Entry**:  
  `ta.crossover(close, bb_upper) and close > ema`
- **Exit**:  
  `ta.crossunder(close, bb_middle) or ta.crossunder(close, ema)`

---

## 📝 Inputs
| Parameter | Description |
|-----------|-------------|
| EMA Length | Period for the Exponential Moving Average |
| BB Length | Period for Bollinger Bands |
| BB Multiplier | Standard deviation multiplier for Bollinger Bands |
| Training Start/End Date | Date range for training backtest |
| Testing Start/End Date | Date range for testing backtest |
| Segment | Choose Training, Testing, or All |

---

## 📊 Performance Metrics(for 1 week Data)

| Metric                 | Value |
|------------------------|-------|
| Net Profit (USDT)       |  96.17     |
| Net Profit (%)          |  9.62    |
| Sharpe Ratio            |  1.239     |
| Sortino Ratio           |  37.415  |
| Profit Factor           |  2.349     |
| Max Drawdown (USDT)     |  8.47     |
| Max Drawdown (%)        |  0.78  |
| Margin Calls            |    0   |



## 📊 Example Chart
<img width="1780" height="812" alt="image" src="https://github.com/user-attachments/assets/4147c211-ae58-437d-9093-e41fd9a16650" />
<img width="1920" height="1080" alt="Screenshot (91)" src="https://github.com/user-attachments/assets/b9a1198c-c0dd-48ed-9ca7-fac25fc94f41" />



---

## ⚠️ Disclaimer
This script is for **educational purposes only**.  
It is not financial advice. Past performance does not guarantee future results.

---

## 📄 License
MIT License – feel free to modify and improve.

---


