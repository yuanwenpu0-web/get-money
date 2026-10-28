# VAMOStrategy

Welcome to **VAMOStrategy**, an open-source Pine Script v6 project for TradingView — perfect for Bitcoin scalping!  
This repository includes both:

- **VAMO Crossover Strategy** – a scalping strategy using momentum & volatility  
- **VAMO Indicator** – a Volatility-Adjusted Momentum Oscillator (VAMO) to visualize signals  

With ATR-based risk management and a volatility filter, this project is built for **fast, precise trades**.  
Join us to use, test, and improve VAMOStrategy!

---

## ✨ Features

- **VAMO Indicator** — Combines momentum (ROC) and volatility (ATR) for adaptive signals  
- **Crossover Signals** — Long/short trades triggered when VAMO crosses its EMA signal line  
- **Risk Management** — ATR-based stop-loss (1× ATR) and take-profit (2× ATR by default)  
- **Volatility Filter** — Skips trades during extreme volatility to reduce false signals  
- **Customizable** — Adjust momentum length, ATR period, and TP multiplier to fit your style  

---

## ⚙️ Setup

1. Open **TradingView** and navigate to **Pine Editor**  
2. **For the Indicator**  
   - Copy `Indicator.pine`  
   - Click **Add to Chart** to visualize VAMO  
3. **For the Strategy**  
   - Copy `Strategy.pine`  
   - Apply it to a Bitcoin chart (`BTC/USD` or `BTC/USDT`) on a **1–5 minute** timeframe  
4. Adjust inputs (Momentum Length, Volatility Length, Take-Profit Multiplier, etc.) in the strategy settings  

---

## 📊 Backtesting Results  

### Latest Backtest (Oct 27 – Nov 2, 2025)  
Tested on BTC/USD (1-minute timeframe) with default settings (Momentum Length=14, Volatility Length=14, TP Multiplier=2.0).  

| **Metric**       | **Value**                 |
|------------------|---------------------------|
| **Net Profit**   | +$56,525.59 (**+5.65%**) |
| **Max Drawdown** | $10,790.87 (**1.05%**)   |
| **Total Trades** | 660                      |
| **Winning Trades** | 39.39% (260/660)        |
| **Profit Factor** | 1.301                    |

### Previous Backtest (Sept 15–20, 2025)  

| **Metric**       | **Value**                 |
|------------------|-------------------------|
| **Net Profit**   | +$14,848.94 (**+1.48%**) |
| **Max Drawdown** | $1,178.97 (**0.82%**)   |
| **Total Trades** | 594                     |
| **Winning Trades** | 39.90% (237/594)       |
| **Profit Factor** | 1.146                  |

> **Note:** These results show strong potential, but always backtest with your own data, timeframe, and risk management. Results can vary based on market conditions, slippage, and commissions.

---

## 💡 Scalping Tips

- **Timeframe:** 1–5 minute Bitcoin charts work best  
- **Optimize Inputs:** Try Momentum Length = `5`, Volatility Length = `5`, Signal Length = `3`  
- **Backtest First:** Test on historical BTC data to refine parameters  
- **Use Leverage Wisely:** 5–20× leverage can amplify profits *and* risks — manage carefully  

---

## 🤝 Contributing

We’d love your help to make VAMOStrategy even better!  

You can contribute by:  
- **Adding Features** — new indicators, filters (volume-based, session filters), or exit logic  
- **Optimizing** — tune parameters for other assets (e.g., ETH) or timeframes  
- **Fixing Bugs** — spot and patch code issues or logic errors  

**How to contribute:**  
- Fork the repo  
- Make your changes  
- Submit a **pull request**  
- Discuss ideas in the **Issues** tab  

> Contributing Guide: *coming soon*

---

## 📂 Repository Structure

- `Strategy.pine` — VAMO Crossover Strategy (Pine Script v6)  
- `Indicator.pine` — VAMO Indicator (Pine Script v6)  
- `LICENSE` — MIT License  
- `README.md` — This file  

---

## 📜 License

Licensed under the **MIT License**.  
See the `LICENSE` file for details.  
© 2025 **TPTBusiness**

---

## ⚠️ Disclaimer

This project is provided **"as is"** with no warranty.  
Trading carries high risk, and **past performance does not guarantee future results**.  
Backtest thoroughly and understand the code before trading with real funds.
