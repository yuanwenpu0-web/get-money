# 📈 CrossMACD Scalper ⚡

**CrossMACD Scalper** is a Pine Script strategy designed for **frequent scalping** and **intraday trading** using **EMA crossover signals** confirmed by **MACD histogram momentum**. This strategy runs entirely within **TradingView’s Strategy Tester** and is ideal for **volatile assets** like BTC, ETH, Forex pairs, and indices.

![strategy preview](preview.png) <!-- Optional: add a screenshot of strategy on chart -->

---

## 🚀 Features

- 🔁 High-frequency trade signals
- 🧠 Trend detection using 9/21 EMA crossover
- 📊 Momentum confirmation with MACD histogram
- 🎯 Built-in take profit & stop loss system
- ⚙️ Simple, adjustable parameters
- 💻 100% TradingView Pine Script — no external setup

---

## 📚 Strategy Logic

| Condition   | Criteria |
|-------------|----------|
| **Buy**     | Fast EMA crosses above Slow EMA **AND** MACD Histogram > 0 |
| **Sell**    | Fast EMA crosses below Slow EMA **AND** MACD Histogram < 0 |
| **Exit**    | TP at +1.2% or SL at -0.6% of entry price |

---

## 🛠️ How to Use

1. Go to [TradingView](https://www.tradingview.com/)
2. Open the **Pine Script Editor**
3. Paste code from `CrossMACD_Scalper.pine`
4. Click **"Add to Chart"**
5. Use **Strategy Tester** to evaluate performance
6. Adjust parameters to fit your trading style

---

## ⚙️ Inputs & Parameters

| Parameter         | Default | Description                            |
|-------------------|---------|----------------------------------------|
| Fast EMA          | 9       | Short-term trend detection             |
| Slow EMA          | 21      | Long-term trend confirmation           |
| MACD Fast Length  | 12      | MACD line short EMA                    |
| MACD Slow Length  | 26      | MACD line long EMA                     |
| MACD Signal Line  | 9       | MACD signal smoothing line             |
| Take Profit (%)   | 1.2     | Exit trade at 1.2% profit              |
| Stop Loss (%)     | 0.6     | Exit trade at 0.6% loss                |

---

## 📊 Best Used On

- 🕐 **Timeframes**: 5min, 15min, 1H
- 💹 **Assets**: BTC/USDT, ETH, Gold, Nasdaq, EUR/USD, etc.
- 🔄 **Style**: Scalping or intraday trend following

---

## 📥 Future Features

- 🔔 Alert system for live webhook trading
- 📤 CSV export of trade performance
- 📈 Visual dashboard for trades and stats

---

## 🧪 Disclaimer

This script is for **educational and testing purposes** only. It does not execute live trades. Use with paper trading and backtesting tools responsibly. Always test your strategy before live deployment.

---

## 📄 License

MIT License — use it freely, fork it, improve it, and share it. Attribution is appreciated!

---

## 👨‍💻 Author

Made by [Loop24Infility (Kshitij Jadhav)]  
🔗 Connect: [Linkedin](https://www.linkedin.com/in/kshitij-jadhav-7b434b338/) | [Twitter](https://twitter.com/yourhandle) | [GitHub](https://github.com/Loop24Infility)

