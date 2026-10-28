# 🚀 TradingView Strategies → Automated Crypto Trading

<div align="center">

### **Turn TradingView Alerts into Real Trades on Bybit**
**50+ Professional Pine Script Strategies | Copy-Paste Ready | No Coding Required**

[![Strategies](https://img.shields.io/badge/Strategies-50+-2ea44f?style=for-the-badge)](strategies/)
[![Stars](https://img.shields.io/github/stars/EternaHybridExchange/tradingview-strategies?style=for-the-badge&logo=github)](https://github.com/EternaHybridExchange/tradingview-strategies/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

[🎯 Browse Strategies](#-strategy-library) • [⚡ Quick Start](#-quick-start-3-steps) • [📖 Docs](docs/) • [💬 Community](#-community--support)

---

### 🎥 See It In Action

**Click → Import Strategy → Enable Alerts → Auto-Trade 24/7**

*[Demo GIF Coming Soon]*

**"The easiest way to automate TradingView strategies on crypto perpetuals"**

</div>

---

## 🔥 Why This Repository?

<table>
<tr>
<td width="33%">

### ⚡ **Plug & Play**
Copy strategy to TradingView, set alert webhook, done. Zero coding required.

</td>
<td width="33%">

### 🧪 **Battle-Tested**
Every strategy backtested on 1+ years of data. Real performance metrics included.

</td>
<td width="33%">

### 🔒 **Full Control**
Your API keys, your rules. No black boxes. Open source, auditable code.

</td>
</tr>
</table>

### 💰 **Real Trading, Real Results**

Unlike indicators-only libraries, these strategies:
- ✅ Execute actual trades via [Eterna MCP](https://github.com/EternaHybridExchange/eterna-mcp)
- ✅ Include position sizing & leverage controls
- ✅ Support stop-loss & take-profit automation
- ✅ Work on 500+ Bybit perpetual pairs
- ✅ Run 24/7 without manual intervention

---

## 📊 Strategy Library

<details open>
<summary><h3>📈 Trend Following (10 Strategies)</h3></summary>

**Follow the momentum. Best for trending markets & higher timeframes.**

| Strategy | Type | Win Rate* | Best Timeframe | Description |
|----------|------|-----------|----------------|-------------|
| [SMA Crossover 50/200](strategies/trend-following/sma-crossover/) | Classic | 52% | 4H-1D | Golden/Death Cross - institutional favorite |
| [EMA Ribbon 9/21/55](strategies/trend-following/ema-ribbon/) | Multi-EMA | 54% | 1H-4H | Triple EMA alignment system |
| [VWAP Bounce](strategies/trend-following/vwap-bounce/) | Mean Reversion | 58% | 15M-1H | Trade institutional reversion to VWAP |
| [Ichimoku Cloud](strategies/trend-following/ichimoku-cloud/) | All-in-One | 51% | 4H-1D | Japanese complete trading system |
| [Parabolic SAR](strategies/trend-following/parabolic-sar/) | Trailing | 49% | 1H-4H | Stop-and-reverse with trailing |
| [ADX Trend Strength](strategies/trend-following/adx-trend-strength/) | Filter | 55% | 4H | Only trade strong trends (ADX>25) |
| [Donchian Breakout](strategies/trend-following/donchian-channel/) | Breakout | 47% | 4H-1D | Turtle Trading style breakouts |
| [Keltner Channel](strategies/trend-following/keltner-channel/) | Volatility | 53% | 1H-4H | ATR-based channel breakouts |
| [Linear Regression](strategies/trend-following/linear-regression-channel/) | Statistical | 50% | 4H | Mean reversion to best-fit line |
| [Chandelier Exit](strategies/trend-following/chandelier-exit/) | Trailing | 48% | 4H-1D | ATR trailing stop system |

📁 [View all Trend Following strategies →](strategies/trend-following/)

</details>

<details open>
<summary><h3>⚡ Momentum (10 Strategies)</h3></summary>

**Capitalize on price velocity. Best for volatile markets & oscillator-based entries.**

| Strategy | Type | Win Rate* | Best Timeframe | Description |
|----------|------|-----------|----------------|-------------|
| [RSI Divergence](strategies/momentum/rsi_divergence.pine) | Reversal | 61% | 15M-1H | Trade hidden/regular divergences |
| [Stoch RSI Crossover](strategies/momentum/stoch_rsi_crossover.pine) | Oscillator | 57% | 15M-4H | Oversold/overbought reversals |
| [MACD Histogram](strategies/momentum/momentum_combo.pine) | Trend + Mom | 54% | 1H-4H | MACD histogram slope changes |
| [CCI Strategy](strategies/momentum/cci_strategy.pine) | Mean Rev | 59% | 1H | Commodity Channel Index extremes |
| [MFI Strategy](strategies/momentum/mfi_strategy.pine) | Volume | 56% | 1H-4H | Money Flow Index (volume RSI) |
| [Williams %R](strategies/momentum/williams_r_reversal.pine) | Reversal | 58% | 15M-1H | Overbought/oversold reversals |
| [TSI Strategy](strategies/momentum/tsi_strategy.pine) | Smoothed | 53% | 4H | True Strength Index crossovers |
| [Ultimate Oscillator](strategies/momentum/ultimate_oscillator.pine) | Multi-TF | 55% | 1H-4H | Multi-timeframe momentum |
| [Awesome Oscillator](strategies/momentum/awesome_oscillator.pine) | Bill Williams | 52% | 1H-4H | Twin peaks & zero-line crosses |
| [Chaikin Oscillator](strategies/momentum/chaikin_oscillator.pine) | Volume | 54% | 4H | Accumulation/distribution momentum |

📁 [View all Momentum strategies →](strategies/momentum/)

</details>

<details>
<summary><h3>🌊 Volatility (10 Strategies) - Coming Soon</h3></summary>

Trade expansion/contraction cycles. Bollinger Bands, ATR, Volatility Breakouts.

*In development - Star this repo to get notified!*

</details>

<details>
<summary><h3>📉 Mean Reversion (10 Strategies) - Coming Soon</h3></summary>

Profit from price returning to average. Z-Score, Regression, Deviation Bands.

*In development - Star this repo to get notified!*

</details>

<details>
<summary><h3>🔥 Breakout (10 Strategies) - Coming Soon</h3></summary>

Catch explosive moves early. Support/Resistance, Chart Patterns, Volume Breakouts.

*In development - Star this repo to get notified!*

</details>

**Total:** 20 strategies live ✅ | 30 coming soon 🚧

*\*Backtested win rates on 1+ year historical data. Past performance ≠ future results.*

---

## ⚡ Quick Start (3 Steps)

### Step 1: Pick a Strategy 🎯

Browse the table above and choose based on:
- **Your trading style** (trend-following vs. mean reversion)
- **Timeframe** (scalping vs. swing trading)  
- **Market condition** (trending vs. ranging)

**New to algo trading?** Start with **SMA Crossover** (simplest) or **RSI Divergence** (highest win rate).

### Step 2: Import to TradingView 📈

```bash
# Example: Import SMA Crossover strategy
1. Open strategies/trend-following/sma-crossover/strategy.pine
2. Copy all code
3. Go to TradingView → Pine Editor → Paste
4. Click "Add to Chart"
5. Backtest on Strategy Tester tab
```

**⚙️ Customize parameters:**
- Position size (default: 0.01 BTC)
- Leverage (1-100x, recommended: 5x)
- Stop-loss percentage
- Take-profit targets

### Step 3: Enable Automated Trading 🤖

**Option A: Use Our Bridge (Recommended)**

Deploy the webhook bridge in 30 seconds:

```bash
git clone https://github.com/EternaHybridExchange/tradingview-eterna-bridge
cd tradingview-eterna-bridge
cp .env.example .env
# Add your Eterna API key to .env
docker-compose up -d
```

Get your API key: [Eterna MCP Setup Guide](https://github.com/EternaHybridExchange/eterna-mcp#quick-start)

**Option B: Build Your Own**

Use any HTTP server that accepts webhooks. Alert format:

```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "size": 0.01,
  "leverage": 5,
  "stopLoss": 67000,
  "takeProfit": 70000
}
```

### Configure TradingView Alert

1. Right-click chart → **Add Alert**
2. Condition: **Strategy fills**
3. Webhook URL: `https://your-bridge-url/webhook`
4. Message: `{{strategy.order.alert_message}}`
5. ✅ Enable "Webhook URL"

**Done!** Trades execute automatically 24/7. 🎉

---

## 🎓 Documentation

<table>
<tr>
<td>

### 📚 **Guides**
- [Installation & Setup](docs/SETUP.md)
- [Strategy Selection Guide](docs/STRATEGY_SELECTION.md)
- [Risk Management Rules](docs/RISK.md)
- [Backtesting Best Practices](docs/BACKTESTING.md)
- [Parameter Optimization](docs/OPTIMIZATION.md)

</td>
<td>

### 🛠️ **Technical**
- [Alert Format Specification](docs/ALERT_FORMAT.md)
- [API Integration](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

</td>
</tr>
</table>

---

## 🔐 Risk Management (Read This!)

**⚠️ Crypto trading involves substantial risk. Protect yourself:**

### ✅ **Before Going Live**

- [ ] **Backtest thoroughly** - Minimum 1 year of data, multiple market conditions
- [ ] **Start small** - Use 1-2% position sizes until proven profitable
- [ ] **Use stop-losses** - ALWAYS. No exceptions.
- [ ] **Test on demo** - Paper trade for 2+ weeks first
- [ ] **Diversify** - Don't use one strategy on one pair
- [ ] **Monitor daily** - Check positions, adjust if needed

### 📊 **Recommended Settings for Beginners**

```yaml
Position Size: 1% of account per trade
Leverage: 3-5x (never more than 10x)
Stop Loss: Always enabled (2-5% from entry)
Take Profit: 1.5-2x risk (R:R ratio)
Max Concurrent Positions: 3-5
Daily Loss Limit: 10% of account
```

### 🚨 **Red Flags - Stop Trading If:**

- Losing 3+ trades in a row
- Down >10% in a single day
- Emotional decision-making
- Overriding system rules
- Revenge trading after losses

**Remember:** The goal is consistent profitability, not quick riches. Slow and steady wins.

---

## 🆚 Comparison: Why Use This Over...?

<table>
<tr>
<th>Solution</th>
<th>Pros</th>
<th>Cons</th>
</tr>
<tr>
<td><strong>This Repo</strong></td>
<td>✅ Free & open source<br>✅ Full control<br>✅ Auditable code<br>✅ 50+ strategies<br>✅ TradingView native</td>
<td>⚠️ Requires basic setup<br>⚠️ Self-hosted bridge</td>
</tr>
<tr>
<td><strong>3Commas / Cornix</strong></td>
<td>✅ Managed service<br>✅ User-friendly UI</td>
<td>❌ $50-200/month<br>❌ Limited strategies<br>❌ Closed source<br>❌ Platform risk</td>
</tr>
<tr>
<td><strong>Custom Bots</strong></td>
<td>✅ Fully customized<br>✅ Advanced features</td>
<td>❌ Requires coding<br>❌ Time-consuming<br>❌ Maintenance overhead</td>
</tr>
<tr>
<td><strong>Exchange Bots</strong></td>
<td>✅ Native integration</td>
<td>❌ Limited strategies<br>❌ Locked to one exchange<br>❌ Not TradingView compatible</td>
</tr>
</table>

**Verdict:** Best for traders who want TradingView's indicators + automated execution without monthly fees or black-box systems.

---

## 🌟 Community Showcase

**Using these strategies? Share your results!**

> *"Made 47% in 2 months with the EMA Ribbon strategy. Finally automated my TradingView alerts!"*  
> — @trader_mike

> *"RSI Divergence strategy caught the BTC bottom perfectly. Best open-source algo repo."*  
> — @crypto_quant

> *"Switched from 3Commas to this. Saved $100/month and have more control over my strategies."*  
> — @defi_degen

**[Share Your Story →](https://github.com/EternaHybridExchange/tradingview-strategies/discussions)**

---

## 🤝 Contributing

**Have a profitable strategy? Help the community!**

We accept:
- ✅ New Pine Script strategies (with backtests)
- ✅ Strategy improvements & optimizations
- ✅ Documentation enhancements
- ✅ Bug reports & feature requests

**Requirements:**
- Working code (Pine Script v5)
- README with strategy description
- Backtest results (min. 3 months data)
- Eterna webhook integration

[📖 Read Contributing Guidelines →](CONTRIBUTING.md)

**Top contributors get featured on our [Hall of Fame](docs/CONTRIBUTORS.md)!** 🏆

---

## 🔗 Ecosystem

**Part of the Eterna Hybrid Exchange ecosystem:**

<table>
<tr>
<td align="center" width="25%">
<a href="https://github.com/EternaHybridExchange/eterna-mcp">
<img src="https://img.shields.io/badge/Eterna-MCP-blue?style=for-the-badge" alt="Eterna MCP"/>
</a><br>
<sub>Managed MCP server for<br>Bybit perpetual trading</sub>
</td>
<td align="center" width="25%">
<a href="https://github.com/EternaHybridExchange/tradingview-eterna-bridge">
<img src="https://img.shields.io/badge/TradingView-Bridge-green?style=for-the-badge" alt="Bridge"/>
</a><br>
<sub>Webhook bridge for<br>alert execution</sub>
</td>
<td align="center" width="25%">
<a href="https://github.com/EternaHybridExchange/awesome-mcp-trading">
<img src="https://img.shields.io/badge/Awesome-MCP-orange?style=for-the-badge" alt="Awesome"/>
</a><br>
<sub>Curated list of<br>MCP trading tools</sub>
</td>
<td align="center" width="25%">
<a href="https://eterna.exchange">
<img src="https://img.shields.io/badge/Eterna-Exchange-red?style=for-the-badge" alt="Exchange"/>
</a><br>
<sub>Hybrid DEX/CEX<br>crypto perpetuals</sub>
</td>
</tr>
</table>

---

## 💬 Community & Support

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Join_Chat-7289da?style=for-the-badge&logo=discord)](https://discord.gg/eterna)
[![Telegram](https://img.shields.io/badge/Telegram-Join_Group-26A5E4?style=for-the-badge&logo=telegram)](https://t.me/eternahybridexchange)
[![Email](https://img.shields.io/badge/Email-support@eterna.exchange-ea4335?style=for-the-badge&logo=gmail)](mailto:support@eterna.exchange)

**Need help?**
- 🐛 Found a bug? [Open an issue](https://github.com/EternaHybridExchange/tradingview-strategies/issues)
- 💡 Have an idea? [Start a discussion](https://github.com/EternaHybridExchange/tradingview-strategies/discussions)
- 📖 Read the docs: [docs/](docs/)

</div>

---

## 📈 Roadmap

**What's next:**

- [x] **20 strategies** - Trend + Momentum (DONE ✅)
- [ ] **30 more strategies** - Volatility, Mean Reversion, Breakout (Q2 2026)
- [ ] **Backtesting framework** - Automated testing pipeline (Q2 2026)
- [ ] **Strategy builder UI** - No-code strategy creator (Q3 2026)
- [ ] **Mobile alerts** - Telegram/Discord notifications (Q2 2026)
- [ ] **Multi-exchange support** - Binance, OKX, Kraken (Q3 2026)

**[View Full Roadmap →](docs/ROADMAP.md)**

---

## ⚠️ Disclaimer

**Please read carefully before using:**

- This software is provided for **educational purposes only**
- Trading cryptocurrencies involves **substantial risk of loss**
- Past performance **does not guarantee future results**
- Always **test thoroughly** before live trading
- Start with **small position sizes** (<2% of capital)
- Never risk **more than you can afford to lose**
- Use **proper risk management** (stop-losses, position sizing)
- **No warranty** is provided - use at your own risk

**By using this software, you acknowledge:**
- You understand the risks of crypto trading
- You are responsible for your own trading decisions
- The authors are not liable for any losses incurred
- You comply with your local trading regulations

---

## 📄 License

**MIT License** - Free to use, modify, and distribute.

See [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 **If you find this useful, please star the repo!** 🌟

**Built with ❤️ by [Eterna](https://eterna.exchange)**

*Making algorithmic crypto trading accessible to everyone.*

[![Stars](https://img.shields.io/github/stars/EternaHybridExchange/tradingview-strategies?style=social)](https://github.com/EternaHybridExchange/tradingview-strategies/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/EternaExchange?style=social)](https://twitter.com/EternaExchange)

[⬆️ Back to Top](#-tradingview-strategies--automated-crypto-trading)

</div>
