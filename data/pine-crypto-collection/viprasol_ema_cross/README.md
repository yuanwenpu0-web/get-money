<p align="center">
  <img src="docs/assets/logo.png" width="120" alt="Viprasol Tech logo">
</p>

<h1 align="center">Pine Script Strategy</h1>

<p align="center">
  <strong>A flagship TradingView Pine Script v6 strategy — EMA crossover with stackable confluence filters, dual stop/target engines, an ATR trailing stop, and an on-chart stats dashboard.</strong>
</p>

<p align="center">
  <em>Built and maintained by <a href="https://viprasol.com">Viprasol Tech</a> — Fintech Experts. Full-Stack Builders.</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Viprasol-Tech/pine-script-strategy?style=flat-square&color=blue" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/TradingView-Pine%20v6-2962FF?style=flat-square" alt="Pine v6">
  <img src="https://img.shields.io/badge/version-0.2.0-00ffaa?style=flat-square" alt="Version 0.2.0">
  <img src="https://img.shields.io/badge/type-strategy-ff3d71?style=flat-square" alt="Strategy">
  <a href="https://t.me/viprasol_help"><img src="https://img.shields.io/badge/Telegram-support-26A5E4?style=flat-square&logo=telegram&logoColor=white" alt="Telegram"></a>
</p>

---

> ## ⚠️ Disclaimer
> This strategy is for **educational purposes only** and is **not financial advice**. Backtest results on TradingView are **not** indicative of future performance and can be affected by repainting, slippage, and commission assumptions. Forward-test on a demo account before risking capital. **Use at your own risk** — Viprasol Tech assumes no responsibility for your trading results.

---

## ✨ Features

- 📈 **EMA crossover core** — long on fast-over-slow, short on fast-under-slow, evaluated on confirmed bars.
- 🔀 **Selectable MA type** — EMA, SMA, WMA, RMA, or HMA, on any source.
- 🧭 **HTF trend filter** — only take longs above (and shorts below) a higher-timeframe trend EMA (no-lookahead, repaint-safe).
- 📊 **RSI filter** — independent long/short momentum thresholds.
- 💪 **ADX filter** — only trade when trend strength clears a minimum.
- 🔊 **Volume filter** — require volume above its moving average × a multiplier.
- 🕒 **Session filter** — restrict entries to a session window + timezone, with optional end-of-session flatten and background shading.
- 🛡️ **Dual stop/target engine** — switch between **ATR-scaled** and **fixed-percent** SL/TP; take-profit can be disabled.
- 🪤 **ATR trailing stop** — configurable activation and trail distances; only ever tightens.
- ↕️ **Direction toggles** — enable longs and shorts independently.
- 📋 **Stats dashboard** (`table.new`) — net profit, win rate, profit factor, trades, max drawdown, live position, and filter status.
- 🔔 **Six alert conditions** for broker / webhook automation.
- 🎨 Viprasol palette (`#00ffaa` / `#ff3d71`).

## 🚀 How to install & run on TradingView

1. Open [TradingView](https://www.tradingview.com/) and pick any chart and timeframe.
2. Open the **Pine Editor** tab (bottom of the screen).
3. Copy the full contents of [`src/viprasol_ma_cross_strategy.pine`](src/viprasol_ma_cross_strategy.pine) into the editor.
4. Click **Add to chart**. The EMAs, entry markers, and dashboard appear immediately.
5. Open the **Strategy Tester** panel to inspect the backtest, then click the ⚙️ gear on the strategy to tune the inputs.
6. To automate, open **Alerts** → create an alert → choose a *Viprasol* condition and paste your webhook URL.

> **Tip:** Filters are off by default except the HTF trend filter — turn them on one at a time and watch the dashboard to see how each affects win rate and profit factor.

## ⚙️ Inputs / parameters

| Group | Input | Default | Notes |
|-------|-------|---------|-------|
| Signal | Fast / Slow MA length | 12 / 26 | Crossover periods |
| Signal | MA type | EMA | EMA / SMA / WMA / RMA / HMA |
| Signal | Signal source | close | Any series |
| Trend Filter | Require HTF trend filter | on | Gate by higher-timeframe trend |
| Trend Filter | Higher timeframe / EMA length | D / 50 | Trend reference |
| RSI Filter | Require RSI filter | off | — |
| RSI Filter | RSI length / long / short | 14 / 50 / 50 | Momentum thresholds |
| ADX Filter | Require ADX filter | off | — |
| ADX Filter | ADX length / smoothing / minimum | 14 / 14 / 20 | Trend strength |
| Volume Filter | Require volume filter | off | — |
| Volume Filter | Volume MA length / multiplier | 20 / 1.2 | Above-average volume |
| Session Filter | Restrict to a session | off | — |
| Session Filter | Window / timezone | 0930-1600 / America/New_York | Entry window |
| Session Filter | Close at session end | off | EOD flatten |
| Risk / Exits | Stop / target mode | ATR | ATR or Percent |
| Risk / Exits | ATR length | 14 | Volatility unit |
| Risk / Exits | SL / TP (× ATR) | 2.0 / 3.0 | ATR mode |
| Risk / Exits | SL / TP (%) | 1.5 / 3.0 | Percent mode |
| Risk / Exits | Use take-profit | on | Toggle TP |
| Trailing Stop | Enable ATR trailing stop | off | — |
| Trailing Stop | Trail / activation (× ATR) | 2.0 / 1.0 | Trail behavior |
| Direction | Allow long / short | on / on | Direction toggles |
| Dashboard | Show dashboard / position | on / Top Right | Stats table |
| Dashboard | Markers / HTF EMA / session shade | on | Visuals |

## 🗺️ Roadmap

- [x] EMA crossover core with HTF trend filter
- [x] ATR-based stop-loss / take-profit
- [x] Selectable MA type and source
- [x] RSI, ADX, volume, and session confluence filters
- [x] Percent-based SL/TP mode
- [x] ATR trailing stop
- [x] On-chart stats dashboard
- [x] Expanded alert conditions
- [ ] Optional pyramiding / scale-in logic
- [ ] Break-even stop after first target
- [ ] Multi-timeframe RSI confluence
- [ ] Companion indicator-only (study) version

## ❓ FAQ

**Does this repaint?**
The HTF trend filter uses `lookahead = barmerge.lookahead_off`, so confirmed-bar signals do not repaint. As with any strategy, the current (unconfirmed) bar can still change until it closes.

**Why are most filters off by default?**
To keep the baseline behavior close to v0.1 and let you add confluence incrementally. Enable filters one at a time and compare the dashboard stats.

**ATR vs Percent stops — which should I use?**
ATR adapts stop distance to current volatility (good across regimes). Percent gives fixed, predictable distances. Switch via the **Stop / target mode** input.

**Can I automate live trades?**
Yes — create a TradingView alert on any *Viprasol* alert condition and point it at your broker/webhook bridge. Always demo-test first.

**Which markets and timeframes?**
It is market-agnostic (stocks, crypto, forex, futures). Defaults suit intraday-to-daily; retune lengths and risk for your instrument.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo and create a feature branch.
2. Keep the code **Pine Script v6** and preserve the `[Viprasol]` branding and palette.
3. Test in the TradingView Strategy Tester and note before/after stats in your PR.
4. Open a pull request describing the change and rationale.

For larger ideas, open an issue first so we can align on scope.

## Contact — Viprasol Tech Private Limited

- Website: [viprasol.com](https://viprasol.com)
- Email: [support@viprasol.com](mailto:support@viprasol.com)
- Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) | WhatsApp: +91 96336 52112
- GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) | [LinkedIn](https://www.linkedin.com/in/viprasol/) | X [@viprasol](https://twitter.com/viprasol)

## License

[MIT](LICENSE) (c) 2025 Viprasol Tech Private Limited
