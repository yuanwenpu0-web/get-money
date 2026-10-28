# Bitcoin Market Regime Detector [RegimeRisk]

A TradingView indicator that classifies Bitcoin price action into five market regimes — **Bull, Bear, Range, Volatility, and Transition** — directly on your chart.

[![Open on TradingView](https://img.shields.io/badge/TradingView-Open%20Indicator-2962FF?logo=tradingview&logoColor=white)](https://www.tradingview.com/script/YOUR-SCRIPT-ID/)
[![Website](https://img.shields.io/badge/regimerisk.com-Live%20Regime%20Dashboard-0b0b0b)](https://regimerisk.com)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://mozilla.org/MPL/2.0/)
[![Pine Script v5](https://img.shields.io/badge/Pine%20Script-v5-00b894)](https://www.tradingview.com/pine-script-docs/)

![Bitcoin Market Regime Detector on a BTCUSD chart](assets/screenshots/hero.png)

> **What this repo is:** an open, transparent **reference implementation** of regime detection in Pine Script. It is intentionally simplified. The production system at **[regimerisk.com](https://regimerisk.com)** uses an ensemble machine-learning model and a proprietary labelling scheme — see [How this relates to RegimeRisk](#how-this-relates-to-regimerisk).

---

## The five regimes

Most indicators tell you *direction*. Regime detection tells you *what kind of market you're in* — which is what decides whether a given strategy should even be running.

| | Regime | Character |
|---|---|---|
| 🟢 | **Bull** | Established uptrend, strong directional movement |
| 🔴 | **Bear** | Established downtrend, strong directional movement |
| ⚪ | **Range** | Weak directional movement, mean-reverting / sideways |
| 🟠 | **Volatility** | Realised volatility in its top band — expansion / stress |
| 🔵 | **Transition** | Conflicting or not-yet-confirmed signals between states |

The indicator paints the chart background by regime, shows the current state in a panel, and fires an alert on every regime change.

---

## How it works

This reference uses **transparent, rule-based** classification — no black box:

1. **Trend** — price relative to a long EMA, plus the EMA's slope over a lookback.
2. **Volatility** — ATR as a percentage of price, ranked against its own recent history (percentile).
3. **Directional strength** — ADX, to separate trending conditions from ranges.

These combine in a fixed priority order (volatility → range → trend → transition). Every threshold is a tunable input. The full logic is ~80 lines of commented Pine in [`src/bitcoin_market_regime_detector.pine`](src/bitcoin_market_regime_detector.pine) — read it, fork it, change it.

> ⚠️ This is deliberately a **baseline**. It is not the model that powers RegimeRisk. See [below](#how-this-relates-to-regimerisk).

---

## Install

**On TradingView (recommended)**
Open the published indicator → **[Bitcoin Market Regime Detector [RegimeRisk]](https://www.tradingview.com/script/YOUR-SCRIPT-ID/)** → add to favourites → apply to any BTC chart.

**From source**
1. Copy the contents of [`src/bitcoin_market_regime_detector.pine`](src/bitcoin_market_regime_detector.pine).
2. In TradingView, open **Pine Editor**, paste, and click **Add to chart**.

Works on any symbol and timeframe; defaults are tuned with BTC daily / 4H in mind.

---

## Inputs

| Input | Default | Controls |
|---|---|---|
| Trend EMA length | 100 | Baseline trend reference |
| Trend slope lookback | 20 | Bars used to measure EMA slope |
| Volatility (ATR) length | 20 | ATR smoothing |
| Volatility percentile window | 200 | History for ranking current volatility |
| High-volatility percentile | 80 | Percentile above which the market is "Volatility" |
| ADX length | 14 | Directional-movement smoothing |
| ADX range threshold | 20 | Below this = Range |
| ADX trend threshold | 25 | Above this = trend can be confirmed |
| Show regime panel | on | Toggles the on-chart status panel |

---

## Alerts

The indicator fires on **regime change**, two ways:

- **Built-in alert** — add an alert with condition *Bitcoin Market Regime Detector → Regime change*.
- **Dynamic alert** — the script also calls `alert()` on bar close with the new regime name, so the message names the state it switched to.

---

## How this relates to RegimeRisk

This indicator is a **free, on-chart taste** of one idea: markets move through distinguishable regimes, and knowing the current one is more useful than another oscillator.

The full system at **[regimerisk.com](https://regimerisk.com)** goes well beyond what a rule-based Pine script can do:

- An **ensemble ML model** — gradient-boosted trees (XGBoost / LightGBM / CatBoost) plus an LSTM, combined by a meta-learner — rather than fixed thresholds.
- **Probabilistic** output: the likelihood of each of the five regimes, not a single hard label.
- A **proprietary labelling scheme** (Ground Truth v2.0) the model is trained against.
- Coverage beyond BTC, a live regime dashboard, history, and alerts.

If this indicator is useful to you, the dashboard is the next step up → **[regimerisk.com](https://regimerisk.com)**

---

## Versioning

Changes are tracked in [`CHANGELOG.md`](CHANGELOG.md). The Pine source carries a matching `@version` annotation in its header.

---

## Disclaimer

Provided for research and educational purposes only. **Not financial advice** and not a recommendation to buy or sell any asset. Regime labels can be wrong and can change at the edges of a state transition. Do your own research and manage your own risk.

---

## License

[Mozilla Public License 2.0](LICENSE) — the standard for open-source TradingView scripts. Free to use, modify, and redistribute under its terms; modifications to MPL-covered files stay under MPL.

---

## About

Built by **RegimeRisk**, a [Nexus AI Labs](https://regimerisk.com) project — crypto market regime detection powered by ensemble machine learning.

[regimerisk.com](https://regimerisk.com) · [TradingView](https://www.tradingview.com/u/YOUR-HANDLE/)
