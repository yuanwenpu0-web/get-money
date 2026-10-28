# BTC 1% Radar

**Know which side of Bitcoin is likelier to print ±1% first — and when the right call is no trade.**

A public-data scorecard for [Binance](https://www.binance.com) USD-M **Bitcoin perpetual futures**. It reads structure, open interest, funding, taker flow, and order-book depth, then prints **LONG**, **SHORT**, or **NO TRADE**. Next to that sits a walk-forward **Nifty 50 / NSE** research lab that would rather keep a simple paper sleeve than promote an overfit one. No API key. No order routing. Not a chat model placing trades.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://github.com/ryaneditraj/trading-ai/actions/workflows/test.yml/badge.svg)](https://github.com/ryaneditraj/trading-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Binance public data](https://img.shields.io/badge/data-Binance%20USD--M%20public-F0B90B)](#tech-stack)
[![Pine Script v5](https://img.shields.io/badge/charts-TradingView%20Pine%20v5-2962FF)](pinescript/README.md)

> Not financial advice. Cards are research output. At 100x, a ±1% target sits past a typical liquidation, and the card says so before it says LONG or SHORT.

## Demo

Real terminal runs of the offline demo (`python -m btc_radar --demo`). The capture network received **HTTP 451** from `fapi.binance.com` (Binance blocks some regions), so these images are the supported offline path, not a mockup and not a live book.

<p align="center">
  <img src="docs/screenshots/btc-radar-demo.png" alt="BTC 1% Radar offline demo: Bitcoin near $97,488, LONG 9.0 out of 10, SHORT 1.0 out of 10, with a 100x liquidation warning that the 1% target is beyond liquidation distance" width="720">
</p>

<p align="center">
  <img src="docs/screenshots/btc-radar-no-trade.png" alt="Same demo snapshot with the macro gate set to elevated: the score is still long, and the signal is NO TRADE" width="720">
</p>

A short refresh loop (`--watch 3` on the same demo snapshot):

<p align="center">
  <img src="docs/screenshots/btc-radar-watch.gif" alt="BTC 1% Radar watch mode clearing and reprinting the LONG scorecard every few seconds" width="720">
</p>

## Features

- **BTC 1% Radar** — ASCII scorecard from Binance USD-M public REST, plus a short sample of the public liquidation websocket
- **Strict 8/10 gate** — a mixed tape prints `NO TRADE` instead of a coin-flip
- **Labeled points** — trend, VWAP, taker CVD, open interest with price, funding, liquidity cluster, optional macro bias
- **100x liquidation warning** — distance is about `1/leverage − maintenance margin`, and a ±1% target inside that band is called out
- **Macro window** — estimated CPI, NFP, and FOMC hours in `btc_radar/macro_events.json`, overridable with `--macro`
- **Walk-forward radar backtest** — did the signal pick the side that hit ±1% first?
- **Offline `--demo`** — synthetic snapshot, no network
- **Nifty paper sleeve** — locked Paper A in `strategies/nifty_edge_paper.py`, with TradingView Pine Script ports
- **Anti-overfit lab** — `research/nifty_edge_8h/` records the gates a sleeve had to clear, and the ones that only *looked* better

## Tech stack

| Layer | Choice |
|--------|--------|
| Language | Python 3.11+ |
| Radar data | Binance USD-M public REST (`fapi.binance.com`) and `!forceOrder@arr` |
| Research data | Yahoo Finance via `yfinance` (NSE names and GIFT overnight studies) |
| Numerics | pandas, numpy |
| Charts | TradingView Pine Script v5 |
| Tests | pytest, offline |

No broker keys. No LLM in the scoring loop. The "AI" in the repo name is the research process: measure, gate, and refuse the cute parameter.

## Quick start

```bash
git clone https://github.com/ryaneditraj/trading-ai.git
cd trading-ai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
python -m btc_radar --demo --macro low
```

`.env.example` is a reminder that credentials are not used. You do not need to copy it for the radar.

## Usage

| Command | What you get |
|---------|----------------|
| `python -m btc_radar --demo --macro low` | Offline card. This is the screenshot above. |
| `python -m btc_radar --demo` | Same snapshot, with the packaged macro calendar |
| `python -m btc_radar` | Live BTCUSDT card from the public API |
| `python -m btc_radar --json` | The same result as JSON |
| `python -m btc_radar --watch 60` | Refresh every 60 seconds |
| `python -m btc_radar --macro elevated` | Force the don't-trade window |
| `python -m btc_radar --leverage 20 --risk-usd 250` | Recompute the liquidation line and the cash reminder |
| `python -m btc_radar --symbol ETHUSDT` | Another USD-M symbol (letters and digits only) |
| `btc-radar --demo` | Same CLI after `pip install .` |
| `python strategies/btc_1pct_radar_backtest.py --days 14` | Walk-forward hit-rate on recent Binance history |
| `python strategies/nifty_edge_paper.py` | Paper sleeve. Downloads Yahoo data on first run. |

`strategies/btc_1pct_radar.py` is the same radar entry point.

Live calls fail with a plain HTTP 451 message in regions Binance blocks. Use `--demo` there. The backtest asks Binance for about 28 days of history; derivatives history is not a deep archive.

## How a card is scored

```
Binance public snapshot
        │
        ▼
 structure │ derivatives │ liquidity │ catalyst
        │
        ▼
   10-point score  →  LONG / SHORT / NO TRADE  →  ASCII or JSON
```

A side prints only at **8/10**, and only if it leads the other side. Macro `ELEVATED` forces `NO TRADE` unless you pass `--force`. Funding that is already crowded withholds the "funding ok" point instead of pretending the crowd is an edge.

## Project structure

```
btc_radar/          Product: fetch, features, score, render, backtest, demo
strategies/         Runnable sleeves. nifty_edge_paper.py is the paper book.
                    nifty_edge_r*.py are dated research passes, not the champion.
research/           Walk-forward engine, India edge search, locked 8-hour campaign
pinescript/         TradingView ports (Turtle, Nifty, paper sleeve)
tests/              Offline pytest suite
docs/screenshots/   Real terminal captures
```

The Nifty campaign write-up is [`research/nifty_edge_8h/FINAL_REPORT.md`](research/nifty_edge_8h/FINAL_REPORT.md). Pine notes are in [`pinescript/README.md`](pinescript/README.md).

## Roadmap

- [ ] Refresh `macro_events.json` from an official calendar instead of estimates
- [ ] Multi-exchange depth, not only the Binance book
- [ ] Save radar cards and a rolling hit-rate
- [ ] Keep the research archive easy to separate if the radar should stand alone
- [x] Offline demo, pytest, and a security pass on the public client

## FAQ

### What is the BTC 1% Radar?

It is a Python CLI that scores Binance USD-M Bitcoin futures and answers a narrow question: which side is likelier to trade ±1% first, long or short? If neither side clears 8/10, the card says NO TRADE.

### Does the BTC 1% Radar need a Binance API key?

No. It uses public futures market data only: klines, premium index, open interest, funding, long/short ratios, taker volume, order-book depth, and the public liquidation stream. It never asks for a key, and it never sends one.

### Does this repository place trades?

No. There is no broker adapter and no order endpoint. `--risk-usd` is a printed reminder of how much cash you said you were willing to lose, not a position size sent anywhere.

### What do LONG, SHORT, and NO TRADE mean?

LONG means the long points cleared the threshold and beat the short points. SHORT is the mirror. NO TRADE means the tape is mixed, or a macro window told the model to stand aside. The percentages on the bars are the score scaled to 10, not a measured win rate.

### Why does 100x warn that the 1% target is past liquidation?

Isolated liquidation distance is roughly `1/leverage` minus maintenance margin. At 100x that is about ±0.60%, which is inside a ±1% target. The warning is the point of the tool: a target you cannot survive is not a target.

### Can I run it offline?

Yes. `python -m btc_radar --demo` builds a synthetic snapshot and scores it locally. `pytest` does not open a socket.

### How is the Nifty research different from the Bitcoin radar?

The radar is a short-horizon crypto scorecard. The Nifty lab is a systematic equity study on NSE large caps: walk-forward splits, cost ladders, and bootstrap gates. The paper sleeve stays the simpler campaign twin when extra filters only tie the last walk-forward window. Read the [final report](research/nifty_edge_8h/FINAL_REPORT.md) before treating any JSON stat as a track record.

### Which files are safe to paper-trade from?

Start with `strategies/nifty_edge_paper.py` and the Pine port in `pinescript/`. The `nifty_edge_r*.py` files are the search history. They are not a stack of live systems.

### What Python version and libraries does it use?

Python 3.11 or newer, with pandas, numpy, requests, yfinance, and pyarrow. Development installs add pytest. See `requirements.txt` and `pyproject.toml`.

### Is a backtest here a promise of future returns?

No. Past windows are not future results. Nifty numbers in the research report assume costs. They are not a live audited record. Crypto leverage can liquidate an account before the 1% print.

### Is there a project website?

The homepage is this GitHub repository: [github.com/ryaneditraj/trading-ai](https://github.com/ryaneditraj/trading-ai). There is no separate marketing site, so there is no HTML sitemap or Open Graph page. [`llms.txt`](llms.txt) is the short machine-readable summary for answer engines.

### How do I report a security problem?

Use a [private GitHub security advisory](https://github.com/ryaneditraj/trading-ai/security/advisories/new). Details and the audit notes are in [SECURITY.md](SECURITY.md). Do not paste API keys into issues.

## Contributing

Issues and pull requests are welcome. Tests must stay offline. Please don't add live order entry. The short version is [CONTRIBUTING.md](CONTRIBUTING.md), and community standards are the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE). Use it, fork it, argue with the score. Do not treat the output as a trade instruction.

## Author

**Ryan Editraj** ([@ryaneditraj](https://github.com/ryaneditraj)) — AI builder and tech creator.

- GitHub: [github.com/ryaneditraj](https://github.com/ryaneditraj)
- Instagram: [instagram.com/ryaneditraj](https://instagram.com/ryaneditraj)
