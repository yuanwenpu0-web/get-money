# ETH Multi-Timeframe Backtester

A Python backtesting and optimization engine for a multi-timeframe EMA trend-following strategy, built to replace manual TradingView parameter testing with systematic, automated search — and to demonstrate a rigorous, honest approach to strategy validation.

**This is a research and engineering portfolio project, not financial advice or a proven trading system.** See [Honest Limitations](#honest-limitations-read-this) before drawing any conclusions from the numbers below.

---

## Why this project

Manually testing parameter combinations in TradingView's Strategy Tester doesn't scale — there's no way to search thousands of combinations, cross-validate against multiple years of data, or catch subtle simulation bugs by hand. This project builds that missing tooling:

- A **Numba-JIT-compiled backtest engine** fast enough to run thousands of parameter trials per search (~2ms per full-history backtest)
- **Optuna (TPE sampler)** for efficient parameter search across a space too large to brute-force
- **Exhaustive grid verification** as a secondary check on any parameter the search claims is optimal
- **Cross-validation against real trade-level execution data** — not just backtest-vs-backtest comparisons

## What's in this repo

| File | Purpose |
|---|---|
| `backtest_engine.py` | Core simulation engine — multi-timeframe EMA stack, entry/exit logic, ATR volatility filter, close-based stop-loss confirmation |
| `optimize.py` | Optuna-based parameter search wrapper, CSV data loader (auto-detects multiple exchange formats) |
| `run_final_config.py` | Standalone runner reproducing the documented final configuration end-to-end |
| `case_study.md` | Full write-up of the methodology, findings, and — importantly — the mistakes caught along the way |

## Methodology

1. **Port the strategy logic** from Pine Script (TradingView) to Python for programmatic testing
2. **Search broadly** — EMA lengths, stop-loss/take-profit scale and ratio, and filter combinations, ranked by net profit with a minimum trade-count floor to avoid overfitting to sparse samples
3. **Verify exhaustively** — every "best" parameter from the search gets re-checked against a full grid of nearby values to confirm it's a genuine local optimum, not a sampling artifact
4. **Validate out-of-sample** — test on data outside the original tuning window (extended from 6 months → 18 months → 30 months across this project) specifically to catch overfitting
5. **Cross-check against reality** — compare backtest trades against actual executed trades from a live paper-trading account, which is what ultimately caught the most important bug in this project (see below)

## The most important part of this project: a bug I caught, not a result I'm proud of

Early results looked excellent — a configuration showing 30/30 profitable months, ~27% win rate, strong net profit. **Comparing against real executed trades revealed the backtest was silently overstating performance by roughly 87%.**

The cause: the engine's stop-loss simulation assumed every stop-loss exit closed at *exactly* the nominal stop distance. In reality, one of the exit mechanisms (a "close-based" stop, designed to avoid getting stopped out by brief price wicks) can only confirm a stop once a full bar closes beyond the level — meaning the real exit price can overshoot the nominal stop significantly during fast moves. The backtest didn't model this overshoot at all.

**How it was caught**: parsing and analyzing real trade-level data from a paper account, chronologically, and noticing that realized losses never matched the theoretical stop-loss value exactly — they clustered around it but were consistently larger, with a long tail of much larger outliers during fast market moves.

**What this demonstrates**: the value of validating a model against ground truth rather than trusting internal consistency checks alone. Every verification step performed *before* this (exhaustive grids, monthly consistency checks, robustness sweeps) passed cleanly — none of them could catch a bug baked into the core simulation assumption itself, because they were all built on top of it.

Full details, including the corrected results and the re-optimization that followed, are in [`case_study.md`](case_study.md).

## What was tested and rejected

A meaningful part of this project was systematically testing ideas that *didn't* work, and documenting why — this is arguably more informative than the ideas that did:

| Approach | Outcome |
|---|---|
| ADX trend-strength filter | Rejected — didn't discriminate between good and bad market conditions |
| Volume confirmation | Rejected — reduced profit at every threshold tested |
| Maximum trade-hold-duration cutoff | Rejected — disproportionately cut the strategy's best trades |
| Consecutive-loss circuit breakers (multiple triggers/cooldowns) | Rejected — worsened drawdown rather than improving it |
| Cross-asset (BTC) confirmation | Rejected — no improvement in trade quality |
| Trend-persistence requirement | Rejected — same failure pattern as other reactive filters |

The pattern that emerged: filters evaluating **present market conditions** (e.g., current volatility relative to its own recent average) tended to help; filters reacting to **recent trade history** (e.g., "pause after N losses") consistently did not. This is documented in detail in the case study, since understanding *why* something fails is often more useful than the failure itself.

## Honest limitations (read this)

- Backtested on historical exchange data; live execution (spread, slippage, commissions, order routing) will differ, sometimes significantly — this project found real evidence of that gap
- No claim of forward performance — 30 months of favorable backtest results is meaningfully more evidence than 6 months, but it is not a guarantee
- This is not investment advice, and the specific strategy parameters are not the point of this repository — the methodology and validation process are

## Tech stack

Python · NumPy · Pandas · Numba (JIT compilation) · Optuna (TPE hyperparameter search) · Pine Script (original strategy source)

## Setup

```bash
pip install numpy pandas numba optuna
python3 run_final_config.py --csv your_1min_ohlcv_data.csv
```

Data loader auto-detects several common formats (labeled CSV with a time column, raw headerless Binance klines, HistData's semicolon-delimited format).

## License

MIT — see [LICENSE](LICENSE).
