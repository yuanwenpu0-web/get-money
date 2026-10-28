# Findings & Decisions

## Requirements
-

## Research Findings
- The built-in parameter selection does not search SL/TP. Existing orders do not reliably retain original stops or signal IDs; realistic order replay is approximate.
- Current order inventory: 26 paper orders, 20 closed, 6 open, with 15 stop exits and 5 target exits. Closed realized PnL is approximately -270.05 USDT. Entry and exit fees are included in realized PnL.
- Evaluate strategies used by current orders, keep configuration and trading database read-only, and separate candidate selection from out-of-sample evaluation.
- Historical order signals have no foreign-key link, but 26/26 have matching near-entry candidate signals; target and slippage-adjusted entry agree. This is a cautious inference, not a verified identity. ATR stored in signal reasons is rounded and cannot be trusted for low-price assets.
- Paper-order replay skips the partial entry bar. Later stop values can reflect trailing changes. Strategy reruns and historical-order counterfactuals have different assumptions and should be reported separately.
- The read-only 1h strategy rerun covered 16 historical order symbol-strategy legs with 180 days of data, chronologically split into train/validation/test. Current 1.5/2.5 ATR yielded +0.7138% mean test return and -4.2772% mean individual-leg drawdown. Validation-selected 2.0/2.5 ATR yielded +0.3912% and -3.5973%, respectively, in the later test. Independent account per leg, not portfolio returns.
- The current 1.5/2.5 parameters also had the highest validation-period net return; this criterion was examined after test results were seen and is exploratory. Do not promote the best test-only candidate as validated.
- Of 20 closed orders, 18 passed matching, ATR and baseline exit checks. Fixed-entry replay on those 18 gave -162.9663 USDT at current parameters and -99.5717 at 1.0/2.5, but ignores changed sizing and later trades. The 1.0/2.5 strategy-level test was -0.3863% vs current +0.7138%, illustrating the danger of fitting a small in-sample set.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
-
