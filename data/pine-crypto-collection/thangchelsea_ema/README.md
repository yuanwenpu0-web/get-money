# EMA 9/45/200 Trend Strategy — Breakout / Pullback

A long-only trend-following strategy built on three EMAs (9 / 45 / 200), with:

- a **TradingView Pine Script v6 strategy** you can paste into the Pine Editor, and
- the **Python backtest** it was researched with (BTC/USDT and PAXG/USDT, Jan 2021 → Sep 2026), including out-of-sample checks and fee sensitivity.

> **Not financial advice.** This is research on historical data. Past performance does not guarantee future results.

## The strategy

| Step | Rule |
|---|---|
| Trend filter | Breakout: EMA9 > EMA45 > EMA200 on the crossover bar. Pullback: waits after **every** bullish cross (so it can catch a missed Breakout) and needs EMA9 > EMA45 > EMA200 on the pullback bar and before entry (input *Pullback trend filter* can switch this to EMA9 > EMA45 and close > EMA200). The first 200 bars are skipped while the EMAs warm up. The result tables below use the original Pullback rule (wait only after an aligned cross) — see *Further research* for the new rule. |
| Signal | EMA9 crosses above EMA45 while the trend filter is true. |
| Entry — **Breakout** | Buy-stop at the high of the crossover bar. |
| Entry — **Pullback** | After any bullish cross (even one that did not meet the trend filter), wait for price to come back to EMA45. Signal bar = the first bar that touches EMA45 but closes above it, **or** (if price closes below EMA45) the first bar that closes back above it. Buy-stop at the high of that bar. |
| Pending order | Cancelled if EMA9 crosses back below EMA45 before it fills. One trade per crossover. |
| Stop loss | Entry − k × ATR(14), active from the entry bar (k = 2.5 by default). |
| Exit | When EMA9 crosses below EMA45: sell-stop at the low of that bar; cancelled if EMA9 crosses back above EMA45 first. |
| Sizing | Fixed % of equity at risk per trade (1% by default). |

Stops and entries are stop orders placed at bar close for the next bar. If the bar opens beyond the level (gap), the fill is at the open. No look-ahead, no repainting.

## Results (after 0.04% fee per side)

Full period **2021-01-01 → 2026-09-23**. Out-of-sample = trades entered after 2025-09-30 (the rules were tuned on data before that date). R = multiples of the initial risk.

### BTC/USDT

| Timeframe | Entry | ATR k | Trades | Win rate | Profit factor | Expectancy | Max DD | Out-of-sample | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **1H** | **Breakout** | **2.5** | 205 | 28.3% | **1.84** | +0.53R | −15.3R | PF 1.18 (28 trades) | **Best** |
| 1H | Pullback | 2.5 | 159 | 26.4% | 1.84 | +0.50R | −13.1R | PF 1.15 (23 trades) | Alternative |
| 4H | Breakout | 2.0 | 60 | 21.7% | 1.46 | +0.31R | −10.8R | 0 / 7 winners | Not recommended |
| 1D | Breakout | 2.5 | 11 | 45.5% | 2.96 | +0.85R | −2.0R | 1 trade | Too few trades |

### Gold (PAXG/USDT)

| Timeframe | Entry | ATR k | Trades | Win rate | Profit factor | Expectancy | Max DD | Out-of-sample | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **4H** | **Breakout** | 2.0 | 56 | 37.5% | 2.51 | +0.74R | −6.5R | PF 6.2 (8 trades) | **Best** |
| **4H** | **Pullback** | 2.5 | 40 | 45.0% | **2.79** | +0.71R | **−3.9R** | PF 2.35 (6 trades) | **Best (lower DD)** |
| 1H | Breakout | 4.0 | 214 | 35.0% | 1.53 | +0.20R | −19.3R | – | Breaks at 0.1% fees |
| 1D | Breakout | 2.5 | 11 | 54.5% | 9.75 | +3.45R | −1.7R | 0 trades | Too few trades |

### $1,000 simulation (BTC 1H Breakout 2.5×, 1% risk per trade, fees included)

| System | Final value | Max DD |
|---|---|---|
| Strategy signals only | $2,274 (×2.27) | **−17.3%** |
| Buy & hold BTC | $2,897 (×2.90) | −77.2% |

The strategy made less than buy & hold over this bull-market period but with a far smaller drawdown.

### Caveats

- **Low win rate (≈25–45%)**: most profit comes from a few large trends; losing streaks of 10+ trades are normal. Keep risk per trade small.
- **Favourable test period**: both BTC and gold rose strongly from 2021 to 2026. Expect weaker results in sideways or falling markets.
- **Small out-of-sample samples** (BTC 1H ≈ 28 trades, gold 4H 6–8 trades) and some **overfitting risk** (several ATR levels, two filters and five pullback variants were tried on the same data).
- **Slippage is not modelled** in the Python backtest — a bigger issue for thin PAXG liquidity. PAXG also trades on weekends, unlike spot gold.
- On 1H there is no lower-timeframe data, so a stop hit on the entry bar is resolved with the worst-case assumption. On 4H/1D the backtest uses 1H bars inside the entry bar.

## TradingView

File: [`tradingview/ema_9_45_200_trend_strategy.pine`](tradingview/ema_9_45_200_trend_strategy.pine) (a version with Vietnamese labels is in `ema_9_45_200_trend_strategy_vi.pine`).

`tradingview/ema_9_45_200_trend_strategy_vi_tradingview.pine` is the extended Vietnamese version the author runs on TradingView: same entry/exit rules, plus a maximum-leverage cap on position size (`maxLev`, with 10x margin so large low-ATR positions still fill), rule-based position tracking from OHLC (so levels, arrows and alerts do not depend on the Strategy Tester's sizing), entry/exit arrows instead of the Strategy Tester trade labels, and its own EMA colours. Its header comment still mentions the earlier "close > EMA200" Pullback default; the code default is EMA9 > EMA45 > EMA200.

1. Open TradingView → **Pine Editor** → create a new strategy and replace the code with the file's contents.
2. **Save**, then **Add to chart** (e.g. BINANCE:BTCUSDT 1H or BINANCE:PAXGUSDT 4H).
3. Check the **Strategy Tester** tab. Adjust *Entry type*, *Pullback trend filter*, *Stop loss = k x ATR(14)* and *Risk per trade* in the settings.

Defaults: initial capital 1,000 USD, commission 0.04% per side, slippage 2 ticks, long only, no pyramiding.

On the chart: green = pending buy-stop, light red = **projected stop loss** (buy-stop − k × ATR, shown as soon as an order is pending), red = active stop loss, fuchsia = pending exit level. Levels are drawn as dots, so a level that exists for a single bar (e.g. a buy-stop that fills on the next bar, or a fill that is stopped out on the same bar) is still visible. On the last bar the active levels are also drawn as dashed lines extending right with price labels. ATR(14) and the stop distance are shown in the Data Window. The script's ATR is an EMA of the true range (same as the Python backtest), so it differs slightly from TradingView's built-in ATR indicator (RMA / Wilder).

### Reading the chart (manual trading)

| On the chart | Meaning | When it shows |
|---|---|---|
| Green ▲ below a bar | Signal: EMA9 crosses above EMA45 with the EMAs aligned | Signal bar |
| **Green** dots + **"Buy stop …"** label | **Entry price**: place a buy-stop order here | From the signal bar close until filled or cancelled |
| **Light red** dots + **"Projected SL …"** label | **Projected stop loss** = buy stop − 2.5 × ATR | Together with the buy stop |
| **"Long"** arrow | Order filled (actual entry price) | Fill bar |
| **Red** dots + **"Stop loss …"** label | **Actual stop loss** = fill price − 2.5 × ATR | While in a position, including a fill stopped out on the same bar |
| **Fuchsia** dots + **"Exit stop …"** label | Exit level (low of the bearish-cross bar) | In a position after EMA9 crosses below EMA45 |
| **"Exit"** arrow | Position closed | Exit bar |

Manual workflow: place the buy-stop and stop loss on your exchange when "Buy stop" / "Projected SL" appear; size = account × risk % ÷ SL distance; cancel the order if the green level disappears; after a fill the stop is fill price − SL distance; move the stop up to "Exit stop" when it is higher. Levels update at bar close only, and the script never places orders itself.

**Alerts:** at bar close the script fires `alert()` messages for: new pending buy-stop (with projected SL and SL distance), buy-stop cancelled, long filled (with stop loss), filled and stopped out on the same bar, new exit stop (with the active stop), exit stop cancelled, and position closed. To enable: on the chart click **Alert** → Condition: this strategy → **"alert() function calls only"** → choose notifications → Create. One alert covers all events. Re-create the alert after editing the script (alerts run on the script version they were created with). The script does not place orders on your exchange. Note: alerts on indicators/strategies need a paid TradingView plan (the free Basic plan allows 0 technical alerts).

TradingView results will differ from the Python backtest: TradingView loads a limited number of historical bars (depending on your plan), its broker emulator decides same-bar stop fills from OHLC (or Bar Magnifier), and if the stop loss and the exit level are both hit in one bar the script fills the higher level. On BTCUSDT 1H (Jan–Sep 2026) the script gave 20 trades / PF 1.26 versus 24 trades / PF 1.33 in Python.

## Python backtest

```bash
pip install -r requirements.txt
cd scripts
python full_track.py          # full track record 2021 → 2026 for 1H / 4H / 1D, ATR 1.0–4.0 → results/full_track_summary.csv
python oos_test.py            # out-of-sample check (Breakout)
python oos_pullback_test.py   # out-of-sample check (Breakout and Pullback)
python pullback_test.py       # Breakout vs Pullback, in-sample
python backtest_main.py       # in-sample 2021 → Sep 2025 statistics + trade lists
python asset_test.py ../data/paxg_h1.csv ../data/paxg_h4.csv PAXG   # any asset
python download_binance.py PAXGUSDT 4h ../data/paxg_h4.csv          # fetch closed candles from the Binance public API
python ema_full_report.py ../data/btc_h1_binance.csv ../data/btc_h4_binance.csv BTC 9 45 [A|B]   # full 1H/4H/1D table for any EMA pair; B = Pullback filter close > EMA200
python ema_tf_report.py ../data/btc_m15_binance.csv BTC 15m 6.0 9 45 [A|B]                        # same for a single timeframe (e.g. 15m)
python ema_pair_test.py ../data/btc_h1_binance.csv ../data/btc_h4_binance.csv BTC 9,45 10,50      # compare EMA pairs
python h4_report.py ../data/btc_h1_binance.csv ../data/btc_h4_binance.csv                         # detailed 4H table
python fx_fee_check.py                                                                             # EUR/USD and USD/EUR with forex-like fees
```

| Folder | Contents |
|---|---|
| `data/` | Binance candles from 2021: BTC/USDT (15m, 1H, 4H, 1D), PAXG/USDT (1H, 4H), EUR/USDT (1H, 4H) and its inverse USD/EUR; `*_binance.csv` files were refreshed in October 2026, end dates per file are listed in the research notes |
| `scripts/` | Backtest engine and experiments (comments and console output are in Vietnamese) |
| `results/` | Summary tables and per-trade lists |
| `tradingview/` | Pine Script strategy |
| `docs/research-notes-vi.md` | Full research log in Vietnamese: every table, the bugs found and fixed (off-by-one bar, unrealistic fills, stop hit on the entry bar, EMA warm-up, data gaps), and independent verification |

Implementation notes: EMAs use pandas `ewm(adjust=False)` (the Pine script seeds its EMAs the same way); ATR is an EMA of the true range; the data is split into segments at gaps (> 6 h on 1H) and each segment gets its own 200-bar warm-up. Fees are converted to R as `fee × (entry + exit) / stop distance`.

## Further research (October 2026)

Details and full tables are in [`docs/research-notes-vi.md`](docs/research-notes-vi.md) (Vietnamese). All figures after 0.04% fee per side; out-of-sample = trades entered from 1 Oct 2025. Each section uses data downloaded on the date it states, so re-running with newer data gives slightly different numbers.

- **Out-of-sample update, BTC 1H EMA9/45 Breakout 2.5×**: with data to 9 Oct 2026, out-of-sample PF fell to 0.84 (35 trades; 6 of the 7 trades from 27 Sep to 6 Oct lost). Not conclusive for a low-win-rate system, but worth monitoring.
- **EMA period sensitivity**: results change a lot with small EMA changes. BTC 1H Breakout 2.5× PF: 8/40 1.33, **9/45 1.77**, 10/50 1.33, 11/55 1.26, 12/60 1.42; BTC 4H: 1.08, 1.42, **2.01**, 1.28, 1.35. Every pair is profitable on BTC 1H, so the trend-following edge is real, but the headline PF of the best pair is likely optimistic.
- **EMA10/50** (`results/ema10_50_*.csv`): best BTC configuration becomes 4H Breakout 2.0× (PF 2.16, 50 trades, only 4 out of sample); gold 4H Breakout 2.0× PF 3.23 and Pullback 3.0× PF 4.28.
- **15-minute BTC** (`results/ema*_btc_15m.csv`): ~1,000 trades, Breakout 2.5× PF 1.28, Max DD −52R; more total R than 1H only because of the trade count and one +94.5R trade (stop only 0.20% of price). Removing the 5 largest trades leaves PF 1.03; PF 0.99 at 0.1% fees. Not better than 1H.
- **Pullback filter "close > EMA200"** (`results/ema9_45_locB_*.csv`): better than the stacked filter on BTC 1H (2.5×: +103R vs +79R, same Max DD) and gold 4H (PF 3.6 vs 2.8), worse on BTC 4H, gold 1H and BTC 15m; the extra trades it adds lost out of sample (small samples).
- **Pullback waits after every bullish cross** (`scripts/pullback_arm_test.py`, `results/pullback_arm_test.csv`, data to 17 Oct 2026): if the cross happens before the EMAs are stacked (no Breakout), a later pullback to EMA45 is still taken once EMA9 > EMA45 > EMA200. BTC 1H 2.5×: 264 trades, PF 1.86, +135.7R, Max DD −12.9R, out of sample PF 2.21 (+31.4R, 47 trades) vs the old rule 163 trades, PF 1.81, +79.4R, out of sample PF 0.95. Gold 4H 2.5×: PF 3.49, +60.1R (old: 2.79, +28.3R). Worse on gold 1H; BTC 4H improves but stays weak out of sample. The stacked filter beats "close > EMA200" under this rule, so it is now the default. The rule was designed after looking at one chart example, so keep monitoring it.
- **EUR/USD and USD/EUR** (`results/eurusd_*.csv`, `results/usdeur_summary.csv`): no edge even without fees (PF below 1 almost everywhere).

## License

[Mozilla Public License 2.0](LICENSE). You may use, modify and share this code; modified versions of these files must stay open-source under the same license.
