# Pine Scripts

TradingView indicators and strategies written in Pine Script v6. Every script in
this repository is open source, commented, and built around one idea: code that
behaves correctly on live bars, not just code that compiles.

## Scripts

| File | What it demonstrates |
|---|---|
| [`mtf-ema-non-repainting.pine`](mtf-ema-non-repainting.pine) | Higher-timeframe values requested without repainting |
| [`alert-engine-webhook.pine`](alert-engine-webhook.pine) | Alerts that fire once per closed bar, with a JSON payload |
| [`strategy-template-risk.pine`](strategy-template-risk.pine) | Risk-based position sizing, ATR stops, targets in R |

## The repainting problem

This is the single most common defect in multi-timeframe scripts, and the reason
most "perfect in backtest" indicators fall apart in live trading.

The direct call looks harmless:

```pine
request.security(syminfo.tickerid, "60", ta.ema(close, 21))
```

On historical bars it returns the value of a **closed** higher-timeframe bar. In
realtime it returns the value of the bar **still forming**, which changes with
every tick. Signals appear, disappear, and reappear inside the same candle.

The fix is to request the `[1]` offset with lookahead enabled:

```pine
request.security(syminfo.tickerid, htf, emaFast[1], lookahead = barmerge.lookahead_on)
```

That combination returns the last closed higher-timeframe bar in both contexts,
so the plotted value is stable. The trade-off is explicit and worth stating: the
line updates one higher-timeframe bar behind. That delay is the price of a signal
that does not lie to you.

`mtf-ema-non-repainting.pine` plots both versions side by side so the difference
is visible rather than asserted. Enable **Overlay repainting version**, open a low
chart timeframe against a much higher HTF setting, and watch the two pairs of
lines while a higher-timeframe bar forms.

## Alert reliability

`alert-engine-webhook.pine` addresses the two failure modes that account for most
"my alerts are broken" support requests:

- **Intrabar firing.** A cross that appears and vanishes before the bar closes
  should never reach the alert. The script gates every signal on
  `barstate.isconfirmed`.
- **Duplicate firing.** The same bar producing two alerts. A `var` counter
  records which bar already fired and blocks the second.

The payload is emitted as JSON so a webhook endpoint can parse it without string
surgery:

```json
{"strategy":"ema-cross","symbol":"BTCUSDT","exchange":"BINANCE","tf":"60","side":"long","price":77315.26,"time":"2026-08-22T05:00:00Z"}
```

## Risk sizing

`strategy-template-risk.pine` solves position size backwards from the stop: if the
stop is hit, the loss equals a fixed percentage of current equity, regardless of
instrument or volatility. A fixed contract count cannot do that — it risks a
different amount on every trade, which is why equity curves built on fixed size
are so hard to interpret.

Commission and slippage are declared in the `strategy()` header rather than added
after the fact, because a backtest that ignores costs is not a backtest.

## Usage

Copy the contents of any `.pine` file into the TradingView Pine Editor and click
**Add to chart**. No installation, no dependencies, no account linking required.

## A necessary disclaimer

These scripts are engineering demonstrations, not trading advice and not trading
systems. The crossovers used as entry signals are placeholders chosen for
readability — they carry no edge, and none is claimed. Nothing here has been
validated as profitable, and no one should trade it as-is.

## License

MIT. Use it, modify it, ship it in client work.
