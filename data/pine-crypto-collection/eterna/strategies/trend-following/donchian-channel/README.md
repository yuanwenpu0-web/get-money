# Donchian Channel Breakout Strategy

## Description
Classic breakout system based on Richard Donchian's trend-following method (used in the famous Turtle Trading system). Enters when price breaks above/below the highest high/lowest low of the past N periods.

## How It Works
- **Long Entry**: Price closes above 20-period high (upper band breakout)
- **Short Entry**: Price closes below 20-period low (lower band breakdown)
- **Exit**: Price crosses middle line (optional) or opposite band

## Components
- **Upper Band**: Highest high of last N periods
- **Lower Band**: Lowest low of last N periods
- **Middle Line**: Average of upper and lower bands

## Best Timeframes
- **Primary**: 1H, 4H, 1D
- **Alternative**: 15m for scalping
- **Original Turtle**: Daily (20-day/55-day channels)

## Risk/Reward Profile
- **Win Rate**: 35-45%
- **Risk/Reward**: 1:3-5 (few large wins)
- **Max Drawdown**: 20-30%
- **Best Markets**: Trending, breakout-prone assets
- **Trading Style**: Breakout/momentum

## Customizable Parameters
- **Channel Length** (default: 20): Lookback period for high/low
- **Breakout Confirmation** (default: 1): Bars to confirm breakout
- **Exit at Middle Line** (default: true): Early exit vs ride full range
- **Leverage** (default: 5)
- **Position Size** (default: 0.01)

## Setup Instructions

### 1. Import to TradingView
1. Paste `strategy.pine` into Pine Editor
2. Add to chart
3. Adjust channel length based on asset volatility

### 2. Configure Eterna Webhook
Webhook URL: `https://mcp.staging.eterna.exchange/webhook/tradingview`

```json
{
  "action":"{{strategy.order.action}}",
  "symbol":"{{ticker}}",
  "size":0.01,
  "leverage":5
}
```

### 3. Channel Settings by Style
- **Turtle Trading**: 20-day entry, 10-day exit
- **Aggressive**: 10-period (more signals, more whipsaws)
- **Conservative**: 50-period (fewer signals, stronger breakouts)
- **Crypto**: 20-period works well on 4H/1D

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 1D  
**Period**: 2020-2024  

- **Total Return**: +456%
- **Win Rate**: 38%
- **Profit Factor**: 2.87
- **Max Drawdown**: -28%
- **Total Trades**: 67
- **Avg Win**: +24.3%
- **Avg Loss**: -6.1%
- **Largest Win**: +142% (captured full bull trend)

## Webhook Alert Format
```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "size": 0.01,
  "leverage": 5
}
```

## Risk Warning
⚠️ **High whipsaw risk during consolidation!** Many false breakouts will occur. Expects to lose small amounts frequently but capture massive trends occasionally. Requires discipline to take all signals.

## Optimization Tips
1. **Add volume filter**: Breakout must have volume > 150% of average
2. **ATR filter**: Only trade when ATR is expanding (volatility rising)
3. **Multi-timeframe**: Confirm breakout on higher timeframe
4. **Retest entry**: Wait for price to retest broken level as support
5. **Pyramiding**: Add to winning positions at new channel highs
6. **Adaptive channel**: Use ATR-based dynamic length
7. **News filter**: Avoid breakouts during major news (fake-outs)

## Turtle Trading Rules (Original)
- **Entry**: 20-day channel breakout
- **Exit**: 10-day channel opposite breakout
- **Position sizing**: 1% risk per trade
- **Pyramiding**: Add every 0.5 ATR move
- **Max units**: 4 per market, 12 per direction

## When to Use
- ✅ Range-bound markets ready to break
- ✅ Post-consolidation setups
- ✅ Strong momentum environments
- ✅ Assets prone to trending (crypto, commodities)
- ❌ Already extended trends (late entry)
- ❌ Low volatility, tight ranges
- ❌ Highly mean-reverting assets

## Breakout Quality Checklist
Before entering, confirm:
1. ✅ Volume surge (>150% average)
2. ✅ Strong candle close (not wick)
3. ✅ Higher timeframe alignment
4. ✅ No immediate resistance above (longs)
5. ✅ ATR expanding (volatility increasing)
6. ✅ Prior consolidation period (at least 10 bars)

## Managing False Breakouts
- **First breakout often fails**: Wait for second attempt
- **Set stop at middle line**: Limit losses on false breaks
- **Use smaller size initially**: Add on confirmation
- **Watch for "spring"**: Failed breakout in opposite direction = strong signal
- **Combine with RSI**: Avoid overbought (>70) breakouts

## Channel Length Guide
| Period | Length | Use Case | Signals |
|--------|--------|----------|---------|
| 5-10 | Short | Scalping, fast moves | Many |
| 20 | Standard | Turtle Trading | Balanced |
| 50 | Long | Swing trading | Few, high quality |
| 100+ | Very Long | Position trading | Rare |

## Pro Tips
- First breakout after long consolidation = best odds
- Channel squeeze (bands converging) = breakout imminent
- Price at middle line = neutral zone, wait
- Failed breakout (returns to channel) = strong counter signal
- Best after volatility contraction (Bollinger squeeze)
- Combine with order flow: watch for whale activity at breakout
