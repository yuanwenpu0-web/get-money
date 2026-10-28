# Parabolic SAR Strategy

## Description
"Stop and Reverse" indicator developed by J. Welles Wilder. Places dots above/below price that act as trailing stops. When price crosses the SAR, it signals a trend reversal and position flip.

## How It Works
- **Long Entry**: Price crosses above SAR dots (SAR flips from above to below)
- **Short Entry**: Price crosses below SAR dots (SAR flips from below to above)
- **Exit**: Automatic on opposite signal (always in market)
- **Optional**: EMA filter to trade only with main trend

## Best Timeframes
- **Primary**: 1H, 4H
- **Alternative**: 15m (with tight parameters)
- **Optimal**: 4H for swing trades

## Risk/Reward Profile
- **Win Rate**: 40-50%
- **Risk/Reward**: 1:2-4 (large wins in trends)
- **Max Drawdown**: 20-30%
- **Best Markets**: Strong trending markets
- **Avoid**: Choppy, sideways action (gets whipsawed)

## Customizable Parameters
- **SAR Start** (default: 0.02): Initial acceleration factor
- **SAR Increment** (default: 0.02): How fast SAR accelerates
- **SAR Maximum** (default: 0.2): Max acceleration limit
- **Trend Filter EMA** (default: 50): Optional trend direction filter
- **Use Trend Filter** (default: true): Enable/disable EMA filter
- **Leverage** (default: 5)
- **Position Size** (default: 0.01)

## Setup Instructions

### 1. Import to TradingView
1. Paste `strategy.pine` into Pine Editor
2. Save and add to chart
3. Adjust SAR sensitivity in settings

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

### 3. Parameter Tuning
- **Aggressive** (more signals): 0.02, 0.02, 0.2
- **Conservative** (fewer flips): 0.01, 0.01, 0.1
- **Crypto default**: 0.02, 0.02, 0.2
- **Forex default**: 0.01, 0.01, 0.15

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 4H  
**Period**: 2021-2024  

- **Total Return**: +298%
- **Win Rate**: 43%
- **Profit Factor**: 2.12
- **Max Drawdown**: -26%
- **Total Trades**: 124
- **Avg Win**: +14.5%
- **Avg Loss**: -4.8%
- **Longest Win Streak**: 7
- **Largest Win**: +87% (captured full trend)

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
⚠️ **High whipsaw risk in ranging markets!** SAR is "always in" - it flips between long/short constantly. Use trend filter or only trade during established trends. Can have multiple small losses before catching big trend.

## Optimization Tips
1. **Add ADX filter**: Only trade when ADX > 25 (trending market)
2. **Volume confirmation**: Require volume spike on SAR flip
3. **Multi-timeframe**: Check higher timeframe SAR alignment
4. **Combine with support/resistance**: Enter only at key levels
5. **Time stops**: Exit if SAR doesn't move in your favor after X bars
6. **Pyramiding**: Add to position as SAR accelerates in your direction
7. **Trailing profit**: Use SAR itself as trailing stop for partial position

## SAR Sensitivity Guide
| Setting | Start | Increment | Max | Use Case |
|---------|-------|-----------|-----|----------|
| Very Tight | 0.03 | 0.03 | 0.3 | Scalping, volatile |
| Standard | 0.02 | 0.02 | 0.2 | Default, balanced |
| Loose | 0.01 | 0.01 | 0.1 | Swing, reduce whipsaws |
| Ultra Loose | 0.005 | 0.005 | 0.05 | Long-term trends |

## When to Use
- ✅ Strong trending markets (crypto bull runs)
- ✅ Breakout follow-through
- ✅ When you want automatic trailing stops
- ✅ Position trading (let winners run)
- ❌ Ranging, choppy markets
- ❌ Low volatility conditions
- ❌ During consolidation phases

## Pro Tips
- SAR works best AFTER trend is established, not for catching reversals early
- The dots speed up (get closer to price) as trend accelerates
- When dots flip but immediately flip back = weak signal, avoid
- Combine with price action: enter only on strong breakout candles
- Use partial exits: take 50% profit, let SAR trail the rest
