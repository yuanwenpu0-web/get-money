# RSI Divergence Strategy

## Description
This strategy identifies bullish and bearish divergences between price action and the Relative Strength Index (RSI). Divergences occur when price and momentum indicators move in opposite directions, often signaling potential trend reversals.

### How It Works
- **Bullish Divergence**: Price makes a lower low while RSI makes a higher low → BUY signal
- **Bearish Divergence**: Price makes a higher high while RSI makes a lower high → SELL signal

## Recommended Timeframes
- **Primary**: 1H, 4H
- **Secondary**: 15m, 1D
- **Not recommended**: < 5m (too noisy)

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| RSI Length | 14 | Calculation period for RSI |
| RSI Overbought | 70 | Upper threshold for bearish divergence |
| RSI Oversold | 30 | Lower threshold for bullish divergence |
| Pivot Lookback | 5 | Bars to identify pivot points |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Setup Instructions

### 1. Add to TradingView
1. Open TradingView chart
2. Click "Pine Editor" at bottom
3. Copy `rsi_divergence.pine` content
4. Click "Save" and "Add to Chart"

### 2. Configure Eterna MCP Webhook
1. Right-click chart → "Add Alert"
2. **Condition**: Select strategy name
3. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
4. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```
5. Click "Create"

### 3. Optimal Settings by Asset
- **BTC/ETH**: RSI Length 14, Pivot 5, Timeframe 4H
- **Altcoins**: RSI Length 14, Pivot 7, Timeframe 1H
- **Forex**: RSI Length 21, Pivot 5, Timeframe 4H

## Risk Management
- ⚠️ Divergences can fail in strong trends
- Use with trend confirmation indicators
- Best in ranging or reversal zones
- Set stop-loss at recent swing high/low

## Performance Notes
- Win rate: ~55-65% in ranging markets
- Risk/Reward: 1:1.5 to 1:2
- Works best when combined with support/resistance levels
