# Awesome Oscillator Strategy (Bill Williams)

## Description
The Awesome Oscillator (AO) is part of Bill Williams' trading system and measures market momentum by calculating the difference between 5-period and 34-period simple moving averages of median prices. It identifies momentum shifts before they appear in price action.

### How It Works
- **BUY**: Zero-line cross up, Twin Peaks bullish pattern, or Saucer pattern above zero
- **SELL**: Zero-line cross down, Twin Peaks bearish pattern, or Saucer pattern below zero
- Histogram color indicates momentum direction (green = bullish, red = bearish)

## Recommended Timeframes
- **Primary**: 4H, 1D
- **Secondary**: 1H, 1W
- **Best for**: Trend trading and momentum plays

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Fast SMA Length | 5 | Short-term median price average |
| Slow SMA Length | 34 | Long-term median price average |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Trading Patterns

### Zero-Line Cross
- **Bullish**: AO crosses above zero → strong buy
- **Bearish**: AO crosses below zero → strong sell

### Twin Peaks
- **Bullish**: Two lows below zero, second higher → reversal buy
- **Bearish**: Two highs above zero, second lower → reversal sell

### Saucer
- **Bullish**: Three bars above zero, middle lowest, last green → continuation buy
- **Bearish**: Three bars below zero, middle highest, last red → continuation sell

## Setup Instructions

### 1. Add to TradingView
1. Pine Editor → New
2. Paste `awesome_oscillator.pine`
3. Save → Add to Chart

### 2. Configure Eterna MCP Webhook
1. Create alert on strategy
2. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```

### 3. Optimal Settings
- **Standard (Bill Williams)**: Fast 5, Slow 34
- **For crypto**: Fast 3, Slow 21
- **For forex**: Fast 5, Slow 34 (standard works well)

## Risk Management
- Best used with other Bill Williams indicators (Accelerator, Alligator)
- Zero-line acts as strong trend filter
- Twin Peaks provide excellent risk/reward setups
- Avoid counter-trend trades

## Performance Notes
- Win rate: ~60-70% in trending markets
- Excellent for catching momentum shifts
- Part of proven Bill Williams trading system
- Works best on longer timeframes (4H+)
