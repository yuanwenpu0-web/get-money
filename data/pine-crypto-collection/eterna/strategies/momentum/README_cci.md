# CCI (Commodity Channel Index) Strategy

## Description
The Commodity Channel Index measures the deviation of price from its statistical mean. Originally developed for commodities, CCI works well across all markets to identify cyclical trends and overbought/oversold conditions.

### How It Works
- **BUY**: CCI crosses above -100 (oversold) or reverses from extreme oversold (< -200)
- **SELL**: CCI crosses below +100 (overbought) or reverses from extreme overbought (> +200)
- **Trend confirmation**: Zero-line crossovers indicate trend direction

## Recommended Timeframes
- **Primary**: 1H, 4H
- **Secondary**: 15m, 1D
- **Best for**: Commodities, crypto, forex

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| CCI Length | 20 | Calculation period |
| Overbought Level | +100 | Standard upper threshold |
| Oversold Level | -100 | Standard lower threshold |
| Extreme Overbought | +200 | Critical reversal level |
| Extreme Oversold | -200 | Critical reversal level |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Setup Instructions

### 1. Add to TradingView
1. Pine Editor → New
2. Paste `cci_strategy.pine`
3. Save → Add to Chart

### 2. Configure Eterna MCP Webhook
1. Create alert on strategy
2. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```

### 3. Optimal Settings
- **Crypto (volatile)**: Length 14, Thresholds ±150
- **Forex**: Length 20, Thresholds ±100
- **Commodities**: Length 20, Thresholds ±100

## Risk Management
- Extreme levels (±200) offer best risk/reward
- Use zero-line for trend bias
- Combine with price action patterns
- Best during range-bound markets

## Performance Notes
- Win rate: ~60-70% from extreme levels
- Works exceptionally well on commodities
- Less prone to whipsaws than RSI
- Excellent for timing entry in established trends
