# Williams %R Reversal Strategy

## Description
Williams %R (Williams Percent Range) measures overbought and oversold levels by comparing the close to the high-low range over a specified period. This strategy identifies reversal points when the indicator crosses key threshold levels.

### How It Works
- **BUY**: Williams %R crosses above -80 (oversold) with bullish price confirmation
- **SELL**: Williams %R crosses below -20 (overbought) with bearish price confirmation

## Recommended Timeframes
- **Primary**: 1H, 4H
- **Secondary**: 15m, 1D
- **Best for**: Swing trading and reversals

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Williams %R Length | 14 | Lookback period |
| Overbought Level | -20 | Upper threshold |
| Oversold Level | -80 | Lower threshold |
| Confirmation Bars | 2 | Bars to confirm reversal |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Setup Instructions

### 1. Add to TradingView
1. Pine Editor → New
2. Paste `williams_r_reversal.pine`
3. Save → Add to Chart

### 2. Configure Eterna MCP Webhook
1. Create alert on strategy
2. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```

### 3. Optimal Settings
- **High volatility**: Overbought -15, Oversold -85
- **Low volatility**: Overbought -25, Oversold -75
- **Trending markets**: Length 21, wider thresholds

## Risk Management
- Works best near support/resistance levels
- Avoid during strong trending conditions
- Use with volume confirmation
- Set stops beyond recent swing points

## Performance Notes
- Win rate: ~55-65% in ranging markets
- Sensitive to false signals in trends
- Best combined with price action patterns
- Excellent for catching V-shaped reversals
