# Stochastic RSI Crossover Strategy

## Description
The Stochastic RSI combines the Stochastic Oscillator with the RSI to identify overbought and oversold conditions with greater sensitivity. This strategy trades crossovers of the %K and %D lines in extreme zones.

### How It Works
- **BUY**: %K crosses above %D while in oversold zone (< 20)
- **SELL**: %K crosses below %D while in overbought zone (> 80)

## Recommended Timeframes
- **Primary**: 15m, 1H
- **Secondary**: 5m, 4H
- **Best for**: Scalping and day trading

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| RSI Length | 14 | RSI calculation period |
| Stochastic Length | 14 | Stochastic calculation period |
| K Smoothing | 3 | %K line smoothing |
| D Smoothing | 3 | %D line smoothing |
| Overbought Level | 80 | Upper threshold |
| Oversold Level | 20 | Lower threshold |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Setup Instructions

### 1. Add to TradingView
1. Open TradingView chart
2. Pine Editor → New script
3. Paste `stoch_rsi_crossover.pine`
4. Save and add to chart

### 2. Configure Eterna MCP Webhook
1. Create alert on strategy
2. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```

### 3. Optimal Settings by Market
- **Crypto (volatile)**: Overbought 85, Oversold 15
- **Crypto (stable)**: Overbought 80, Oversold 20
- **Forex**: Overbought 75, Oversold 25

## Risk Management
- ⚠️ Very sensitive - generates frequent signals
- Use in trending markets with caution
- Best combined with trend filters
- Consider wider stop-losses due to sensitivity

## Performance Notes
- Win rate: ~50-60% with proper filtering
- Best during range-bound conditions
- Prone to whipsaws in choppy markets
- Quick entries and exits required
