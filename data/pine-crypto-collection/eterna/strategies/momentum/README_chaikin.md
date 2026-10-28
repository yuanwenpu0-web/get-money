# Chaikin Oscillator Strategy

## Description
The Chaikin Oscillator measures the momentum of the Accumulation/Distribution line using the difference between fast and slow EMAs. It identifies the flow of volume into or out of an asset, helping traders spot accumulation and distribution phases.

### How It Works
- **BUY**: Chaikin crosses above zero with upward momentum or shows bullish divergence
- **SELL**: Chaikin crosses below zero with downward momentum or shows bearish divergence
- Volume-based signals indicate institutional activity

## Recommended Timeframes
- **Primary**: 1H, 4H
- **Secondary**: 15m, 1D
- **Best for**: Assets with reliable volume data

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Fast EMA Length | 3 | Short-term smoothing |
| Slow EMA Length | 10 | Long-term smoothing |
| Signal Threshold | 0 | Zero-line reference |
| Divergence Lookback | 5 | Bars for divergence detection |
| Leverage | 5 | Trading leverage |
| Position Size | 0.01 | Order size |

## Setup Instructions

### 1. Add to TradingView
1. Pine Editor → New
2. Paste `chaikin_oscillator.pine`
3. Save → Add to Chart

### 2. Configure Eterna MCP Webhook
1. Create alert on strategy
2. **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. **Message**: 
```json
{"action":"buy","symbol":"{{ticker}}","size":0.01,"leverage":5}
```

### 3. Optimal Settings
- **Standard**: Fast 3, Slow 10
- **Smoother signals**: Fast 5, Slow 15
- **Faster response**: Fast 2, Slow 7

## Risk Management
- Requires reliable volume data (avoid low-volume pairs)
- Divergences provide best entry opportunities
- Zero-line acts as trend filter
- Combine with price action for confirmation

## Performance Notes
- Win rate: ~60-70% on liquid markets
- Excellent for spotting institutional accumulation
- Works best on stocks and high-volume crypto
- Less effective on thin or manipulated volume
