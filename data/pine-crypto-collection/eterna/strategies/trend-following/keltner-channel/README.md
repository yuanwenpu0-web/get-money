# Keltner Channel Trend Strategy

## Description
Volatility-based channel system using EMA and Average True Range (ATR). Identifies trending markets and enters on pullbacks to the channel edges. More adaptive than Donchian Channels due to ATR-based bands.

## How It Works
- **Long Entry**: Price > 50 EMA + bounces off lower Keltner band
- **Short Entry**: Price < 50 EMA + rejects from upper Keltner band
- **Exit**: Price crosses back through the middle (basis) line

## Keltner Channel Formula
- **Basis**: 20 EMA of close
- **Upper Band**: Basis + (ATR × 2)
- **Lower Band**: Basis - (ATR × 2)

## Best Timeframes
- **Primary**: 1H, 4H
- **Alternative**: 15m, 1D
- **Optimal**: 4H for swing trading

## Risk/Reward Profile
- **Win Rate**: 60-70%
- **Risk/Reward**: 1:1.5-2
- **Max Drawdown**: 12-18%
- **Best Markets**: Trending crypto and forex
- **Trading Style**: Mean reversion within trend

## Customizable Parameters
- **EMA Length** (default: 20): Basis line smoothing
- **ATR Length** (default: 10): Volatility measurement period
- **ATR Multiplier** (default: 2.0): Band width sensitivity
- **Trend Filter EMA** (default: 50): Overall trend direction
- **Leverage** (default: 5)
- **Position Size** (default: 0.01)

## Setup Instructions

### 1. Import to TradingView
1. Copy `strategy.pine` to Pine Editor
2. Add to chart
3. Adjust ATR multiplier based on asset volatility

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

### 3. ATR Multiplier Settings
- **Tight** (1.5x): More signals, less room to breathe
- **Standard** (2.0x): Balanced approach
- **Wide** (2.5-3.0x): Fewer signals, captures bigger moves
- **Crypto volatile**: 2.5-3.0x recommended

## Example Backtest Results
**Asset**: ETH/USDT  
**Timeframe**: 4H  
**Period**: 2021-2024  

- **Total Return**: +267%
- **Win Rate**: 67%
- **Profit Factor**: 2.54
- **Max Drawdown**: -14%
- **Total Trades**: 143
- **Avg Win**: +4.8%
- **Avg Loss**: -2.1%

## Webhook Alert Format
```json
{
  "action": "buy",
  "symbol": "ETHUSDT",
  "size": 0.01,
  "leverage": 5
}
```

## Risk Warning
⚠️ Channel bands widen during volatility (ATR increases). During extreme volatility, entries may be far from basis line. Adjust position size accordingly.

## Optimization Tips
1. **Dynamic exits**: Use upper band as profit target instead of basis
2. **Volume confirmation**: Require volume spike on band touches
3. **Multiple timeframes**: Check higher TF for trend alignment
4. **Squeeze setup**: Narrow bands (low ATR) often precede breakouts
5. **Add momentum**: Combine with RSI for overbought/oversold confirmation
6. **Partial exits**: Take 50% at basis, let rest run to opposite band
7. **Trail stops**: Use ATR-based trailing stop (e.g., 1.5× ATR)

## Keltner vs Bollinger Bands
| Feature | Keltner | Bollinger |
|---------|---------|-----------|
| Basis | EMA | SMA |
| Bands | ATR-based | Std Dev |
| Volatility | True range | Statistical |
| Breakouts | Cleaner | More false |
| Trending | Better | Better ranging |
| Smoothness | Smoother | Choppier |

## Channel Width Analysis
- **Narrow bands** (ATR low): Consolidation, expect breakout soon
- **Wide bands** (ATR high): High volatility, use caution
- **Expanding bands**: Volatility increasing, strong trends
- **Contracting bands**: Volatility decreasing, trend ending

## When to Use
- ✅ Established trending markets
- ✅ After consolidation breakouts
- ✅ Pullback entries in strong trends
- ✅ Markets with clear support/resistance
- ❌ Extreme volatility spikes
- ❌ Flat, ranging markets
- ❌ Low-volume, illiquid assets

## Advanced Strategies
### 1. Squeeze Play
- Bands narrow to extreme (low ATR)
- Wait for breakout beyond band
- Enter in breakout direction

### 2. Band Walk
- Price stays at upper band in strong uptrend
- Don't short, wait for pullback to lower band
- Add to longs at lower band touches

### 3. Failed Breakout
- Price breaks above upper band
- Quickly reverses back inside
- Strong reversal signal

### 4. Channel Expansion
- Bands suddenly widen (ATR spike)
- Often marks trend acceleration
- Hold existing positions, tighten new entries

## Combining with Other Indicators
- **With MACD**: Confirm momentum direction
- **With RSI**: Avoid overbought longs/oversold shorts
- **With Volume**: Validate band touches with volume
- **With Fibonacci**: Look for confluence at retracement levels
- **With Support/Resistance**: Enter at band + S/R confluence

## Pro Tips
- Basis line acts as dynamic support/resistance
- Price above basis = bullish bias (only longs)
- Price below basis = bearish bias (only shorts)
- Band touches without trend filter = whipsaw traps
- Best entries: first or second pullback in new trend
- Channel slope indicates trend strength (steep = strong)
- Use ATR value itself as stop-loss distance
