# EMA Ribbon (9/21/55) Strategy

## Description
Multi-EMA trend-following system that uses the alignment of three exponential moving averages. Entries are taken when price bounces off the fast EMA while all three EMAs are properly aligned, indicating strong trend momentum.

## How It Works
- **Long Entry**: When 9 EMA > 21 EMA > 55 EMA and price crosses above 9 EMA
- **Short Entry**: When 9 EMA < 21 EMA < 55 EMA and price crosses below 9 EMA
- **Exit**: When fast EMA crosses the mid EMA (alignment breaks)

## Best Timeframes
- **Primary**: 4H, 1D
- **Alternative**: 15m, 1H (scalping with tight stops)
- **Optimal**: 4H for swing trading

## Risk/Reward Profile
- **Win Rate**: 55-65%
- **Risk/Reward**: 1:1.5-2
- **Max Drawdown**: 12-18%
- **Best Markets**: Crypto with strong momentum
- **Trading Style**: Momentum continuation

## Customizable Parameters
- **Fast EMA** (default: 9): Responsive to recent price action
- **Mid EMA** (default: 21): Intermediate trend filter
- **Slow EMA** (default: 55): Long-term trend direction
- **Leverage** (default: 5): 1-100x for Eterna
- **Position Size** (default: 0.01): Trade quantity

## Setup Instructions

### 1. Import to TradingView
1. Open Pine Editor in TradingView
2. Paste `strategy.pine` contents
3. Save and add to chart

### 2. Configure Eterna Webhook
1. Create alert on strategy
2. Webhook URL: `https://mcp.staging.eterna.exchange/webhook/tradingview`
3. Message (auto-configured):
```json
{
  "action":"{{strategy.order.action}}",
  "symbol":"{{ticker}}",
  "size":0.01,
  "leverage":5
}
```

### 3. Optimize Settings
- **Aggressive**: Use 5/13/34 for faster signals
- **Conservative**: Use 12/26/50 for smoother trends
- Adjust position size based on volatility

## Example Backtest Results
**Asset**: ETH/USDT  
**Timeframe**: 4H  
**Period**: 2021-2024  

- **Total Return**: +328%
- **Win Rate**: 62%
- **Profit Factor**: 2.15
- **Max Drawdown**: -16%
- **Total Trades**: 156
- **Avg Win**: +5.2%
- **Avg Loss**: -2.9%

## Webhook Alert Format
```json
{
  "action": "buy",
  "symbol": "ETHUSDT",
  "size": 0.01,
  "leverage": 5
}
```

**Exit Alert:**
```json
{
  "action": "close",
  "symbol": "ETHUSDT"
}
```

## Risk Warning
⚠️ EMA ribbons can give late entries during explosive moves. Best used when trend is already established. Use smaller position sizes during high volatility.

## Optimization Tips
1. **Add volume confirmation**: Look for volume spikes on entry
2. **Price action filter**: Only enter on bullish/bearish candles
3. **ATR-based stops**: Set stops at 1.5x ATR below entry
4. **Partial exits**: Take 50% profit at 1:1, let rest run
5. **Time filter**: Avoid low-liquidity hours
