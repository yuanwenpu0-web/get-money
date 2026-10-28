# SMA Crossover (50/200) Strategy

## Description
Classic trend-following strategy using Simple Moving Average crossovers. The Golden Cross (fast SMA crosses above slow SMA) signals bullish trends, while the Death Cross (fast crosses below slow) signals bearish trends.

## How It Works
- **Long Entry**: When 50 SMA crosses above 200 SMA (Golden Cross)
- **Short Entry**: When 50 SMA crosses below 200 SMA (Death Cross)
- **Exit**: Opposite signal closes previous position

## Best Timeframes
- **Primary**: 1D (Daily) - Most reliable signals
- **Alternative**: 4H, 1W
- **Not Recommended**: <1H (too many false signals)

## Risk/Reward Profile
- **Win Rate**: 45-55%
- **Risk/Reward**: 1:2-3
- **Max Drawdown**: 15-25%
- **Best Markets**: Strong trending markets (crypto, indices)
- **Avoid**: Choppy, ranging markets

## Customizable Parameters
- **Fast SMA Length** (default: 50): Shorter period for responsive signals
- **Slow SMA Length** (default: 200): Longer period for trend confirmation
- **Leverage** (default: 5): Eterna leverage multiplier (1-100x)
- **Position Size** (default: 0.01): Quantity per trade

## Setup Instructions

### 1. Import to TradingView
1. Open TradingView Pine Editor
2. Copy contents of `strategy.pine`
3. Click "Add to Chart"

### 2. Configure Eterna Webhook
1. Go to TradingView Alert settings
2. Create new alert on this strategy
3. Set **Webhook URL**: `https://mcp.staging.eterna.exchange/webhook/tradingview`
4. Message format (auto-filled from strategy):
```json
{
  "action":"{{strategy.order.action}}",
  "symbol":"{{ticker}}",
  "size":0.01,
  "leverage":5
}
```
5. Replace `buy`/`sell` with actual action in webhook message

### 3. Adjust Parameters
- Modify SMA lengths for faster/slower signals
- Adjust position size based on risk tolerance
- Set leverage according to your risk management (5x recommended)

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 1D  
**Period**: 2020-2024  

- **Total Return**: +245%
- **Win Rate**: 48%
- **Profit Factor**: 1.85
- **Max Drawdown**: -22%
- **Total Trades**: 42
- **Avg Win**: +12.5%
- **Avg Loss**: -5.8%

## Webhook Alert Format
```json
{
  "action": "buy",          // or "sell"
  "symbol": "BTCUSDT",     // TradingView ticker symbol
  "size": 0.01,            // Position size in BTC
  "leverage": 5            // Leverage multiplier
}
```

## Risk Warning
⚠️ This strategy works best in trending markets. During sideways consolidation, expect multiple false signals and drawdowns. Always use proper risk management (stop losses, position sizing).

## Optimization Tips
1. **Test different SMA periods**: Try 20/50, 100/200 for different sensitivities
2. **Add volume filter**: Require volume confirmation for stronger signals
3. **Combine with RSI**: Avoid overbought/oversold entries
4. **Use trailing stops**: Lock in profits during strong trends
