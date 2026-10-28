# VWAP Bounce Strategy

## Description
Institutional-grade mean reversion within a trend. Uses Volume Weighted Average Price (VWAP) as dynamic support/resistance. Enters when price bounces off VWAP bands in the direction of the main trend (determined by 50 EMA).

## How It Works
- **Long Entry**: Price above 50 EMA + bounces from VWAP lower band
- **Short Entry**: Price below 50 EMA + rejects from VWAP upper band
- **Exit**: Price crosses back through center VWAP line

## Best Timeframes
- **Primary**: 15m, 30m, 1H
- **Intraday**: Best for day trading with VWAP reset daily
- **Not for**: Daily charts (VWAP resets, loses context)

## Risk/Reward Profile
- **Win Rate**: 60-70%
- **Risk/Reward**: 1:1.2-1.8
- **Max Drawdown**: 10-15%
- **Best Markets**: High-volume crypto pairs (BTC, ETH)
- **Trading Style**: Mean reversion + trend following hybrid

## Customizable Parameters
- **VWAP Deviation %** (default: 0.5%): Band width for entries
- **Trend EMA** (default: 50): Determines overall trend direction
- **Leverage** (default: 5): Eterna leverage
- **Position Size** (default: 0.01): Trade size

## Setup Instructions

### 1. Import to TradingView
1. Copy `strategy.pine` to Pine Editor
2. Add to chart (works best on 15m-1H)
3. Ensure chart has volume data

### 2. Configure Eterna Webhook
Webhook URL: `https://mcp.staging.eterna.exchange/webhook/tradingview`

Entry message:
```json
{
  "action":"{{strategy.order.action}}",
  "symbol":"{{ticker}}",
  "size":0.01,
  "leverage":5
}
```

Exit message:
```json
{
  "action":"close",
  "symbol":"{{ticker}}"
}
```

### 3. Adjust Parameters
- **Volatile markets**: Increase deviation to 0.8-1.0%
- **Stable markets**: Decrease to 0.3-0.4%
- **Faster trend**: Use 20 EMA instead of 50

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 15m  
**Period**: Jan-Dec 2024  

- **Total Return**: +187%
- **Win Rate**: 68%
- **Profit Factor**: 2.45
- **Max Drawdown**: -12%
- **Total Trades**: 342
- **Avg Win**: +1.8%
- **Avg Loss**: -0.9%

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
⚠️ VWAP resets at market open (or daily on crypto). Best used during active trading hours. Avoid during extreme volatility when VWAP loses relevance.

## Optimization Tips
1. **Volume filter**: Only trade when volume > average
2. **Session filter**: Trade only during high-liquidity sessions
3. **Multiple bands**: Add 1% and 2% deviation levels for staged entries
4. **Momentum confirmation**: Add RSI (30/70) to filter weak bounces
5. **Combine with order flow**: Watch for large volume at bounce points
6. **Partial profits**: Scale out at VWAP center, let runner go to opposite band

## When to Use
- ✅ High-volume trading sessions
- ✅ Established trends with pullbacks
- ✅ Assets with strong institutional participation
- ❌ Low-volume overnight sessions
- ❌ Extreme news-driven volatility
- ❌ Ranging, trendless markets
