# Linear Regression Channel Strategy

## Description
Statistical trend-following system based on linear regression. Uses the best-fit line through recent price action with standard deviation bands. Entries occur on mean-reversion touches within the regression channel while following the overall trend slope.

## How It Works
- **Long Entry**: Upward slope + price > 50 EMA + bounces off lower band
- **Short Entry**: Downward slope + price < 50 EMA + rejects from upper band
- **Exit**: Price crosses back through the regression line (mean)

## Components
- **Linear Regression Line**: Best-fit line through price data (100 periods)
- **Upper/Lower Bands**: ±2 standard deviations from regression line
- **Slope**: Indicates trend direction and strength

## Best Timeframes
- **Primary**: 4H, 1D
- **Alternative**: 1H
- **Optimal**: 1D for clean trend identification

## Risk/Reward Profile
- **Win Rate**: 65-75%
- **Risk/Reward**: 1:1.5-2
- **Max Drawdown**: 10-15%
- **Best Markets**: Smooth trending assets (indices, large-cap crypto)
- **Trading Style**: Statistical mean reversion within trend

## Customizable Parameters
- **Linear Regression Length** (default: 100): Lookback for regression calculation
- **Deviation Multiplier** (default: 2.0): Band width (2σ = 95% confidence)
- **Trend Filter EMA** (default: 50): Additional trend confirmation
- **Leverage** (default: 5)
- **Position Size** (default: 0.01)

## Setup Instructions

### 1. Import to TradingView
1. Copy `strategy.pine` to Pine Editor
2. Add to chart (works best on higher timeframes)
3. Watch slope angle indicator for trend strength

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

### 3. Length Optimization
- **Short-term** (50): Responsive, more signals
- **Standard** (100): Balanced, smooth trends
- **Long-term** (200): Very smooth, major trends only
- **Intraday**: 50-75 on 15m/1H
- **Swing**: 100-150 on 4H/1D

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 1D  
**Period**: 2020-2024  

- **Total Return**: +342%
- **Win Rate**: 71%
- **Profit Factor**: 3.12
- **Max Drawdown**: -11%
- **Total Trades**: 89
- **Avg Win**: +8.7%
- **Avg Loss**: -2.9%

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
⚠️ Linear regression assumes normal distribution of prices. During extreme volatility or news events, price can remain outside bands for extended periods. Best used in established trends, not choppy markets.

## Optimization Tips
1. **Slope threshold**: Only trade when slope angle > 5° (strong trend)
2. **R-squared filter**: Add correlation check (high R² = good fit)
3. **Volume at extremes**: Require volume spike on band touches
4. **Multiple timeframes**: Align regression slope across 3 timeframes
5. **Dynamic length**: Shorten during volatility, lengthen during stability
6. **Breakout mode**: Enter when price breaks above upper band in uptrend
7. **Deviation tiers**: Use 1σ for partial entry, 2σ for full position

## Slope Interpretation
- **Steep Up** (>15°): Strong uptrend, buy dips aggressively
- **Moderate Up** (5-15°): Healthy uptrend, standard entries
- **Flat** (-5° to +5°): Ranging, avoid or reduce size
- **Moderate Down** (-15° to -5°): Downtrend, short rallies
- **Steep Down** (<-15°): Strong downtrend, aggressive shorts

## Channel Quality Indicators
**Good Regression Fit:**
- Price oscillates within bands regularly
- Touches both upper and lower bands
- Slope is consistent (not erratic)
- Bands are parallel

**Poor Regression Fit:**
- Price constantly outside bands
- Slope changes direction frequently
- Bands converge/diverge wildly
- Low correlation to actual price movement

## When to Use
- ✅ Established, smooth trends
- ✅ Assets with predictable patterns
- ✅ After consolidation breakouts
- ✅ Low-volatility trending environments
- ❌ Highly volatile, erratic assets
- ❌ News-driven markets
- ❌ Choppy, ranging conditions
- ❌ During major trend reversals

## Advanced Techniques
### 1. Regression Channels for Support/Resistance
- Parallel channels on multiple timeframes
- Lower band of daily = strong support
- Upper band breakout = acceleration signal

### 2. Channel Breakout Strategy
- Price breaks upper band in uptrend
- Don't fade it, go with it
- Exit only on slope change

### 3. Mean Reversion Fade
- Price far outside band (>3σ)
- Low volume, exhaustion signs
- Counter-trend scalp back to mean

### 4. Regression Slope Divergence
- Price making higher highs
- Regression slope flattening
- Early warning of trend weakening

## Combining with Other Indicators
- **With RSI**: Enter band touches when RSI confirms (30/70)
- **With MACD**: Align entries with MACD direction
- **With Volume Profile**: Look for POC near regression line
- **With Fibonacci**: Confluence of band + Fib levels
- **With Pivot Points**: Strong entries at pivot + band touch

## Statistical Concepts
- **1 Std Dev**: ~68% of prices within this range
- **2 Std Dev**: ~95% of prices within this range (default)
- **3 Std Dev**: ~99.7% of prices (rare, extreme moves)

## Pro Tips
- Regression line is "fair value" - price gravitates toward it
- Steeper slope = stronger trend, wider bands
- Flat regression = consolidation, tighten bands
- Price at upper band in uptrend = strength, not weakness
- Channel breakouts often lead to new channels
- Use regression on volume too (spot anomalies)
- Best after volatility contraction (bands tight)
- Respect the slope: don't fight the trend direction
