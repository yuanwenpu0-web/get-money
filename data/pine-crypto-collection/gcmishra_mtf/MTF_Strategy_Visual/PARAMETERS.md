# Parameter Guide and Tuning

This guide explains all parameters in the MTF Strategy Visual indicator and provides recommendations for different trading scenarios.

## Table of Contents

1. [EMA Settings](#ema-settings)
2. [ATR Filter Settings](#atr-filter-settings)
3. [Timeframe Settings](#timeframe-settings)
4. [Visual Settings](#visual-settings)
5. [Risk Management](#risk-management)
6. [Tuning Strategies](#tuning-strategies)

---

## EMA Settings

### EMA 1-6 Periods (Default: 5, 8, 13, 21, 34, 55)

**Description**: Six exponential moving averages that form the EMA cloud. These are Fibonacci-based periods that work well together.

**How It Works**:
- EMAs create a cloud that acts as dynamic support/resistance
- Alignment (all ascending or descending) indicates strong trend
- Convergence (coming together) indicates sideways market

**Tuning Tips**:

| Market Type | Fast Settings | Standard | Conservative |
|-------------|---------------|----------|--------------|
| Crypto      | 3,5,8,13,21,34 | 5,8,13,21,34,55 | 8,13,21,34,55,89 |
| Forex       | 5,8,13,21,34,55 | 8,13,21,34,55,89 | 13,21,34,55,89,144 |
| Stocks      | 8,13,21,34,55,89 | 13,21,34,55,89,144 | 21,34,55,89,144,233 |

**Guidelines**:
- **Faster EMAs (lower periods)**: More signals, more noise, faster response
- **Slower EMAs (higher periods)**: Fewer signals, smoother, less noise
- Keep the proportional relationship (Fibonacci sequence recommended)

---

## ATR Filter Settings

### ATR Period (Default: 14)

**Description**: Number of periods used to calculate Average True Range (volatility measure).

**How It Works**:
- ATR measures market volatility
- Higher ATR = more volatile market = larger stop losses and targets
- Lower ATR = calmer market = tighter stops and targets

**Tuning Tips**:
- **Scalping (M1-M5)**: 7-10 periods (faster response)
- **Day Trading (M5-M30)**: 14 periods (standard)
- **Swing Trading (H1-D1)**: 20-30 periods (smoother)

### ATR MA Period (Default: 14)

**Description**: Moving average period applied to ATR to smooth volatility readings.

**How It Works**:
- Creates a baseline volatility level
- Helps filter out abnormal spikes
- Used to determine if current volatility is above/below average

**Tuning Tips**:
- Keep same as ATR Period for consistency
- Increase for smoother filtering (up to 20-30)
- Decrease for more responsive filtering (down to 7-10)

### ATR Threshold Multiplier (Default: 1.0)

**Description**: Minimum volatility requirement for active trading. ATR must exceed ATR_MA × threshold.

**How It Works**:
- `1.0` means ATR must be equal to its moving average
- `1.2` means ATR must be 20% higher than average
- Prevents trading in low-volatility conditions

**Tuning Tips**:

| Scenario | Recommended Setting |
|----------|---------------------|
| Want more signals | 0.8 - 0.9 |
| Balanced approach | 1.0 - 1.1 |
| Only high-volatility signals | 1.2 - 1.5 |
| Very selective (breakouts only) | 1.5 - 2.0 |

### Sideways Threshold (Default: 0.8)

**Description**: ATR contraction level that indicates sideways market. Below this level, market is considered choppy.

**How It Works**:
- `0.8` means if ATR drops below 80% of its MA, market is sideways
- Lower values = more conservative (fewer sideways detections)
- Higher values = more aggressive (more sideways detections)

**Tuning Tips**:
- **Too many false signals?** Increase to 0.9-1.0 (detect more sideways markets)
- **Missing good signals?** Decrease to 0.6-0.7 (detect fewer sideways markets)
- **Crypto markets**: 0.7-0.8 (more volatile)
- **Forex markets**: 0.8-0.9 (smoother)

---

## Timeframe Settings

### Higher Timeframe (HTF) - Default: 60 minutes

**Description**: The highest timeframe used for determining overall market bias/direction.

**How It Works**:
- Sets the primary trend direction
- Must have clear EMA alignment (all ascending or descending)
- Filters out trades against the main trend

**Recommendations**:

| Chart Timeframe | HTF Setting | Purpose |
|-----------------|-------------|---------|
| 1-3 minutes | 15-30 min | Scalping |
| 5-15 minutes | 60 min (1H) | Day trading |
| 30-60 minutes | 240 min (4H) | Swing trading |
| 1-4 hours | Daily (D) | Position trading |

**Rules**:
- HTF should be 4-12x your chart timeframe
- Too close = repainting risk
- Too far = missing intermediate trends

### Medium Timeframe (MTF) - Default: 10 minutes

**Description**: Intermediate timeframe that confirms HTF direction and provides entry alignment.

**How It Works**:
- Acts as a filter between HTF and LTF
- Must align with HTF bias
- Provides earlier entry signals than HTF

**Recommendations**:

| HTF Setting | MTF Setting | Chart Timeframe |
|-------------|-------------|-----------------|
| 15 min | 5 min | 1-3 min |
| 60 min | 10-15 min | 3-5 min |
| 240 min (4H) | 60 min (1H) | 15-30 min |
| Daily | 240 min (4H) | 60 min |

**Rules**:
- MTF should be between Chart and HTF
- Typical ratio: Chart × 2-4 = MTF, HTF × 0.25-0.5 = MTF
- Example: Chart=5min, MTF=15min, HTF=60min

### Lower Timeframe (LTF)

**Description**: Your current chart timeframe. This is where entry signals are generated.

**How It Works**:
- Shows precise entry points
- Triggers BUY/SELL when price breaks EMA cloud
- Must have HTF and MTF alignment before signaling

---

## Visual Settings

### Colors

**Bullish Zone Color (Default: Green with 90% transparency)**
- Background color when HTF bias is bullish

**Bearish Zone Color (Default: Red with 90% transparency)**
- Background color when HTF bias is bearish

**Sideways Zone Color (Default: Gray with 85% transparency)**
- Background color when market is sideways/choppy

**Buy/Sell Signal Colors (Default: Lime, Red - 0% transparency)**
- Colors for BUY and SELL labels

**Customization Tips**:
- Keep transparency high (80-95%) to see price action clearly
- Use contrasting colors for different zones
- Adjust based on chart theme (dark/light mode)

### Display Options

All display options default to `true` (enabled):

- **Show EMA Zones**: Background coloring based on market bias
- **Show Entry Signals**: BUY/SELL labels on chart
- **Show Sideways Detection**: Boxes around sideways market zones
- **Show Dashboard**: Information table in top-right corner
- **Show TP/SL Levels**: Horizontal lines for take profit and stop loss

**Usage Tips**:
- Disable EMA Zones if chart becomes too cluttered
- Disable Sideways Detection boxes if using smaller timeframes (can overlap)
- Dashboard is useful for quick market assessment
- TP/SL levels help with trade management

---

## Risk Management

### TP1, TP2, TP3 Multipliers (Default: 1.5, 2.5, 4.0)

**Description**: Take Profit levels as multiples of ATR from entry price.

**How It Works**:
- TP1 = Entry + (ATR × 1.5) for buys, Entry - (ATR × 1.5) for sells
- TP2 = Entry + (ATR × 2.5)
- TP3 = Entry + (ATR × 4.0)

**Strategy Examples**:

| Strategy Type | TP1 | TP2 | TP3 | Description |
|---------------|-----|-----|-----|-------------|
| Conservative | 1.0 | 1.5 | 2.0 | Small, frequent profits |
| Balanced | 1.5 | 2.5 | 4.0 | Standard risk/reward |
| Aggressive | 2.0 | 3.5 | 6.0 | Large targets, fewer wins |
| Scalping | 0.5 | 1.0 | 1.5 | Very tight targets |

**Position Management**:
- Take 50% profit at TP1, move SL to breakeven
- Take 30% profit at TP2, trail stop to TP1
- Let 20% run to TP3 with trailing stop

### Stop Loss Multiplier (Default: 1.5)

**Description**: Stop Loss distance as multiple of ATR from entry price.

**How It Works**:
- SL = Entry - (ATR × 1.5) for buys
- SL = Entry + (ATR × 1.5) for sells

**Tuning Tips**:

| Risk Tolerance | SL Multiplier | Risk:Reward at TP1 |
|----------------|---------------|---------------------|
| Tight (aggressive) | 1.0 | 1:1.5 |
| Balanced | 1.5 | 1:1.0 |
| Wide (conservative) | 2.0 | 1:0.75 |

**Guidelines**:
- Tighter SL (1.0-1.2): More stopped out, but smaller losses
- Wider SL (2.0-2.5): Fewer stop outs, but larger potential losses
- Match SL to TP1 for 1:1 risk/reward minimum
- Crypto: 1.5-2.0 (more volatile)
- Forex: 1.2-1.5 (less volatile)

---

## Tuning Strategies

### For Scalping (M1-M5)

```
EMA: 3, 5, 8, 13, 21, 34
HTF: 15 minutes
MTF: 5 minutes
Chart: 1-3 minutes
ATR Period: 7
ATR Threshold: 1.2
Sideways Threshold: 0.8
TP1/TP2/TP3: 0.5, 1.0, 1.5
SL: 1.0
```

**Characteristics**: Fast signals, high frequency, small profits per trade

### For Day Trading (M5-M30)

```
EMA: 5, 8, 13, 21, 34, 55
HTF: 60 minutes
MTF: 15 minutes
Chart: 5-15 minutes
ATR Period: 14
ATR Threshold: 1.0
Sideways Threshold: 0.8
TP1/TP2/TP3: 1.5, 2.5, 4.0
SL: 1.5
```

**Characteristics**: Balanced approach, multiple signals per day, moderate profits

### For Swing Trading (H1-H4)

```
EMA: 8, 13, 21, 34, 55, 89
HTF: Daily
MTF: 4 hours
Chart: 1 hour
ATR Period: 20
ATR Threshold: 0.9
Sideways Threshold: 0.9
TP1/TP2/TP3: 2.0, 3.5, 6.0
SL: 2.0
```

**Characteristics**: Fewer but higher-quality signals, hold for days/weeks

### For Volatile Markets (Crypto)

```
ATR Threshold: 0.9-1.0 (lower to catch moves)
Sideways Threshold: 0.7 (less restrictive)
SL Multiplier: 2.0 (wider stops)
TP Multipliers: 2.0, 3.5, 6.0 (bigger targets)
```

### For Ranging Markets (Forex Major Pairs)

```
ATR Threshold: 1.2-1.5 (higher to avoid chop)
Sideways Threshold: 0.9-1.0 (more restrictive)
SL Multiplier: 1.2 (tighter stops)
TP Multipliers: 1.0, 1.5, 2.5 (smaller targets)
```

---

## Advanced Tips

### Avoiding Repainting

The indicator uses `barstate.isconfirmed` for labels and objects, which minimizes repainting. However:

1. Always wait for bar close before taking action
2. Use alerts to get notifications after bar confirmation
3. Avoid using MTF/HTF too close to chart timeframe

### Optimizing for Specific Assets

**High Beta Stocks (TSLA, NVDA)**:
- Increase ATR threshold (1.2-1.5)
- Wider stop loss (2.0-2.5 ATR)
- Larger TP targets

**Low Volatility Pairs (EURCHF, AUDNZD)**:
- Decrease ATR threshold (0.8-0.9)
- Tighter stop loss (1.0-1.2 ATR)
- Smaller TP targets

**Indices (SPX, NDX)**:
- Standard settings work well
- Consider session times (London/New York open)
- Increase ATR threshold during low-volume periods

### Combining with Other Indicators

This indicator works well with:
- **Volume Profile**: Confirm signals at key volume nodes
- **Support/Resistance**: Trade signals near major levels
- **RSI/MACD**: Add momentum confirmation
- **Trend Lines**: Only take signals in direction of macro trend

---

## Troubleshooting Common Issues

### Issue: Too Many Signals

**Solution**:
1. Increase ATR Threshold to 1.2-1.5
2. Increase Sideways Threshold to 0.9-1.0
3. Use higher timeframes (HTF=4H instead of 1H)
4. Tighten EMA convergence detection

### Issue: No Signals at All

**Solution**:
1. Decrease ATR Threshold to 0.8-0.9
2. Check that chart TF < MTF TF < HTF TF
3. Verify EMAs are not all converged (sideways market)
4. Lower Sideways Threshold to 0.6-0.7

### Issue: Signals Not Matching Dashboard

**Solution**:
- Signals require ALL conditions (HTF+MTF+LTF+ATR)
- Dashboard shows individual component status
- All must align for signal generation

### Issue: Stop Loss Too Tight/Wide

**Solution**:
- Adjust SL Multiplier based on asset volatility
- Check ATR value - may need different ATR period
- Consider using fixed SL instead of ATR-based

---

## Backtesting Recommendations

When backtesting different parameter combinations:

1. **Use at least 3-6 months of data**: Covers different market conditions
2. **Test multiple assets**: Parameters that work on BTC may not work on EUR
3. **Consider drawdown**: Not just profit, but maximum drawdown matters
4. **Win rate vs. R:R ratio**: 40% win rate with 1:3 R:R beats 60% with 1:1
5. **Account for slippage**: Real trading has costs
6. **Forward test**: Test on out-of-sample data after optimization

---

## Quick Reference Table

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| ATR Threshold | 1.2-1.5 | 1.0 | 0.8-0.9 |
| Sideways Threshold | 0.9-1.0 | 0.8 | 0.6-0.7 |
| SL Multiplier | 2.0-2.5 | 1.5 | 1.0-1.2 |
| TP1 Multiplier | 1.0-1.5 | 1.5-2.0 | 0.5-1.0 |
| EMA Periods | Higher (8-144) | Standard (5-55) | Lower (3-34) |
| Signal Frequency | Low | Medium | High |

---

## Summary

The key to successful parameter tuning is:

1. **Start with defaults**: They work for most scenarios
2. **Change one parameter at a time**: Understand each impact
3. **Match to your trading style**: Scalping ≠ Swing trading
4. **Adapt to market conditions**: Volatile ≠ Ranging
5. **Keep a trading journal**: Track what works and what doesn't
6. **Be patient**: Good parameters take time to validate

Remember: **No parameter set works 100% of the time**. The goal is consistency over many trades, not perfection on every trade.
