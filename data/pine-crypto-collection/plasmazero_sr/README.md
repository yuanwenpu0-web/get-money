# Support/Resistance & Liquidity Zones Indicator
Advanced TradingView Pine Script indicator that automatically identifies key price levels where institutional money flows and major reversals occur. Maps critical support and resistance with liquidity zone highlighting.

## Key Features
- Automatic S/R level detection using pivot points
- Liquidity zone highlighting (buy/sell imbalances)
- Order block identification via volume analysis
- Multi-timeframe analysis capability
- Level strength scoring system
- Break & retest alerts
- Customizable lookback periods for clean charts

## Visuals
- **Support Zones (S)**  
  - Color: Green (semi-transparent boxes)
  - Drawn at pivot lows with zone width
  - Shows areas where buying pressure historically emerged
  
- **Resistance Zones (R)**  
  - Color: Red (semi-transparent boxes)
  - Drawn at pivot highs with zone width
  - Shows areas where selling pressure typically increases
  
- **Liquidity Zones (L)**  
  - Color: Blue (semi-transparent boxes)
  - High-volume areas with institutional interest
  - Thicker borders to highlight significance
  
- **Pivot Markers**
  - Green triangles: Mark pivot lows (support formation points)
  - Red triangles: Mark pivot highs (resistance formation points)

## Settings Overview
### Pivot Settings
- **Pivot Left/Right Bars** (default: 5/5): Controls pivot detection sensitivity
  - Higher values = fewer, more significant levels
  - Lower values = more responsive, more levels detected
- **Max Levels to Show** (default: 5): Limits visual clutter by showing only the most recent levels
- **Lookback Period** (default: 100 bars): Only displays pivots within this recent range
- **Extend Zones Right** (default: 50 bars): How far zones project into the future

### Display Options
- Toggle Support, Resistance, and Liquidity zones independently
- Show/hide zone labels (S, R, L markers)
- Show/hide level strength indicators
- Customizable colors for each zone type

### Liquidity Settings
- **Volume Threshold %** (default: 150%): Determines what qualifies as "high volume"
  - Volume must exceed this % of 20-period average to create liquidity zones
- **Zone Width %** (default: 0.5%): Thickness of support/resistance zones

## Best Used For
- Day trading and scalping strategies
- Range-bound market identification
- Breakout trading setups
- High-volume instruments (BTC, ETH, SPY)
- Order flow analysis
- Identifying confluence areas for entries/exits

## How It Works

### Level Detection Algorithm
1. **Scans historical price data** to identify significant pivot points based on left/right bar settings
2. **Analyzes volume clusters** to confirm level strength using 20-period moving average
3. **Maps liquidity zones** where institutional orders likely exist (high volume + pivot)
4. **Dynamically updates levels** as market structure changes and new pivots form
5. **Tracks price interactions** with levels to build strength scoring

### Zone Classifications
- **Support Zones**: Areas where buying pressure historically emerged (pivot lows)
- **Resistance Zones**: Areas where selling pressure typically increases (pivot highs)
- **Liquidity Zones**: High-volume areas with institutional interest (blue highlighting)

## Usage Notes
- Works on **any symbol and timeframe** supported by TradingView
  - Higher timeframes (4H, Daily) = stronger levels with higher precedence
  - Lower timeframes (5m, 15m) = faster reactions, more granular zones
- **Lookback period prevents chart clutter** by focusing on recent, relevant levels
- Zones are **fixed-length projections**, not infinite extensions
- **Volume analysis** helps identify where smart money is likely positioned
- Larger pivot settings filter noise; smaller settings catch micro-levels
- This is an **indicator**, not a strategy. It does not enter or exit trades by itself
- Combine with your own entry logic (RSI, MACD, trend filters, SMT, price action, etc.)
- Levels are derived from historical price action; they do not guarantee future support or resistance

## Alert Conditions
- **Resistance Break**: Triggers when price breaks above recent resistance
- **Support Break**: Triggers when price breaks below recent support

## Pro Tips
- Look for **liquidity zones (L)** near support/resistance for highest-probability trades
- Levels with **●●● strength** have been tested multiple times and are more reliable
- Reduce "Max Levels" to 3-5 for the cleanest, most actionable view
- Increase "Lookback Period" to 200+ for swing trading; decrease to 50 for scalping
- Watch for **confluence** between multiple zones stacking at similar prices
- Price often reacts strongest at the first touch of a fresh level

## WARNINGS
**ANY LEVELS, ENTRIES/EXITS ARE NOT FINANCIAL ADVICE.**

**ALWAYS USE MULTIPLE CONFLUENCES AND DON'T ENTER BLINDLY BASED OFF THIS INDICATOR.**

**THIS IS NOT FINANCIAL ADVICE. TRADE AT YOUR OWN RISK.**

Support and resistance levels are probabilities, not certainties. Always use proper risk management, stop losses, and position sizing. Past performance does not guarantee future results.

# License & Credits
This Pine Script® code is subject to the terms of the Mozilla Public License 2.0:
https://mozilla.org/MPL/2.0/

Copyright © PlasmaZero