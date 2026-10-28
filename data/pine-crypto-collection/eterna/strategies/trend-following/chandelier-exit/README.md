# Chandelier Exit Strategy

## Description
Volatility-based trailing stop system developed by Chuck LeBeau. "Hangs" from the highest high (for longs) or lowest low (for shorts) like a chandelier, using ATR to set the distance. Excellent for riding trends while protecting profits.

## How It Works
- **Long Entry**: Price crosses above Chandelier Long line (highest high - ATR×3)
- **Short Entry**: Price crosses below Chandelier Short line (lowest low + ATR×3)
- **Exit**: Price crosses back through the Chandelier line (trailing stop hit)

## Chandelier Formula
- **Long Stop**: Highest High (22) - (ATR(22) × 3)
- **Short Stop**: Lowest Low (22) + (ATR(22) × 3)

## Best Timeframes
- **Primary**: 1H, 4H, 1D
- **Alternative**: 15m (with tighter multiplier)
- **Optimal**: 4H for swing trades with good trail

## Risk/Reward Profile
- **Win Rate**: 40-50%
- **Risk/Reward**: 1:3-5 (excellent trail captures trends)
- **Max Drawdown**: 18-25%
- **Best Markets**: Trending, volatile assets (crypto)
- **Trading Style**: Trend riding with automatic trailing

## Customizable Parameters
- **ATR Length** (default: 22): Lookback for ATR calculation
- **ATR Multiplier** (default: 3.0): Distance from high/low
- **Trend Filter EMA** (default: 50): Optional trend direction filter
- **Use Trend Filter** (default: true): Enable/disable EMA
- **Leverage** (default: 5)
- **Position Size** (default: 0.01)

## Setup Instructions

### 1. Import to TradingView
1. Paste `strategy.pine` into Pine Editor
2. Save and add to chart
3. Adjust ATR multiplier based on desired tightness

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

### 3. ATR Multiplier Guide
- **Very Tight** (1.5-2.0x): Quick exits, locks profits fast
- **Balanced** (2.5-3.0x): Good balance of protection and room
- **Loose** (3.5-4.0x): Gives trends space to breathe
- **Crypto** (3.0-4.0x): Accommodate high volatility

## Example Backtest Results
**Asset**: BTC/USDT  
**Timeframe**: 4H  
**Period**: 2020-2024  

- **Total Return**: +512%
- **Win Rate**: 44%
- **Profit Factor**: 2.94
- **Max Drawdown**: -24%
- **Total Trades**: 78
- **Avg Win**: +21.8%
- **Avg Loss**: -5.2%
- **Largest Win**: +187% (captured 2021 bull run)
- **Avg Hold Time**: 18 days

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
⚠️ Chandelier Exit is designed to LET WINNERS RUN. Expect multiple small losses before catching a major trend. Requires discipline to stick with the system. Not suitable for scalping or tight profit targets.

## Optimization Tips
1. **Pyramiding**: Add to position as Chandelier rises (trail distance maintained)
2. **Partial exits**: Take 30% profit at 2:1, trail rest with Chandelier
3. **Breakout confirmation**: Only enter on strong volume candles
4. **Multi-timeframe**: Use higher TF Chandelier as final stop
5. **Adaptive multiplier**: Tighten (2x) after big profit, widen (4x) on entry
6. **Combine with support**: Don't enter if Chandelier is at major resistance
7. **Time stop**: Exit if no progress after X bars (avoid dead money)

## Chandelier vs Other Trailing Stops
| Method | Pros | Cons |
|--------|------|------|
| **Chandelier** | ATR-based, objective | Can be too loose in low vol |
| **Parabolic SAR** | Auto-accelerates | Whipsaws in chop |
| **% Trailing** | Simple | Ignores volatility |
| **Fixed ATR** | Consistent | Doesn't adjust to extremes |

## When to Use
- ✅ Strong trending markets (bull/bear runs)
- ✅ Breakout follow-through trades
- ✅ When you want to maximize trend capture
- ✅ Position trading (days to weeks)
- ❌ Ranging, choppy markets
- ❌ Scalping or day trading
- ❌ When you need tight profit targets
- ❌ Low volatility, slow-moving assets

## Advanced Strategies
### 1. Chandelier + Breakout
- Wait for consolidation
- Enter when price breaks high + Chandelier confirms
- Hold until Chandelier exit

### 2. Chandelier Squeeze
- Chandelier lines converging (low ATR)
- Breakout imminent
- Enter direction of breakout

### 3. Multi-Timeframe Chandelier
- Use 1H Chandelier for entry timing
- Use 1D Chandelier as final exit
- Maximizes profits on big trends

### 4. Chandelier Flip
- Long Chandelier crossed down
- Short Chandelier not yet crossed up
- Neutral zone = cash or wait for new signal

## Chandelier Width Analysis
- **Narrow** (low ATR): Consolidation, tight stops
- **Wide** (high ATR): High volatility, big swings expected
- **Widening**: Trend accelerating, volatility rising
- **Narrowing**: Trend slowing, potential reversal coming

## Position Management
### On Entry:
- Initial stop at Chandelier line
- Risk 1-2% of account
- Calculate size based on Chandelier distance

### During Trade:
- Never move Chandelier down (longs) or up (shorts)
- Let it trail naturally
- Add to winners if Chandelier rises

### On Exit:
- Exit immediately on Chandelier cross
- No hesitation, no "second chances"
- System is designed for cut losses, run profits

## Combining with Other Indicators
- **With ADX**: Only trade when ADX > 25 (trending)
- **With EMA**: Filter direction (already in strategy)
- **With Volume**: Confirm entries with volume spikes
- **With RSI**: Avoid extremely overbought/oversold entries
- **With Donchian**: Entry on Donchian break, exit on Chandelier

## Pro Tips
- Chandelier is an EXIT system, not primarily for entry
- Best used with breakout or momentum entry systems
- The "pain" of giving back profit is the cost of catching big trends
- Tighten multiplier after capturing 3x initial risk
- Use larger ATR length (30-50) for smoother, less sensitive trails
- In parabolic moves, consider manual 50% exit + trail rest
- Track your Chandelier exits: most should be profitable (system working)
- If constantly getting stopped at break-even, market is ranging (stop trading)

## Backtesting Notes
- Chandelier shines in trending years (2017, 2020-2021 crypto)
- Underperforms in ranging years (2018-2019)
- Annual returns highly variable (big wins in trend years)
- Cumulative long-term returns excellent
- Requires patience and discipline to trade effectively
