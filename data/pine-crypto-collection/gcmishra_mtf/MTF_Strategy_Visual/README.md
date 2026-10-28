# Multi-Timeframe Strategy Visual Indicator

A professional Pine Script indicator for TradingView that implements a multi-timeframe EMA cloud strategy with ATR volatility filtering.

## Features

✅ **Multi-Timeframe Analysis** (HTF/MTF/LTF)  
✅ **EMA Cloud Visualization** (6 EMAs with color-coded zones)  
✅ **ATR Volatility Filter** (avoid low-volatility periods)  
✅ **Sideways Market Detection** (prevents false signals)  
✅ **Entry Signal Labels** (BUY/SELL with entry price)  
✅ **TP/SL Levels** (automatic calculation based on ATR)  
✅ **Real-Time Dashboard** (shows current market status)  
✅ **TradingView Alerts** (get notified of signals)

## Installation

1. Open TradingView
2. Click on "Pine Editor" at the bottom of the chart
3. Copy the entire code from `indicator.pine`
4. Click "Add to Chart"
5. Configure parameters as needed

## Strategy Logic

### Higher Timeframe (HTF) - Default: 1 Hour
- Defines overall market bias/direction
- EMAs must be in perfect alignment (all ascending or descending)

### Medium Timeframe (MTF) - Default: 10 Minutes
- Confirms HTF direction
- ATR filter must be active (sufficient volatility)

### Lower Timeframe (LTF) - Current Chart
- Triggers entry signals
- Price must break through EMA cloud with confirmation

### Signal Requirements

**BUY Signal:**
- HTF: Bullish (EMAs ascending)
- MTF: Aligned with HTF + ATR active
- LTF: Price breaks above EMA cloud
- No sideways market condition

**SELL Signal:**
- HTF: Bearish (EMAs descending)
- MTF: Aligned with HTF + ATR active
- LTF: Price breaks below EMA cloud
- No sideways market condition

## Parameters

See `PARAMETERS.md` for detailed parameter explanations and tuning guide.

## Visual Elements

- **Green Background**: Bullish market bias
- **Red Background**: Bearish market bias
- **Gray Background**: Sideways/choppy market
- **Green Label "BUY"**: Long entry signal
- **Red Label "SELL"**: Short entry signal
- **Blue Line**: Entry price
- **Red Dashed**: Stop loss
- **Green Dashed**: Take profit levels (TP1, TP2, TP3)

## Dashboard

Top-right corner shows:
- Current status (Active/Waiting/Sideways)
- ATR value and volatility level
- HTF bias (Bullish/Bearish/Sideways)
- MTF alignment status
- LTF trend direction
- Last signal generated

## Alerts

Configure alerts for:
- Buy signals
- Sell signals
- Sideways market warnings

## Recommended Settings

### For Crypto (BTCUSD, ETHUSD)
- HTF: 60 minutes
- MTF: 10 minutes
- Chart: 3 minutes
- ATR Period: 14

### For Forex (EURUSD, GBPUSD)
- HTF: 240 minutes (4H)
- MTF: 60 minutes (1H)
- Chart: 15 minutes
- ATR Period: 14

### For Stocks
- HTF: Daily (1D)
- MTF: 60 minutes (1H)
- Chart: 15 minutes
- ATR Period: 14

## Risk Management

Always use:
- Stop loss at SL level (1.5x ATR)
- Take partial profits at TP1 (1.5x ATR)
- Move stop to breakeven after TP1
- Trail remaining position to TP3

## Usage Tips

1. **Wait for Alignment**: All three timeframes should align before taking trades
2. **Check the Dashboard**: Verify HTF bias and MTF alignment before entering
3. **Avoid Gray Zones**: Don't trade when background is gray (sideways market)
4. **Use Higher Timeframes**: Larger timeframes generally produce more reliable signals
5. **ATR Filter**: Ensure ATR is active (volatility is sufficient) before trading

## Troubleshooting

### No Signals Appearing
- Check that all three timeframes are properly aligned
- Verify ATR filter is not too restrictive
- Ensure chart timeframe is lower than MTF and HTF

### Too Many False Signals
- Increase ATR threshold multiplier (try 1.2-1.5)
- Use higher timeframes for HTF/MTF
- Tighten sideways detection threshold

### Dashboard Not Showing
- Enable "Show Dashboard" in Display Options
- Check that indicator is properly loaded on chart

## Support

For issues or questions, open an issue on GitHub.

## License

MIT License - Free to use and modify

## Disclaimer

This indicator is for educational and informational purposes only. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always use proper risk management and never risk more than you can afford to lose.
