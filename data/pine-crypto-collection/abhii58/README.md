**# Btc-trading-1---5-min-strategy-script**
This TradingView Pine Script indicator provides high-probability scalping signals for Bitcoin (BTC) on 1 to 5 minute timeframes. It combines multiple technical indicators like EMA crossovers, RSI, and MACD to generate buy/sell signals, making it ideal for short-term crypto traders.

**# Strategy Overview**

The script uses-

   Fast and Slow EMA crossover

   RSI filter to confirm momentum

   MACD line vs signal line to validate trend strength

   Candle body direction (close > open or vice versa)


When all conditions align, a Buy or Sell signal is triggered and visually plotted on the chart.


**Features**

✅ Buy/Sell Signal Alerts
✅ Configurable EMA, RSI, MACD values
✅ Signal Labels: BUY 📈 and SELL 📉
✅ Visual shapes (triangles) to avoid clutter
✅ Clean chart overlay with Fast and Slow EMAs
✅ Works best on 1-min to 5-min BTC/USD charts


**Logic Behind Signals**

Buy Signal is generated when:

    EMA Fast crosses above EMA Slow

    RSI is above Buy Threshold (default: 50)

    MACD Line is above Signal Line

    Candle is bullish (Close > Open)

Sell Signal is generated when:

    EMA Fast crosses below EMA Slow

    RSI is below Sell Threshold (default: 50)

    MACD Line is below Signal Line

    Candle is bearish (Close < Open)

 How to Use

    Open TradingView

    Go to Pine Editor and paste the script.

    Add the indicator to your chart.

    Use with BTC pairs on 1min–5min charts.

    Enable alerts using the Buy Alert and Sell Alert conditions.


### ⚠️ Disclaimer

This indicator is **for educational purposes only**. Trading cryptocurrencies involves significant risk. Always test strategies on demo accounts and use proper risk management. The author is not responsible for financial losses.

   
