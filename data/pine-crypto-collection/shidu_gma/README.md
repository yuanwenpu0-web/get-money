# Strategy 1 : MA + Filters + RIS + ADX + ATR - BTC/USDT (15-Min Timeframe)

![CAP1](/cap1.png)
*image from backtesting environment using july to august 2024*

### Overview

This Pine Script strategy, is a working progress of my algotrading journey and learning process and it is designed for trading BTC/USDT on a 15-minute timeframe. It incorporates advanced moving averages and filters, trend confirmation using RSI (Relative Strength Index) and ADX (Average Directional Index), and dynamic risk management based on volatility using the ATR (Average True Range). The source of knowledge and the credits belongs to loxx, you can reach him on his profile at tradingview.com.

### Key Features:

1.	Gaussian Moving Average (GMA):
The script uses a Gaussian Moving Average with Fibonacci weighting to smooth price data and reduce the impact of outliers. This technique helps identify trends by assigning higher importance to central price values and lower importance to outliers.
2.	2-Pole Super Smoother Filter (2PSSF):
To further reduce noise, the GMA is combined with John Ehlers’ 2-Pole Super Smoother Filter, which closely follows price trends, ensuring smoother signals and avoiding false reversals in volatile conditions.
3.	RSI + ADX Confirmation:
Entry decisions are confirmed using a combination of RSI and ADX. The RSI measures price momentum, and the ADX assesses trend strength. Using both indicators together ensures that trades are entered during strong trends while avoiding weak or choppy markets.
4.	Dynamic Trailing Stop with ATR:
To manage risk dynamically, the stop-loss level is adjusted using the ATR. The strategy includes a volatility threshold to switch between aggressive and conservative risk settings. When volatility is high (ATR exceeds a threshold), the script applies a more conservative stop-loss distance to avoid premature exits.

### Strategy Logic:

Indicators:

* Gaussian Moving Average (GMA): A Fibonacci-weighted moving average that reduces noise while identifying the core trend. The strategy calculates this using Fibonacci series values to give more weight to the most recent prices.
* RSI: Relative Strength Index is used to gauge momentum and determine overbought/oversold conditions.
* ADX: The Average Directional Index is used to assess the strength of the trend. An ADX value above 25 suggests a strong trend.

Entry Conditions:

	•	Long Entry:
	•	The Gaussian Moving Average crosses above its signal line.
	•	RSI values (current, previous, and two bars back) are above 45.
	•	ADX value is above 25, confirming a strong trend.
	•	Short Entry:
	•	The Gaussian Moving Average crosses below its signal line.
	•	RSI values (current, previous, and two bars back) are below 55.
	•	ADX confirms trend strength (above 25).

Exit Conditions:

	•	Long Exit:
	•	The price reaches the target price (calculated using a spread from the entry price).
	•	The trailing stop, dynamically adjusted using ATR, is hit.
	•	Short Exit:
	•	The price reaches the target price or trailing stop.
	•	The trailing stop is based on ATR and adjusted according to volatility.

### Risk Management:

The script uses dynamic risk management based on ATR. If the ATR exceeds 1.5x, the stop-loss distance is widened using a 2.0x ATR multiplier. This adjustment accounts for increased market volatility, providing more room for trades to develop.

### Alerts:

The strategy generates custom webhook alerts to integrate with external platforms, allowing users to automate trade execution via their preferred broker/exchange. Alerts are triggered for entry and exit conditions for both long and short trades.

### Customization Options:

The following parameters can be adjusted:

	•	ATR Period and Multiplier: Fine-tune the sensitivity of the dynamic trailing stop.
	•	RSI & ADX Levels: Modify the thresholds for momentum and trend strength validation.
	•	Trade Size & Risk: Set the percentage of equity used for each trade and other risk-related parameters.

### Backtesting and Performance:

The strategy includes a built-in backtesting framework with performance tracking for:

	•	Win rate
	•	Profit/loss metrics
	•	Drawdown
	•	Average win/loss
	•	Commission and net returns

How to Use:

	1.	Add the script to a 15-minute BTC/USDT chart.
	2.	Adjust the strategy parameters as necessary for your risk tolerance and trading preferences.
	3.	Enable alerts to automate trade execution using webhook integration.
	4.	Backtest the strategy with historical data and analyze the results using the built-in performance table.

### Conclusion:

This script combines advanced technical analysis tools like Fibonacci-weighted Gaussian averages and trend confirmation indicators (RSI, ADX) with dynamic risk management, making it a robust strategy for trading in volatile crypto markets like BTC/USDT. The use of trailing stops based on ATR ensures trades adapt to changing market conditions, offering both protection and opportunity capture.

### Disclaimer: 

Do not use this script in a real world scenario because it is intended for educational and learning purposes. I'll not be responsible for any loss or damage that it can cause. 

### License

This script is licensed under the MIT License.
