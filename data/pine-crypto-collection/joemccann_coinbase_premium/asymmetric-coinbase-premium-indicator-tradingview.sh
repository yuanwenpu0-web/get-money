//@version=5
//
//    _____                                        __         .__        
//   /  _  \   _________.__. _____   _____   _____/  |________|__| ____  
//  /  /_\  \ /  ___<   |  |/     \ /     \_/ __ \   __\_  __ \  |/ ___\ 
// /    |    \\\___ \ \___  |  Y Y  \  Y Y  \  ___/|  |  |  | \/  \  \___ 
// \____|__  /____  >/ ____|__|_|  /__|_|  /\___  >__|  |__|  |__|\___  >
//         \/     \/ \/          \/      \/     \/                    \/ 
//
// Disclaimers: https://t.co/kOsUzy5pCB

indicator("Asymmetric Coinbase Premium Histogram with SMA", overlay=false)

// Extract the base currency from the current chart (e.g., "BTC" from "BTCUSD")
base_asset = syminfo.basecurrency

// Construct Coinbase and Binance symbols dynamically
coinbase_symbol = "COINBASE:" + base_asset + "USD"
binance_symbol = "BINANCE:" + base_asset + "USDT"

// Fetch prices from Coinbase and Binance
coinbase_price = request.security(coinbase_symbol, timeframe.period, close)
binance_price = request.security(binance_symbol, timeframe.period, close)

// Calculate the premium (absolute and percentage)
premium = coinbase_price - binance_price
percent_premium = (premium / binance_price) * 100

// Input to select between notional and percentage premium
show_percent = input.bool(false, title="Show Percentage Premium")

// Select the value to plot based on user input
histogram_value = show_percent ? percent_premium : premium

// Define the color based on the premium value
bar_color = histogram_value > 0 ? color.green : color.red

// Plot the histogram
plot(histogram_value, style=plot.style_histogram, color=bar_color, title="Asymmetric Coinbase Premium Histogram")

// Input for the moving average length
sma_length = input.int(14, title="SMA Length", minval=1)

// Calculate the simple moving average (SMA) of the premium
sma_premium = ta.sma(histogram_value, sma_length)

// Plot the SMA line on the same chart
plot(sma_premium, color=color.gray, linewidth=2, title="SMA of Premium")
