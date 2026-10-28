# pinescript-harmonic-detector

## Introduction

The **pinescript-harmonic-detector** is a TradingView Pine Script™ indicator specifically designed to automatically identify and provide trading signals for harmonic chart patterns in cryptocurrency markets. This tool helps traders identify potential reversal zones based on geometric price patterns and Fibonacci relationships that often occur in trading charts.

**What Problem Does It Solve?**
- Eliminates the need for manual harmonic pattern identification
- Provides objective entry, stop-loss, and target levels
- Helps traders find high-probability reversal zones in crypto markets
- Reduces emotional trading through systematic pattern recognition


## Compatibility

- **Markets**: Optimized for Cryptocurrency markets (BTC, ETH, etc.)
- **Timeframes**: Works on all timeframes (1m to Monthly)
- **TradingView Version**: Compatible with TradingView Pine Script™ v5+

## Supported Harmonic Patterns

This indicator can identify the following harmonic patterns:

1. **Bat Pattern** - Features a 0.886 retracement at point D
2. **Butterfly Pattern** - Extended pattern with a 1.27 or 1.618 extension
3. **Crab Pattern** - Known for its extreme projection with 3.14 and 3.618 ratios
4. **Shark Pattern** - Features a first leg retracement of 0.886
5. **Gartley Pattern** - Classic pattern with 0.618 retracement of XA
6. **Cypher Pattern** - Contains specific BC projection

Each pattern is validated against strict Fibonacci ratio criteria for accuracy.

## Project Structure

```
pinescript-harmonic-detector/
├── src/
│   └── harmonic_patterns.pine  # Main indicator file (for TradingView publishing)
└── docs/
    └── ...                     # Documentation (coming soon)
```

## Installation & Usage

TradingView requires indicators to be published as single-file scripts:

1. Open the TradingView platform
2. Click the "Pine Editor" button in the bottom panel
3. Copy the contents of `src/harmonic_patterns.pine` into the editor
4. Click "Save" to save the script to your personal scripts
5. Click "Add to Chart" to apply the indicator to your chart

## Configuration

The indicator includes several customizable settings:

- **Pattern Selection**: Choose which patterns to detect
- **Display Options**: Configure visual elements and information display
- **Alert Settings**: Set up custom alerts for pattern detection
- **Sensitivity Controls**: Adjust tolerance levels for more or fewer signals

## Limitations and Considerations

- **Not a Standalone System**: This indicator works best as part of a comprehensive trading strategy
- **False Signals**: No indicator is 100% accurate; always confirm with other technical analysis tools
- **Market Conditions**: Performance may vary in different market conditions (trending vs ranging)
- **Repainting**: The indicator may adjust patterns as new price data becomes available
- **CPU Usage**: Detection of multiple patterns across many bars may impact chart performance

## Contributing

Contributions are welcome! If you'd like to improve the indicator:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.