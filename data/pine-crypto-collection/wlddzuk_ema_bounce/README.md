# TradingView-to-Telegram Signal Bot

A production-ready FastAPI server that receives webhook alerts from TradingView Pine Script strategies and forwards formatted trading signals to Telegram channels with London timezone display and comprehensive timeframe support.

## 🚀 Features

- **📊 Pine Script Integration**: Receives webhooks from TradingView EMA Bounce + VWAP + MACD strategy
- **🕐 London Timezone**: Displays all timestamps in Europe/London (GMT/BST with DST support)
- **⏰ 5 Timeframes**: Supports 5m, 15m, 1h, 4h, 1D trading timeframes
- **💰 Smart Price Formatting**: USD pairs ($65,000.12), BTC pairs (0.05234567 BTC)
- **🔄 Retry Logic**: Exponential backoff for reliable message delivery
- **🛡️ Security**: Shared secret authentication and admin-only commands
- **📈 Dynamic Pair Management**: Add/remove trading pairs via Telegram commands
- **🎯 Chat Routing**: Route different symbols/timeframes to different channels
- **📊 Statistics**: Comprehensive signal tracking and reporting
- **🐳 Docker Ready**: Production containerization with health checks

## 📋 Prerequisites

- Python 3.11+
- TradingView Premium account (for webhook alerts)
- Telegram Bot Token (from @BotFather)
- Telegram Channel/Group to send signals

## 🛠️ Quick Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd tradingview-telegram-bot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (see Configuration section)
```

### 3. Run Locally

```bash
python main.py
```

### 4. Deploy to Production (Render.com)

```bash
# Push to GitHub, then connect to Render.com
# Set environment variables in Render dashboard
```

## ⚙️ Configuration

### Required Environment Variables

```bash
# TradingView webhook security
TV_SHARED_SECRET=your_webhook_secret_here_make_it_long_and_random

# Telegram Bot Configuration  
TELEGRAM_BOT_TOKEN=1234567890:your_bot_token_here
TELEGRAM_CHAT_ID_DEFAULT=-1001234567890

# Admin users (comma-separated)
TELEGRAM_ADMIN_IDS=123456789,987654321
```

### Optional Configuration

```bash
# Timezone (default: Europe/London)
TZ_DISPLAY=Europe/London

# Trading pairs (default: BTCUSDT,ETHUSDT,ETHBTC,ADAUSDT)
DEFAULT_PAIRS=BTCUSDT,ETHUSDT,ETHBTC,ADAUSDT

# Timeframes (default: 5,15,60,240,D)
SUPPORTED_TIMEFRAMES=5,15,60,240,D

# Advanced chat routing (JSON format)
TELEGRAM_TF_CHAT_MAP={"5": "-1001111111111", "60": "-1001222222222"}
TELEGRAM_SYMBOL_CHAT_MAP={"BTCUSDT": "-1001111111111", "ETHBTC": "-1001222222222"}
```

## 📱 TradingView Setup

### 1. Install Pine Script Strategy

Copy the `ema_bounce_vwap_macd_strategy.pine` file content and add it as a new indicator in TradingView.

### 2. Create Webhook Alerts

For each symbol and timeframe combination, create an alert with:

- **Condition**: EMA Bounce Buy
- **Webhook URL**: `https://your-domain.com/api/v1/tv-webhook`
- **Message**: Leave empty (handled by Pine Script)
- **Headers**: `X-TV-Secret: your_webhook_secret_here`

### Required Alerts (20 total)

| Symbol   | Timeframes      |
|----------|-----------------|
| BTCUSDT  | 5m, 15m, 1h, 4h, 1D |
| ETHUSDT  | 5m, 15m, 1h, 4h, 1D |
| ETHBTC   | 5m, 15m, 1h, 4h, 1D |
| ADAUSDT  | 5m, 15m, 1h, 4h, 1D |

## 🤖 Telegram Commands

### User Commands

- `/start` - Welcome message and bot info
- `/help` - Show available commands
- `/status` - Show active pairs and recent signals
- `/signals` - Show recent signals (last 24h)
- `/chart <SYMBOL> <TF>` - Generate TradingView chart link

### Admin Commands

- `/add <SYMBOL>` - Add new coin pair to monitoring
- `/remove <SYMBOL>` - Remove coin pair from monitoring
- `/list` - Show all monitored pairs
- `/stats` - Show detailed signal statistics
- `/config` - Show current bot configuration

### Command Examples

```bash
/chart BTCUSDT 1h        # Get BTC 1h chart
/add ADAUSDT             # Add ADA pair (admin only)
/remove ETHBTC           # Remove ETH/BTC pair (admin only)
/signals                 # Show recent signals
```

## 📊 Signal Format

```
🚀 **EMA BOUNCE SIGNAL** 🚀

💰 **COIN PAIR**: BTC/USDT
⏰ **TIMEFRAME**: 5M
📅 **Signal Time**: 2025-08-21 09:00 (GMT)

📈 **TRADE DETAILS**:
🔵 **ENTRY**: $65,000.12
🔴 **STOP LOSS**: $64,200.45
🟢 **TAKE PROFIT**: $68,999.77

📊 **RISK METRICS**:
💸 **Risk**: 1.23% (Entry to Stop)
🎯 **Reward**: 3.0R (3:1 Risk/Reward)

🔗 **Chart**: [View on TradingView](https://tradingview.com/chart/...)

🆔 Signal ID: BTCUSDT_5_1734567890000
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t tv-telegram-bot .

# Run container
docker run -d \
  --name tv-telegram-bot \
  -p 8000:8000 \
  --env-file .env \
  -v ./data:/app/data \
  tv-telegram-bot
```

### Docker Compose

```yaml
version: '3.8'
services:
  tv-telegram-bot:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🌐 Render.com Deployment

### 1. Connect Repository

1. Sign up at [Render.com](https://render.com)
2. Connect your GitHub repository
3. Create new Web Service

### 2. Configure Service

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`
- **Environment**: Python 3.11

### 3. Set Environment Variables

In Render dashboard, add all required environment variables from `.env.example`.

### 4. Deploy

Render will automatically deploy when you push to your main branch.

## 🧪 Testing

### Test Webhook Endpoint

```bash
# Test webhook connectivity
curl -X GET https://your-domain.com/api/v1/webhook-test

# Test webhook with sample data
curl -X POST https://your-domain.com/api/v1/tv-webhook \
  -H "Content-Type: application/json" \
  -H "X-TV-Secret: your_secret_here" \
  -d '{
    "event": "EMA_BOUNCE_BUY",
    "symbol": "BTCUSDT",
    "timeframe": "60",
    "bar_time": 1734567890000,
    "entry": 65000.12,
    "stop": 64200.45,
    "target": 68999.77,
    "rr": 3.0,
    "signal_id": "BTCUSDT_60_1734567890000"
  }'
```

### Health Checks

```bash
# Basic health check
curl https://your-domain.com/healthz

# Comprehensive readiness check
curl https://your-domain.com/readyz
```

## 📁 Project Structure

```
├── main.py                    # FastAPI server entry point
├── webhook_handler.py         # TradingView webhook processing
├── telegram_bot.py           # Telegram message formatting and sending
├── telegram_handlers.py      # Bot command handlers
├── database.py               # Database operations and schema
├── config.py                 # Configuration and environment
├── formatters.py             # Message and price formatting
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .env.example             # Environment variables template
├── README.md                # This file
├── CLAUDE.md                # Project documentation
├── tasks/todo.md            # Implementation checklist
└── ema_bounce_vwap_macd_strategy.pine  # TradingView Pine Script
```

## 🔧 Advanced Configuration

### Chat Routing

Route different symbols or timeframes to different Telegram channels:

```bash
# Route by timeframe
TELEGRAM_TF_CHAT_MAP={"5": "-100111111", "60": "-100222222", "D": "-100333333"}

# Route by symbol  
TELEGRAM_SYMBOL_CHAT_MAP={"BTCUSDT": "-100111111", "ETHBTC": "-100222222"}
```

Routing priority: Symbol-specific → Timeframe-specific → Default chat

### Retry Configuration

```bash
TELEGRAM_MAX_RETRIES=3
TELEGRAM_RETRY_DELAYS=1,2,4  # Exponential backoff delays in seconds
```

### Database Settings

```bash
DATABASE_URL=sqlite:///./signals.db
IDEMPOTENCY_TTL_DAYS=7  # How long to keep signals for deduplication
```

## 🛡️ Security

- **Webhook Authentication**: Shared secret validation with constant-time comparison
- **Admin Authorization**: Telegram user ID whitelist for management commands
- **Rate Limiting**: Configurable per-IP rate limits
- **Input Validation**: Comprehensive Pydantic models for all inputs
- **SQL Injection Protection**: Parameterized queries throughout

## 📊 Monitoring

### Health Endpoints

- `GET /healthz` - Basic liveness check
- `GET /readyz` - Comprehensive readiness check (DB + Telegram connectivity)
- `GET /` - API information and configuration overview

### Structured Logging

All events logged in JSON format with relevant context:

```json
{
  "timestamp": "2025-08-21T09:00:00Z",
  "level": "info", 
  "event": "webhook_received",
  "signal_id": "BTCUSDT_60_1734567890000",
  "symbol": "BTCUSDT",
  "timeframe": "60"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Signals not sending**
   - Check Telegram bot token and chat ID
   - Verify webhook secret matches
   - Check `/readyz` endpoint for connectivity

2. **TradingView alerts not received**
   - Verify webhook URL is accessible
   - Check `X-TV-Secret` header in alert settings
   - Review webhook endpoint logs

3. **Wrong timezone display**
   - Verify `TZ_DISPLAY=Europe/London` in environment
   - Check system timezone if running locally

4. **Permission errors**
   - Ensure bot is added to Telegram channel
   - Verify admin user IDs are correct
   - Check bot has message sending permissions

### Debug Mode

```bash
DEBUG=true python main.py
```

Enables detailed logging and CORS for development.

## 📚 API Documentation

When running, visit `http://localhost:8000/docs` for interactive API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TradingView for Pine Script v6 platform
- Telegram Bot API for messaging capabilities
- FastAPI for high-performance async web framework
- Contributors and testers

---

**Ready to receive trading signals!** 🚀📈

For support, please open an issue on GitHub or contact the maintainers.