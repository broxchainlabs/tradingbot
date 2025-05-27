# flyingbuddha-scalping-bot
Scalping bot for Indian market


# 🚀 FlyingBuddha Scalping Bot

**Deploy-ready scalping bot for Indian stock markets with real-time paper/live trading capabilities.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## ✨ Features

- **📈 Real-time Market Scanning** - Scans 40+ NSE stocks for scalping opportunities
- **⚡ Ultra-fast Signals** - 1-minute and 5-minute momentum strategies
- **🟢 Paper Trading** - Risk-free testing with simulated trades
- **🔴 Live Trading Ready** - Switch to live trading with broker API
- **📊 Live Dashboard** - Real-time P&L, positions, and performance metrics
- **🎯 Smart Position Management** - Auto stop-loss and target management
- **📱 Mobile Responsive** - Trade from anywhere

## 🚀 Quick Deploy

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repo
5. **Start Trading!** 🎯

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/flyingbuddha-scalping-bot.git
cd flyingbuddha-scalping-bot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 How It Works

### Market Scanning
- Scans 40+ liquid NSE stocks every minute
- Scores stocks based on momentum + volume
- Focuses on high-probability scalping setups

### Signal Generation
- **1-Minute Momentum**: Quick reversals with >0.15% moves
- **5-Minute Breakouts**: Volume-backed directional moves >0.3%
- **Smart Filtering**: Only trades when volume > 1.5x average

### Risk Management
- **Position Sizing**: 1% risk per trade
- **Stop Loss**: 0.25% automatic stop loss
- **Target**: 0.5% profit target
- **Time Exit**: Max 3-8 minutes per position
- **Max Positions**: 3 concurrent positions

## 🎮 Usage

### Paper Trading (Default)
1. Click **🚀 Start** to begin scanning
2. Watch live opportunities and signals
3. Monitor P&L and performance
4. **Zero risk** - all trades simulated

### Live Trading
1. Switch to **🔴 Live** mode
2. Enter your broker API credentials
3. **Real money trades** executed automatically
4. Monitor live positions and P&L

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Capital** | Current available capital + P&L |
| **Total Trades** | Number of completed trades |
| **Win Rate** | Percentage of profitable trades |
| **Active Positions** | Currently open positions |

## 🔧 Configuration

### Stock Universe
- **Banking**: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
- **IT**: TCS, INFY, WIPRO, HCLTECH, TECHM
- **Large Cap**: RELIANCE, ITC, BHARTIARTL, MARUTI
- **Others**: 25+ additional liquid stocks

### Trading Parameters
```python
CAPITAL = 100000           # Starting capital
MAX_POSITIONS = 3          # Max concurrent positions
RISK_PER_TRADE = 1%        # Risk per trade
STOP_LOSS = 0.25%          # Stop loss percentage
TARGET = 0.5%              # Profit target
MAX_HOLD_TIME = 3-8 min    # Maximum hold time
```

## 🔐 Broker Integration

### Supported Brokers
- **Zerodha** (Primary)
- **Upstox** (Coming Soon)
- **Angel Broking** (Coming Soon)

### API Setup for Live Trading
1. Get API credentials from your broker
2. Enter in the **🔗 API Setup** section
3. Click **Connect** to activate live trading

## 📱 Screenshots

### Main Dashboard
- Live market opportunities
- Active positions tracking
- Real-time P&L updates
- Recent trades history

### Signal Generation
- Real-time signal alerts
- Confidence scoring
- Strategy identification
- Execution tracking

## ⚠️ Risk Disclaimer

**IMPORTANT**: 
- This bot trades real money in live mode
- Scalping involves high frequency trading with inherent risks
- Past performance doesn't guarantee future results
- Use paper trading first to understand the bot
- Only trade with money you can afford to lose
- The creators are not responsible for trading losses

## 🛠️ Technical Details

### Data Sources
- **Paper Mode**: Yahoo Finance (free, 15-min delay)
- **Live Mode**: Broker real-time feeds

### Technology Stack
- **Frontend**: Streamlit
- **Data**: yfinance, pandas, numpy
- **Visualization**: Plotly
- **Database**: SQLite
- **Hosting**: Streamlit Cloud (free)

### System Requirements
- Python 3.7+
- Internet connection
- Modern web browser

## 🚀 Deployment Options

### Free Hosting
- **Streamlit Cloud**: Completely free, easy setup
- **Heroku**: Free tier available
- **Railway**: Free tier with good performance

### VPS Hosting (For 24/7 Operation)
- **DigitalOcean**: $5/month
- **Linode**: $5/month
- **AWS EC2**: Free tier available

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/flyingbuddha-scalping-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flyingbuddha-scalping-bot/discussions)
- **Email**: your-email@domain.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **yfinance** for market data
- **Streamlit** for the amazing framework
- **Plotly** for beautiful charts
- **Indian Stock Market** for opportunities

---

### 🎯 Ready to Deploy?

1. **Fork this repo**
2. **Deploy on Streamlit Cloud**
3. **Start with paper trading**
4. **Scale to live trading**

**Happy Scalping! 🚀📈**
