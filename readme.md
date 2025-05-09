# Momentum Trading Application

A comprehensive Python application for executing medium and long-term momentum trading strategies. This system is designed to be extensible, well-structured, and suitable for both research and live trading.

## 🎯 Features

- **Multiple Data Sources**: Integrated support for Tiingo, yfinance, Alpha Vantage, and Polygon.io
- **Tiingo Integration**: Primary data source with comprehensive market data coverage
- **Flexible Strategy Framework**: Easily implement new momentum strategies
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio risk controls
- **Backtesting Engine**: Comprehensive backtesting with performance metrics
- **Paper Trading**: Test strategies with simulated trading
- **Performance Monitoring**: Real-time dashboard and detailed performance tracking
- **Extensible Architecture**: Modular design for easy customization

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/momentum-trading.git
   cd momentum-trading
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e .[dev]
   ```

2. **Configure API Keys**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (especially TIINGO_API_KEY)
   ```

3. **Run a Backtest with Tiingo**:
   ```bash
   python scripts/backtest.py --strategy classic_momentum --start 2020-01-01 --end 2023-12-31 --source tiingo
   ```

4. **Run Tiingo Examples**:
   ```bash
   python scripts/tiingo_example.py
   ```

## 📁 Project Structure

```
momentum-trading/
├── config/              # Configuration files
│   ├── config.yaml     # Main configuration
│   └── strategies/     # Strategy configurations
├── src/                # Source code
│   ├── core/           # Core components (Portfolio, Risk Manager)
│   ├── data/           # Data fetching and processing
│   ├── analysis/       # Technical indicators and analysis
│   ├── strategies/     # Trading strategy implementations
│   ├── execution/      # Order execution and broker interfaces
│   └── monitoring/     # Performance tracking and dashboards
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── notebooks/          # Research and analysis notebooks

## 🔧 Core Components

### 1. Data Collection Module
- **DataFetcher**: Unified interface for multiple data sources
- **TiingoDataFetcher**: Specialized module for Tiingo integration
- **Caching System**: Reduces API calls and improves performance
- **Market Calendar**: Trading day awareness
- **Data Sources**: Tiingo (primary), yfinance, Alpha Vantage, Polygon.io

### 2. Strategy Framework
- **BaseStrategy**: Abstract base class for all strategies
- **ClassicMomentumStrategy**: Implementation of traditional momentum approach
- **Signal Generation**: Structured trading signals with metadata

### 3. Risk Management
- **Position Sizing**: Multiple methods (risk parity, Kelly criterion)
- **Risk Limits**: Maximum position size, stop-loss, drawdown controls
- **Portfolio Constraints**: Sector exposure, correlation limits

### 4. Portfolio Management
- **Position Tracking**: Real-time P&L calculation
- **Trade History**: Complete audit trail
- **Performance Metrics**: Comprehensive performance analysis

### 5. Monitoring System
- **Performance Tracker**: Real-time metrics calculation
- **Alert System**: Configurable notifications
- **Dashboard**: Visual monitoring interface

## 📈 Available Strategies

### Classic Momentum
- Ranks assets by past returns
- Invests in top performers
- Monthly rebalancing

### Dual Momentum (Coming Soon)
- Combines absolute and relative momentum
- Risk-on/risk-off switching
- Adaptive asset allocation

### Adaptive Momentum (Coming Soon)
- Dynamic lookback periods
- Regime detection
- Volatility scaling

## 🎮 Usage Examples

### Run a Backtest
```python
# Using the command line
python scripts/backtest.py \
    --strategy classic_momentum \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --capital 100000

# Or programmatically
from src.strategies.classic_momentum import ClassicMomentumStrategy
from src.core.portfolio import Portfolio
from src.data.data_fetcher import DataFetcher

# Initialize components
data_fetcher = DataFetcher(config)
portfolio = Portfolio(100000)
strategy = ClassicMomentumStrategy(config, portfolio)

# Fetch data
market_data = await data_fetcher.fetch_price_data(
    symbols=['SPY', 'QQQ', 'IWM'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Run strategy
signals = strategy.generate_signals(market_data)
strategy.execute_signals(signals, market_data)
```

### Paper Trading
```bash
python scripts/live_trading.py \
    --paper \
    --strategy classic_momentum \
    --capital 100000
```

### Performance Analysis
```python
from src.monitoring.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
tracker.update(portfolio, datetime.now())

# Get metrics
metrics = tracker.calculate_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Generate report
report = tracker.generate_report()
print(report)
```

## ⚙️ Configuration

The system uses YAML configuration files for flexibility:

```yaml
# config/config.yaml
trading:
  universe:
    equities: ["SPY", "QQQ", "IWM"]
  risk:
    max_position_pct: 0.10
    stop_loss_pct: 0.05
    
strategies:
  classic_momentum:
    lookback_period: 252
    top_n: 10
    rebalance_frequency: "monthly"
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_strategies.py
```

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:
- **Returns**: Total, annualized, volatility
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Drawdown**: Maximum drawdown, recovery time
- **Win/Loss**: Win rate, profit factor, average win/loss

## 🔌 Tiingo Integration

The application uses Tiingo as the primary data source, offering:

### Available Data Types
- **Daily OHLCV**: Historical daily price data with adjustments
- **Intraday Data**: 1-minute to hourly bars
- **Fundamental Data**: Company financials and metadata
- **News Data**: Real-time news articles and sentiment
- **Cryptocurrency Data**: Major crypto pairs

### Usage Examples

```python
# Fetch daily data
from src.data.tiingo_fetcher import TiingoDataFetcher

fetcher = TiingoDataFetcher(config)
daily_data = await fetcher.fetch_daily_data(
    symbols=['AAPL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    adjusted=True
)

# Fetch intraday data
intraday_data = await fetcher.fetch_intraday_data(
    symbols=['SPY'],
    start_date='2023-12-01',
    end_date='2023-12-31',
    frequency='1hour'
)

# Fetch fundamental data
fundamentals = await fetcher.fetch_fundamental_data(['AAPL'])
```

### Configuration

```yaml
# config/config.yaml
data:
  sources:
    tiingo:
      enabled: true
      api_key: "${TIINGO_API_KEY}"
      rate_limit: 600  # requests per hour
      cache:
        enabled: true
        ttl: 3600
```

## 🔮 Future Enhancements

- [ ] Additional momentum strategies (dual momentum, sector momentum)
- [ ] Machine learning integration for signal generation
- [ ] Real-time market data streaming
- [ ] Advanced order types and execution algorithms
- [ ] Multi-asset class support (futures, crypto)
- [ ] Cloud deployment and scaling
- [ ] Mobile monitoring app

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- [Architecture Overview](docs/architecture.md)
- [Strategy Development Guide](docs/strategies.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- Thanks to the open-source community for the excellent libraries
- Inspired by quantitative finance research and momentum studies
- Built with Python and modern software engineering practices

---

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.
