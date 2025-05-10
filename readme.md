# Momentum Trading Application

A comprehensive Python application for executing medium and long-term momentum trading strategies with integrated technical, fundamental, and sentiment analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🚀 Features

- **Advanced Momentum Strategies**: Multiple strategy implementations including classic momentum, dual momentum, adaptive momentum, and enhanced advanced momentum
- **Multi-Source Data Integration**: Primary integration with Tiingo for market data, fundamentals, and news sentiment
- **Comprehensive Analysis**: Technical indicators, fundamental analysis (Dow 30), and news sentiment scoring
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio risk controls
- **Performance Monitoring**: Real-time metrics, drawdown analysis, and comprehensive reporting
- **Universe Support**: S&P 500, Dow 30, and custom universe selection
- **Backtesting Engine**: Full backtesting capabilities with walk-forward analysis
- **Optimized Data Fetching**: Parallel processing, caching, and rate limiting

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Available Strategies](#available-strategies)
- [Data Sources](#data-sources)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/momentum-trading.git
cd momentum-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your TIINGO_API_KEY

# Run enhanced momentum analysis
python scripts/run_enhanced_advanced_momentum.py --universe sp500 --timeframe both
```

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/momentum-trading.git
   cd momentum-trading
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```
   TIINGO_API_KEY=your_tiingo_api_key_here
   ```

5. **Initialize data directories**:
   ```bash
   python scripts/setup_project.sh
   ```

## ⚙️ Configuration

The application uses YAML configuration files located in the `config/` directory:

### Main Configuration (`config/config.yaml`)

```yaml
data:
  sources:
    tiingo:
      enabled: true
      api_key: "${TIINGO_API_KEY}"
      rate_limit: 600
      batch_size: 50
      
trading:
  universe:
    equities: ["SPY", "QQQ", "IWM", "VTI"]
  risk:
    max_position_pct: 0.10
    stop_loss_pct: 0.05
    
strategies:
  enhanced_advanced_momentum:
    lookback_period: 252
    top_n: 20
    rebalance_frequency: "weekly"
```

### Strategy Configuration

Strategies can be configured in individual YAML files:

```yaml
# config/strategies/enhanced_advanced_momentum.yaml
strategies:
  enhanced_intermediate_momentum:
    timeframe: "intermediate"
    technical_weight: 0.5
    fundamental_weight: 0.3
    sentiment_weight: 0.2
```

## 📈 Available Strategies

### 1. Classic Momentum
Traditional momentum strategy based on past price performance:
```python
from src.strategies.classic_momentum import ClassicMomentumStrategy

strategy = ClassicMomentumStrategy(config)
```

### 2. Enhanced Advanced Momentum
Comprehensive strategy combining technical, fundamental, and sentiment analysis:
```python
from src.strategies.enhanced_advanced_momentum import EnhancedAdvancedMomentumStrategy

strategy = EnhancedAdvancedMomentumStrategy(
    config=config,
    tiingo_fetcher=tiingo_fetcher
)
```

Features:
- Technical indicators (RSI, MACD, Moving Averages)
- Fundamental analysis (Dow 30 companies)
- News sentiment scoring
- Multi-timeframe analysis (intermediate and long-term)
- Comprehensive scoring system

## 📊 Data Sources

### Tiingo (Primary)

The application primarily uses Tiingo for:
- Daily and intraday price data
- Fundamental data (Dow 30 companies)
- News and sentiment analysis

```python
from src.data.enhanced_tiingo_fetcher import EnhancedTiingoDataFetcher

tiingo_config = {
    'api_key': os.getenv('TIINGO_API_KEY'),
    'rate_limit': 10000,
    'batch_size': 50
}

fetcher = EnhancedTiingoDataFetcher(tiingo_config)

# Fetch price data
market_data = await fetcher.fetch_daily_data(
    symbols=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Fetch fundamental data (Dow 30 only)
fundamentals = await fetcher.fetch_fundamental_data(
    symbols=['AAPL', 'JPM', 'JNJ']
)

# Fetch news sentiment
news = await fetcher.fetch_news(
    symbols=['AAPL'],
    start_date='2023-12-01',
    limit=100
)
```

### Universe Selection

The application supports multiple universes:
- **S&P 500**: Full S&P 500 constituents
- **Dow 30**: Dow Jones Industrial Average components
- **Custom**: User-defined symbol lists

## 📚 Usage Examples

### Run Enhanced Momentum Analysis

```bash
# Analyze both timeframes for S&P 500
python scripts/run_enhanced_advanced_momentum.py --universe sp500 --timeframe both

# Analyze intermediate timeframe only
python scripts/run_enhanced_advanced_momentum.py --universe default --timeframe intermediate

# Run backtest
python scripts/backtest.py --strategy enhanced_advanced_momentum --start 2020-01-01 --end 2023-12-31
```

### Programmatic Usage

```python
import asyncio
from src.strategies.enhanced_advanced_momentum import EnhancedAdvancedMomentumStrategy
from src.data.enhanced_tiingo_fetcher import EnhancedTiingoDataFetcher
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager

async def run_strategy():
    # Initialize components
    tiingo_fetcher = EnhancedTiingoDataFetcher(tiingo_config)
    portfolio = Portfolio(initial_capital=100000)
    risk_manager = RiskManager(risk_config)
    
    # Create strategy
    strategy = EnhancedAdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager,
        tiingo_fetcher=tiingo_fetcher
    )
    
    # Fetch market data
    market_data = await fetch_market_data(symbols)
    
    # Enhance with Tiingo data
    await strategy.enhance_with_tiingo_data(symbols)
    
    # Generate signals
    signals = strategy.generate_signals(market_data)
    
    # Execute strategy
    strategy.execute_signals(signals, market_data)
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics()
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Run the strategy
asyncio.run(run_strategy())
```

## 📂 Project Structure

```
momentum-trading/
├── config/                          # Configuration files
│   ├── config.yaml                  # Main configuration
│   ├── strategies/                  # Strategy-specific configs
│   └── logging.yaml                 # Logging configuration
├── src/                             # Source code
│   ├── core/                        # Core components
│   │   ├── base_strategy.py         # Base strategy class
│   │   ├── portfolio.py             # Portfolio management
│   │   └── risk_manager.py          # Risk management
│   ├── data/                        # Data fetching and processing
│   │   ├── enhanced_tiingo_fetcher.py
│   │   └── data_fetcher_optimized.py
│   ├── strategies/                  # Trading strategies
│   │   ├── classic_momentum.py
│   │   └── enhanced_advanced_momentum.py
│   ├── analysis/                    # Technical indicators
│   │   └── momentum_indicators.py
│   └── monitoring/                  # Performance tracking
│       └── performance_tracker.py
├── scripts/                         # Utility scripts
│   ├── run_enhanced_advanced_momentum.py
│   ├── backtest.py
│   └── fetch_sp500_tickers.py
├── data/                            # Data storage
│   └── universe/                    # Universe definitions
│       ├── sp500_symbols.json
│       └── dow30_symbols.json
├── tests/                           # Test suite
├── notebooks/                       # Research notebooks
└── logs/                            # Log files
```

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

```python
metrics = tracker.calculate_metrics()
```

Available metrics:
- **Returns**: Total, annualized, monthly
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, recovery time
- **Win/Loss**: Win rate, profit factor, average win/loss

## 📈 Output Examples

The enhanced momentum analysis produces comprehensive reports:

```csv
Rank,Symbol,Action,Comprehensive Score,Technical Score,Fundamental Score,Sentiment Score
1,NVDA,BUY,0.812,0.875,0.750,0.690
2,AAPL,BUY,0.789,0.823,0.812,0.654
3,MSFT,HOLD,0.654,0.698,0.723,0.512
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test module
pytest tests/test_strategies.py
```

## 📈 Monitoring and Alerts

The application includes monitoring capabilities:

```yaml
monitoring:
  alerts:
    enabled: true
    triggers:
      - type: "drawdown"
        threshold: 0.10
      - type: "volatility_spike"
        threshold: 2.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- Tiingo for providing comprehensive financial data APIs
- The open-source community for excellent libraries
- Contributors and users of this project

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/momentum-trading/issues)
- Email: veeramm@hotmail.com

---

Built with ❤️ for the quantitative trading community
