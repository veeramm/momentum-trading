# Tiingo Integration Guide

This guide covers the integration of Tiingo as a data source for the Momentum Trading Application.

## Overview

Tiingo is a financial data platform that provides high-quality market data through a REST API. The application uses Tiingo as its primary data source due to its comprehensive coverage, reliability, and cost-effectiveness.

## Setup

### 1. Get a Tiingo API Key

1. Sign up for a free account at [https://www.tiingo.com](https://www.tiingo.com)
2. Navigate to your account page
3. Copy your API token

### 2. Configure Environment

Add your Tiingo API key to the `.env` file:

```bash
TIINGO_API_KEY=your_tiingo_api_key_here
```

### 3. Update Configuration

Ensure Tiingo is enabled in `config/config.yaml`:

```yaml
data:
  sources:
    tiingo:
      enabled: true
      api_key: "${TIINGO_API_KEY}"
      rate_limit: 600  # requests per hour
      use_iex: false  # Use IEX feed for real-time data
      cache:
        enabled: true
        ttl: 3600
```

## Available Data Types

### 1. Daily Price Data

```python
from src.data.tiingo_fetcher import TiingoDataFetcher

fetcher = TiingoDataFetcher(config)

# Fetch adjusted daily prices
daily_data = await fetcher.fetch_daily_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    adjusted=True  # Use adjusted prices (recommended)
)
```

### 2. Intraday Data

```python
# Fetch hourly data
intraday_data = await fetcher.fetch_intraday_data(
    symbols=['SPY', 'QQQ'],
    start_date='2023-12-01',
    end_date='2023-12-31',
    frequency='1hour',  # Options: 1min, 5min, 15min, 30min, 1hour, 4hour
    use_iex=False  # Set to True for IEX data (if subscribed)
)
```

### 3. Fundamental Data

```python
# Fetch company fundamentals
fundamentals = await fetcher.fetch_fundamental_data(['AAPL', 'MSFT'])

# Access different types of fundamental data
for symbol, data in fundamentals.items():
    meta = data.get('meta')  # Company metadata
    statements = data.get('statements')  # Financial statements
    daily_metrics = data.get('daily_metrics')  # Daily fundamental metrics
```

### 4. News Data

```python
# Fetch recent news
news = await fetcher.fetch_news(
    symbols=['AAPL', 'TSLA'],
    start_date='2023-12-01',
    limit=100
)
```

### 5. Cryptocurrency Data

```python
# Fetch crypto prices
crypto_data = await fetcher.get_crypto_prices(
    tickers=['btcusd', 'ethusd'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    frequency='1day'
)
```

## Integration with Strategies

### Using Tiingo in Backtesting

```bash
# Run backtest with Tiingo data
python scripts/backtest.py \
    --strategy classic_momentum \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --source tiingo
```

### In Strategy Code

```python
from src.data.data_fetcher import DataFetcher

class MyStrategy(BaseStrategy):
    async def prepare_data(self):
        # DataFetcher will automatically use Tiingo if configured
        data_fetcher = DataFetcher(self.config['data'])
        
        market_data = await data_fetcher.fetch_price_data(
            symbols=self.universe,
            start_date=self.start_date,
            end_date=self.end_date,
            source='tiingo'  # Explicitly use Tiingo
        )
        
        return market_data
```

## Rate Limits and Best Practices

### Rate Limiting

Tiingo has the following rate limits:
- Free tier: 1,000 requests/hour
- Paid tiers: Higher limits based on plan

The application automatically handles rate limiting:

```python
# Rate limiter is built into TiingoDataFetcher
fetcher = TiingoDataFetcher(config)
# Automatically respects rate limits
```

### Caching

Enable caching to reduce API calls:

```yaml
data:
  cache:
    enabled: true
    backend: "redis"  # or "memory"
    ttl: 3600  # Cache for 1 hour
```

### Best Practices

1. **Use Adjusted Prices**: Always use adjusted prices for historical analysis
2. **Batch Requests**: Fetch multiple symbols in one call when possible
3. **Cache Data**: Enable caching to avoid redundant API calls
4. **Error Handling**: The fetcher includes retry logic for transient errors
5. **Data Quality**: Tiingo provides high-quality, cleaned data

## Advanced Features

### Custom Data Processing

```python
class CustomTiingoFetcher(TiingoDataFetcher):
    def _process_daily_data(self, df: pd.DataFrame, adjusted: bool) -> pd.DataFrame:
        # Custom processing
        df = super()._process_daily_data(df, adjusted)
        
        # Add custom indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
```

### Real-time Data (IEX)

If you have access to IEX data:

```python
# Configure for IEX real-time data
config = {
    'tiingo': {
        'api_key': api_key,
        'use_iex': True
    }
}

# Fetch real-time quotes
realtime_data = await fetcher.fetch_intraday_data(
    symbols=['SPY'],
    start_date=datetime.now() - timedelta(minutes=5),
    end_date=datetime.now(),
    frequency='1min',
    use_iex=True
)
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: TIINGO_API_KEY not found in environment variables
   ```
   Solution: Ensure the API key is in your `.env` file

2. **Rate Limit Exceeded**
   ```
   Error: 429 Too Many Requests
   ```
   Solution: The application will automatically wait and retry

3. **Invalid Symbol**
   ```
   Error: Symbol not found
   ```
   Solution: Check symbol format (use exchange suffix if needed)

### Debugging

Enable debug logging to see API calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Complete Example: Momentum Analysis with Tiingo

```python
import asyncio
from datetime import datetime, timedelta
from src.data.tiingo_fetcher import TiingoDataFetcher
from src.analysis.momentum_indicators import calculate_momentum

async def analyze_momentum():
    # Setup
    config = {
        'api_key': 'your_api_key',
        'cache': {'enabled': True}
    }
    fetcher = TiingoDataFetcher(config)
    
    # Define universe
    symbols = ['SPY', 'QQQ', 'IWM', 'EEM', 'TLT']
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = await fetcher.fetch_daily_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        adjusted=True
    )
    
    # Calculate momentum
    momentum_scores = {}
    for symbol, df in data.items():
        momentum = calculate_momentum(df['close'], period=252)
        momentum_scores[symbol] = momentum.iloc[-1]
    
    # Rank by momentum
    ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Momentum Rankings:")
    for rank, (symbol, score) in enumerate(ranked, 1):
        print(f"{rank}. {symbol}: {score:.4f}")

# Run the analysis
asyncio.run(analyze_momentum())
```

## API Reference

### TiingoDataFetcher Class

```python
class TiingoDataFetcher:
    def __init__(self, config: dict)
    
    async def fetch_daily_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjusted: bool = True
    ) -> Dict[str, pd.DataFrame]
    
    async def fetch_intraday_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = '5min',
        use_iex: bool = False
    ) -> Dict[str, pd.DataFrame]
    
    async def fetch_fundamental_data(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict]
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> List[Dict]
    
    async def get_crypto_prices(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        frequency: str = '1day'
    ) -> Dict[str, pd.DataFrame]
```

## Resources

- [Tiingo API Documentation](https://api.tiingo.com/documentation)
- [Tiingo Python Client](https://github.com/hydrosquall/tiingo-python)
- [Momentum Trading Application Docs](../README.md)
