"""
Enhanced Tiingo Data Fetcher Module

This module provides comprehensive data fetching from Tiingo,
including fundamentals, news, and real-time data for paid subscriptions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import aiohttp
import pandas as pd
from loguru import logger


class EnhancedTiingoDataFetcher:
    """
    Enhanced Tiingo data fetcher with support for advanced features.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Tiingo data fetcher.
        
        Args:
            config: Tiingo configuration dictionary
        """
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Tiingo API key required")
        
        self.base_url = config.get('base_url', 'https://api.tiingo.com')
        self.use_iex = config.get('use_iex', False)
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 600)
        self.rate_limit_window = config.get('rate_limit_window', 3600)
        self.request_count = 0
        self.request_times = []
        
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an async request to Tiingo API with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        # Check rate limit
        await self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = datetime.now()
        
        # Remove old requests outside the window
        self.request_times = [
            t for t in self.request_times 
            if (current_time - t).total_seconds() < self.rate_limit_window
        ]
        
        # If at limit, wait
        if len(self.request_times) >= self.rate_limit:
            oldest_request = self.request_times[0]
            wait_time = self.rate_limit_window - (current_time - oldest_request).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add current request
        self.request_times.append(current_time)
    
    async def fetch_daily_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjusted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily price data for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            adjusted: Use adjusted prices
            
        Returns:
            Dictionary of DataFrames by symbol
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        results = {}
        
        for symbol in symbols:
            try:
                endpoint = f"/tiingo/daily/{symbol}/prices"
                params = {
                    'startDate': start_date,
                    'endDate': end_date
                }
                
                data = await self._make_request(endpoint, params)
                df = pd.DataFrame(data)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Use adjusted prices if requested
                    if adjusted and 'adjClose' in df.columns:
                        df['close'] = df['adjClose']
                        df['open'] = df['adjOpen'] if 'adjOpen' in df.columns else df['open']
                        df['high'] = df['adjHigh'] if 'adjHigh' in df.columns else df['high']
                        df['low'] = df['adjLow'] if 'adjLow' in df.columns else df['low']
                        df['volume'] = df['adjVolume'] if 'adjVolume' in df.columns else df['volume']
                    
                    results[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return results
    
    async def fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch fundamental data for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of fundamental data by symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch company meta data
                meta_endpoint = f"/tiingo/daily/{symbol}"
                meta_data = await self._make_request(meta_endpoint)
                
                # Fetch fundamental statements
                statements_endpoint = f"/tiingo/fundamentals/{symbol}/statements"
                statements = await self._make_request(statements_endpoint)
                
                # Fetch daily fundamental metrics
                metrics_endpoint = f"/tiingo/fundamentals/{symbol}/daily"
                daily_metrics = await self._make_request(metrics_endpoint)
                
                results[symbol] = {
                    'meta': meta_data,
                    'statements': statements,
                    'daily_metrics': daily_metrics
                }
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        
        return results
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles.
        
        Args:
            symbols: Optional list of symbols to filter by
            start_date: Optional start date
            end_date: Optional end date
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        endpoint = "/tiingo/news"
        params = {'limit': limit}
        
        if symbols:
            params['tickers'] = ','.join(symbols)
        
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            params['startDate'] = start_date
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            params['endDate'] = end_date
        
        try:
            news = await self._make_request(endpoint, params)
            
            # Process news for better structure
            processed_news = []
            for article in news:
                processed_article = {
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'publishedDate': article.get('publishedDate'),
                    'url': article.get('url'),
                    'source': article.get('source'),
                    'symbols': article.get('tickers', []),
                    'tags': article.get('tags', []),
                    'sentiment': self._extract_sentiment(article)
                }
                processed_news.append(processed_article)
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _extract_sentiment(self, article: Dict) -> float:
        """
        Extract sentiment score from article.
        
        Args:
            article: News article data
            
        Returns:
            Sentiment score (-1 to 1)
        """
        # Tiingo might provide sentiment in different ways
        # Check for direct sentiment score
        if 'sentiment' in article:
            return article['sentiment']
        
        # Check for sentiment in tags or other fields
        tags = article.get('tags', [])
        
        # Simple sentiment classification based on tags
        positive_tags = ['bullish', 'upgrade', 'beat', 'positive', 'growth']
        negative_tags = ['bearish', 'downgrade', 'miss', 'negative', 'decline']
        
        positive_count = sum(1 for tag in tags if any(pos in tag.lower() for pos in positive_tags))
        negative_count = sum(1 for tag in tags if any(neg in tag.lower() for neg in negative_tags))
        
        if positive_count + negative_count > 0:
            return (positive_count - negative_count) / (positive_count + negative_count)
        
        return 0.0  # Neutral if no sentiment indicators
    
    async def fetch_intraday_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = '5min',
        use_iex: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            frequency: Data frequency (1min, 5min, etc.)
            use_iex: Use IEX data feed
            
        Returns:
            Dictionary of DataFrames by symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                if use_iex or self.use_iex:
                    endpoint = f"/iex/{symbol}/prices"
                    params = {
                        'startDate': start_date.isoformat() if isinstance(start_date, datetime) else start_date,
                        'endDate': end_date.isoformat() if isinstance(end_date, datetime) else end_date,
                        'resampleFreq': frequency
                    }
                else:
                    endpoint = f"/tiingo/crypto/{symbol}/prices"
                    params = {
                        'startDate': start_date,
                        'endDate': end_date,
                        'resampleFreq': frequency
                    }
                
                data = await self._make_request(endpoint, params)
                df = pd.DataFrame(data)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    results[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error fetching intraday data for {symbol}: {e}")
        
        return results
    
    async def fetch_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch real-time quotes for symbols (requires IEX subscription).
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of real-time quotes by symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                endpoint = f"/iex/{symbol}/quote"
                quote = await self._make_request(endpoint)
                
                results[symbol] = {
                    'last': quote.get('last'),
                    'bid': quote.get('bidPrice'),
                    'ask': quote.get('askPrice'),
                    'volume': quote.get('volume'),
                    'timestamp': quote.get('timestamp')
                }
                
            except Exception as e:
                logger.error(f"Error fetching real-time quote for {symbol}: {e}")
        
        return results
