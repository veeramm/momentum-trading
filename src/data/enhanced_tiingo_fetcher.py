# src/data/enhanced_tiingo_fetcher.py

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
from tqdm.asyncio import tqdm


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
        
        # Batch settings
        self.max_concurrent_requests = config.get('max_concurrent_requests', 10)
        self.batch_size = config.get('batch_size', 50)
        
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
    
    async def _fetch_fundamental_single(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a single symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Fundamental data dictionary
        """
        try:
            # Fetch company meta data
            meta_endpoint = f"/tiingo/daily/{symbol}"
            meta_data = await self._make_request(meta_endpoint)
            
            # Try to fetch fundamental statements (might fail for DOW 30)
            statements = {}
            daily_metrics = {}
            
            try:
                statements_endpoint = f"/tiingo/fundamentals/{symbol}/statements"
                statements = await self._make_request(statements_endpoint)
            except Exception as e:
                if "DOW 30" in str(e) or "404" in str(e):
                    # Silently handle DOW 30 limitation
                    pass
                else:
                    logger.debug(f"Error fetching statements for {symbol}: {e}")
            
            try:
                metrics_endpoint = f"/tiingo/fundamentals/{symbol}/daily"
                daily_metrics = await self._make_request(metrics_endpoint)
            except Exception as e:
                if "DOW 30" in str(e) or "404" in str(e):
                    # Silently handle DOW 30 limitation
                    pass
                else:
                    logger.debug(f"Error fetching daily metrics for {symbol}: {e}")
            
            return {
                'meta': meta_data,
                'statements': statements,
                'daily_metrics': daily_metrics,
                'has_fundamentals': bool(statements or daily_metrics)
            }
            
        except Exception as e:
            logger.debug(f"Error fetching fundamentals for {symbol}: {e}")
            return {'error': str(e), 'has_fundamentals': False}
    
    async def fetch_fundamental_data(
        self, 
        symbols: List[str], 
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Fetch fundamental data for symbols with batch processing.
        
        Args:
            symbols: List of symbols
            show_progress: Show progress bar
            
        Returns:
            Dictionary of fundamental data by symbol
        """
        results = {}
        
        # Split into batches
        batches = [symbols[i:i+self.batch_size] 
                  for i in range(0, len(symbols), self.batch_size)]
        
        # Process batches with progress bar
        if show_progress:
            pbar = tqdm(total=len(symbols), desc="Fetching fundamentals", unit="symbols")
        
        for batch in batches:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def fetch_with_semaphore(symbol):
                async with semaphore:
                    return symbol, await self._fetch_fundamental_single(symbol)
            
            # Fetch batch concurrently
            tasks = [fetch_with_semaphore(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.debug(f"Error in batch processing: {result}")
                else:
                    symbol, data = result
                    results[symbol] = data
            
            if show_progress:
                pbar.update(len(batch))
            
            # Small delay between batches
            if batch != batches[-1]:
                await asyncio.sleep(0.5)
        
        if show_progress:
            pbar.close()
        
        return results
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles with improved error handling.
        
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
            logger.warning(f"Error fetching news: {e}. Returning empty news list.")
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
