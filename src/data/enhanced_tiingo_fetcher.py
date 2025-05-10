"""
Enhanced Tiingo Data Fetcher

This module provides enhanced Tiingo data fetching with parallel processing,
Dow 30 filtering, and comprehensive data retrieval.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import aiohttp
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class EnhancedTiingoDataFetcher:
    """
    Enhanced Tiingo data fetcher with parallel processing and advanced features.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the enhanced Tiingo data fetcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Tiingo API key required")
        
        # Load configuration parameters
        self.base_url = config.get('base_url', 'https://api.tiingo.com')
        self.use_iex = config.get('use_iex', False)
        self.rate_limit = config.get('rate_limit', 600)
        self.rate_limit_window = config.get('rate_limit_window', 3600)
        
        # Parallel processing settings
        self.batch_size = config.get('batch_size', 50)
        self.max_concurrent_requests = config.get('max_concurrent_requests', 10)
        self.max_workers = config.get('max_workers', 10)
        
        # Network settings
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Load Dow 30 symbols
        self.dow30_symbols = self._load_dow30_symbols()
        logger.info(f"Loaded {len(self.dow30_symbols)} Dow 30 symbols")
        
        # Initialize session
        self.session = None
        
        # Initialize rate limiter
        self._rate_limiter = AsyncRateLimiter(
            max_calls=self.rate_limit,
            time_window=self.rate_limit_window
        )
    
    def _load_dow30_symbols(self) -> set:
        """Load Dow 30 symbols from file."""
        try:
            project_root = Path(__file__).parent.parent.parent
            symbols_file = project_root / 'data' / 'universe' / 'dow30_symbols.json'
            
            if symbols_file.exists():
                with open(symbols_file, 'r') as f:
                    data = json.load(f)
                    return set(data['symbols'])
            else:
                logger.warning("Dow 30 symbols file not found")
                return set()
        except Exception as e:
            logger.error(f"Error loading Dow 30 symbols: {e}")
            return set()
    
    def _is_dow30(self, symbol: str) -> bool:
        """Check if symbol is in Dow 30."""
        return symbol.upper() in self.dow30_symbols
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.api_key}'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: int = 100,
        retry_count: int = 0
    ) -> List[Dict]:
        """
        Fetch news data from Tiingo with retry logic.
        
        Args:
            symbols: List of symbols to get news for
            start_date: Start date for news
            end_date: End date for news
            limit: Maximum number of news items
            retry_count: Current retry attempt
            
        Returns:
            List of news articles with sentiment data
        """
        session = await self._get_session()
        
        # Build URL
        url = f"{self.base_url}/tiingo/news"
        
        # Build parameters
        params = {
            'token': self.api_key,
            'limit': limit
        }
        
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
            await self._rate_limiter.acquire()
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status in [502, 503, 504] and retry_count < self.max_retries:
                    # Retry for server errors
                    logger.warning(f"Server error {response.status}, retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay * (retry_count + 1))
                    return await self.fetch_news(symbols, start_date, end_date, limit, retry_count + 1)
                else:
                    error_msg = await response.text()
                    logger.error(f"Error fetching news: {response.status} - {error_msg}")
                    return []
                    
        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                logger.warning(f"Request timeout, retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay * (retry_count + 1))
                return await self.fetch_news(symbols, start_date, end_date, limit, retry_count + 1)
            else:
                logger.error("Timeout error after all retries")
                return []
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []
    
    async def fetch_fundamental_data(
        self, 
        symbols: List[str], 
        show_progress: bool = True,
        force_all: bool = False
    ) -> Dict[str, Dict]:
        """
        Fetch fundamental data for symbols with optimized parallel processing.
        
        Args:
            symbols: List of symbols
            show_progress: Show progress bar
            force_all: Force fetch for all symbols, not just Dow 30
            
        Returns:
            Dictionary of fundamental data by symbol
        """
        results = {}
        
        # Filter symbols to only Dow 30 unless forced
        if not force_all:
            dow30_symbols = [s for s in symbols if self._is_dow30(s)]
            non_dow30_symbols = [s for s in symbols if not self._is_dow30(s)]
            
            if non_dow30_symbols:
                logger.info(f"Skipping {len(non_dow30_symbols)} non-Dow 30 symbols for fundamentals")
            
            symbols_to_fetch = dow30_symbols
        else:
            symbols_to_fetch = symbols
        
        if not symbols_to_fetch:
            logger.warning("No Dow 30 symbols found in request")
            return results
        
        logger.info(f"Fetching fundamentals for {len(symbols_to_fetch)} Dow 30 symbols")
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=len(symbols_to_fetch), desc="Fetching fundamentals", unit="symbols")
        
        # Process in parallel batches
        batch_size = min(self.batch_size, self.max_workers)
        batches = [symbols_to_fetch[i:i+batch_size] 
                  for i in range(0, len(symbols_to_fetch), batch_size)]
        
        for batch in batches:
            # Create tasks for parallel execution
            tasks = []
            for symbol in batch:
                task = asyncio.create_task(self._fetch_fundamental_single(symbol))
                tasks.append((symbol, task))
            
            # Wait for batch to complete
            for symbol, task in tasks:
                try:
                    data = await task
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching fundamentals for {symbol}: {e}")
                    results[symbol] = {'error': str(e), 'has_fundamentals': False}
                
                if show_progress:
                    pbar.update(1)
            
            # Small delay between batches to respect rate limits
            if batch != batches[-1]:
                await asyncio.sleep(0.5)
        
        if show_progress:
            pbar.close()
        
        # Add placeholder for non-Dow 30 symbols if not forcing all
        if not force_all:
            for symbol in non_dow30_symbols:
                results[symbol] = {
                    'meta': {'symbol': symbol, 'note': 'Fundamentals available for Dow 30 only'},
                    'statements': {},
                    'daily_metrics': {},
                    'has_fundamentals': False
                }
        
        logger.info(f"Completed fundamental data fetch for {len(results)} symbols")
        return results
    
    async def _fetch_fundamental_single(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a single symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of fundamental data
        """
        session = await self._get_session()
        
        # URLs for different fundamental endpoints
        urls = {
            'meta': f"{self.base_url}/tiingo/fundamentals/{symbol}/meta",
            'statements': f"{self.base_url}/tiingo/fundamentals/{symbol}/statements",
            'daily': f"{self.base_url}/tiingo/fundamentals/{symbol}/daily"
        }
        
        fundamental_data = {
            'symbol': symbol,
            'has_fundamentals': False
        }
        
        try:
            for data_type, url in urls.items():
                await self._rate_limiter.acquire()
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        fundamental_data[data_type] = data
                        fundamental_data['has_fundamentals'] = True
                    else:
                        logger.debug(f"No {data_type} data for {symbol}: {response.status}")
                        fundamental_data[data_type] = {}
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {'error': str(e), 'has_fundamentals': False}


class AsyncRateLimiter:
    """Async rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            now = datetime.now()
            
            # Remove old calls outside the window
            cutoff_time = now - timedelta(seconds=self.time_window)
            self.calls = [call_time for call_time in self.calls 
                         if call_time > cutoff_time]
            
            # If at limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = (self.calls[0] + timedelta(seconds=self.time_window) - now).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            # Add current call
            self.calls.append(now)
