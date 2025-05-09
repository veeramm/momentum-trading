"""
Optimized Data Fetcher Module with Parallel Processing

This module handles data retrieval from various sources with
parallel processing and progress tracking.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import aiohttp
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm.asyncio import tqdm
from ..utils.cache import DataCache


class OptimizedDataFetcher:
    """
    Handles data retrieval with parallel processing and progress tracking.
    """
    
    def __init__(self, config: dict):
        """
        Initialize data fetcher with configuration.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.cache = DataCache(config.get('cache', {}))
        self.sources = self._initialize_sources(config.get('sources', {}))
        
        # Configure parallel processing
        self.max_concurrent_requests = config.get('max_concurrent_requests', 10)
        self.batch_size = config.get('batch_size', 50)
        self.timeout = config.get('timeout', 30)
        
    def _initialize_sources(self, sources_config: dict) -> dict:
        """Initialize data source connections"""
        sources = {}
        
        # Initialize Tiingo
        if sources_config.get('tiingo', {}).get('enabled', False):
            tiingo_config = sources_config.get('tiingo', {})
            if tiingo_config.get('api_key'):
                sources['tiingo'] = TiingoSourceOptimized(tiingo_config)
        
        # Initialize yfinance (no API key needed)
        if sources_config.get('yfinance', {}).get('enabled', True):
            sources['yfinance'] = YFinanceSourceOptimized(sources_config.get('yfinance', {}))
            
        logger.info(f"Initialized data sources: {list(sources.keys())}")
        return sources
    
    async def fetch_price_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = '1d',
        source: str = 'auto',
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for multiple symbols with parallel processing.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1h, etc.)
            source: Data source to use or 'auto'
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of DataFrames by symbol
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Check cache first
        cached_data = {}
        uncached_symbols = []
        
        logger.info(f"Checking cache for {len(symbols)} symbols...")
        
        for symbol in symbols:
            cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
            cached = self.cache.get(cache_key)
            
            if cached is not None:
                cached_data[symbol] = cached
            else:
                uncached_symbols.append(symbol)
        
        logger.info(f"Found {len(cached_data)} cached symbols, need to fetch {len(uncached_symbols)}")
        
        # Fetch uncached data with parallel processing
        if uncached_symbols:
            # Determine which source to use
            if source == 'auto':
                source_name = self._select_best_source(interval)
            else:
                source_name = source
                
            if source_name not in self.sources:
                raise ValueError(f"Data source '{source_name}' not available")
                
            # Fetch data from source
            data_source = self.sources[source_name]
            
            # Split into batches for better rate limit handling
            batches = [uncached_symbols[i:i+self.batch_size] 
                      for i in range(0, len(uncached_symbols), self.batch_size)]
            
            all_new_data = {}
            
            # Process batches with progress bar
            if show_progress:
                pbar = tqdm(total=len(uncached_symbols), desc="Fetching data", unit="symbols")
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} symbols)")
                
                # Fetch batch with parallel processing
                batch_data = await data_source.fetch_batch_parallel(
                    batch, start_date, end_date, interval,
                    max_concurrent=self.max_concurrent_requests
                )
                
                # Cache the new data
                for symbol, data in batch_data.items():
                    cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
                    self.cache.set(cache_key, data)
                    all_new_data[symbol] = data
                
                if show_progress:
                    pbar.update(len(batch))
                
                # Small delay between batches to respect rate limits
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(1)
            
            if show_progress:
                pbar.close()
            
            # Combine with cached data
            cached_data.update(all_new_data)
            
        return cached_data
    
    def _select_best_source(self, interval: str) -> str:
        """Select the best data source based on interval and availability"""
        # Prefer Tiingo if available
        if 'tiingo' in self.sources:
            return 'tiingo'
            
        # For daily data, prefer yfinance, then alphavantage
        if interval == '1d':
            if 'yfinance' in self.sources:
                return 'yfinance'
            elif 'alphavantage' in self.sources:
                return 'alphavantage'
                
        # Return first available source
        return list(self.sources.keys())[0]


class BaseDataSourceOptimized:
    """Base class for optimized data sources"""
    
    def __init__(self, config: dict):
        self.config = config
        self.rate_limiter = AsyncRateLimiter(
            max_calls=config.get('rate_limit', 60),
            time_window=config.get('rate_limit_window', 60)
        )
        
    async def fetch_batch_parallel(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str,
        max_concurrent: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel"""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all symbols
        tasks = []
        for symbol in symbols:
            task = self._fetch_with_semaphore(
                semaphore, symbol, start_date, end_date, interval
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
            elif result is not None:
                data[symbol] = result
        
        return data
    
    async def _fetch_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch single symbol with semaphore control"""
        async with semaphore:
            await self.rate_limiter.acquire()
            return await self._fetch_single(symbol, start_date, end_date, interval)
    
    async def _fetch_single(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol - to be implemented by subclasses"""
        raise NotImplementedError


class TiingoSourceOptimized(BaseDataSourceOptimized):
    """Optimized Tiingo data source implementation"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Tiingo API key required")
        
        # Initialize Tiingo client
        from tiingo import TiingoClient
        self.client = TiingoClient({
            'session': True,
            'api_key': self.api_key
        })
        
        self.base_url = config.get('base_url', 'https://api.tiingo.com')
        self.use_iex = config.get('use_iex', False)
        
    async def _fetch_single(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol from Tiingo"""
        try:
            if interval == '1d':
                # Use async wrapper for Tiingo call
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.get_dataframe(
                        symbol,
                        startDate=start_date.strftime('%Y-%m-%d'),
                        endDate=end_date.strftime('%Y-%m-%d')
                    )
                )
                
                if not df.empty:
                    # Standardize column names and handle duplicates
                    df = self._standardize_columns(df)
                    return df
                else:
                    logger.warning(f"No data found for {symbol}")
                    return None
            else:
                logger.warning(f"Intraday data not available for free tier: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Tiingo column names to match our format"""
        # Set date as index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        
        # Remove duplicate columns if they exist
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Standardize column names
        column_mapping = {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'adjOpen': 'adj_open',
            'adjHigh': 'adj_high',
            'adjLow': 'adj_low',
            'adjClose': 'adj_close',
            'adjVolume': 'adj_volume'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Select standard columns
        standard_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in standard_columns if col in df.columns]
        
        return df[available_columns]


class YFinanceSourceOptimized(BaseDataSourceOptimized):
    """Optimized yfinance data source implementation"""
    
    async def _fetch_single(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol from yfinance"""
        try:
            # Use async wrapper for yfinance call
            ticker = yf.Ticker(symbol)
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
            )
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df.index.name = 'date'
            
            if not df.empty:
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None


class AsyncRateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
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
