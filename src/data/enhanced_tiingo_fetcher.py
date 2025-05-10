# Add these imports at the top

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
    def __init__(self, config: dict):
        # ... existing initialization code ...
        
        # Load Dow 30 symbols
        self.dow30_symbols = self._load_dow30_symbols()
        logger.info(f"Loaded {len(self.dow30_symbols)} Dow 30 symbols")
        
        # Parallel processing settings
        self.max_workers = config.get('max_workers', 10)
    
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
    
    async def _fetch_fundamental_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch fundamentals for a batch of symbols using parallel processing.
        
        Args:
            symbols: List of symbols in batch
            
        Returns:
            Dictionary of results
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return symbol, await self._fetch_fundamental_single(symbol)
        
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
            else:
                symbol, data = result
                batch_results[symbol] = data
        
        return batch_results
