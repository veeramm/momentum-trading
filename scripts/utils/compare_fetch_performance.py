#!/usr/bin/env python
"""
Compare Sequential vs Parallel Data Fetching Performance

This script demonstrates the performance improvement of parallel
data fetching over sequential fetching.
"""

import asyncio
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
from loguru import logger
import os
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_fetcher import DataFetcher
from src.data.parallel_data_fetcher import OptimizedDataFetcher


async def test_sequential_fetch(symbols, start_date, end_date, data_config):
    """Test sequential data fetching"""
    logger.info("\n=== Testing Sequential Fetch ===")
    fetcher = DataFetcher(data_config)
    
    start_time = time.time()
    data = await fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='tiingo'
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Sequential fetch completed: {len(data)} symbols in {elapsed:.2f} seconds")
    logger.info(f"Average time per symbol: {elapsed/len(symbols):.2f} seconds")
    
    return elapsed


async def test_parallel_fetch(symbols, start_date, end_date, data_config):
    """Test parallel data fetching"""
    logger.info("\n=== Testing Parallel Fetch ===")
    fetcher = OptimizedDataFetcher(data_config)
    
    start_time = time.time()
    data = await fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='tiingo'
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Parallel fetch completed: {len(data)} symbols in {elapsed:.2f} seconds")
    logger.info(f"Average time per symbol: {elapsed/len(symbols):.2f} seconds")
    
    return elapsed


async def main():
    """Main function"""
    load_dotenv()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Test configuration
    symbols = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B',
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
        'CRM', 'XOM', 'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'CVX', 'WMT',
        'ABT', 'NKE', 'TMO', 'AVGO', 'ACN', 'MRK', 'LLY', 'ORCL', 'DHR',
        'VZ', 'CMCSA', 'PM', 'NEE', 'RTX', 'WFC', 'IBM', 'QCOM', 'BMY',
        'TXN', 'HON', 'COST', 'COP', 'UPS'
    ]  # 50 stocks for testing
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data_config = {
        'sources': {
            'tiingo': {
                'enabled': True,
                'api_key': os.getenv('TIINGO_API_KEY'),
                'cache': {
                    'enabled': False,  # Disable cache for fair comparison
                    'backend': 'memory'
                },
                'rate_limit': 600
            }
        }
    }
    
    logger.info(f"Testing with {len(symbols)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Test sequential fetch
    seq_time = await test_sequential_fetch(symbols, start_date, end_date, data_config)
    
    # Clear cache if enabled
    if data_config['sources']['tiingo']['cache']['enabled']:
        logger.info("Clearing cache between tests...")
    
    # Test parallel fetch
    par_time = await test_parallel_fetch(symbols, start_date, end_date, data_config)
    
    # Performance comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Sequential time: {seq_time:.2f} seconds")
    logger.info(f"Parallel time: {par_time:.2f} seconds")
    logger.info(f"Speed improvement: {seq_time/par_time:.2f}x faster")
    logger.info(f"Time saved: {seq_time - par_time:.2f} seconds ({(seq_time - par_time)/seq_time*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
