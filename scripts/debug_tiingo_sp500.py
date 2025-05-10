#!/usr/bin/env python
"""
Tiingo API Debugger for S&P 500 Fundamental and News Data

This script tests the Tiingo API's ability to fetch fundamental and news data
for S&P 500 stocks to identify what data is available.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from loguru import logger
import os
from dotenv import load_dotenv
import aiohttp
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.enhanced_tiingo_fetcher import EnhancedTiingoDataFetcher


class TiingoDebugger:
    """Debug tool for testing Tiingo API capabilities."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Tiingo configuration
        self.tiingo_config = {
            'api_key': os.getenv('TIINGO_API_KEY'),
            'rate_limit': 1000,  # Lower limit for testing
            'rate_limit_window': 3600,
            'timeout': 30,
            'max_retries': 3
        }
        
        if not self.tiingo_config['api_key']:
            raise ValueError("TIINGO_API_KEY environment variable not set")
        
        self.fetcher = EnhancedTiingoDataFetcher(self.tiingo_config)
        
        # Load universe data
        self.sp500_symbols = self.load_sp500_symbols()
        self.dow30_symbols = self.load_dow30_symbols()
        
        # Results storage
        self.results = {
            'fundamental_success': {},
            'fundamental_errors': {},
            'news_success': {},
            'news_errors': {},
            'summary': {}
        }
    
    def load_sp500_symbols(self):
        """Load S&P 500 symbols from file."""
        symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                data = json.load(f)
                return data['symbols']
        else:
            logger.error("S&P 500 symbols file not found")
            return []
    
    def load_dow30_symbols(self):
        """Load Dow 30 symbols from file."""
        symbols_file = project_root / 'data' / 'universe' / 'dow30_symbols.json'
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                data = json.load(f)
                return set(data['symbols'])
        else:
            logger.warning("Dow 30 symbols file not found")
            return set()
    
    async def test_fundamental_data(self, symbols):
        """Test fundamental data fetching for given symbols."""
        logger.info(f"Testing fundamental data for {len(symbols)} symbols...")
        
        # Test direct API access for a few samples
        sample_symbols = random.sample(symbols, min(5, len(symbols)))
        
        for symbol in sample_symbols:
            try:
                # Test individual endpoints
                results = await self._test_fundamental_endpoints(symbol)
                
                if any(results.values()):
                    self.results['fundamental_success'][symbol] = results
                    logger.info(f"✓ {symbol}: Fundamental data available")
                else:
                    self.results['fundamental_errors'][symbol] = "No data available"
                    logger.warning(f"✗ {symbol}: No fundamental data")
                
            except Exception as e:
                self.results['fundamental_errors'][symbol] = str(e)
                logger.error(f"✗ {symbol}: Error - {e}")
        
        # Also test batch fetch
        try:
            batch_results = await self.fetcher.fetch_fundamental_data(sample_symbols, force_all=True)
            logger.info(f"Batch fetch completed for {len(batch_results)} symbols")
            
            # Analyze batch results
            for symbol, data in batch_results.items():
                if data.get('has_fundamentals', False):
                    self.results['fundamental_success'][symbol] = {
                        'has_meta': bool(data.get('meta')),
                        'has_statements': bool(data.get('statements')),
                        'has_daily': bool(data.get('daily_metrics'))
                    }
                else:
                    self.results['fundamental_errors'][symbol] = data.get('error', 'No fundamentals')
        
        except Exception as e:
            logger.error(f"Batch fundamental fetch error: {e}")
    
    async def _test_fundamental_endpoints(self, symbol):
        """Test individual fundamental endpoints for a symbol."""
        session = await self.fetcher._get_session()
        
        endpoints = {
            'meta': f"{self.fetcher.base_url}/tiingo/fundamentals/{symbol}/meta",
            'statements': f"{self.fetcher.base_url}/tiingo/fundamentals/{symbol}/statements",
            'daily': f"{self.fetcher.base_url}/tiingo/fundamentals/{symbol}/daily"
        }
        
        results = {}
        
        for endpoint_name, url in endpoints.items():
            try:
                await self.fetcher._rate_limiter.acquire()
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[endpoint_name] = bool(data)
                    else:
                        results[endpoint_name] = False
                        
            except Exception as e:
                logger.debug(f"Error fetching {endpoint_name} for {symbol}: {e}")
                results[endpoint_name] = False
        
        return results
    
    async def test_news_data(self, symbols):
        """Test news data fetching for given symbols."""
        logger.info(f"Testing news data for {len(symbols)} symbols...")
        
        # Test samples
        sample_symbols = random.sample(symbols, min(10, len(symbols)))
        
        # Test individual symbols
        for symbol in sample_symbols[:5]:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                news = await self.fetcher.fetch_news(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    limit=10
                )
                
                if news:
                    self.results['news_success'][symbol] = {
                        'count': len(news),
                        'has_sentiment': any(article.get('sentiment') for article in news),
                        'sample_title': news[0].get('title', '') if news else ''
                    }
                    logger.info(f"✓ {symbol}: {len(news)} news articles found")
                else:
                    self.results['news_errors'][symbol] = "No news found"
                    logger.warning(f"✗ {symbol}: No news data")
                    
            except Exception as e:
                self.results['news_errors'][symbol] = str(e)
                logger.error(f"✗ {symbol}: News error - {e}")
        
        # Test batch news fetch
        try:
            batch_news = await self.fetcher.fetch_news(
                symbols=sample_symbols,
                start_date=datetime.now() - timedelta(days=7),
                limit=50
            )
            
            if batch_news:
                logger.info(f"Batch news fetch returned {len(batch_news)} articles")
                
                # Analyze which symbols have news
                symbol_news_count = {}
                for article in batch_news:
                    symbols_in_article = article.get('symbols', [])
                    for sym in symbols_in_article:
                        symbol_news_count[sym] = symbol_news_count.get(sym, 0) + 1
                
                for sym, count in symbol_news_count.items():
                    if sym in sample_symbols:
                        self.results['news_success'][sym] = {'count': count, 'batch_test': True}
                        
        except Exception as e:
            logger.error(f"Batch news fetch error: {e}")
    
    async def test_general_news(self):
        """Test general market news without specific symbols."""
        logger.info("Testing general market news...")
        
        try:
            news = await self.fetcher.fetch_news(
                start_date=datetime.now() - timedelta(days=1),
                limit=20
            )
            
            if news:
                self.results['summary']['general_news_count'] = len(news)
                self.results['summary']['general_news_works'] = True
                logger.info(f"✓ General news: {len(news)} articles found")
            else:
                self.results['summary']['general_news_works'] = False
                logger.warning("✗ No general news found")
                
        except Exception as e:
            self.results['summary']['general_news_error'] = str(e)
            logger.error(f"General news error: {e}")
    
    def generate_report(self):
        """Generate a comprehensive debug report."""
        report = []
        report.append("=== Tiingo API Debug Report ===")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total S&P 500 symbols: {len(self.sp500_symbols)}")
        report.append(f"Dow 30 symbols identified: {len(self.dow30_symbols)}")
        report.append("")
        
        # Fundamental data summary
        report.append("=== Fundamental Data Summary ===")
        dow30_success = sum(1 for sym in self.results['fundamental_success'] 
                           if sym in self.dow30_symbols)
        non_dow30_success = len(self.results['fundamental_success']) - dow30_success
        
        report.append(f"Successful fetches: {len(self.results['fundamental_success'])}")
        report.append(f"  - Dow 30: {dow30_success}")
        report.append(f"  - Non-Dow 30: {non_dow30_success}")
        report.append(f"Failed fetches: {len(self.results['fundamental_errors'])}")
        
        if self.results['fundamental_success']:
            report.append("\nSample successful fundamental fetches:")
            for symbol, data in list(self.results['fundamental_success'].items())[:5]:
                is_dow30 = " (Dow 30)" if symbol in self.dow30_symbols else ""
                report.append(f"  {symbol}{is_dow30}: {data}")
        
        # News data summary
        report.append("\n=== News Data Summary ===")
        report.append(f"Successful news fetches: {len(self.results['news_success'])}")
        report.append(f"Failed news fetches: {len(self.results['news_errors'])}")
        
        if self.results['news_success']:
            report.append("\nSample successful news fetches:")
            for symbol, data in list(self.results['news_success'].items())[:5]:
                report.append(f"  {symbol}: {data.get('count', 0)} articles")
        
        # General summary
        report.append("\n=== General Summary ===")
        report.append(f"General news works: {self.results['summary'].get('general_news_works', False)}")
        if 'general_news_count' in self.results['summary']:
            report.append(f"General news articles: {self.results['summary']['general_news_count']}")
        
        # Conclusions
        report.append("\n=== Conclusions ===")
        report.append("1. Fundamental Data:")
        if dow30_success > 0 and non_dow30_success == 0:
            report.append("   - Only available for Dow 30 companies")
        elif non_dow30_success > 0:
            report.append("   - Available for some non-Dow 30 companies (unexpected)")
        else:
            report.append("   - Not available for tested symbols")
        
        report.append("\n2. News Data:")
        if len(self.results['news_success']) > 0:
            report.append("   - Available for S&P 500 symbols")
            report.append("   - Both individual and batch fetching work")
        else:
            report.append("   - Issues with news data fetching")
        
        return "\n".join(report)
    
    async def run_debug_session(self):
        """Run complete debug session."""
        try:
            logger.info("Starting Tiingo API debug session...")
            
            # Test a subset of S&P 500 symbols
            test_symbols = random.sample(self.sp500_symbols, min(20, len(self.sp500_symbols)))
            
            # Ensure we include some Dow 30 symbols
            dow30_test = [sym for sym in test_symbols if sym in self.dow30_symbols]
            if len(dow30_test) < 5:
                additional_dow30 = random.sample(list(self.dow30_symbols), 
                                               min(5, len(self.dow30_symbols)))
                test_symbols.extend(additional_dow30)
                test_symbols = list(set(test_symbols))
            
            logger.info(f"Testing {len(test_symbols)} symbols "
                       f"(including {len([s for s in test_symbols if s in self.dow30_symbols])} Dow 30)")
            
            # Run tests
            await self.test_fundamental_data(test_symbols)
            await self.test_news_data(test_symbols)
            await self.test_general_news()
            
            # Generate report
            report = self.generate_report()
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = project_root / 'logs' / f'tiingo_debug_report_{timestamp}.txt'
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Also save detailed results as JSON
            results_file = project_root / 'logs' / f'tiingo_debug_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Report saved to: {report_file}")
            logger.info(f"Detailed results saved to: {results_file}")
            
            # Print report to console
            print("\n" + report)
            
        finally:
            # Ensure session is closed
            await self.fetcher.close()


async def main():
    """Main function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    # Create and run debugger
    debugger = TiingoDebugger()
    await debugger.run_debug_session()


if __name__ == "__main__":
    asyncio.run(main())
