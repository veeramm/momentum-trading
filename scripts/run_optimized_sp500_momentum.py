#!/usr/bin/env python
"""
Optimized S&P 500 Momentum Strategy with Parallel Processing

This script runs the momentum strategy on S&P 500 stocks with
parallel data fetching for much faster execution.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd
from loguru import logger
import os
from dotenv import load_dotenv
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.parallel_data_fetcher import OptimizedDataFetcher
from src.strategies.advanced_momentum import AdvancedMomentumStrategy
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.monitoring.performance_tracker import PerformanceTracker


async def run_optimized_sp500_momentum(strategy_type='intermediate'):
    """
    Run momentum strategy with parallel data fetching
    """
    start_time = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # Load strategy configuration
    config_path = project_root / 'config' / 'strategies' / 'advanced_momentum.yaml'
    if not config_path.exists():
        # Use default config
        strategy_config = {
            'timeframe': strategy_type,
            'lookback_period': 252,
            'top_n': 30 if strategy_type == 'intermediate' else 20,
            'rebalance_frequency': 'weekly' if strategy_type == 'intermediate' else 'monthly',
            'min_holding_period': 14 if strategy_type == 'intermediate' else 60,
            'max_holding_period': 56 if strategy_type == 'intermediate' else 180,
            'ma_short': 20,
            'ma_medium': 50,
            'ma_long': 200,
            'min_price_performance': 0.10 if strategy_type == 'intermediate' else 0.15,
            'volume_surge_threshold': 1.5 if strategy_type == 'intermediate' else 1.3,
            'rsi_min': 50 if strategy_type == 'intermediate' else 55,
            'rsi_max': 75 if strategy_type == 'intermediate' else 70,
            'trailing_stop_ma': 20,
            'weekly_ma_exit': 50,
            'max_position_size': 0.033 if strategy_type == 'intermediate' else 0.05,
            'stop_loss_pct': 0.07 if strategy_type == 'intermediate' else 0.10,
            'min_price': 10.0,
            'min_volume': 5000000
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            strategy_config = config['strategies'][f'{strategy_type}_momentum']
    
    # Load S&P 500 symbols
    symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            symbols = data['symbols']
        logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
    else:
        # Use hardcoded list if file doesn't exist
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM']
        logger.warning("S&P 500 symbols file not found, using limited list")
    
    # Initialize data configuration
    data_config = {
        'sources': {
            'tiingo': {
                'enabled': True,
                'api_key': os.getenv('TIINGO_API_KEY'),
                'rate_limit': 600,  # Tiingo free tier limit
                'cache': {
                    'enabled': True,
                    'backend': 'memory',
                    'ttl': 86400  # 24 hours
                }
            }
        }
    }
    
    # Initialize optimized data fetcher
    data_fetcher = OptimizedDataFetcher(data_config)
    
    # Initialize components
    portfolio = Portfolio(initial_capital=1000000)
    risk_manager = RiskManager(strategy_config)
    strategy = AdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager
    )
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # Get enough historical data
    
    logger.info(f"Running {strategy_type} momentum strategy on S&P 500")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial capital: ${portfolio.initial_capital:,.2f}")
    logger.info(f"Total symbols to analyze: {len(symbols)}")
    
    # Progress tracking callback
    def progress_callback(batch_num, total_batches, fetched_count):
        elapsed = time.time() - start_time
        rate = fetched_count / elapsed if elapsed > 0 else 0
        eta = (len(symbols) - fetched_count) / rate if rate > 0 else 0
        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"{fetched_count}/{len(symbols)} symbols fetched "
            f"({rate:.1f} symbols/sec, ETA: {eta:.0f}s)"
        )
    
    # Fetch data in parallel
    logger.info("Starting parallel data fetch...")
    fetch_start = time.time()
    
    market_data = await data_fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='tiingo'
    )
    
    fetch_time = time.time() - fetch_start
    logger.info(f"Data fetch completed in {fetch_time:.1f} seconds")
    logger.info(f"Successfully fetched: {len(market_data)}/{len(symbols)} symbols")
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = strategy.generate_signals(market_data)
    
    # Display results
    buy_signals = [s for s in signals if s.direction == 'long']
    sell_signals = [s for s in signals if s.direction == 'neutral']
    
    logger.info(f"\nGenerated {len(signals)} signals:")
    
    if buy_signals:
        logger.info(f"\nBUY signals ({len(buy_signals)}):")
        buy_signals.sort(key=lambda x: x.strength, reverse=True)
        
        for i, signal in enumerate(buy_signals[:10]):
            indicators = signal.metadata.get('indicators', {})
            logger.info(
                f"  {i+1}. {signal.symbol}: "
                f"Score={signal.strength:.2f}, "
                f"1M Return={indicators.get('return_1m', 0):.1%}, "
                f"RSI={indicators.get('rsi', 0):.1f}, "
                f"Volume Ratio={indicators.get('volume_ratio', 0):.1f}x, "
                f"52W High={indicators.get('pct_from_52w_high', 0):.1%}"
            )
    
    if sell_signals:
        logger.info(f"\nSELL signals ({len(sell_signals)}):")
        for signal in sell_signals:
            logger.info(
                f"  {signal.symbol}: "
                f"Reason={signal.metadata.get('reason', 'unknown')}, "
                f"Days Held={signal.metadata.get('days_held', 0)}"
            )
    
    # Create analysis report
    analysis_data = []
    for signal in buy_signals:
        indicators = signal.metadata.get('indicators', {})
        analysis_data.append({
            'Symbol': signal.symbol,
            'Signal Strength': signal.strength,
            '1M Return': indicators.get('return_1m', 0),
            '3M Return': indicators.get('return_3m', 0),
            'RSI': indicators.get('rsi', 0),
            'Volume Ratio': indicators.get('volume_ratio', 0),
            'Above MA20': indicators.get('close', 0) > indicators.get('ma_20', 0),
            'Above MA50': indicators.get('close', 0) > indicators.get('ma_50', 0),
            'Above MA200': indicators.get('close', 0) > indicators.get('ma_200', 0),
            'MACD Signal': indicators.get('macd_histogram', 0) > 0,
            '52W High %': indicators.get('pct_from_52w_high', 0),
            'ADX': indicators.get('adx', 0)
        })
    
    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df = analysis_df.sort_values('Signal Strength', ascending=False)
        
        # Save to CSV
        report_file = f"optimized_sp500_momentum_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analysis_df.to_csv(report_file, index=False)
        logger.info(f"\nDetailed analysis saved to: {report_file}")
    
    # Performance summary
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.1f} seconds")
    logger.info(f"Data fetch time: {fetch_time:.1f} seconds ({fetch_time/total_time*100:.1f}%)")
    logger.info(f"Analysis time: {total_time-fetch_time:.1f} seconds ({(total_time-fetch_time)/total_time*100:.1f}%)")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized S&P 500 momentum strategy")
    parser.add_argument(
        "--timeframe", 
        choices=['intermediate', 'long_term'],
        default='intermediate',
        help="Strategy timeframe"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(f"logs/optimized_sp500_momentum_{args.timeframe}.log", level="DEBUG", rotation="10 MB")
    
    await run_optimized_sp500_momentum(args.timeframe)


if __name__ == "__main__":
    asyncio.run(main())
