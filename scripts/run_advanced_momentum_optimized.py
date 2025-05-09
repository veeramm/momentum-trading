#!/usr/bin/env python
"""
Run Advanced Momentum Strategy - Optimized Version

This script uses parallel data fetching with progress tracking
for faster execution.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Update this line to import the optimized version
from src.data.data_fetcher_optimized import OptimizedDataFetcher
from src.strategies.advanced_momentum import AdvancedMomentumStrategy
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.monitoring.performance_tracker import PerformanceTracker


async def run_advanced_momentum(strategy_type='intermediate', universe='default'):
    """
    Run the advanced momentum strategy with optimized data fetching
    
    Args:
        strategy_type: 'intermediate' or 'long_term'
        universe: 'default', 'sp500', or 'custom'
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config_path = project_root / 'config' / 'strategies' / 'advanced_momentum.yaml'
    
    # Create default config if file doesn't exist
    if not config_path.exists():
        default_config = {
            'strategies': {
                'intermediate_momentum': {
                    'enabled': True,
                    'timeframe': 'intermediate',
                    'lookback_period': 252,
                    'top_n': 15,
                    'rebalance_frequency': 'weekly',
                    'min_holding_period': 14,
                    'max_holding_period': 56,
                    'ma_short': 20,
                    'ma_medium': 50,
                    'ma_long': 200,
                    'min_price_performance': 0.10,
                    'volume_surge_threshold': 1.5,
                    'rsi_min': 50,
                    'rsi_max': 75,
                    'trailing_stop_ma': 20,
                    'max_position_size': 0.067,
                    'stop_loss_pct': 0.07,
                    'min_price': 10.0,
                    'min_volume': 1000000
                },
                'long_term_momentum': {
                    'enabled': True,
                    'timeframe': 'long_term',
                    'lookback_period': 252,
                    'top_n': 10,
                    'rebalance_frequency': 'monthly',
                    'min_holding_period': 60,
                    'max_holding_period': 180,
                    'ma_short': 20,
                    'ma_medium': 50,
                    'ma_long': 200,
                    'min_price_performance': 0.15,
                    'volume_surge_threshold': 1.3,
                    'rsi_min': 55,
                    'rsi_max': 70,
                    'weekly_ma_exit': 50,
                    'max_position_size': 0.10,
                    'stop_loss_pct': 0.10,
                    'min_price': 15.0,
                    'min_volume': 2000000
                }
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get strategy-specific configuration
    strategy_config = config['strategies'][f'{strategy_type}_momentum']
    
    # Initialize data configuration with Tiingo - optimized settings
    data_config = {
        'sources': {
            'tiingo': {
                'enabled': True,
                'api_key': os.getenv('TIINGO_API_KEY'),
                'cache': {
                    'enabled': True,
                    'backend': 'memory',
                    'ttl': 86400  # Cache for 24 hours
                },
                'rate_limit': 10000,  # Tiingo hourly limit
                'rate_limit_window': 3600  # 1 hour window
            }
        },
        'max_concurrent_requests': 25,  # Parallel requests
        'batch_size': 50,  # Symbols per batch
        'timeout': 30
    }
    
    # Initialize components - use OptimizedDataFetcher
    data_fetcher = OptimizedDataFetcher(data_config)
    portfolio = Portfolio(initial_capital=100000)
    risk_config = config.get('risk_limits', {})
    risk_manager = RiskManager(risk_config)
    performance_tracker = PerformanceTracker()
    
    # Initialize strategy
    strategy = AdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager
    )
    
    # Define universe
    if universe == 'sp500':
        # Try to load S&P 500 symbols from file
        symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                data = json.load(f)
                symbols = data['symbols']
            logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
        else:
            # Fallback to top S&P 500 stocks
            symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM',
                      'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM',
                      'XOM', 'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'CVX', 'WMT', 'ABT', 'NKE']
            logger.warning("S&P 500 symbols file not found, using top 30 S&P 500 stocks")
    elif universe == 'custom':
        # Load custom universe from config
        symbols = config.get('universe', {}).get('symbols', [])
        if not symbols:
            raise ValueError("No symbols defined in custom universe")
    else:
        # Default universe (top 30 for demo)
        symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM',
            'XOM', 'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'CVX', 'WMT', 'ABT', 'NKE'
        ]
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # Get enough historical data
    
    logger.info(f"Running {strategy_type} momentum strategy")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial capital: ${portfolio.initial_capital:,.2f}")
    logger.info(f"Total symbols: {len(symbols)}")
    
    # Fetch historical data with progress bar
    logger.info(f"Starting optimized data fetch for {len(symbols)} symbols...")
    start_time = datetime.now()
    
    market_data = await data_fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='tiingo',
        show_progress=True
    )
    
    fetch_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Data fetch completed in {fetch_time:.1f} seconds")
    logger.info(f"Successfully fetched data for {len(market_data)} symbols")
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = strategy.generate_signals(market_data)
    
    # Display signals
    logger.info(f"\nGenerated {len(signals)} signals:")
    
    buy_signals = [s for s in signals if s.direction == 'long']
    sell_signals = [s for s in signals if s.direction == 'neutral']
    
    if buy_signals:
        logger.info(f"\nBUY signals ({len(buy_signals)}):")
        for signal in buy_signals:
            indicators = signal.metadata.get('indicators', {})
            logger.info(
                f"  {signal.symbol}: "
                f"Score={signal.strength:.2f}, "
                f"1M Return={indicators.get('return_1m', 0):.1%}, "
                f"RSI={indicators.get('rsi', 0):.1f}, "
                f"Volume Ratio={indicators.get('volume_ratio', 0):.1f}x"
            )
    
    if sell_signals:
        logger.info(f"\nSELL signals ({len(sell_signals)}):")
        for signal in sell_signals:
            logger.info(
                f"  {signal.symbol}: "
                f"Reason={signal.metadata.get('reason', 'unknown')}, "
                f"Days Held={signal.metadata.get('days_held', 0)}"
            )
    
    # Simulate execution (simplified)
    for signal in buy_signals[:strategy_config['top_n']]:
        try:
            position_size = portfolio.cash * strategy_config['max_position_size']
            shares = int(position_size / signal.metadata['indicators']['close'])
            
            if shares > 0:
                portfolio.open_position(
                    symbol=signal.symbol,
                    quantity=shares,
                    price=signal.metadata['indicators']['close'],
                    timestamp=signal.timestamp
                )
        except Exception as e:
            logger.error(f"Error opening position for {signal.symbol}: {e}")
    
    # Display portfolio summary
    if portfolio.positions:
        summary = portfolio.get_position_summary()
        logger.info("\nPortfolio Summary:")
        logger.info(f"Number of positions: {len(portfolio.positions)}")
        logger.info(f"Total value: ${portfolio.total_equity:,.2f}")
        logger.info(f"Cash remaining: ${portfolio.cash:,.2f}")
        
        logger.info("\nTop Holdings:")
        for _, row in summary.head(5).iterrows():
            logger.info(
                f"  {row['symbol']}: "
                f"{row['quantity']} shares @ ${row['current_price']:.2f} "
                f"(${row['market_value']:,.2f})"
            )
    
    # Strategy-specific metrics
    metrics = strategy.get_strategy_specific_metrics()
    logger.info("\nStrategy Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Create a detailed report
    report_data = []
    for symbol, position in portfolio.positions.items():
        if symbol in market_data:
            indicators = strategy.calculate_indicators(market_data[symbol])
            report_data.append({
                'Symbol': symbol,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Return': f"{position.unrealized_pnl_pct:.1%}",
                'RSI': f"{indicators['rsi']:.1f}",
                'Volume Ratio': f"{indicators['volume_ratio']:.1f}x",
                'MACD': indicators['macd'] > indicators['macd_signal'],
                'Above MA20': indicators['close'] > indicators['ma_20'],
                'Above MA50': indicators['close'] > indicators['ma_50'],
                '52W High %': f"{indicators['pct_from_52w_high']:.1%}"
            })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_file = f"momentum_report_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"\nDetailed report saved to: {report_file}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced momentum strategy with optimized data fetching")
    parser.add_argument(
        "--timeframe", 
        choices=['intermediate', 'long_term'],
        default='intermediate',
        help="Strategy timeframe"
    )
    parser.add_argument(
        "--universe",
        choices=['default', 'sp500', 'custom'],
        default='default',
        help="Universe of stocks to analyze"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(f"logs/advanced_momentum_{args.timeframe}_{args.universe}.log", level="DEBUG", rotation="10 MB")
    
    await run_advanced_momentum(args.timeframe, args.universe)


if __name__ == "__main__":
    asyncio.run(main())
