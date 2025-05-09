#!/usr/bin/env python
"""
Run Advanced Momentum Strategy - Enhanced Analysis Version

This script analyzes all stocks and outputs a comprehensive CSV with scores
and trading recommendations including BUY/HOLD/SELL based on technical analysis.
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

from src.data.data_fetcher_optimized import OptimizedDataFetcher
from src.strategies.advanced_momentum import AdvancedMomentumStrategy
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.monitoring.performance_tracker import PerformanceTracker


async def run_advanced_momentum(strategy_type='intermediate', universe='default'):
    """
    Run the advanced momentum strategy with comprehensive stock analysis
    
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
    
    # Calculate indicators and scores for ALL stocks
    all_stocks_analysis = {}
    for symbol, data in market_data.items():
        if len(data) < 252:  # Need at least 1 year of data
            continue
        
        try:
            indicators = strategy.calculate_indicators(data)
            buy_signal, signal_strength = strategy.check_buy_criteria(indicators, symbol)
            
            # Calculate additional metrics for better decision making
            # Momentum decay (comparing short vs long term momentum)
            momentum_1m = indicators.get('return_1m', 0)
            momentum_3m = indicators.get('return_3m', 0)
            momentum_decay = momentum_1m - momentum_3m if momentum_3m != 0 else momentum_1m
            
            # Moving average convergence
            ma_convergence = (indicators['ma_20'] - indicators['ma_50']) / indicators['ma_50'] * 100
            
            # Distance from 52-week high (already calculated)
            distance_from_high = indicators.get('pct_from_52w_high', 0)
            
            # Volume trend (comparing current volume to longer-term average)
            volume_trend = indicators.get('volume_ratio', 1.0)
            
            # Determine current action (BUY/HOLD/SELL)
            action = determine_action(indicators, signal_strength, strategy_config)
            
            all_stocks_analysis[symbol] = {
                'score': signal_strength,
                'indicators': indicators,
                'buy_signal': buy_signal,
                'momentum_decay': momentum_decay,
                'ma_convergence': ma_convergence,
                'distance_from_high': distance_from_high,
                'volume_trend': volume_trend,
                'action': action
            }
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            continue
    
    # Sort all stocks by score
    sorted_stocks = sorted(all_stocks_analysis.items(), 
                          key=lambda x: x[1]['score'], 
                          reverse=True)
    
    # Create comprehensive report
    report_data = []
    for rank, (symbol, analysis) in enumerate(sorted_stocks, 1):
        indicators = analysis['indicators']
        action = analysis['action']
        
        # Only allocate capital to top N BUY recommendations
        if action == 'BUY' and rank <= strategy_config['top_n']:
            position_size = portfolio.cash * strategy_config['max_position_size']
            current_price = indicators['close']
            shares = int(position_size / current_price)
            amount = shares * current_price
            if strategy_type == 'intermediate':
                holding_period = f"{strategy_config['min_holding_period']}-{strategy_config['max_holding_period']} days"
            else:
                holding_period = f"{strategy_config['min_holding_period']}-{strategy_config['max_holding_period']} days"
        else:
            amount = 0
            shares = 0
            holding_period = 'N/A'
        
        report_data.append({
            'Rank': rank,
            'Symbol': symbol,
            'Action': action,
            'Signal Strength': f"{analysis['score']:.3f}",
            'Buy Criteria Met': 'Yes' if analysis['buy_signal'] else 'No',
            'Current Price': f"${indicators['close']:.2f}",
            '1M Return': f"{indicators.get('return_1m', 0):.1%}",
            '3M Return': f"{indicators.get('return_3m', 0):.1%}",
            'RSI': f"{indicators['rsi']:.1f}",
            'Volume Ratio': f"{indicators['volume_ratio']:.1f}x",
            'MACD': 'Bullish' if indicators['macd'] > indicators['macd_signal'] else 'Bearish',
            'Above MA20': 'Yes' if indicators['close'] > indicators['ma_20'] else 'No',
            'Above MA50': 'Yes' if indicators['close'] > indicators['ma_50'] else 'No',
            'Above MA200': 'Yes' if indicators['close'] > indicators['ma_200'] else 'No',
            '52W High %': f"{indicators.get('pct_from_52w_high', 0):.1%}",
            'ADX': f"{indicators.get('adx', 0):.1f}",
            'Momentum Decay': f"{analysis['momentum_decay']:.1%}",
            'MA Convergence': f"{analysis['ma_convergence']:.1%}",
            'Volume Trend': f"{analysis['volume_trend']:.1f}x",
            'BUY/SELL': action,
            '$ Amount': f"${amount:,.2f}" if amount > 0 else "-",
            'Shares Purchased': shares if shares > 0 else "-",
            'Retaining Period': holding_period
        })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_file = f"full_momentum_analysis_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"\nComprehensive analysis saved to: {report_file}")
        
        # Display summary
        logger.info("\n=== Analysis Summary ===")
        logger.info(f"Total stocks analyzed: {len(report_data)}")
        logger.info(f"BUY recommendations: {len([r for r in report_data if r['BUY/SELL'] == 'BUY'])}")
        logger.info(f"SELL recommendations: {len([r for r in report_data if r['BUY/SELL'] == 'SELL'])}")
        logger.info(f"HOLD recommendations: {len([r for r in report_data if r['BUY/SELL'] == 'HOLD'])}")
        
        # Show top 10 stocks
        logger.info("\nTop 10 Stocks by Signal Strength:")
        for i in range(min(10, len(report_data))):
            row = report_data[i]
            logger.info(f"{row['Rank']}. {row['Symbol']}: Score={row['Signal Strength']}, Action={row['Action']}")
        
        # Show SELL recommendations
        sell_stocks = [r for r in report_data if r['BUY/SELL'] == 'SELL']
        if sell_stocks:
            logger.info("\nSELL Recommendations:")
            for row in sell_stocks[:10]:  # Show up to 10 SELL recommendations
                logger.info(f"{row['Symbol']}: RSI={row['RSI']}, 52W High={row['52W High %']}, Momentum Decay={row['Momentum Decay']}")


def determine_action(indicators, signal_strength, config):
    """
    Determine the action (BUY/HOLD/SELL) based on technical indicators
    
    Args:
        indicators: Dictionary of calculated indicators
        signal_strength: Overall signal strength score
        config: Strategy configuration
        
    Returns:
        Action string: 'BUY', 'HOLD', or 'SELL'
    """
    # SELL criteria
    sell_conditions = []
    
    # 1. Extreme overbought (RSI > 75)
    if indicators['rsi'] > 75:
        sell_conditions.append('extreme_rsi')
    
    # 2. Significant decline from 52-week high (more than 20%)
    if indicators.get('pct_from_52w_high', 0) < -0.20:
        sell_conditions.append('far_from_high')
    
    # 3. MACD bearish crossover with negative histogram
    if indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
        sell_conditions.append('macd_bearish')
    
    # 4. Price below key moving averages
    below_ma_count = 0
    if indicators['close'] < indicators['ma_20']:
        below_ma_count += 1
    if indicators['close'] < indicators['ma_50']:
        below_ma_count += 1
    if indicators['close'] < indicators['ma_200']:
        below_ma_count += 1
    
    if below_ma_count >= 2:
        sell_conditions.append('below_ma')
    
    # 5. Weak ADX (trend strength) and negative momentum
    if indicators.get('adx', 25) < 25 and indicators.get('return_1m', 0) < 0:
        sell_conditions.append('weak_trend')
    
    # 6. Volume divergence (price up but volume significantly down)
    if indicators.get('return_1m', 0) > 0 and indicators.get('volume_ratio', 1) < 0.5:
        sell_conditions.append('volume_divergence')
    
    # If multiple sell conditions are met, recommend SELL
    if len(sell_conditions) >= 2:
        return 'SELL'
    
    # BUY criteria
    buy_conditions = []
    
    # 1. Strong signal strength
    if signal_strength >= 0.5:
        buy_conditions.append('strong_signal')
    
    # 2. RSI in momentum range
    if config['rsi_min'] <= indicators['rsi'] <= config['rsi_max']:
        buy_conditions.append('good_rsi')
    
    # 3. Price above key moving averages
    above_ma_count = 0
    if indicators['close'] > indicators['ma_20']:
        above_ma_count += 1
    if indicators['close'] > indicators['ma_50']:
        above_ma_count += 1
    if indicators['close'] > indicators['ma_200']:
        above_ma_count += 1
    
    if above_ma_count >= 2:
        buy_conditions.append('above_ma')
    
    # 4. Positive momentum
    if indicators.get('return_1m', 0) > config['min_price_performance']:
        buy_conditions.append('positive_momentum')
    
    # 5. Volume confirmation
    if indicators.get('volume_ratio', 0) > config['volume_surge_threshold']:
        buy_conditions.append('volume_surge')
    
    # 6. MACD bullish
    if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
        buy_conditions.append('macd_bullish')
    
    # If multiple buy conditions are met, recommend BUY
    if len(buy_conditions) >= 4:
        return 'BUY'
    
    # Default to HOLD
    return 'HOLD'


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced momentum strategy with full stock analysis")
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
    logger.add(f"logs/advanced_momentum_full_{args.timeframe}_{args.universe}.log", level="DEBUG", rotation="10 MB")
    
    await run_advanced_momentum(args.timeframe, args.universe)


if __name__ == "__main__":
    asyncio.run(main())
