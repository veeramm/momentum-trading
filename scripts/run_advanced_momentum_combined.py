#!/usr/bin/env python
"""
Run Advanced Momentum Strategy - Combined Timeframe Analysis

This script analyzes all stocks for both intermediate and long-term timeframes
and outputs comprehensive CSVs with scores and trading recommendations.
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


async def run_advanced_momentum_analysis(timeframe, universe, market_data, portfolio):
    """
    Run the advanced momentum strategy for a specific timeframe
    
    Args:
        timeframe: 'intermediate' or 'long_term'
        universe: 'default', 'sp500', or 'custom'
        market_data: Pre-fetched market data
        portfolio: Portfolio instance
    """
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
    strategy_config = config['strategies'][f'{timeframe}_momentum']
    
    # Initialize strategy components
    risk_config = config.get('risk_limits', {})
    risk_manager = RiskManager(risk_config)
    
    # Initialize strategy
    strategy = AdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager
    )
    
    logger.info(f"\nAnalyzing {timeframe} timeframe...")
    
    # Generate signals
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
            momentum_1m = indicators.get('return_1m', 0)
            momentum_3m = indicators.get('return_3m', 0)
            momentum_decay = momentum_1m - momentum_3m if momentum_3m != 0 else momentum_1m
            
            ma_convergence = (indicators['ma_20'] - indicators['ma_50']) / indicators['ma_50'] * 100
            distance_from_high = indicators.get('pct_from_52w_high', 0)
            volume_trend = indicators.get('volume_ratio', 1.0)
            
            # Determine current action (BUY/HOLD/SELL)
            action = determine_action(indicators, signal_strength, strategy_config, buy_signal)
            
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
            holding_period = f"{strategy_config['min_holding_period']}-{strategy_config['max_holding_period']} days"
        else:
            amount = 0
            shares = 0
            holding_period = 'N/A'
        
        report_data.append({
            'Rank': rank,
            'Symbol': symbol,
            'Timeframe': timeframe.replace('_', ' ').title(),
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
    
    return report_data


async def run_combined_momentum(universe='default'):
    """
    Run the advanced momentum strategy for both timeframes
    
    Args:
        universe: 'default', 'sp500', or 'custom'
    """
    # Load environment variables
    load_dotenv()
    
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
    
    # Initialize components
    data_fetcher = OptimizedDataFetcher(data_config)
    portfolio = Portfolio(initial_capital=100000)
    
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
        symbols = []  # Would load from config
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
    
    logger.info(f"Running combined momentum analysis")
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
    
    # Run analysis for both timeframes
    intermediate_results = await run_advanced_momentum_analysis('intermediate', universe, market_data, portfolio)
    long_term_results = await run_advanced_momentum_analysis('long_term', universe, market_data, portfolio)
    
    # Create data directory if it doesn't exist
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Save individual timeframe results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save intermediate results
    intermediate_df = pd.DataFrame(intermediate_results)
    intermediate_file = data_dir / f"momentum_analysis_intermediate_{timestamp}.csv"
    intermediate_df.to_csv(intermediate_file, index=False)
    logger.info(f"\nIntermediate analysis saved to: {intermediate_file}")
    
    # Save long-term results
    long_term_df = pd.DataFrame(long_term_results)
    long_term_file = data_dir / f"momentum_analysis_long_term_{timestamp}.csv"
    long_term_df.to_csv(long_term_file, index=False)
    logger.info(f"Long-term analysis saved to: {long_term_file}")
    
    # Create combined analysis
    combined_results = intermediate_results + long_term_results
    combined_df = pd.DataFrame(combined_results)
    combined_file = data_dir / f"momentum_analysis_combined_{timestamp}.csv"
    combined_df.to_csv(combined_file, index=False)
    logger.info(f"Combined analysis saved to: {combined_file}")
    
    # Create a pivot comparison of both timeframes
    comparison_data = []
    for symbol in set(intermediate_df['Symbol']):
        int_row = intermediate_df[intermediate_df['Symbol'] == symbol].iloc[0]
        lt_row = long_term_df[long_term_df['Symbol'] == symbol].iloc[0]
        
        comparison_data.append({
            'Symbol': symbol,
            'Intermediate Action': int_row['Action'],
            'Intermediate Score': int_row['Signal Strength'],
            'Long-term Action': lt_row['Action'],
            'Long-term Score': lt_row['Signal Strength'],
            'Consensus': 'BUY' if int_row['Action'] == 'BUY' and lt_row['Action'] == 'BUY' else
                        'SELL' if int_row['Action'] == 'SELL' and lt_row['Action'] == 'SELL' else 'HOLD',
            'Current Price': int_row['Current Price'],
            'RSI': int_row['RSI'],
            '1M Return': int_row['1M Return'],
            '52W High %': int_row['52W High %']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Symbol')
    comparison_file = data_dir / f"momentum_analysis_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Timeframe comparison saved to: {comparison_file}")
    
    # Display summary
    logger.info("\n=== Combined Analysis Summary ===")
    
    # Intermediate summary
    int_buy = len([r for r in intermediate_results if r['BUY/SELL'] == 'BUY'])
    int_sell = len([r for r in intermediate_results if r['BUY/SELL'] == 'SELL'])
    int_hold = len([r for r in intermediate_results if r['BUY/SELL'] == 'HOLD'])
    
    logger.info(f"\nIntermediate Timeframe:")
    logger.info(f"BUY: {int_buy}, SELL: {int_sell}, HOLD: {int_hold}")
    
    # Long-term summary
    lt_buy = len([r for r in long_term_results if r['BUY/SELL'] == 'BUY'])
    lt_sell = len([r for r in long_term_results if r['BUY/SELL'] == 'SELL'])
    lt_hold = len([r for r in long_term_results if r['BUY/SELL'] == 'HOLD'])
    
    logger.info(f"\nLong-term Timeframe:")
    logger.info(f"BUY: {lt_buy}, SELL: {lt_sell}, HOLD: {lt_hold}")
    
    # Consensus summary
    consensus_buy = len([r for r in comparison_data if r['Consensus'] == 'BUY'])
    consensus_sell = len([r for r in comparison_data if r['Consensus'] == 'SELL'])
    consensus_hold = len([r for r in comparison_data if r['Consensus'] == 'HOLD'])
    
    logger.info(f"\nConsensus (Both Timeframes Agree):")
    logger.info(f"BUY: {consensus_buy}, SELL: {consensus_sell}, HOLD: {consensus_hold}")
    
    # Show top consensus BUY stocks
    consensus_buys = comparison_df[comparison_df['Consensus'] == 'BUY'].head(10)
    if not consensus_buys.empty:
        logger.info("\nTop Consensus BUY Recommendations:")
        for _, row in consensus_buys.iterrows():
            logger.info(f"{row['Symbol']}: Int Score={row['Intermediate Score']}, LT Score={row['Long-term Score']}")


def determine_action(indicators, signal_strength, config, buy_criteria_met):
    """
    Determine the action (BUY/HOLD/SELL) based on technical indicators
    
    Args:
        indicators: Dictionary of calculated indicators
        signal_strength: Overall signal strength score
        config: Strategy configuration
        buy_criteria_met: Boolean indicating if strategy buy criteria are met
        
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
    
    # If multiple buy conditions are met AND strategy buy criteria are met, recommend BUY
    if len(buy_conditions) >= 4 and buy_criteria_met:
        return 'BUY'
    
    # Default to HOLD
    return 'HOLD'


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run combined momentum strategy analysis")
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
    logger.add(f"logs/momentum_analysis_combined_{args.universe}.log", level="DEBUG", rotation="10 MB")
    
    await run_combined_momentum(args.universe)


if __name__ == "__main__":
    asyncio.run(main())
