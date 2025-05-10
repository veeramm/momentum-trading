#!/usr/bin/env python
"""
Run Enhanced Advanced Momentum Strategy

This script runs the enhanced advanced momentum strategy with integrated
technical, fundamental, and sentiment analysis using optimized data fetching.
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

from src.strategies.enhanced_advanced_momentum import EnhancedAdvancedMomentumStrategy
from src.data.enhanced_tiingo_fetcher import EnhancedTiingoDataFetcher
from src.data.data_fetcher_optimized import OptimizedDataFetcher
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager


async def run_enhanced_momentum_analysis(timeframe='intermediate', universe='default'):
    """
    Run the enhanced advanced momentum strategy analysis.
    
    Args:
        timeframe: 'intermediate' or 'long_term'
        universe: 'default', 'sp500', or 'custom'
    """
    # Load environment variables
    load_dotenv()
    
    # Tiingo configuration for enhanced strategy
    tiingo_config = {
        'api_key': os.getenv('TIINGO_API_KEY'),
        'use_iex': True,
        'rate_limit': 10000,
        'rate_limit_window': 3600
    }
    
    # Data configuration for optimized fetching
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
                'rate_limit': 10000,
                'rate_limit_window': 3600
            }
        },
        'max_concurrent_requests': 25,
        'batch_size': 50,
        'timeout': 30
    }
    
    # Strategy configuration
    strategy_config = {
        'timeframe': timeframe,
        'lookback_period': 252,
        'top_n': 20 if timeframe == 'intermediate' else 10,
        'rebalance_frequency': 'weekly' if timeframe == 'intermediate' else 'monthly',
        'ma_short': 20,
        'ma_medium': 50,
        'ma_long': 200,
        'min_price_performance': 0.10 if timeframe == 'intermediate' else 0.15,
        'volume_surge_threshold': 1.5 if timeframe == 'intermediate' else 1.3,
        'rsi_min': 50 if timeframe == 'intermediate' else 55,
        'rsi_max': 75 if timeframe == 'intermediate' else 70,
        'technical_weight': 0.5,
        'fundamental_weight': 0.3,
        'sentiment_weight': 0.2
    }
    
    # Initialize components
    data_fetcher = OptimizedDataFetcher(data_config)
    tiingo_fetcher = EnhancedTiingoDataFetcher(tiingo_config)
    portfolio = Portfolio(initial_capital=100000)
    risk_manager = RiskManager({})
    
    # Initialize enhanced advanced strategy
    strategy = EnhancedAdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager,
        tiingo_fetcher=tiingo_fetcher
    )
    
    # Define universe
    if universe == 'sp500':
        symbols = load_sp500_symbols()
    elif universe == 'custom':
        symbols = []  # Would load from config
    else:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM',
                  'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM',
                  'XOM', 'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'CVX', 'WMT', 'ABT', 'NKE']
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    logger.info(f"Running enhanced advanced momentum analysis for {timeframe} timeframe")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Universe: {len(symbols)} symbols")
    
    # Fetch market data
    logger.info("Fetching market data...")
    market_data = await data_fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='tiingo',
        show_progress=True
    )
    
    logger.info(f"Successfully fetched data for {len(market_data)} symbols")
    
    # Enhance with Tiingo fundamental and sentiment data
    await strategy.enhance_with_tiingo_data(list(market_data.keys()))
    
    # Get comprehensive analysis
    analyses = strategy.get_analysis_for_reporting(market_data)
    
    # Create report data with enhanced metrics
    report_data = []
    for rank, analysis in enumerate(analyses, 1):
        indicators = analysis['indicators']
        
        report_data.append({
            'Rank': rank,
            'Symbol': analysis['symbol'],
            'Timeframe': timeframe.replace('_', ' ').title(),
            'Action': analysis['action'],
            'Comprehensive Score': f"{analysis['comprehensive_score']:.3f}",
            'Technical Score': f"{analysis['technical_score']:.3f}",
            'Fundamental Score': f"{analysis['fundamental_score']:.3f}",
            'Sentiment Score': f"{analysis['sentiment_score']:.3f}",
            'Buy Criteria Met': 'Yes' if analysis['buy_criteria_met'] else 'No',
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
            'MA Convergence': f"{(indicators['ma_20'] - indicators['ma_50']) / indicators['ma_50'] * 100:.1f}%",
            'Volume Trend': f"{indicators.get('volume_ratio', 1.0):.2f}x"
        })
    
    return report_data


async def run_combined_analysis(universe='default'):
    """
    Run analysis for both timeframes and create comparison.
    
    Args:
        universe: 'default', 'sp500', or 'custom'
    """
    # Run analysis for both timeframes
    intermediate_results = await run_enhanced_momentum_analysis('intermediate', universe)
    long_term_results = await run_enhanced_momentum_analysis('long_term', universe)
    
    # Create data directory
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if we have results before creating dataframes
    if not intermediate_results:
        logger.warning("No intermediate results found")
        intermediate_df = pd.DataFrame()
    else:
        intermediate_df = pd.DataFrame(intermediate_results)
    
    if not long_term_results:
        logger.warning("No long-term results found")
        long_term_df = pd.DataFrame()
    else:
        long_term_df = pd.DataFrame(long_term_results)
    
    # Save individual results if not empty
    if not intermediate_df.empty:
        intermediate_file = data_dir / f"enhanced_advanced_momentum_intermediate_{timestamp}.csv"
        intermediate_df.to_csv(intermediate_file, index=False)
        logger.info(f"Intermediate analysis saved to: {intermediate_file}")
    
    if not long_term_df.empty:
        long_term_file = data_dir / f"enhanced_advanced_momentum_long_term_{timestamp}.csv"
        long_term_df.to_csv(long_term_file, index=False)
        logger.info(f"Long-term analysis saved to: {long_term_file}")
    
    # Create comprehensive comparison only if both dataframes have data
    if intermediate_df.empty or long_term_df.empty:
        logger.warning("Cannot create comparison - missing data for one or both timeframes")
        return
    
    comparison_data = []
    
    # Check if Symbol column exists
    if 'Symbol' not in intermediate_df.columns or 'Symbol' not in long_term_df.columns:
        logger.error("Symbol column not found in results")
        return
    
    symbols = set(intermediate_df['Symbol'].tolist() + long_term_df['Symbol'].tolist())
    
    for symbol in symbols:
        int_data = intermediate_df[intermediate_df['Symbol'] == symbol]
        lt_data = long_term_df[long_term_df['Symbol'] == symbol]
        
        if not int_data.empty and not lt_data.empty:
            int_row = int_data.iloc[0]
            lt_row = lt_data.iloc[0]
            
            # Parse scores
            int_comp_score = float(int_row['Comprehensive Score'])
            lt_comp_score = float(lt_row['Comprehensive Score'])
            int_tech_score = float(int_row['Technical Score'])
            lt_tech_score = float(lt_row['Technical Score'])
            int_fund_score = float(int_row['Fundamental Score'])
            lt_fund_score = float(lt_row['Fundamental Score'])
            int_sent_score = float(int_row['Sentiment Score'])
            lt_sent_score = float(lt_row['Sentiment Score'])
            
            # Calculate averages
            avg_comp_score = (int_comp_score + lt_comp_score) / 2
            avg_tech_score = (int_tech_score + lt_tech_score) / 2
            avg_fund_score = (int_fund_score + lt_fund_score) / 2
            avg_sent_score = (int_sent_score + lt_sent_score) / 2
            
            # Determine consensus
            if int_row['Action'] == 'BUY' and lt_row['Action'] == 'BUY':
                consensus = 'STRONG BUY' if avg_comp_score > 0.7 else 'BUY'
            elif int_row['Action'] == 'SELL' and lt_row['Action'] == 'SELL':
                consensus = 'STRONG SELL' if avg_comp_score < 0.3 else 'SELL'
            elif int_row['Action'] == lt_row['Action']:
                consensus = int_row['Action']
            else:
                consensus = 'MIXED'
            
            comparison_data.append({
                'Symbol': symbol,
                'Intermediate Action': int_row['Action'],
                'Intermediate Comp Score': int_row['Comprehensive Score'],
                'Intermediate Tech Score': int_row['Technical Score'],
                'Intermediate Fund Score': int_row['Fundamental Score'],
                'Intermediate Sent Score': int_row['Sentiment Score'],
                'Long-term Action': lt_row['Action'],
                'Long-term Comp Score': lt_row['Comprehensive Score'],
                'Long-term Tech Score': lt_row['Technical Score'],
                'Long-term Fund Score': lt_row['Fundamental Score'],
                'Long-term Sent Score': lt_row['Sentiment Score'],
                'Average Comp Score': f"{avg_comp_score:.3f}",
                'Average Tech Score': f"{avg_tech_score:.3f}",
                'Average Fund Score': f"{avg_fund_score:.3f}",
                'Average Sent Score': f"{avg_sent_score:.3f}",
                'Consensus': consensus,
                'Current Price': int_row['Current Price'],
                'RSI': int_row['RSI'],
                '1M Return': int_row['1M Return'],
                '3M Return': int_row['3M Return'],
                '52W High %': int_row['52W High %'],
                'Volume Ratio': int_row['Volume Ratio']
            })
    
    # Sort by average comprehensive score
    comparison_data.sort(key=lambda x: float(x['Average Comp Score']), reverse=True)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = data_dir / f"enhanced_advanced_momentum_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Comparison saved to: {comparison_file}")
    
    # Display comprehensive summary
    logger.info("\n=== Enhanced Advanced Momentum Analysis Summary ===")
    
    # Count actions by timeframe
    for timeframe, df in [('Intermediate', intermediate_df), ('Long-term', long_term_df)]:
        buy_count = len(df[df['Action'] == 'BUY'])
        sell_count = len(df[df['Action'] == 'SELL'])
        hold_count = len(df[df['Action'] == 'HOLD'])
        
        logger.info(f"\n{timeframe} Timeframe:")
        logger.info(f"BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")
        
        # Average scores
        avg_comp = df['Comprehensive Score'].apply(float).mean()
        avg_tech = df['Technical Score'].apply(float).mean()
        avg_fund = df['Fundamental Score'].apply(float).mean()
        avg_sent = df['Sentiment Score'].apply(float).mean()
        
        logger.info(f"Average Scores - Comprehensive: {avg_comp:.3f}, Technical: {avg_tech:.3f}, "
                   f"Fundamental: {avg_fund:.3f}, Sentiment: {avg_sent:.3f}")
    
    # Consensus summary
    if not comparison_df.empty:
        strong_buy = len(comparison_df[comparison_df['Consensus'] == 'STRONG BUY'])
        buy = len(comparison_df[comparison_df['Consensus'] == 'BUY'])
        sell = len(comparison_df[comparison_df['Consensus'] == 'SELL'])
        strong_sell = len(comparison_df[comparison_df['Consensus'] == 'STRONG SELL'])
        mixed = len(comparison_df[comparison_df['Consensus'] == 'MIXED'])
        hold = len(comparison_df[comparison_df['Consensus'] == 'HOLD'])
        
        logger.info(f"\nConsensus Analysis:")
        logger.info(f"STRONG BUY: {strong_buy}, BUY: {buy}, HOLD: {hold}, "
                   f"SELL: {sell}, STRONG SELL: {strong_sell}, MIXED: {mixed}")
        
        # Top consensus picks
        top_picks = comparison_df[comparison_df['Consensus'].str.contains('BUY')].head(10)
        if not top_picks.empty:
            logger.info("\nTop 10 Consensus BUY Recommendations:")
            for _, row in top_picks.iterrows():
                logger.info(f"{row['Symbol']}: {row['Consensus']} "
                          f"(Avg Comp Score: {row['Average Comp Score']}, "
                          f"Tech: {row['Average Tech Score']}, "
                          f"Fund: {row['Average Fund Score']}, "
                          f"Sent: {row['Average Sent Score']})")


def load_sp500_symbols():
    """Load S&P 500 symbols from file."""
    symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            return data['symbols']
    else:
        logger.warning("S&P 500 symbols file not found, using default universe")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM',
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM',
                'XOM', 'NFLX', 'CSCO', 'PFE', 'KO', 'PEP', 'CVX', 'WMT', 'ABT', 'NKE']


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced advanced momentum strategy analysis")
    parser.add_argument(
        "--universe",
        choices=['default', 'sp500', 'custom'],
        default='default',
        help="Universe of stocks to analyze"
    )
    parser.add_argument(
        "--timeframe",
        choices=['intermediate', 'long_term', 'both'],
        default='both',
        help="Timeframe to analyze"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(f"logs/enhanced_advanced_momentum_{args.universe}_{args.timeframe}.log", 
               level="DEBUG", rotation="10 MB")
    
    if args.timeframe == 'both':
        await run_combined_analysis(args.universe)
    else:
        results = await run_enhanced_momentum_analysis(args.timeframe, args.universe)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(results)
        output_file = project_root / 'data' / f"enhanced_advanced_momentum_{args.timeframe}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
