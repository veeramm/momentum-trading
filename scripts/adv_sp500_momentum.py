#!/usr/bin/env python
"""
Run Advanced Momentum Strategy on S&P 500

This script runs the advanced momentum strategy on all S&P 500 stocks
with proper batching and rate limit handling.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_fetcher import DataFetcher
from src.strategies.advanced_momentum import AdvancedMomentumStrategy
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.monitoring.performance_tracker import PerformanceTracker


# S&P 500 symbols list (as of 2024)
SP500_SYMBOLS = [
    'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE',
    'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIV',
    'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR',
    'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON',
    'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO',
    'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBY', 'BDX', 'BEN',
    'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR',
    'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
    'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CERN',
    'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP',
    'COST', 'CPB', 'CPRT', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTSH', 'CTVA',
    'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG',
    'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV',
    'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH',
    'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETFC', 'ETN', 'ETR', 'ETSY', 'EVRG',
    'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB', 'FBHS',
    'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLS', 'FLT',
    'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD',
    'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS',
    'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES',
    'HFC', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC',
    'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN',
    'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM',
    'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR',
    'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX',
    'KO', 'KR', 'KSU', 'L', 'LB', 'LDOS', 'LEG', 'LEN', 'LH', 'LHX',
    'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV',
    'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP',
    'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM',
    'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRO',
    'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'MXIM', 'MYL', 'NBL',
    'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NLSN', 'NOC',
    'NOV', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL',
    'NWS', 'NWSA', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OTIS',
    'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE',
    'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC',
    'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC',
    'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN',
    'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
    'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SLG',
    'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ',
    'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH',
    'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TIF', 'TJX', 'TMO', 'TMUS', 'TPR',
    'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TWTR', 'TXN',
    'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM',
    'UNP', 'UPS', 'URI', 'USB', 'V', 'VAR', 'VFC', 'VIAC', 'VLO', 'VMC',
    'VNO', 'VNT', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
    'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT',
    'WRB', 'WRK', 'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY',
    'XRX', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'
]


async def run_sp500_momentum(strategy_type='intermediate'):
    """
    Run the advanced momentum strategy on S&P 500 stocks
    
    Args:
        strategy_type: 'intermediate' or 'long_term'
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
                    'top_n': 30,  # Select top 30 from S&P 500
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
                    'max_position_size': 0.033,  # 30 positions = ~100%
                    'stop_loss_pct': 0.07,
                    'min_price': 10.0,
                    'min_volume': 5000000  # Higher volume for S&P 500
                },
                'long_term_momentum': {
                    'enabled': True,
                    'timeframe': 'long_term',
                    'lookback_period': 252,
                    'top_n': 20,  # Select top 20 from S&P 500
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
                    'max_position_size': 0.05,  # 20 positions = 100%
                    'stop_loss_pct': 0.10,
                    'min_price': 15.0,
                    'min_volume': 10000000  # Higher for long-term
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
    
    # Initialize data configuration with Tiingo
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
                'rate_limit': 600  # Tiingo hourly limit
            }
        }
    }
    
    # Initialize components
    data_fetcher = DataFetcher(data_config)
    portfolio = Portfolio(initial_capital=1000000)  # $1M for S&P 500 portfolio
    risk_config = config.get('risk_limits', {})
    risk_manager = RiskManager(risk_config)
    performance_tracker = PerformanceTracker()
    
    # Initialize strategy
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
    logger.info(f"Total symbols to analyze: {len(SP500_SYMBOLS)}")
    
    # Fetch market data in batches to avoid overwhelming the API
    batch_size = 25  # Process 25 symbols at a time
    all_market_data = {}
    failed_symbols = []
    
    for i in range(0, len(SP500_SYMBOLS), batch_size):
        batch = SP500_SYMBOLS[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(SP500_SYMBOLS) + batch_size - 1) // batch_size
        
        logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} symbols)...")
        
        try:
            batch_data = await data_fetcher.fetch_price_data(
                symbols=batch,
                start_date=start_date,
                end_date=end_date,
                interval='1d',
                source='tiingo'
            )
            
            # Add to overall market data
            all_market_data.update(batch_data)
            
            # Log progress
            successful = len([s for s in batch if s in batch_data])
            logger.info(f"Batch {batch_num}: {successful}/{len(batch)} symbols fetched successfully")
            
        except Exception as e:
            logger.error(f"Error fetching batch {batch_num}: {e}")
            failed_symbols.extend(batch)
        
        # Add delay between batches to respect rate limits
        if i + batch_size < len(SP500_SYMBOLS):
            await asyncio.sleep(2)  # 2 second delay between batches
    
    logger.info(f"Data fetch complete. Successfully fetched: {len(all_market_data)}/{len(SP500_SYMBOLS)} symbols")
    
    if failed_symbols:
        logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = strategy.generate_signals(all_market_data)
    
    # Display signals
    logger.info(f"\nGenerated {len(signals)} signals:")
    
    buy_signals = [s for s in signals if s.direction == 'long']
    sell_signals = [s for s in signals if s.direction == 'neutral']
    
    if buy_signals:
        logger.info(f"\nBUY signals ({len(buy_signals)}):")
        buy_signals.sort(key=lambda x: x.strength, reverse=True)  # Sort by strength
        
        for i, signal in enumerate(buy_signals[:10]):  # Show top 10
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
    
    # Create detailed analysis report
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
        report_file = f"sp500_momentum_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analysis_df.to_csv(report_file, index=False)
        logger.info(f"\nDetailed analysis saved to: {report_file}")
        
        # Display summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"Average Signal Strength: {analysis_df['Signal Strength'].mean():.3f}")
        logger.info(f"Average 1M Return: {analysis_df['1M Return'].mean():.1%}")
        logger.info(f"Average RSI: {analysis_df['RSI'].mean():.1f}")
        logger.info(f"% Above MA50: {analysis_df['Above MA50'].mean():.1%}")
        logger.info(f"% with Bullish MACD: {analysis_df['MACD Signal'].mean():.1%}")
        
        # Top sectors analysis (if we had sector data)
        # This would require additional fundamental data
    
    # Strategy-specific metrics
    metrics = strategy.get_strategy_specific_metrics()
    logger.info("\nStrategy Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        else:
            logger.info(f"  {key}: {value}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced momentum strategy on S&P 500")
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
    logger.add(f"logs/sp500_momentum_{args.timeframe}.log", level="DEBUG", rotation="10 MB")
    
    await run_sp500_momentum(args.timeframe)


if __name__ == "__main__":
    asyncio.run(main())
