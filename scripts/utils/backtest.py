#!/usr/bin/env python
"""
Backtest Script for Momentum Trading Strategies

This script runs backtests for configured momentum strategies
and generates performance reports.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.data.data_fetcher import DataFetcher
from src.strategies.classic_momentum import ClassicMomentumStrategy
from src.monitoring.performance_tracker import PerformanceTracker


class Backtester:
    """
    Backtesting engine for momentum strategies.
    """
    
    def __init__(self, config_path: str):
        """Initialize backtester with configuration"""
        self.config = self._load_config(config_path)
        self.data_fetcher = DataFetcher(self.config['data'])
        self.performance_tracker = PerformanceTracker()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_backtest(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        data_source: str = "auto"
    ):
        """
        Run backtest for a specific strategy.
        
        Args:
            strategy_name: Name of strategy to test
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        """
        logger.info(f"Starting backtest for {strategy_name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Get strategy configuration
        strategy_config = self.config['strategies'].get(strategy_name)
        if not strategy_config:
            raise ValueError(f"Strategy '{strategy_name}' not found in config")
        
        # Initialize portfolio and risk manager
        portfolio = Portfolio(initial_capital)
        risk_manager = RiskManager(self.config['trading']['risk'])
        
        # Initialize strategy
        strategy_class = self._get_strategy_class(strategy_name)
        strategy = strategy_class(
            config=strategy_config,
            portfolio_manager=portfolio,
            risk_manager=risk_manager
        )
        
        # Get universe of assets
        universe = self.config['trading']['universe']
        symbols = universe.get('equities', []) + universe.get('bonds', [])
        
        # Fetch historical data
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        market_data = await self.data_fetcher.fetch_price_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d',
            source=data_source
        )
        
        # Run backtest day by day
        all_dates = sorted(set().union(*[data.index for data in market_data.values()]))
        
        logger.info(f"Running backtest over {len(all_dates)} trading days...")
        
        for current_date in all_dates:
            # Get data up to current date
            current_data = {}
            for symbol, data in market_data.items():
                mask = data.index <= current_date
                if mask.any():
                    current_data[symbol] = data[mask]
            
            # Update strategy
            strategy.update(current_data, current_date)
            
            # Update portfolio values
            portfolio.update_prices(current_data, current_date)
            
            # Track performance
            self.performance_tracker.update(portfolio, current_date)
            
            # Log progress every month
            if current_date.day == 1:
                equity = portfolio.total_equity
                logger.info(f"{current_date.date()}: Portfolio value: ${equity:,.2f}")
        
        # Generate performance report
        logger.info("Generating performance report...")
        self._generate_report(strategy_name, start_date, end_date)
    
    def _get_strategy_class(self, strategy_name: str):
        """Get strategy class by name"""
        strategy_map = {
            'classic_momentum': ClassicMomentumStrategy,
            # Add more strategies here
        }
        
        return strategy_map.get(strategy_name)
    
    def _generate_report(self, strategy_name: str, start_date: str, end_date: str):
        """Generate and save performance report"""
        metrics = self.performance_tracker.calculate_metrics()
        
        # Create report
        report = f"""
Momentum Trading Backtest Report
================================

Strategy: {strategy_name}
Period: {start_date} to {end_date}

Performance Metrics:
-------------------
Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Calmar Ratio: {metrics['calmar_ratio']:.2f}

Win Rate: {metrics['win_rate']:.2%}
Average Win: {metrics['avg_win']:.2%}
Average Loss: {metrics['avg_loss']:.2%}
Profit Factor: {metrics['profit_factor']:.2f}

Best Month: {metrics['best_month']:.2%}
Worst Month: {metrics['worst_month']:.2%}
        """
        
        # Save report
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"{strategy_name}_backtest_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")
        
        # Save detailed results
        results_df = self.performance_tracker.get_results_dataframe()
        results_file = report_dir / f"{strategy_name}_results_{timestamp}.csv"
        results_df.to_csv(results_file)
        
        logger.info(f"Detailed results saved to {results_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Backtest momentum trading strategies")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--strategy", required=True, help="Strategy name to backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--source", default="auto", help="Data source (tiingo, yfinance, auto)")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/backtest.log", level="DEBUG", rotation="10 MB")
    
    # Run backtest
    backtester = Backtester(args.config)
    await backtester.run_backtest(
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        data_source=args.source
    )


if __name__ == "__main__":
    asyncio.run(main())
