#!/usr/bin/env python
"""
Standalone Enhanced Advanced Momentum Strategy Script

This script includes all necessary components in one file for immediate use.
Run this to get the enhanced momentum analysis without needing to create multiple files.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger
import os
from dotenv import load_dotenv
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import aiohttp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.base_strategy import BaseStrategy, Signal
from src.analysis.momentum_indicators import calculate_rsi
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager


class EnhancedTiingoDataFetcher:
    """Enhanced Tiingo data fetcher with support for advanced features."""
    
    def __init__(self, config: dict):
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Tiingo API key required")
        
        self.base_url = config.get('base_url', 'https://api.tiingo.com')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make an async request to Tiingo API."""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
    
    async def fetch_daily_data(self, symbols: List[str], start_date, end_date, adjusted: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch daily price data for multiple symbols."""
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        results = {}
        
        for symbol in symbols:
            try:
                endpoint = f"/tiingo/daily/{symbol}/prices"
                params = {
                    'startDate': start_date,
                    'endDate': end_date
                }
                
                data = await self._make_request(endpoint, params)
                df = pd.DataFrame(data)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    if adjusted and 'adjClose' in df.columns:
                        df['close'] = df['adjClose']
                        df['open'] = df['adjOpen'] if 'adjOpen' in df.columns else df['open']
                        df['high'] = df['adjHigh'] if 'adjHigh' in df.columns else df['high']
                        df['low'] = df['adjLow'] if 'adjLow' in df.columns else df['low']
                        df['volume'] = df['adjVolume'] if 'adjVolume' in df.columns else df['volume']
                    
                    results[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return results
    
    async def fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch fundamental data - simplified version."""
        # Return dummy data for now
        return {symbol: {'meta': {}, 'statements': {}, 'daily_metrics': {}} for symbol in symbols}
    
    async def fetch_news(self, symbols: List[str], start_date=None, limit: int = 100) -> List[Dict]:
        """Fetch news articles - simplified version."""
        # Return dummy data for now
        return []


class EnhancedAdvancedMomentumStrategy(BaseStrategy):
    """Enhanced momentum strategy with integrated action determination."""
    
    def __init__(self, config: dict, portfolio_manager=None, risk_manager=None, tiingo_fetcher=None):
        super().__init__(config, portfolio_manager, risk_manager)
        
        self.tiingo_fetcher = tiingo_fetcher
        self.timeframe = config.get('timeframe', 'intermediate')
        self.top_n = config.get('top_n', 20)
        
        self.ma_short = config.get('ma_short', 20)
        self.ma_medium = config.get('ma_medium', 50)
        self.ma_long = config.get('ma_long', 200)
        
        self.min_price_performance = config.get('min_price_performance', 0.10)
        self.volume_surge_threshold = config.get('volume_surge_threshold', 1.5)
        self.rsi_min = config.get('rsi_min', 50)
        self.rsi_max = config.get('rsi_max', 80)
        
        self.technical_weight = config.get('technical_weight', 0.5)
        self.fundamental_weight = config.get('fundamental_weight', 0.3)
        self.sentiment_weight = config.get('sentiment_weight', 0.2)
        
        self.fundamental_cache = {}
        self.news_cache = {}
    
    def calculate_momentum_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum scores - required by BaseStrategy."""
        analysis = self.calculate_comprehensive_score('UNKNOWN', data)
        return pd.Series([analysis['comprehensive_score']])
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Handle duplicate columns
        close = data['close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        volume = data.get('volume', pd.Series(0, index=close.index))
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        
        # Moving Averages
        indicators['ma_20'] = close.rolling(window=20).mean().iloc[-1]
        indicators['ma_50'] = close.rolling(window=50).mean().iloc[-1]
        indicators['ma_200'] = close.rolling(window=200).mean().iloc[-1]
        
        # Current price
        indicators['close'] = close.iloc[-1]
        
        # RSI
        rsi = calculate_rsi(close, period=14)
        indicators['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        
        # 52-week high
        high_52w = close.rolling(window=252).max().iloc[-1]
        indicators['pct_from_52w_high'] = (close.iloc[-1] - high_52w) / high_52w
        
        # Volume indicators
        indicators['volume_ma_20'] = volume.rolling(window=20).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_ma_20'] if indicators['volume_ma_20'] > 0 else 1
        
        # Price performance
        if len(close) >= 21:
            indicators['return_1m'] = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
        else:
            indicators['return_1m'] = 0
            
        if len(close) >= 63:
            indicators['return_3m'] = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
        else:
            indicators['return_3m'] = 0
        
        # Simplified ADX
        indicators['adx'] = 30  # Placeholder
        
        return indicators
    
    def calculate_comprehensive_score(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive score using technical, fundamental, and sentiment data."""
        indicators = self.calculate_indicators(market_data)
        technical_score = self._calculate_technical_score(indicators)
        
        # Simplified scoring for fundamental and sentiment
        fundamental_score = 0.5  # Neutral
        sentiment_score = 0.5    # Neutral
        
        comprehensive_score = (
            technical_score * self.technical_weight +
            fundamental_score * self.fundamental_weight +
            sentiment_score * self.sentiment_weight
        )
        
        action = self._determine_action(comprehensive_score, technical_score, indicators)
        buy_criteria_met = self._check_buy_criteria(comprehensive_score, technical_score, indicators)
        
        return {
            'symbol': symbol,
            'comprehensive_score': comprehensive_score,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
            'action': action,
            'buy_criteria_met': buy_criteria_met,
            'indicators': indicators,
            'timeframe': self.timeframe
        }
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Calculate technical score based on indicators."""
        score = 0.0
        max_score = 100
        
        # Price momentum (30 points)
        if indicators.get('return_1m', 0) > self.min_price_performance:
            score += 15
        if indicators.get('return_3m', 0) > self.min_price_performance * 1.5:
            score += 15
        
        # Moving average strength (25 points)
        if indicators['close'] > indicators['ma_20']:
            score += 8
        if indicators['close'] > indicators['ma_50']:
            score += 8
        if indicators['close'] > indicators['ma_200']:
            score += 9
        
        # RSI in optimal range (15 points)
        rsi = indicators.get('rsi', 50)
        if self.rsi_min <= rsi <= self.rsi_max:
            rsi_optimal = 60
            rsi_diff = abs(rsi - rsi_optimal)
            rsi_score = max(0, 15 - rsi_diff * 0.5)
            score += rsi_score
        
        # MACD bullish (10 points)
        if indicators['macd'] > indicators['macd_signal']:
            score += 10
        
        # Volume confirmation (10 points)
        if indicators.get('volume_ratio', 1.0) > self.volume_surge_threshold:
            score += 10
        
        # Trend strength (10 points)
        if indicators.get('adx', 25) > 30:
            score += 10
        
        return score / max_score
    
    def _determine_action(self, comprehensive_score: float, technical_score: float, 
                         indicators: Dict[str, float]) -> str:
        """Determine trading action based on comprehensive analysis."""
        if comprehensive_score >= 0.7 and technical_score >= 0.6:
            return 'BUY'
        elif comprehensive_score >= 0.6 and technical_score >= 0.5:
            if (indicators['close'] > indicators['ma_50'] and 
                self.rsi_min <= indicators['rsi'] <= self.rsi_max and
                indicators.get('volume_ratio', 1.0) > 1.0):
                return 'BUY'
        elif comprehensive_score < 0.3 or technical_score < 0.3:
            return 'SELL'
        elif (indicators['rsi'] > 80 or
              indicators['close'] < indicators['ma_50'] or
              indicators.get('pct_from_52w_high', 0) < -0.25):
            return 'SELL'
        
        return 'HOLD'
    
    def _check_buy_criteria(self, comprehensive_score: float, technical_score: float,
                           indicators: Dict[str, float]) -> bool:
        """Check if buy criteria are met."""
        if indicators.get('return_1m', 0) <= 0:
            return False
        if comprehensive_score < 0.5 or technical_score < 0.4:
            return False
        if indicators['close'] < indicators['ma_50']:
            return False
        if not (self.rsi_min <= indicators['rsi'] <= self.rsi_max):
            return False
        return True
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals - required by BaseStrategy."""
        analyses = []
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                continue
            try:
                analysis = self.calculate_comprehensive_score(symbol, data)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        analyses.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        signals = []
        current_time = datetime.now()
        
        for i, analysis in enumerate(analyses[:self.top_n]):
            if analysis['action'] == 'BUY' and analysis['buy_criteria_met']:
                signal = Signal(
                    symbol=analysis['symbol'],
                    direction='long',
                    strength=analysis['comprehensive_score'],
                    timestamp=current_time,
                    metadata={'analysis': analysis, 'rank': i + 1}
                )
                signals.append(signal)
        
        return signals
    
    def get_analysis_for_reporting(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Get comprehensive analysis for all stocks for reporting."""
        analyses = []
        
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                continue
            
            try:
                analysis = self.calculate_comprehensive_score(symbol, data)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        analyses.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        return analyses


async def run_enhanced_momentum_analysis(timeframe='intermediate', universe='default'):
    """Run the enhanced momentum strategy analysis."""
    load_dotenv()
    
    tiingo_config = {
        'api_key': os.getenv('TIINGO_API_KEY'),
        'use_iex': True,
        'rate_limit': 10000,
        'rate_limit_window': 3600
    }
    
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
        'technical_weight': 0.7,  # Increased since we have limited data
        'fundamental_weight': 0.15,
        'sentiment_weight': 0.15
    }
    
    tiingo_fetcher = EnhancedTiingoDataFetcher(tiingo_config)
    portfolio = Portfolio(initial_capital=100000)
    risk_manager = RiskManager({})
    
    strategy = EnhancedAdvancedMomentumStrategy(
        config=strategy_config,
        portfolio_manager=portfolio,
        risk_manager=risk_manager,
        tiingo_fetcher=tiingo_fetcher
    )
    
    # Define universe
    if universe == 'sp500':
        symbols = load_sp500_symbols()
    else:
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    logger.info(f"Running enhanced momentum analysis for {timeframe} timeframe")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Universe: {len(symbols)} symbols")
    
    market_data = await tiingo_fetcher.fetch_daily_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        adjusted=True
    )
    
    logger.info(f"Successfully fetched data for {len(market_data)} symbols")
    
    analyses = strategy.get_analysis_for_reporting(market_data)
    
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
            'ADX': f"{indicators.get('adx', 0):.1f}"
        })
    
    return report_data


def load_sp500_symbols():
    """Load S&P 500 symbols from file."""
    symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            return data['symbols']
    else:
        logger.warning("S&P 500 symbols file not found, using default universe")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced momentum strategy analysis")
    parser.add_argument(
        "--universe",
        choices=['default', 'sp500', 'custom'],
        default='default',
        help="Universe of stocks to analyze"
    )
    parser.add_argument(
        "--timeframe",
        choices=['intermediate', 'long_term', 'both'],
        default='intermediate',
        help="Timeframe to analyze"
    )
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    if args.timeframe == 'both':
        # Run both timeframes
        intermediate_results = await run_enhanced_momentum_analysis('intermediate', args.universe)
        long_term_results = await run_enhanced_momentum_analysis('long_term', args.universe)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_dir = project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        
        intermediate_df = pd.DataFrame(intermediate_results)
        intermediate_file = data_dir / f"enhanced_momentum_intermediate_{timestamp}.csv"
        intermediate_df.to_csv(intermediate_file, index=False)
        logger.info(f"Intermediate analysis saved to: {intermediate_file}")
        
        long_term_df = pd.DataFrame(long_term_results)
        long_term_file = data_dir / f"enhanced_momentum_long_term_{timestamp}.csv"
        long_term_df.to_csv(long_term_file, index=False)
        logger.info(f"Long-term analysis saved to: {long_term_file}")
        
        # Create comparison
        comparison_data = []
        symbols = set(df['Symbol'] for df in [intermediate_df, long_term_df])
        
        for symbol in symbols:
            int_data = intermediate_df[intermediate_df['Symbol'] == symbol]
            lt_data = long_term_df[long_term_df['Symbol'] == symbol]
            
            if not int_data.empty and not lt_data.empty:
                int_row = int_data.iloc[0]
                lt_row = lt_data.iloc[0]
                
                int_score = float(int_row['Comprehensive Score'])
                lt_score = float(lt_row['Comprehensive Score'])
                avg_score = (int_score + lt_score) / 2
                
                if int_row['Action'] == 'BUY' and lt_row['Action'] == 'BUY':
                    consensus = 'STRONG BUY' if avg_score > 0.7 else 'BUY'
                elif int_row['Action'] == 'SELL' and lt_row['Action'] == 'SELL':
                    consensus = 'STRONG SELL' if avg_score < 0.3 else 'SELL'
                elif int_row['Action'] == lt_row['Action']:
                    consensus = int_row['Action']
                else:
                    consensus = 'MIXED'
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Intermediate Action': int_row['Action'],
                    'Intermediate Score': int_row['Comprehensive Score'],
                    'Long-term Action': lt_row['Action'],
                    'Long-term Score': lt_row['Comprehensive Score'],
                    'Average Score': f"{avg_score:.3f}",
                    'Consensus': consensus,
                    'Current Price': int_row['Current Price'],
                    'RSI': int_row['RSI'],
                    '1M Return': int_row['1M Return']
                })
        
        comparison_data.sort(key=lambda x: float(x['Average Score']), reverse=True)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = data_dir / f"enhanced_momentum_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Comparison saved to: {comparison_file}")
        
    else:
        results = await run_enhanced_momentum_analysis(args.timeframe, args.universe)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(results)
        output_file = project_root / 'data' / f"enhanced_momentum_{args.timeframe}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
