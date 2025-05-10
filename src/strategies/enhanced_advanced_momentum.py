"""
Enhanced Advanced Momentum Strategy with Tiingo Data Integration

This module implements a comprehensive momentum strategy with integrated
action determination and advanced Tiingo data utilization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from ..core.base_strategy import BaseStrategy, Signal
from ..analysis.momentum_indicators import (
    calculate_momentum, calculate_rsi, calculate_rate_of_change
)


class EnhancedAdvancedMomentumStrategy(BaseStrategy):
    """
    Enhanced momentum strategy with integrated action determination
    and advanced Tiingo data utilization.
    """
    
    def __init__(self, config: dict, portfolio_manager=None, risk_manager=None, tiingo_fetcher=None):
        super().__init__(config, portfolio_manager, risk_manager)
        
        # Tiingo data fetcher for advanced features
        self.tiingo_fetcher = tiingo_fetcher
        
        # Strategy parameters
        self.timeframe = config.get('timeframe', 'intermediate')
        self.top_n = config.get('top_n', 20)
        
        # Indicator periods
        self.ma_short = config.get('ma_short', 20)
        self.ma_medium = config.get('ma_medium', 50)
        self.ma_long = config.get('ma_long', 200)
        
        # Entry/Exit criteria thresholds
        self.min_price_performance = config.get('min_price_performance', 0.10)
        self.volume_surge_threshold = config.get('volume_surge_threshold', 1.5)
        self.rsi_min = config.get('rsi_min', 50)
        self.rsi_max = config.get('rsi_max', 80)
        
        # Action determination weights
        self.technical_weight = config.get('technical_weight', 0.5)
        self.fundamental_weight = config.get('fundamental_weight', 0.3)
        self.sentiment_weight = config.get('sentiment_weight', 0.2)
        
        # Cache for fundamental and news data
        self.fundamental_cache = {}
        self.news_cache = {}
        
    async def enhance_with_tiingo_data(self, symbols: List[str]):
        """
        Fetch and cache advanced Tiingo data for symbols.
        
        Args:
            symbols: List of symbols to fetch data for
        """
        if not self.tiingo_fetcher:
            logger.warning("No Tiingo fetcher configured, skipping advanced data")
            return
            
        # Fetch fundamental data
        try:
            fundamentals = await self.tiingo_fetcher.fetch_fundamental_data(symbols)
            self.fundamental_cache.update(fundamentals)
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {e}")
        
        # Fetch recent news and sentiment
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            news = await self.tiingo_fetcher.fetch_news(
                symbols=symbols,
                start_date=start_date,
                limit=100
            )
            
            # Process news for sentiment scores
            self.news_cache = self._process_news_sentiment(news, symbols)
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
    
    def _process_news_sentiment(self, news: List[Dict], symbols: List[str]) -> Dict:
        """
        Process news data to extract sentiment scores.
        
        Args:
            news: List of news articles
            symbols: List of symbols
            
        Returns:
            Dictionary of sentiment scores by symbol
        """
        sentiment_scores = {symbol: {'count': 0, 'avg_sentiment': 0.0} for symbol in symbols}
        
        for article in news:
            # Extract symbols mentioned in the article
            mentioned_symbols = article.get('symbols', [])
            
            # Get sentiment score (if provided by Tiingo)
            sentiment = article.get('sentiment', 0.0)
            
            for symbol in mentioned_symbols:
                if symbol in sentiment_scores:
                    current = sentiment_scores[symbol]
                    current['count'] += 1
                    # Running average of sentiment
                    current['avg_sentiment'] = (
                        (current['avg_sentiment'] * (current['count'] - 1) + sentiment) 
                        / current['count']
                    )
        
        return sentiment_scores
    
    def calculate_comprehensive_score(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive score using technical, fundamental, and sentiment data.
        
        Args:
            symbol: Stock symbol
            market_data: Market data DataFrame
            
        Returns:
            Dictionary containing scores and analysis
        """
        # Calculate technical indicators
        indicators = self.calculate_indicators(market_data)
        technical_score = self._calculate_technical_score(indicators)
        
        # Calculate fundamental score
        fundamental_score = self._calculate_fundamental_score(symbol)
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(symbol)
        
        # Weighted comprehensive score
        comprehensive_score = (
            technical_score * self.technical_weight +
            fundamental_score * self.fundamental_weight +
            sentiment_score * self.sentiment_weight
        )
        
        # Determine action based on comprehensive analysis
        action = self._determine_action(
            comprehensive_score=comprehensive_score,
            technical_score=technical_score,
            indicators=indicators
        )
        
        # Check if buy criteria are met
        buy_criteria_met = self._check_buy_criteria(
            comprehensive_score=comprehensive_score,
            technical_score=technical_score,
            indicators=indicators
        )
        
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
        """
        Calculate technical score based on indicators.
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            Technical score (0-1)
        """
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
            # Scale based on how optimal the RSI is
            rsi_optimal = 60  # Ideal RSI
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
    
    def _calculate_fundamental_score(self, symbol: str) -> float:
        """
        Calculate fundamental score using Tiingo data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Fundamental score (0-1)
        """
        if symbol not in self.fundamental_cache:
            return 0.5  # Neutral score if no data
        
        fundamentals = self.fundamental_cache[symbol]
        score = 0.0
        components = 0
        
        # Extract key metrics
        metrics = fundamentals.get('daily_metrics', {})
        if metrics:
            latest_metrics = metrics[-1] if isinstance(metrics, list) else metrics
            
            # Revenue growth
            rev_growth = latest_metrics.get('revenueGrowth', 0)
            if rev_growth > 0.1:  # 10% growth
                score += 1
                components += 1
            elif rev_growth > 0:
                score += 0.5
                components += 1
            
            # Earnings growth
            eps_growth = latest_metrics.get('epsGrowth', 0)
            if eps_growth > 0.15:  # 15% growth
                score += 1
                components += 1
            elif eps_growth > 0:
                score += 0.5
                components += 1
            
            # Profit margins
            profit_margin = latest_metrics.get('profitMargin', 0)
            if profit_margin > 0.15:  # 15% margin
                score += 1
                components += 1
            elif profit_margin > 0.10:
                score += 0.5
                components += 1
            
            # Return on equity
            roe = latest_metrics.get('returnOnEquity', 0)
            if roe > 0.20:  # 20% ROE
                score += 1
                components += 1
            elif roe > 0.15:
                score += 0.5
                components += 1
        
        # Normalize score
        return score / components if components > 0 else 0.5
    
    def _calculate_sentiment_score(self, symbol: str) -> float:
        """
        Calculate sentiment score from news data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment score (0-1)
        """
        if symbol not in self.news_cache:
            return 0.5  # Neutral score if no data
        
        sentiment_data = self.news_cache[symbol]
        
        # Convert sentiment to 0-1 scale
        # Assuming sentiment ranges from -1 to 1
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        news_count = sentiment_data.get('count', 0)
        
        # Weight by news volume (more news = more confidence)
        confidence = min(news_count / 10, 1.0)  # Max confidence at 10+ articles
        
        # Convert to 0-1 scale
        normalized_sentiment = (avg_sentiment + 1) / 2
        
        # Blend with neutral based on confidence
        return normalized_sentiment * confidence + 0.5 * (1 - confidence)
    
    def _determine_action(self, comprehensive_score: float, technical_score: float, 
                         indicators: Dict[str, float]) -> str:
        """
        Determine trading action based on comprehensive analysis.
        
        Args:
            comprehensive_score: Overall weighted score
            technical_score: Technical analysis score
            indicators: Technical indicators
            
        Returns:
            Action: 'BUY', 'HOLD', or 'SELL'
        """
        # Strong buy threshold
        if comprehensive_score >= 0.7 and technical_score >= 0.6:
            return 'BUY'
        
        # Buy threshold
        elif comprehensive_score >= 0.6 and technical_score >= 0.5:
            # Additional checks for buy
            if (indicators['close'] > indicators['ma_50'] and 
                self.rsi_min <= indicators['rsi'] <= self.rsi_max and
                indicators.get('volume_ratio', 1.0) > 1.0):
                return 'BUY'
        
        # Sell conditions
        elif comprehensive_score < 0.3 or technical_score < 0.3:
            return 'SELL'
        
        # Specific sell triggers
        elif (indicators['rsi'] > 80 or  # Extremely overbought
              indicators['close'] < indicators['ma_50'] or  # Below key MA
              indicators.get('pct_from_52w_high', 0) < -0.25):  # 25% below high
            return 'SELL'
        
        # Default to hold
        return 'HOLD'
    
    def _check_buy_criteria(self, comprehensive_score: float, technical_score: float,
                           indicators: Dict[str, float]) -> bool:
        """
        Check if buy criteria are met.
        
        Args:
            comprehensive_score: Overall weighted score
            technical_score: Technical analysis score
            indicators: Technical indicators
            
        Returns:
            True if buy criteria are met
        """
        # Must have positive momentum
        if indicators.get('return_1m', 0) <= 0:
            return False
        
        # Must have minimum scores
        if comprehensive_score < 0.5 or technical_score < 0.4:
            return False
        
        # Must be above key moving average
        if indicators['close'] < indicators['ma_50']:
            return False
        
        # RSI must be in acceptable range
        if not (self.rsi_min <= indicators['rsi'] <= self.rsi_max):
            return False
        
        return True
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals with comprehensive analysis.
        
        Args:
            market_data: Dictionary of market data by symbol
            
        Returns:
            List of trading signals
        """
        # Enhance with Tiingo data
        symbols = list(market_data.keys())
        await self.enhance_with_tiingo_data(symbols)
        
        # Analyze all stocks
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
        
        # Sort by comprehensive score
        analyses.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        # Generate signals
        signals = []
        current_time = datetime.now()
        
        # Buy signals for top stocks
        for i, analysis in enumerate(analyses[:self.top_n]):
            if analysis['action'] == 'BUY' and analysis['buy_criteria_met']:
                signal = Signal(
                    symbol=analysis['symbol'],
                    direction='long',
                    strength=analysis['comprehensive_score'],
                    timestamp=current_time,
                    metadata={
                        'analysis': analysis,
                        'rank': i + 1
                    }
                )
                signals.append(signal)
        
        # Sell signals for existing positions
        for symbol, position in self.positions.items():
            if symbol in market_data:
                analysis = self.calculate_comprehensive_score(symbol, market_data[symbol])
                
                if analysis['action'] == 'SELL':
                    signal = Signal(
                        symbol=symbol,
                        direction='neutral',
                        strength=0.0,
                        timestamp=current_time,
                        metadata={
                            'analysis': analysis,
                            'reason': 'sell_criteria_met'
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def get_analysis_for_reporting(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Get comprehensive analysis for all stocks for reporting.
        
        Args:
            market_data: Dictionary of market data by symbol
            
        Returns:
            List of analysis dictionaries for reporting
        """
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
        
        # Sort by comprehensive score
        analyses.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        return analyses
