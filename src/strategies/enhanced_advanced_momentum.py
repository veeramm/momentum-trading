# src/strategies/enhanced_advanced_momentum.py

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
            
        # Check for Dow 30 symbols
        dow30_count = sum(1 for s in symbols if self.tiingo_fetcher._is_dow30(s))
        logger.info(f"Found {dow30_count} Dow 30 symbols out of {len(symbols)} total")
    
        # Fetch fundamental data (automatically filtered to Dow 30)
        try:
            logger.info("Fetching fundamental data for Dow 30 companies...")
            fundamentals = await self.tiingo_fetcher.fetch_fundamental_data(symbols, show_progress=True)
            self.fundamental_cache.update(fundamentals)
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {e}")

        # Fetch recent news and sentiment
        try:
            logger.info("Fetching news and sentiment data...")
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
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all momentum indicators for a single stock.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of indicator values
        """
        indicators = {}
        
        # Handle duplicate columns
        close = data['close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        volume = data['volume']
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
        indicators['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
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
        
        # ADX (simplified)
        high = data['high']
        low = data['low']
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, 
                       abs(high - close.shift(1)), 
                       abs(low - close.shift(1))], axis=1).max(axis=1)
        
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=14).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=14).mean()
        
        indicators['adx'] = adx.iloc[-1] if len(adx) > 0 else 25
        
        return indicators
    
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
        
        # Check if fundamental data is available
        has_fundamentals = (symbol in self.fundamental_cache and 
                           self.fundamental_cache[symbol].get('has_fundamentals', False))
        
        # Adjust weights based on data availability
        if has_fundamentals:
            # Use configured weights
            tech_weight = self.technical_weight
            fund_weight = self.fundamental_weight
            sent_weight = self.sentiment_weight
        else:
            # Redistribute fundamental weight to technical and sentiment
            tech_weight = self.technical_weight + (self.fundamental_weight * 0.7)
            fund_weight = 0
            sent_weight = self.sentiment_weight + (self.fundamental_weight * 0.3)
            
            # Normalize weights
            total_weight = tech_weight + fund_weight + sent_weight
            tech_weight /= total_weight
            sent_weight /= total_weight
        
        # Weighted comprehensive score
        comprehensive_score = (
            technical_score * tech_weight +
            fundamental_score * fund_weight +
            sentiment_score * sent_weight
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
        
        # Check if we have an error or no fundamentals
        if 'error' in fundamentals or not fundamentals.get('has_fundamentals', False):
            return 0.5  # Return neutral score for unavailable data
        
        score = 0.5  # Start with neutral
        
        try:
            # Analyze fundamental data if available
            statements = fundamentals.get('statements', {})
            daily_metrics = fundamentals.get('daily_metrics', {})
            
            # Simple scoring based on available metrics
            # This is a placeholder - you would implement more sophisticated analysis
            if statements:
                # Add points for positive fundamentals
                score += 0.1
            
            if daily_metrics:
                # Add points for good daily metrics
                score += 0.1
            
            # Ensure score stays within bounds
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"Error calculating fundamental score for {symbol}: {e}")
            return 0.5
        
        return score
    
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
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals with comprehensive analysis.
        
        Args:
            market_data: Dictionary of market data by symbol
            
        Returns:
            List of trading signals
        """
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
    
    def calculate_momentum_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for assets.
        Required implementation of abstract method from BaseStrategy.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with momentum scores
        """
        # This is called by the base class for single asset scoring
        # We'll use our comprehensive scoring approach
        
        # Handle duplicate columns
        close_data = data['close']
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        
        if len(close_data) < self.lookback_period:
            return pd.Series([np.nan])
        
        # For single asset analysis, we'll return technical score
        # The full comprehensive score requires symbol information for fundamental/sentiment
        indicators = self.calculate_indicators(data)
        technical_score = self._calculate_technical_score(indicators)
        
        return pd.Series([technical_score])
