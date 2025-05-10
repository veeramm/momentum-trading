# src/strategies/enhanced_advanced_momentum.py

"""
Enhanced Advanced Momentum Strategy with Balanced Weight Adjustment

This module implements a comprehensive momentum strategy with integrated
action determination, advanced Tiingo data utilization, and balanced weight
adjustment based on data availability.
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
    Enhanced momentum strategy with integrated action determination,
    advanced Tiingo data utilization, and balanced weight adjustment.
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
        
        # Base weights (will be adjusted based on data availability)
        self.base_weights = {
            'technical': config.get('technical_weight', 0.5),
            'fundamental': config.get('fundamental_weight', 0.3),
            'sentiment': config.get('sentiment_weight', 0.2)
        }
        
        # Confidence and adjustment parameters
        self.min_confidence = config.get('min_confidence', 0.2)  # Lowered from 0.3
        self.strictness_level = config.get('strictness_level', 'balanced')  # 'strict', 'balanced', 'relaxed'
        self.use_adaptive_thresholds = config.get('use_adaptive_thresholds', True)
        
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
    
    def adjust_weights_for_missing_data(self, data_availability: Dict[str, bool]) -> Dict[str, float]:
        """
        Dynamically adjust weights based on available data with balanced approach
        
        Args:
            data_availability: Dictionary indicating which data types are available
            
        Returns:
            Dictionary of adjusted weights
        """
        # Count available data types
        available_count = sum(data_availability.values())
        
        if available_count == 0:
            raise ValueError("No data available for analysis")
        
        # Start with base weights
        adjusted_weights = self.base_weights.copy()
        
        # Handle missing data based on strictness level
        if self.strictness_level == 'strict':
            # Original strict approach
            for data_type, is_available in data_availability.items():
                if not is_available:
                    # Redistribute weight more aggressively
                    weight_to_redistribute = adjusted_weights[data_type]
                    adjusted_weights[data_type] = 0
                    
                    # Redistribute to available types
                    available_types = [t for t, avail in data_availability.items() if avail and t != data_type]
                    if available_types:
                        for target_type in available_types:
                            adjusted_weights[target_type] += weight_to_redistribute / len(available_types)
        
        elif self.strictness_level == 'relaxed':
            # More lenient approach - keep some weight even for missing data
            for data_type, is_available in data_availability.items():
                if not is_available:
                    # Reduce weight but don't eliminate it
                    adjusted_weights[data_type] *= 0.3
        
        else:  # 'balanced' (default)
            # Balanced approach - partial redistribution
            for data_type, is_available in data_availability.items():
                if not is_available:
                    original_weight = adjusted_weights[data_type]
                    # Keep 20% of original weight (as neutral score)
                    adjusted_weights[data_type] = original_weight * 0.2
                    
                    # Redistribute 80% of the weight
                    weight_to_redistribute = original_weight * 0.8
                    
                    # Define redistribution preferences
                    redistribution_priority = {
                        'technical': {'fundamental': 0.3, 'sentiment': 0.7},
                        'fundamental': {'technical': 0.7, 'sentiment': 0.3},
                        'sentiment': {'technical': 0.8, 'fundamental': 0.2}
                    }
                    
                    for target_type, priority in redistribution_priority[data_type].items():
                        if data_availability.get(target_type, False):
                            adjusted_weights[target_type] += weight_to_redistribute * priority
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        for data_type in adjusted_weights:
            adjusted_weights[data_type] /= total_weight
        
        return adjusted_weights
    
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
    
    def _calculate_confidence(self, data_availability: Dict[str, bool], 
                             technical_score: float, 
                             indicators: Dict[str, float]) -> float:
        """
        Calculate confidence level based on data availability and quality
        
        Args:
            data_availability: Dictionary indicating which data types are available
            technical_score: Technical analysis score
            indicators: Technical indicators
            
        Returns:
            Confidence level (0-1)
        """
        # Base confidence from data availability
        available_count = sum(data_availability.values())
        base_confidence = available_count / 3.0
        
        # Adjust for data quality
        quality_multiplier = 1.0
        
        # Technical quality check
        if data_availability['technical']:
            # Check for strong technical signals
            if technical_score > 0.7:
                quality_multiplier *= 1.1
            elif technical_score < 0.3:
                quality_multiplier *= 0.9
        
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            quality_multiplier *= 1.05
        elif volume_ratio < 0.8:
            quality_multiplier *= 0.95
        
        # Final confidence calculation
        confidence = base_confidence * quality_multiplier
        
        # Apply strictness level adjustments
        if self.strictness_level == 'strict':
            confidence *= 0.9  # More conservative
        elif self.strictness_level == 'relaxed':
            confidence *= 1.1  # More lenient
        
        return min(1.0, max(0.0, confidence))
    
    def _determine_action(self, comprehensive_score: float, 
                         technical_score: float, 
                         indicators: Dict[str, float],
                         confidence: float,
                         data_availability: Dict[str, bool]) -> str:
        """
        Determine trading action with balanced approach
        
        Args:
            comprehensive_score: Overall weighted score
            technical_score: Technical analysis score
            indicators: Technical indicators
            confidence: Confidence level
            data_availability: Dictionary indicating which data types are available
            
        Returns:
            Action: 'BUY', 'HOLD', or 'SELL'
        """
        # Base thresholds
        buy_threshold = 0.6
        sell_threshold = 0.3
        
        # Adjust thresholds based on timeframe
        if self.timeframe == 'long_term':
            buy_threshold = 0.65
            sell_threshold = 0.25
        elif self.timeframe == 'short_term':
            buy_threshold = 0.55
            sell_threshold = 0.35
        
        # Adaptive threshold adjustments based on data availability
        if self.use_adaptive_thresholds:
            missing_count = 3 - sum(data_availability.values())
            if missing_count > 0:
                # More modest adjustment than before
                threshold_adjustment = missing_count * 0.02  # Reduced from 0.05
                buy_threshold += threshold_adjustment
                sell_threshold -= threshold_adjustment
        
        # Primary action determination
        if comprehensive_score >= buy_threshold:
            # Additional buy confirmation checks
            if technical_score >= 0.5 and indicators['close'] > indicators['ma_50']:
                action = 'BUY'
            else:
                action = 'HOLD'
        elif comprehensive_score < sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Confidence-based adjustments (less aggressive than before)
        if confidence < self.min_confidence:
            if action == 'BUY':
                # Only downgrade to HOLD if confidence is very low
                if confidence < self.min_confidence * 0.7:
                    action = 'HOLD'
            elif action == 'SELL':
                # Be more cautious about selling with low confidence
                if confidence < self.min_confidence * 0.8:
                    action = 'HOLD'
        
        # Special case handling with less strict requirements
        if action == 'BUY' and not data_availability['fundamental']:
            # Reduced technical requirement when fundamentals missing
            if technical_score < 0.6 or indicators['rsi'] > 75:  # Relaxed from 0.7 and 70
                action = 'HOLD'
        
        return action
    
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
        
        # Check data availability
        data_availability = {
            'technical': True,  # Always available
            'fundamental': (symbol in self.fundamental_cache and 
                          self.fundamental_cache[symbol].get('has_fundamentals', False)),
            'sentiment': (symbol in self.news_cache and 
                         self.news_cache[symbol].get('count', 0) > 0)
        }
        
        # Get adjusted weights based on data availability
        adjusted_weights = self.adjust_weights_for_missing_data(data_availability)
        
        # Weighted comprehensive score
        comprehensive_score = (
            technical_score * adjusted_weights['technical'] +
            fundamental_score * adjusted_weights['fundamental'] +
            sentiment_score * adjusted_weights['sentiment']
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(data_availability, technical_score, indicators)
        
        # Determine action
        action = self._determine_action(
            comprehensive_score=comprehensive_score,
            technical_score=technical_score,
            indicators=indicators,
            confidence=confidence,
            data_availability=data_availability
        )
        
        # Check if buy criteria are met
        buy_criteria_met = self._check_buy_criteria(
            comprehensive_score=comprehensive_score,
            technical_score=technical_score,
            indicators=indicators,
            confidence=confidence
        )
        
        return {
            'symbol': symbol,
            'comprehensive_score': comprehensive_score,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
            'action': action,
            'confidence': confidence,
            'buy_criteria_met': buy_criteria_met,
            'indicators': indicators,
            'timeframe': self.timeframe,
            'data_availability': data_availability,
            'adjusted_weights': adjusted_weights
        }
    
    def _check_buy_criteria(self, comprehensive_score: float, technical_score: float,
                           indicators: Dict[str, float], confidence: float) -> bool:
        """
        Check if buy criteria are met with balanced approach.
        
        Args:
            comprehensive_score: Overall weighted score
            technical_score: Technical analysis score
            indicators: Technical indicators
            confidence: Confidence level
            
        Returns:
            True if buy criteria are met
        """
        # Must have positive momentum
        if indicators.get('return_1m', 0) <= 0:
            return False
        
        # Balanced score requirements
        min_comp_score = 0.5
        min_tech_score = 0.4
        
        # Only apply stricter requirements for very low confidence
        if confidence < self.min_confidence * 0.5:
            min_comp_score += 0.1
            min_tech_score += 0.1
        
        if comprehensive_score < min_comp_score or technical_score < min_tech_score:
            return False
        
        # Must be above key moving average
        if indicators['close'] < indicators['ma_50']:
            return False
        
        # RSI must be in acceptable range
        if not (self.rsi_min <= indicators['rsi'] <= self.rsi_max):
            return False
        
        # More lenient confidence requirement
        if confidence < self.min_confidence * 0.5:  # Only reject if confidence is very low
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
        
        # Sort by comprehensive score (not multiplied by confidence for better balance)
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
                        'rank': i + 1,
                        'confidence': analysis['confidence']
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
                            'reason': 'sell_criteria_met',
                            'confidence': analysis['confidence']
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
        
        # Sort by comprehensive score (not multiplied by confidence)
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
