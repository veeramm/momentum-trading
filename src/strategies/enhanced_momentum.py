"""
Enhanced Momentum Strategy

This module implements an enhanced momentum strategy that integrates
technical, fundamental, and sentiment analysis with cohesive action determination.
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


class EnhancedMomentumStrategy(BaseStrategy):
    """
    Enhanced momentum strategy with integrated action determination
    and advanced data utilization.
    """
    
    def __init__(self, config: dict, portfolio_manager=None, risk_manager=None):
        super().__init__(config, portfolio_manager, risk_manager)
        
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
        self.technical_weight = config.get('technical_weight', 0.7)
        self.fundamental_weight = config.get('fundamental_weight', 0.2)
        self.sentiment_weight = config.get('sentiment_weight', 0.1)
        
    def calculate_momentum_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for assets.
        Required implementation of abstract method from BaseStrategy.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with momentum scores
        """
        # Handle duplicate columns
        close_data = data['close']
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        
        # Calculate basic momentum
        if len(close_data) < self.lookback_period:
            return pd.Series([np.nan])
        
        # Calculate indicators for the full dataset
        indicators = self.calculate_indicators(data)
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(indicators)
        
        # For now, use technical score as the momentum score
        # (in the enhanced version, this would integrate with fundamental and sentiment)
        return pd.Series([technical_score])
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all momentum indicators for a single stock"""
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
        indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_ma_20']
        
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
    
    def _determine_action(self, technical_score: float, indicators: Dict[str, float]) -> str:
        """
        Determine trading action based on technical analysis.
        
        Args:
            technical_score: Technical analysis score
            indicators: Technical indicators
            
        Returns:
            Action: 'BUY', 'HOLD', or 'SELL'
        """
        # Strong buy threshold
        if technical_score >= 0.7:
            return 'BUY'
        
        # Buy threshold
        elif technical_score >= 0.6:
            # Additional checks for buy
            if (indicators['close'] > indicators['ma_50'] and 
                self.rsi_min <= indicators['rsi'] <= self.rsi_max and
                indicators.get('volume_ratio', 1.0) > 1.0):
                return 'BUY'
        
        # Sell conditions
        elif technical_score < 0.3:
            return 'SELL'
        
        # Specific sell triggers
        elif (indicators['rsi'] > 80 or  # Extremely overbought
              indicators['close'] < indicators['ma_50'] or  # Below key MA
              indicators.get('pct_from_52w_high', 0) < -0.25):  # 25% below high
            return 'SELL'
        
        # Default to hold
        return 'HOLD'
    
    def _check_buy_criteria(self, technical_score: float, indicators: Dict[str, float]) -> bool:
        """
        Check if buy criteria are met.
        
        Args:
            technical_score: Technical analysis score
            indicators: Technical indicators
            
        Returns:
            True if buy criteria are met
        """
        # Must have positive momentum
        if indicators.get('return_1m', 0) <= 0:
            return False
        
        # Must have minimum scores
        if technical_score < 0.4:
            return False
        
        # Must be above key moving average
        if indicators['close'] < indicators['ma_50']:
            return False
        
        # RSI must be in acceptable range
        if not (self.rsi_min <= indicators['rsi'] <= self.rsi_max):
            return False
        
        return True
    
    def analyze_stock(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Analyze a single stock and return comprehensive analysis.
        
        Args:
            symbol: Stock symbol
            market_data: Market data DataFrame
            
        Returns:
            Dictionary containing analysis results
        """
        # Calculate technical indicators
        indicators = self.calculate_indicators(market_data)
        technical_score = self._calculate_technical_score(indicators)
        
        # Determine action based on technical analysis
        action = self._determine_action(technical_score, indicators)
        
        # Check if buy criteria are met
        buy_criteria_met = self._check_buy_criteria(technical_score, indicators)
        
        return {
            'symbol': symbol,
            'technical_score': technical_score,
            'action': action,
            'buy_criteria_met': buy_criteria_met,
            'indicators': indicators,
            'timeframe': self.timeframe,
            'signal_strength': technical_score
        }
    
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
                analysis = self.analyze_stock(symbol, data)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by technical score
        analyses.sort(key=lambda x: x['technical_score'], reverse=True)
        
        # Generate signals
        signals = []
        current_time = datetime.now()
        
        # Buy signals for top stocks
        for i, analysis in enumerate(analyses[:self.top_n]):
            if analysis['action'] == 'BUY' and analysis['buy_criteria_met']:
                signal = Signal(
                    symbol=analysis['symbol'],
                    direction='long',
                    strength=analysis['technical_score'],
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
                analysis = self.analyze_stock(symbol, market_data[symbol])
                
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
                analysis = self.analyze_stock(symbol, data)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by technical score
        analyses.sort(key=lambda x: x['technical_score'], reverse=True)
        
        return analyses
