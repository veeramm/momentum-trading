"""
Advanced Momentum Strategy

This module implements a comprehensive momentum strategy with multiple
indicators for both intermediate and long-term timeframes.
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


class AdvancedMomentumStrategy(BaseStrategy):
    """
    Advanced momentum strategy implementation with multiple indicators
    and timeframes (intermediate and long-term).
    """
    
    def __init__(self, config: dict, portfolio_manager=None, risk_manager=None):
        super().__init__(config, portfolio_manager, risk_manager)
        
        # Strategy parameters
        self.timeframe = config.get('timeframe', 'intermediate')  # 'intermediate' or 'long_term'
        self.top_n = config.get('top_n', 20)
        
        # Indicator periods
        self.ma_short = config.get('ma_short', 20)
        self.ma_medium = config.get('ma_medium', 50)
        self.ma_long = config.get('ma_long', 200)
        
        # Holding periods
        self.min_holding_period = config.get('min_holding_period', 14)  # days
        self.max_holding_period = config.get('max_holding_period', 180)  # days
        
        # Entry criteria
        self.min_price_performance = config.get('min_price_performance', 0.10)  # 10%
        self.volume_surge_threshold = config.get('volume_surge_threshold', 1.5)  # 50% above average
        self.rsi_min = config.get('rsi_min', 50)
        self.rsi_max = config.get('rsi_max', 80)
        
        # Exit criteria
        self.trailing_stop_ma = config.get('trailing_stop_ma', 20)  # for intermediate
        self.weekly_ma_exit = config.get('weekly_ma_exit', 50)  # for long-term
        
        # Track entry dates for holding period
        self.entry_dates = {}
    
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
        
        # Calculate basic momentum for compatibility with base class
        if len(close_data) < self.lookback_period:
            return pd.Series([np.nan])
        
        # Calculate momentum score based on multiple indicators
        indicators = self.calculate_indicators(data)
        
        # Create composite momentum score
        momentum_score = 0.0
        
        # Price momentum component (40%)
        if 'return_1m' in indicators:
            momentum_score += indicators['return_1m'] * 0.4
        
        # Technical strength component (30%)
        if 'rsi' in indicators and self.rsi_min <= indicators['rsi'] <= self.rsi_max:
            momentum_score += 0.3
        
        # Trend alignment component (20%)
        if indicators.get('close', 0) > indicators.get('ma_50', 0):
            momentum_score += 0.2
        
        # Volume confirmation component (10%)
        if indicators.get('volume_ratio', 0) > 1.0:
            momentum_score += 0.1
        
        return pd.Series([momentum_score])
    
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
        
        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()
        indicators['obv'] = obv.iloc[-1]
        indicators['obv_ma_20'] = obv.rolling(window=20).mean().iloc[-1]
        
        # VWAP (simplified daily VWAP)
        typical_price = (data['high'] + data['low'] + close) / 3
        if isinstance(typical_price, pd.DataFrame):
            typical_price = typical_price.iloc[:, 0]
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        indicators['vwap'] = vwap.iloc[-1]
        
        # Price performance (1 month and 3 months)
        if len(close) >= 21:
            indicators['return_1m'] = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
        else:
            indicators['return_1m'] = 0
            
        if len(close) >= 63:
            indicators['return_3m'] = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
        else:
            indicators['return_3m'] = 0
        
        # Trend strength (ADX simplified)
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
    
    def check_buy_criteria(self, indicators: Dict[str, float], symbol: str) -> Tuple[bool, float]:
        """Check if stock meets buy criteria and return signal strength"""
        score = 0
        max_score = 100
        
        # Price above moving averages (30 points)
        if indicators['close'] > indicators['ma_20']:
            score += 10
        if indicators['close'] > indicators['ma_50']:
            score += 10
        if indicators['close'] > indicators['ma_200']:
            score += 10
            
        # Moving average alignment (10 points)
        if indicators['ma_20'] > indicators['ma_50'] > indicators['ma_200']:
            score += 10
        
        # RSI in momentum range (10 points)
        if self.rsi_min <= indicators['rsi'] <= self.rsi_max:
            score += 10
        
        # MACD bullish (10 points)
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
            score += 10
        
        # Near 52-week high (10 points)
        if indicators['pct_from_52w_high'] > -0.05:  # Within 5% of 52-week high
            score += 10
        
        # Volume confirmation (10 points)
        if indicators['volume_ratio'] > self.volume_surge_threshold:
            score += 10
        
        # OBV trend (5 points)
        if indicators['obv'] > indicators['obv_ma_20']:
            score += 5
        
        # Price above VWAP (5 points)
        if indicators['close'] > indicators['vwap']:
            score += 5
        
        # Price performance (10 points)
        if indicators['return_1m'] > self.min_price_performance:
            score += 10
        
        # Minimum criteria check
        min_criteria_met = (
            indicators['close'] > indicators['ma_50'] and
            indicators['rsi'] > self.rsi_min and
            indicators['return_1m'] > 0 and
            indicators['volume_ratio'] > 1.0
        )
        
        signal_strength = score / max_score
        return min_criteria_met and score >= 50, signal_strength
    
    def check_sell_criteria(self, indicators: Dict[str, float], symbol: str, 
                           entry_date: datetime, current_date: datetime) -> bool:
        """Check if position should be closed"""
        days_held = (current_date - entry_date).days
        
        # Force exit if maximum holding period exceeded
        if days_held > self.max_holding_period:
            return True
        
        if self.timeframe == 'intermediate':
            # Exit if price closes below trailing stop MA
            if indicators['close'] < indicators['ma_20']:
                return True
            
            # Exit on RSI divergence or extreme overbought
            if indicators['rsi'] > 80:
                return True
            
            # Exit on MACD bearish crossover
            if indicators['macd'] < indicators['macd_signal']:
                return True
                
        else:  # long_term
            # Use weekly equivalent (5x daily)
            ma_weekly = indicators[f'ma_{self.weekly_ma_exit}']
            
            # Exit if price closes below key MA
            if indicators['close'] < ma_weekly:
                return True
            
            # Exit on major trend reversal
            if indicators['ma_20'] < indicators['ma_50']:
                return True
            
            # Exit if far from 52-week high and momentum weakening
            if indicators['pct_from_52w_high'] < -0.20 and indicators['rsi'] < 40:
                return True
        
        # Exit on volume divergence (price up, volume down significantly)
        if indicators['volume_ratio'] < 0.5 and indicators['return_1m'] > 0:
            return True
        
        return False
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals based on advanced momentum criteria"""
        signals = []
        current_time = datetime.now()
        
        # Calculate indicators for all stocks
        stock_scores = {}
        for symbol, data in market_data.items():
            if len(data) < 252:  # Need at least 1 year of data
                continue
            
            try:
                indicators = self.calculate_indicators(data)
                buy_signal, signal_strength = self.check_buy_criteria(indicators, symbol)
                
                if buy_signal:
                    stock_scores[symbol] = {
                        'score': signal_strength,
                        'indicators': indicators,
                        'momentum': indicators['return_1m']
                    }
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                continue
        
        # Rank by score and momentum
        ranked_stocks = sorted(stock_scores.items(), 
                             key=lambda x: (x[1]['score'], x[1]['momentum']), 
                             reverse=True)
        
        # Generate buy signals for top N stocks
        for i, (symbol, data) in enumerate(ranked_stocks[:self.top_n]):
            if symbol not in self.positions:
                signal = Signal(
                    symbol=symbol,
                    direction='long',
                    strength=data['score'],
                    timestamp=current_time,
                    metadata={
                        'indicators': data['indicators'],
                        'rank': i + 1,
                        'timeframe': self.timeframe
                    }
                )
                signals.append(signal)
                self.entry_dates[symbol] = current_time
        
        # Check sell signals for existing positions
        for symbol, position in self.positions.items():
            if symbol in market_data:
                try:
                    indicators = self.calculate_indicators(market_data[symbol])
                    entry_date = self.entry_dates.get(symbol, position.entry_time)
                    
                    if self.check_sell_criteria(indicators, symbol, entry_date, current_time):
                        signal = Signal(
                            symbol=symbol,
                            direction='neutral',
                            strength=0.0,
                            timestamp=current_time,
                            metadata={
                                'reason': 'sell_criteria_met',
                                'indicators': indicators,
                                'days_held': (current_time - entry_date).days
                            }
                        )
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error checking sell criteria for {symbol}: {e}")
        
        return signals
    
    def get_strategy_specific_metrics(self) -> Dict[str, float]:
        """Get metrics specific to advanced momentum strategy"""
        if not self.positions:
            return {}
        
        total_score = 0
        total_momentum = 0
        positions_near_high = 0
        
        for symbol, position in self.positions.items():
            if position.metadata and 'indicators' in position.metadata:
                indicators = position.metadata['indicators']
                total_score += position.metadata.get('score', 0)
                total_momentum += indicators.get('return_1m', 0)
                
                if indicators.get('pct_from_52w_high', -1) > -0.10:
                    positions_near_high += 1
        
        num_positions = len(self.positions)
        
        return {
            'average_signal_score': total_score / num_positions if num_positions > 0 else 0,
            'average_momentum': total_momentum / num_positions if num_positions > 0 else 0,
            'pct_near_52w_high': positions_near_high / num_positions if num_positions > 0 else 0,
            'positions_count': num_positions,
            'strategy_timeframe': self.timeframe
        }
