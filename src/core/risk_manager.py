"""
Risk Management Module

This module provides risk management functionality including
position sizing, risk limits, and portfolio constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from loguru import logger

from .base_strategy import Signal


class RiskManager:
    """
    Risk manager that enforces risk limits and calculates position sizes.
    """
    
    def __init__(self, config: dict):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration dictionary
        """
        self.config = config
        
        # Risk limits
        self.max_position_pct = config.get('max_position_pct', 0.10)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.30)
        self.max_correlation = config.get('max_correlation', 0.70)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        
        # Position sizing parameters
        self.position_sizing = config.get('position_sizing', {})
        self.sizing_method = self.position_sizing.get('method', 'risk_parity')
        self.volatility_target = self.position_sizing.get('volatility_target', 0.10)
        self.lookback_period = self.position_sizing.get('lookback_period', 252)
        
        logger.info(f"Initialized risk manager with method: {self.sizing_method}")
    
    def calculate_position_size(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        portfolio
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            market_data: Market data for the asset
            portfolio: Portfolio instance
            
        Returns:
            Position size in units
        """
        # Get current price
        current_price = market_data['close'].iloc[-1]
        
        # Calculate base position value
        max_position_value = portfolio.total_equity * self.max_position_pct
        
        # Apply sizing method
        if self.sizing_method == 'equal_weight':
            position_value = max_position_value
            
        elif self.sizing_method == 'risk_parity':
            position_value = self._risk_parity_sizing(
                market_data, 
                max_position_value,
                self.volatility_target
            )
            
        elif self.sizing_method == 'kelly':
            position_value = self._kelly_sizing(
                signal,
                market_data,
                max_position_value
            )
            
        else:
            raise ValueError(f"Unknown sizing method: {self.sizing_method}")
        
        # Adjust by signal strength
        position_value *= signal.strength
        
        # Convert to units
        position_size = position_value / current_price
        
        # Apply risk checks
        position_size = self._apply_risk_limits(
            position_size,
            signal.symbol,
            current_price,
            portfolio
        )
        
        return int(position_size)
    
    def _risk_parity_sizing(
        self,
        market_data: pd.DataFrame,
        max_value: float,
        target_vol: float
    ) -> float:
        """Calculate position size using risk parity approach"""
        # Calculate asset volatility
        returns = market_data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Size inversely proportional to volatility
        if volatility > 0:
            position_value = (target_vol / volatility) * max_value
            # Cap at maximum position value
            position_value = min(position_value, max_value)
        else:
            position_value = max_value
        
        return position_value
    
    def _kelly_sizing(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        max_value: float
    ) -> float:
        """Calculate position size using Kelly criterion"""
        # Estimate win probability and payoff
        returns = market_data['close'].pct_change()
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            win_prob = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            # Kelly fraction
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
            
            # Apply Kelly fraction (often reduced for safety)
            position_value = max_value * kelly_fraction * 0.25  # Quarter Kelly
        else:
            position_value = max_value * 0.5  # Default to half position
        
        return position_value
    
    def _apply_risk_limits(
        self,
        position_size: float,
        symbol: str,
        price: float,
        portfolio
    ) -> float:
        """Apply risk limits to position size"""
        # Check maximum position size
        position_value = position_size * price
        max_allowed_value = portfolio.total_equity * self.max_position_pct
        
        if position_value > max_allowed_value:
            position_size = max_allowed_value / price
            logger.debug(f"Position size limited by max position rule: {symbol}")
        
        # Check portfolio drawdown
        current_drawdown = (portfolio.initial_capital - portfolio.total_equity) / portfolio.initial_capital
        if current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            position_size *= 0.5  # Reduce position size
            logger.warning(f"Position size reduced due to drawdown: {symbol}")
        
        # Ensure we have enough cash
        required_cash = position_size * price
        if required_cash > portfolio.cash:
            position_size = portfolio.cash / price
            logger.debug(f"Position size limited by available cash: {symbol}")
        
        return position_size
    
    def validate_trade(self, trade: Dict, portfolio) -> bool:
        """
        Validate if a trade meets risk requirements.
        
        Args:
            trade: Trade details
            portfolio: Portfolio instance
            
        Returns:
            True if trade is valid
        """
        # Check if we're within drawdown limits
        current_drawdown = (portfolio.initial_capital - portfolio.total_equity) / portfolio.initial_capital
        if current_drawdown > self.max_drawdown:
            logger.warning("Trade rejected: Maximum drawdown exceeded")
            return False
        
        # Check position concentration
        position_value = trade['quantity'] * trade['price']
        position_pct = position_value / portfolio.total_equity
        
        if position_pct > self.max_position_pct:
            logger.warning(f"Trade rejected: Position too large ({position_pct:.1%})")
            return False
        
        return True
    
    def calculate_portfolio_risk(self, portfolio) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics.
        
        Args:
            portfolio: Portfolio instance
            
        Returns:
            Dictionary of risk metrics
        """
        if not portfolio.positions:
            return {
                'portfolio_volatility': 0,
                'portfolio_var': 0,
                'concentration_risk': 0,
                'drawdown': 0
            }
        
        # Calculate position weights
        total_value = portfolio.total_equity
        weights = {}
        returns_data = {}
        
        # Placeholder for actual returns calculation
        # In practice, you'd fetch historical returns for each position
        portfolio_returns = []
        
        # Calculate concentration risk (Herfindahl index)
        concentration_risk = 0
        for symbol, position in portfolio.positions.items():
            weight = position.market_value / total_value
            weights[symbol] = weight
            concentration_risk += weight ** 2
        
        # Calculate current drawdown
        drawdown = (portfolio.initial_capital - portfolio.total_equity) / portfolio.initial_capital
        
        return {
            'portfolio_volatility': 0,  # Placeholder
            'portfolio_var': 0,  # Placeholder
            'concentration_risk': concentration_risk,
            'drawdown': drawdown,
            'largest_position': max(weights.values()) if weights else 0,
            'num_positions': len(portfolio.positions)
        }
    
    def check_rebalancing_needed(self, portfolio, target_weights: Dict[str, float]) -> bool:
        """
        Check if portfolio needs rebalancing.
        
        Args:
            portfolio: Portfolio instance
            target_weights: Target weight for each asset
            
        Returns:
            True if rebalancing is needed
        """
        threshold = self.config.get('rebalancing', {}).get('threshold', 0.05)
        
        # Calculate current weights
        total_value = portfolio.total_equity
        current_weights = {}
        
        for symbol, position in portfolio.positions.items():
            current_weights[symbol] = position.market_value / total_value
        
        # Check deviation from target
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > threshold:
                logger.info(f"Rebalancing needed for {symbol}: {deviation:.1%} deviation")
                return True
        
        return False
    
    def calculate_stop_loss(self, entry_price: float, position_type: str = 'long') -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        if position_type == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
