"""
Base Strategy Class for Momentum Trading

This module provides the abstract base class for all momentum trading strategies.
It defines the interface and common functionality that all strategies must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.validators import validate_market_data


@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    direction: str  # 'long', 'short', or 'neutral'
    strength: float  # Signal strength between 0 and 1
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.direction not in ['long', 'short', 'neutral']:
            raise ValueError(f"Invalid direction: {self.direction}")
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Signal strength must be between 0 and 1, got {self.strength}")


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    position_type: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None
    
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


class BaseStrategy(ABC):
    """
    Abstract base class for all momentum trading strategies.
    
    This class provides the framework for implementing momentum-based
    trading strategies with risk management and position sizing.
    """
    
    def __init__(self, config: dict, portfolio_manager=None, risk_manager=None):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration dictionary
            portfolio_manager: Portfolio management instance
            risk_manager: Risk management instance
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        
        # Strategy parameters
        self.name = config.get('name', self.__class__.__name__)
        self.universe = config.get('universe', [])
        self.lookback_period = config.get('lookback_period', 252)
        self.rebalance_frequency = config.get('rebalance_frequency', 'monthly')
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', None)
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.last_rebalance_date: Optional[datetime] = None
        
        logger.info(f"Initialized {self.name} strategy")
    
    @abstractmethod
    def calculate_momentum_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for each asset.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Series with momentum scores indexed by symbol
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on momentum analysis.
        
        Args:
            market_data: Dictionary of market data by symbol
            
        Returns:
            List of trading signals
        """
        pass
    
    def filter_universe(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Filter the trading universe based on criteria.
        
        Args:
            market_data: Dictionary of market data by symbol
            
        Returns:
            List of symbols that pass the filters
        """
        filtered_symbols = []
        
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                logger.debug(f"Insufficient data for {symbol}")
                continue
                
            # Volume filter
            if 'volume' in data.columns:
                # Handle duplicate columns by taking first column if multiple exist
                volume_data = data['volume']
                if isinstance(volume_data, pd.DataFrame):
                    volume_data = volume_data.iloc[:, 0]
                
                avg_volume_series = volume_data.rolling(20).mean()
                avg_volume = float(avg_volume_series.iloc[-1]) if len(avg_volume_series) > 0 and not pd.isna(avg_volume_series.iloc[-1]) else 0
                min_volume = self.config.get('min_volume', 1000000)
                if avg_volume < min_volume:
                    logger.debug(f"Low volume for {symbol}: {avg_volume}")
                    continue
            
            # Price filter
            close_data = data['close']
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
            
            current_price = float(close_data.iloc[-1]) if len(close_data) > 0 and not pd.isna(close_data.iloc[-1]) else 0
            min_price = self.config.get('min_price', 5.0)
            if current_price < min_price:
                logger.debug(f"Low price for {symbol}: {current_price}")
                continue
            
            filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def calculate_position_size(self, signal: Signal, market_data: pd.DataFrame) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal: Trading signal
            market_data: Market data for the symbol
            
        Returns:
            Position size in units of the asset
        """
        if self.risk_manager:
            # Use risk manager for position sizing
            return self.risk_manager.calculate_position_size(
                signal=signal,
                market_data=market_data,
                portfolio=self.portfolio_manager
            )
        
        # Simple position sizing based on signal strength
        max_position_value = self.portfolio_manager.total_equity * self.max_position_size
        current_price = market_data['close'].iloc[-1]
        base_position_size = max_position_value / current_price
        
        # Adjust by signal strength
        position_size = base_position_size * signal.strength
        
        # Apply volatility scaling if configured
        if self.config.get('volatility_scaling', False):
            volatility = market_data['close'].pct_change().std() * np.sqrt(252)
            target_volatility = self.config.get('target_volatility', 0.15)
            volatility_scalar = min(target_volatility / volatility, 1.0)
            position_size *= volatility_scalar
        
        return int(position_size)
    
    def should_rebalance(self, current_date: datetime) -> bool:
        """
        Check if portfolio should be rebalanced.
        
        Args:
            current_date: Current date
            
        Returns:
            True if rebalancing is needed
        """
        if self.last_rebalance_date is None:
            return True
        
        if self.rebalance_frequency == 'daily':
            return True
        elif self.rebalance_frequency == 'weekly':
            return (current_date - self.last_rebalance_date).days >= 7
        elif self.rebalance_frequency == 'monthly':
            return (current_date.month != self.last_rebalance_date.month or
                    current_date.year != self.last_rebalance_date.year)
        elif self.rebalance_frequency == 'quarterly':
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (self.last_rebalance_date.month - 1) // 3
            return (current_quarter != last_quarter or
                    current_date.year != self.last_rebalance_date.year)
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")
    
    def execute_signals(self, signals: List[Signal], market_data: Dict[str, pd.DataFrame]):
        """
        Execute trading signals to update positions.
        
        Args:
            signals: List of trading signals
            market_data: Current market data
        """
        for signal in signals:
            symbol = signal.symbol
            current_data = market_data.get(symbol)
            
            if current_data is None:
                logger.warning(f"No market data for {symbol}")
                continue
            
            current_price = current_data['close'].iloc[-1]
            
            # Check if we have an existing position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Update position with current price
                position.current_price = current_price
                
                # Check for exit conditions
                if signal.direction == 'neutral' or self._check_exit_conditions(position):
                    self._close_position(symbol)
                    continue
            
            # Open new position
            if signal.direction != 'neutral' and symbol not in self.positions:
                position_size = self.calculate_position_size(signal, current_data)
                
                if position_size > 0:
                    self._open_position(
                        symbol=symbol,
                        direction=signal.direction,
                        size=position_size,
                        price=current_price
                    )
    
    def _open_position(self, symbol: str, direction: str, size: float, price: float):
        """Open a new position"""
        stop_loss = None
        if self.stop_loss_pct:
            if direction == 'long':
                stop_loss = price * (1 - self.stop_loss_pct)
            else:
                stop_loss = price * (1 + self.stop_loss_pct)
        
        take_profit = None
        if self.take_profit_pct:
            if direction == 'long':
                take_profit = price * (1 + self.take_profit_pct)
            else:
                take_profit = price * (1 - self.take_profit_pct)
        
        position = Position(
            symbol=symbol,
            quantity=size,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price,
            position_type=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        logger.info(f"Opened {direction} position in {symbol}: {size} @ {price}")
    
    def _close_position(self, symbol: str):
        """Close an existing position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            pnl = position.unrealized_pnl
            pnl_pct = position.unrealized_pnl_pct
            
            del self.positions[symbol]
            logger.info(f"Closed position in {symbol}: P&L {pnl:.2f} ({pnl_pct:.2%})")
    
    def _check_exit_conditions(self, position: Position) -> bool:
        """Check if position should be closed"""
        # Stop loss check
        if position.stop_loss:
            if position.position_type == 'long' and position.current_price <= position.stop_loss:
                logger.info(f"Stop loss triggered for {position.symbol}")
                return True
            elif position.position_type == 'short' and position.current_price >= position.stop_loss:
                logger.info(f"Stop loss triggered for {position.symbol}")
                return True
        
        # Take profit check
        if position.take_profit:
            if position.position_type == 'long' and position.current_price >= position.take_profit:
                logger.info(f"Take profit triggered for {position.symbol}")
                return True
            elif position.position_type == 'short' and position.current_price <= position.take_profit:
                logger.info(f"Take profit triggered for {position.symbol}")
                return True
        
        return False
    
    def update(self, market_data: Dict[str, pd.DataFrame], current_date: datetime):
        """
        Update strategy with new market data.
        
        Args:
            market_data: Current market data
            current_date: Current date
        """
        # Update existing positions with current prices
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]['close'].iloc[-1]
        
        # Check if rebalancing is needed
        if self.should_rebalance(current_date):
            # Generate new signals
            signals = self.generate_signals(market_data)
            self.signals = signals
            
            # Execute signals
            self.execute_signals(signals, market_data)
            
            # Update last rebalance date
            self.last_rebalance_date = current_date
        else:
            # Just check exit conditions for existing positions
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                if self._check_exit_conditions(position):
                    self._close_position(symbol)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        return {
            'total_pnl': total_pnl,
            'total_value': total_value,
            'num_positions': len(self.positions),
            'avg_pnl_pct': np.mean([pos.unrealized_pnl_pct for pos in self.positions.values()]) if self.positions else 0
        }
