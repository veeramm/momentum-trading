"""
Portfolio Management Module

This module handles portfolio tracking, position management,
and performance calculations.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .base_strategy import Position


class Portfolio:
    """
    Portfolio manager that tracks positions, cash, and performance.
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []
        
        logger.info(f"Initialized portfolio with ${initial_capital:,.2f}")
    
    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L"""
        return self.total_equity - self.initial_capital
    
    @property
    def total_pnl_pct(self) -> float:
        """Calculate total P&L percentage"""
        return self.total_pnl / self.initial_capital
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        position_type: str = 'long',
        timestamp: Optional[datetime] = None
    ) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Asset symbol
            quantity: Position size
            price: Entry price
            position_type: 'long' or 'short'
            timestamp: Entry timestamp
            
        Returns:
            Created position
        """
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")
        
        # Calculate position cost
        position_cost = quantity * price
        
        # Check if we have enough cash
        if position_cost > self.cash:
            raise ValueError(f"Insufficient cash for position: need ${position_cost:.2f}, have ${self.cash:.2f}")
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp or datetime.now(),
            current_price=price,
            position_type=position_type
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash -= position_cost
        
        # Record trade
        self.trade_history.append({
            'timestamp': position.entry_time,
            'symbol': symbol,
            'action': 'buy' if position_type == 'long' else 'sell',
            'quantity': quantity,
            'price': price,
            'value': position_cost,
            'type': 'open'
        })
        
        logger.info(f"Opened {position_type} position: {symbol} x{quantity} @ ${price:.2f}")
        return position
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Close an existing position.
        
        Args:
            symbol: Asset symbol
            price: Exit price
            timestamp: Exit timestamp
            
        Returns:
            Trade summary
        """
        if symbol not in self.positions:
            raise ValueError(f"No position exists for {symbol}")
        
        position = self.positions[symbol]
        position.current_price = price
        
        # Calculate P&L
        pnl = position.unrealized_pnl
        pnl_pct = position.unrealized_pnl_pct
        
        # Update cash
        exit_value = position.quantity * price
        self.cash += exit_value
        
        # Move to closed positions
        del self.positions[symbol]
        self.closed_positions.append(position)
        
        # Record trade
        trade_summary = {
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'sell' if position.position_type == 'long' else 'buy',
            'quantity': position.quantity,
            'price': price,
            'value': exit_value,
            'type': 'close',
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        self.trade_history.append(trade_summary)
        
        logger.info(f"Closed position: {symbol} P&L=${pnl:.2f} ({pnl_pct:.2%})")
        return trade_summary
    
    def update_prices(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """
        Update positions with current market prices.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                position.current_price = current_price
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.total_equity,
            'cash': self.cash,
            'positions_value': self.total_equity - self.cash,
            'pnl': self.total_pnl,
            'pnl_pct': self.total_pnl_pct
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (self.total_equity - prev_equity) / prev_equity
            self.daily_returns.append({
                'timestamp': timestamp,
                'return': daily_return
            })
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'position_type': position.position_type
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history DataFrame"""
        return pd.DataFrame(self.trade_history)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_daily_returns(self) -> pd.DataFrame:
        """Get daily returns DataFrame"""
        return pd.DataFrame(self.daily_returns)
    
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.daily_returns:
            return {}
        
        returns_df = pd.DataFrame(self.daily_returns)
        returns = returns_df['return']
        
        # Basic metrics
        total_return = self.total_pnl_pct
        days = len(returns)
        years = days / 252
        
        # Annualized metrics
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # Risk metrics
        volatility = returns.std() * (252 ** 0.5)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
        
        # Sharpe and Sortino ratios
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        equity_curve = pd.DataFrame(self.equity_curve)
        peak = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win/loss metrics
        trades = [t for t in self.trade_history if t.get('type') == 'close']
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and winning_trades else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'days_traded': days
        }
