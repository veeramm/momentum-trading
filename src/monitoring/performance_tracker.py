"""
Performance Tracking Module

This module provides performance tracking and metrics calculation
for the momentum trading system.
"""

from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
from loguru import logger


class PerformanceTracker:
    """
    Tracks and calculates performance metrics for trading strategies.
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.metrics_history = []
        
    def update(self, portfolio, timestamp: datetime):
        """
        Update performance tracking with current portfolio state.
        
        Args:
            portfolio: Portfolio instance
            timestamp: Current timestamp
        """
        # Record equity curve point
        equity_point = {
            'timestamp': timestamp,
            'equity': portfolio.total_equity,
            'cash': portfolio.cash,
            'positions_value': portfolio.total_equity - portfolio.cash,
            'num_positions': len(portfolio.positions),
            'pnl': portfolio.total_pnl,
            'pnl_pct': portfolio.total_pnl_pct
        }
        self.equity_curve.append(equity_point)
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (portfolio.total_equity - prev_equity) / prev_equity
            self.daily_returns.append({
                'timestamp': timestamp,
                'return': daily_return
            })
    
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_curve:
            return {}
        
        # Convert to DataFrames for easier calculation
        equity_df = pd.DataFrame(self.equity_curve)
        returns_df = pd.DataFrame(self.daily_returns) if self.daily_returns else pd.DataFrame()
        
        # Basic metrics
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Time metrics
        start_date = equity_df['timestamp'].iloc[0]
        end_date = equity_df['timestamp'].iloc[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Annualized metrics
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        if not returns_df.empty:
            returns = returns_df['return']
            volatility = returns.std() * np.sqrt(252)
            
            # Downside risk
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Sharpe and Sortino ratios
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
            
            # Maximum drawdown
            peak = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # Recovery time
            max_dd_idx = drawdown.idxmin()
            if max_dd_idx < len(drawdown) - 1:
                recovery_idx = drawdown[max_dd_idx:].idxmax()
                recovery_time = (equity_df.loc[recovery_idx, 'timestamp'] - 
                               equity_df.loc[max_dd_idx, 'timestamp']).days
            else:
                recovery_time = None
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win/loss statistics
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
            
            # Profit factor
            total_wins = positive_returns.sum()
            total_losses = abs(negative_returns.sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Monthly returns analysis
            if len(returns_df) > 30:
                monthly_returns = returns_df.set_index('timestamp')['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
                best_month = monthly_returns.max()
                worst_month = monthly_returns.min()
                positive_months = (monthly_returns > 0).sum()
                total_months = len(monthly_returns)
                monthly_win_rate = positive_months / total_months if total_months > 0 else 0
            else:
                best_month = worst_month = monthly_win_rate = 0
            
        else:
            # Default values if no returns data
            volatility = sharpe_ratio = sortino_ratio = max_drawdown = calmar_ratio = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
            best_month = worst_month = monthly_win_rate = 0
            recovery_time = None
        
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
            'best_month': best_month,
            'worst_month': worst_month,
            'monthly_win_rate': monthly_win_rate,
            'recovery_time': recovery_time,
            'days_traded': days,
            'years_traded': years
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_returns_dataframe(self) -> pd.DataFrame:
        """Get daily returns as DataFrame"""
        return pd.DataFrame(self.daily_returns)
    
    def plot_equity_curve(self) -> None:
        """Plot equity curve (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            equity_df = self.get_results_dataframe()
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['timestamp'], equity_df['equity'], label='Portfolio Equity')
            plt.fill_between(equity_df['timestamp'], 
                           equity_df['cash'], 
                           equity_df['equity'], 
                           alpha=0.3, label='Positions Value')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio Equity Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed, cannot plot equity curve")
    
    def generate_report(self) -> str:
        """Generate a text performance report"""
        metrics = self.calculate_metrics()
        
        report = f"""
Performance Report
==================

Period: {metrics.get('years_traded', 0):.2f} years ({metrics.get('days_traded', 0)} days)

Returns:
--------
Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Volatility: {metrics['volatility']:.2%}

Risk-Adjusted Returns:
---------------------
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Calmar Ratio: {metrics['calmar_ratio']:.2f}

Drawdown:
---------
Maximum Drawdown: {metrics['max_drawdown']:.2%}
Recovery Time: {metrics.get('recovery_time', 'N/A')} days

Win/Loss Statistics:
-------------------
Win Rate: {metrics['win_rate']:.2%}
Average Win: {metrics['avg_win']:.2%}
Average Loss: {metrics['avg_loss']:.2%}
Profit Factor: {metrics['profit_factor']:.2f}

Monthly Performance:
-------------------
Best Month: {metrics['best_month']:.2%}
Worst Month: {metrics['worst_month']:.2%}
Monthly Win Rate: {metrics['monthly_win_rate']:.2%}
"""
        return report
