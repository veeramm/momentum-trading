"""
Momentum Indicators Module

This module provides various momentum indicators and calculations
for use in momentum trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from loguru import logger


def calculate_momentum(
    prices: pd.Series,
    period: int,
    method: str = 'return'
) -> pd.Series:
    """
    Calculate basic momentum indicator.
    
    Args:
        prices: Price series
        period: Lookback period
        method: Calculation method ('return', 'log_return', 'rate_of_change')
        
    Returns:
        Momentum series
    """
    # Handle DataFrames with duplicate columns
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]  # Take first column
    
    if method == 'return':
        momentum = (prices - prices.shift(period)) / prices.shift(period)
    elif method == 'log_return':
        momentum = np.log(prices / prices.shift(period))
    elif method == 'rate_of_change':
        momentum = prices / prices.shift(period) - 1
    else:
        raise ValueError(f"Unknown momentum method: {method}")
        
    return momentum


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI series
    """
    # Handle DataFrames with duplicate columns
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]  # Take first column
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_rate_of_change(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        ROC series
    """
    return (prices - prices.shift(period)) / prices.shift(period) * 100


def calculate_tsi(
    prices: pd.Series,
    fast_period: int = 25,
    slow_period: int = 13
) -> pd.Series:
    """
    Calculate True Strength Index (TSI).
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        TSI series
    """
    momentum = prices.diff()
    
    # Double smoothed momentum
    ema_momentum = momentum.ewm(span=fast_period, adjust=False).mean()
    double_smoothed_momentum = ema_momentum.ewm(span=slow_period, adjust=False).mean()
    
    # Double smoothed absolute momentum
    abs_momentum = momentum.abs()
    ema_abs_momentum = abs_momentum.ewm(span=fast_period, adjust=False).mean()
    double_smoothed_abs_momentum = ema_abs_momentum.ewm(span=slow_period, adjust=False).mean()
    
    tsi = 100 * (double_smoothed_momentum / double_smoothed_abs_momentum)
    
    return tsi


def calculate_dual_momentum(
    prices: pd.Series,
    absolute_period: int,
    relative_period: int,
    benchmark_prices: Optional[pd.Series] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate dual momentum (absolute and relative).
    
    Args:
        prices: Asset price series
        absolute_period: Period for absolute momentum
        relative_period: Period for relative momentum
        benchmark_prices: Benchmark price series for relative momentum
        
    Returns:
        Tuple of (absolute_momentum, relative_momentum)
    """
    # Absolute momentum
    absolute_momentum = calculate_momentum(prices, absolute_period)
    
    # Relative momentum (vs benchmark if provided)
    if benchmark_prices is not None:
        asset_return = calculate_momentum(prices, relative_period)
        benchmark_return = calculate_momentum(benchmark_prices, relative_period)
        relative_momentum = asset_return - benchmark_return
    else:
        relative_momentum = calculate_momentum(prices, relative_period)
    
    return absolute_momentum, relative_momentum


def calculate_sharpe_momentum(
    prices: pd.Series,
    period: int,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Calculate Sharpe ratio-adjusted momentum.
    
    Args:
        prices: Price series
        period: Lookback period
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe-adjusted momentum series
    """
    returns = prices.pct_change()
    
    # Calculate rolling Sharpe ratio
    rolling_mean = returns.rolling(window=period).mean()
    rolling_std = returns.rolling(window=period).std()
    
    # Annualize
    annual_return = rolling_mean * 252
    annual_std = rolling_std * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std
    
    return sharpe_ratio


def calculate_volatility_adjusted_momentum(
    prices: pd.Series,
    momentum_period: int,
    volatility_period: int,
    target_volatility: float = 0.15
) -> pd.Series:
    """
    Calculate volatility-adjusted momentum.
    
    Args:
        prices: Price series
        momentum_period: Period for momentum calculation
        volatility_period: Period for volatility calculation
        target_volatility: Target annual volatility
        
    Returns:
        Volatility-adjusted momentum series
    """
    # Calculate basic momentum
    momentum = calculate_momentum(prices, momentum_period)
    
    # Calculate volatility
    returns = prices.pct_change()
    volatility = returns.rolling(window=volatility_period).std() * np.sqrt(252)
    
    # Adjust momentum by volatility
    vol_scalar = target_volatility / volatility
    vol_adjusted_momentum = momentum * vol_scalar.clip(upper=1.0)
    
    return vol_adjusted_momentum


def calculate_composite_momentum(
    prices: pd.Series,
    volume: Optional[pd.Series] = None,
    periods: list = [20, 60, 120, 252],
    weights: Optional[list] = None
) -> pd.Series:
    """
    Calculate composite momentum score using multiple timeframes.
    
    Args:
        prices: Price series
        volume: Volume series (optional)
        periods: List of periods to use
        weights: Optional weights for each period
        
    Returns:
        Composite momentum score
    """
    if weights is None:
        weights = [1.0] * len(periods)
    
    if len(weights) != len(periods):
        raise ValueError("Weights must match number of periods")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Calculate momentum for each period
    momentum_scores = []
    for period in periods:
        momentum = calculate_momentum(prices, period)
        momentum_scores.append(momentum)
    
    # Combine scores
    composite_score = pd.Series(0.0, index=prices.index)
    for i, score in enumerate(momentum_scores):
        composite_score += score * weights[i]
    
    # Add volume momentum if available
    if volume is not None:
        volume_momentum = calculate_momentum(volume, max(periods))
        # Positive volume momentum amplifies price momentum
        composite_score *= (1 + volume_momentum.clip(lower=0))
    
    return composite_score


def calculate_trend_strength(
    prices: pd.Series,
    period: int,
    method: str = 'adx'
) -> pd.Series:
    """
    Calculate trend strength indicator.
    
    Args:
        prices: Price series
        period: Lookback period
        method: Method to use ('adx', 'linear_regression', 'efficiency_ratio')
        
    Returns:
        Trend strength series
    """
    if method == 'adx':
        # Simplified ADX calculation
        high = prices.rolling(window=2).max()
        low = prices.rolling(window=2).min()
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = high - low
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    elif method == 'linear_regression':
        # Trend strength based on R-squared of linear regression
        def calculate_r_squared(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            p = np.poly1d(coeffs)
            yhat = p(x)
            ybar = np.mean(y)
            ssreg = np.sum((yhat - ybar) ** 2)
            sstot = np.sum((y - ybar) ** 2)
            return ssreg / sstot if sstot != 0 else 0
        
        r_squared = prices.rolling(window=period).apply(calculate_r_squared)
        return r_squared
    
    elif method == 'efficiency_ratio':
        # Kaufman's Efficiency Ratio
        change = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility
        return efficiency_ratio
    
    else:
        raise ValueError(f"Unknown trend strength method: {method}")


def rank_by_momentum(
    data: pd.DataFrame,
    momentum_column: str,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank assets by momentum scores.
    
    Args:
        data: DataFrame with momentum scores
        momentum_column: Column name containing momentum scores
        ascending: Sort order
        
    Returns:
        DataFrame sorted by momentum with rank column
    """
    sorted_data = data.sort_values(momentum_column, ascending=ascending)
    sorted_data['momentum_rank'] = range(1, len(sorted_data) + 1)
    return sorted_data
