Below is detailed explanation of the core analysis logic in the Enhanced Advanced Momentum Strategy:

## Core Analysis Components

### 1. Technical Analysis Score (0-1 scale)

The technical score is calculated based on multiple indicators with specific weightings:

```python
def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
    score = 0.0
    max_score = 100
    
    # Price momentum (30 points)
    - 1-month return > 10%: 15 points
    - 3-month return > 15%: 15 points
    
    # Moving average strength (25 points)
    - Price > 20-day MA: 8 points
    - Price > 50-day MA: 8 points  
    - Price > 200-day MA: 9 points
    
    # RSI in optimal range (15 points)
    - RSI between 50-75 (intermediate) or 55-70 (long-term)
    - Optimal RSI = 60, scoring decreases with distance from optimal
    
    # MACD bullish (10 points)
    - MACD line above signal line: 10 points
    
    # Volume confirmation (10 points)
    - Volume ratio > 1.5x (intermediate) or 1.3x (long-term): 10 points
    
    # Trend strength (10 points)
    - ADX > 30: 10 points
    
    return score / max_score  # Normalized to 0-1
```

### 2. Fundamental Analysis Score (0-1 scale)

The fundamental score is based on Tiingo fundamental data, **but only for Dow 30 companies**:

```python
def _calculate_fundamental_score(self, symbol: str) -> float:
    if symbol not in self.fundamental_cache:
        return 0.5  # Neutral score if no data
    
    fundamentals = self.fundamental_cache[symbol]
    
    # Check if we have an error or no fundamentals
    if 'error' in fundamentals or not fundamentals.get('has_fundamentals', False):
        return 0.5  # Return neutral score for unavailable data
    
    score = 0.5  # Start with neutral
    
    # Analyze fundamental data if available
    statements = fundamentals.get('statements', {})
    daily_metrics = fundamentals.get('daily_metrics', {})
    
    # Simple scoring based on available metrics
    if statements:
        score += 0.1
    
    if daily_metrics:
        score += 0.1
    
    return score  # Currently returns 0.5-0.7
```

**Note**: The fundamental analysis is currently a placeholder implementation that gives a slight positive bias if data exists.

### 3. Sentiment Analysis Score (0-1 scale)

The sentiment score is based on news analysis from Tiingo:

```python
def _calculate_sentiment_score(self, symbol: str) -> float:
    if symbol not in self.news_cache:
        return 0.5  # Neutral score if no data
    
    sentiment_data = self.news_cache[symbol]
    
    # Convert sentiment to 0-1 scale
    avg_sentiment = sentiment_data.get('avg_sentiment', 0)
    news_count = sentiment_data.get('count', 0)
    
    # Weight by news volume (more news = more confidence)
    confidence = min(news_count / 10, 1.0)  # Max confidence at 10+ articles
    
    # Convert to 0-1 scale (assuming sentiment ranges from -1 to 1)
    normalized_sentiment = (avg_sentiment + 1) / 2
    
    # Blend with neutral based on confidence
    return normalized_sentiment * confidence + 0.5 * (1 - confidence)
```

## Handling Missing Data

When fundamental or sentiment data is unavailable, the system intelligently adjusts the weights:

```python
def calculate_comprehensive_score(self, symbol: str, market_data: pd.DataFrame) -> Dict:
    # Calculate individual scores
    technical_score = self._calculate_technical_score(indicators)
    fundamental_score = self._calculate_fundamental_score(symbol)
    sentiment_score = self._calculate_sentiment_score(symbol)
    
    # Check if fundamental data is available
    has_fundamentals = (symbol in self.fundamental_cache and 
                       self.fundamental_cache[symbol].get('has_fundamentals', False))
    
    # Adjust weights based on data availability
    if has_fundamentals:
        # Use configured weights (default: 0.5, 0.3, 0.2)
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
```

## Weight Distribution Examples

### With All Data Available:
- Technical: 50%
- Fundamental: 30%
- Sentiment: 20%

### Without Fundamental Data (Non-Dow 30):
- Technical: 71% (50% + 21% from redistributed fundamental)
- Fundamental: 0%
- Sentiment: 29% (20% + 9% from redistributed fundamental)

The redistribution gives 70% of the missing fundamental weight to technical analysis and 30% to sentiment.

## Final Action Determination

The comprehensive score is used to determine trading actions:

```python
def _determine_action(self, comprehensive_score: float, technical_score: float, 
                     indicators: Dict[str, float]) -> str:
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
```

## Summary

1. **Technical Analysis**: Always available and most heavily weighted. Based on price momentum, moving averages, RSI, MACD, volume, and trend strength.

2. **Fundamental Analysis**: Only available for Dow 30 companies. Currently a placeholder implementation that provides slight positive bias when data exists.

3. **Sentiment Analysis**: Based on news article count and sentiment scores. More articles increase confidence in the sentiment score.

4. **Missing Data Handling**: When fundamental data is unavailable (non-Dow 30 stocks), the weight is redistributed 70% to technical and 30% to sentiment analysis.

5. **Action Determination**: Based on comprehensive score thresholds with additional technical checks for buy signals and specific triggers for sell signals.

This design ensures the system can analyze any stock while providing enhanced analysis for Dow 30 companies where fundamental data is available.
