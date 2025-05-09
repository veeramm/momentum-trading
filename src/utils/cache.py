"""
Data Cache Module

This module provides caching functionality for market data
to reduce API calls and improve performance.
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from loguru import logger


class DataCache:
    """
    Data cache implementation with support for different backends.
    """
    
    def __init__(self, config: dict):
        """
        Initialize cache with configuration.
        
        Args:
            config: Cache configuration dictionary
        """
        self.enabled = config.get('enabled', True)
        self.ttl = config.get('ttl', 3600)  # Default 1 hour
        self.backend = config.get('backend', 'memory')
        
        # Initialize backend
        if self.backend == 'memory':
            self._cache = {}
        elif self.backend == 'redis':
            self._init_redis(config.get('redis', {}))
        else:
            raise ValueError(f"Unknown cache backend: {self.backend}")
            
        logger.info(f"Initialized {self.backend} cache with TTL={self.ttl}s")
    
    def _init_redis(self, redis_config: dict):
        """Initialize Redis connection"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=False  # We'll handle encoding/decoding
            )
            # Test connection
            self.redis_client.ping()
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
            
        try:
            if self.backend == 'memory':
                return self._get_from_memory(key)
            elif self.backend == 'redis':
                return self._get_from_redis(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        if not self.enabled:
            return
            
        ttl = ttl or self.ttl
        
        try:
            if self.backend == 'memory':
                self._set_in_memory(key, value, ttl)
            elif self.backend == 'redis':
                self._set_in_redis(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled:
            return
            
        try:
            if self.backend == 'memory':
                self._cache.pop(key, None)
            elif self.backend == 'redis':
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        if not self.enabled:
            return
            
        try:
            if self.backend == 'memory':
                self._cache.clear()
            elif self.backend == 'redis':
                self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.now() < expiry:
                return value
            else:
                # Expired, remove it
                del self._cache[key]
        return None
    
    def _set_in_memory(self, key: str, value: Any, ttl: int):
        """Set value in memory cache"""
        expiry = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        data = self.redis_client.get(key)
        if data:
            return self._deserialize(data)
        return None
    
    def _set_in_redis(self, key: str, value: Any, ttl: int):
        """Set value in Redis cache"""
        data = self._serialize(value)
        self.redis_client.setex(key, ttl, data)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if isinstance(value, pd.DataFrame):
            # Special handling for DataFrames
            return pickle.dumps(value)
        elif isinstance(value, (dict, list)):
            # JSON for simple types
            return json.dumps(value).encode('utf-8')
        else:
            # Pickle for everything else
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)


class CacheKeyBuilder:
    """Helper class to build consistent cache keys"""
    
    @staticmethod
    def price_data_key(
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Build cache key for price data"""
        return f"price_{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
    
    @staticmethod
    def fundamental_data_key(symbol: str, metric: Optional[str] = None) -> str:
        """Build cache key for fundamental data"""
        if metric:
            return f"fundamental_{symbol}_{metric}"
        return f"fundamental_{symbol}_all"
    
    @staticmethod
    def indicator_key(
        symbol: str,
        indicator: str,
        period: int,
        **params
    ) -> str:
        """Build cache key for technical indicators"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"indicator_{symbol}_{indicator}_{period}_{param_str}"
