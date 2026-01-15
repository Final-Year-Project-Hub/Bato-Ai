"""
Simple in-memory cache with TTL.

Simplified from multi-level cache - removed Redis complexity.
"""

import logging
import time
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, maxsize: int = 1000, default_ttl: int = 86400):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of cached items
            default_ttl: Default TTL in seconds (24 hours)
        """
        self.cache: Dict[str, tuple[Any, float]] = {}  # {key: (value, expiry)}
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                self.hits += 1
                return value
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        
        # LRU eviction if full
        if len(self.cache) >= self.maxsize:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


# Global cache instance
_cache: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = SimpleCache()
    return _cache
