"""
Redis-backed cache for roadmap responses.

Replaces the previous SimpleCache with Redis for production use.
"""

import logging
from typing import Optional, Any, Dict, List

from app.core.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Redis-backed cache wrapper for backward compatibility.
    
    This maintains the same interface as the old SimpleCache
    but uses Redis as the backend.
    """
    
    def __init__(self, maxsize: int = 1000, default_ttl: int = 86400):
        """
        Initialize cache.
        
        Args:
            maxsize: Not used (Redis handles size management)
            default_ttl: Default TTL in seconds (24 hours)
        """
        self.default_ttl = default_ttl
        self._redis = get_redis_cache()
        
        logger.info("✅ SimpleCache initialized with Redis backend")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            val = self._redis.get(key)
            if val:
                logger.info(f"✅ Cache HIT for key: {key}")
            else:
                logger.info(f"❌ Cache MISS for key: {key}")
            return val
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            # Log size of value for debugging
            size_info = "unknown size"
            if isinstance(value, (str, bytes)):
                size_info = f"{len(value)} bytes"
            elif isinstance(value, dict):
                size_info = f"{len(str(value))} chars (approx)"
            
            logger.info(f"💾 Caching key: {key} (TTL: {ttl}s, Size: {size_info})")
            self._redis.set(key, value, ttl=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cached items."""
        try:
            logger.warning("🧹 Clearing entire cache")
            self._redis.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def generate_cache_key(
        self,
        goal: str,
        intent: str,
        proficiency: str,
        tech_stack: Optional[List[str]],
        user_id: Optional[str] = None,
    ) -> str:
        """Generate a stable cache key for roadmap requests."""
        return self._redis.generate_cache_key(
            goal=goal,
            intent=intent,
            proficiency=proficiency,
            tech_stack=tech_stack,
            user_id=user_id
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_stats = self._redis.get_stats()
            return {
                "size": redis_stats.get("total_keys", 0),
                "hits": redis_stats.get("hits", 0),
                "misses": redis_stats.get("misses", 0),
                "hit_rate": redis_stats.get("hit_rate", 0.0),
                "backend": "redis",
                "memory_used": redis_stats.get("used_memory_human", "unknown")
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {
                "size": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "error": str(e)
            }


# Global cache instance
_cache: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = SimpleCache()
    return _cache
