"""
Redis-based caching for LLM responses and roadmaps.

This module provides a Redis-backed cache with:
- Connection pooling for performance
- Automatic serialization/deserialization
- TTL support for automatic expiration
- Health check functionality
"""

import hashlib
import json
import logging
import pickle
from typing import Optional, Any, Dict, List

from redis import Redis, ConnectionPool, RedisError

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache for LLM responses and roadmaps.
    
    Features:
    - Connection pooling
    - Automatic serialization with pickle
    - TTL support
    - Health checks
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        password: Optional[str] = None
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            password: Redis password (if required)
        """
        try:
            # Create connection pool
            pool = ConnectionPool.from_url(
                redis_url,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                password=password,
                decode_responses=False  # We handle encoding with pickle
            )
            
            self.client = Redis(connection_pool=pool)
            
            # Test connection
            self.client.ping()
            logger.info(f"✅ Redis cache connected: {redis_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise RuntimeError(f"Redis connection failed: {e}") from e
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except RedisError as e:
            logger.error(f"Redis get error for key {key[:20]}...: {e}")
            raise
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        try:
            data = pickle.dumps(value)
            if ttl:
                self.client.setex(key, ttl, data)
            else:
                self.client.set(key, data)
        except RedisError as e:
            logger.error(f"Redis set error for key {key[:20]}...: {e}")
            raise
    
    def delete(self, key: str) -> None:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        try:
            self.client.delete(key)
        except RedisError as e:
            logger.error(f"Redis delete error for key {key[:20]}...: {e}")
            raise
    
    def clear(self) -> None:
        """Clear all keys in current database."""
        try:
            self.client.flushdb()
            logger.info("Redis cache cleared")
        except RedisError as e:
            logger.error(f"Redis clear error: {e}")
            raise
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self.client.exists(key))
        except RedisError as e:
            logger.error(f"Redis exists error for key {key[:20]}...: {e}")
            raise
    
    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        try:
            return self.client.ttl(key)
        except RedisError as e:
            logger.error(f"Redis TTL error for key {key[:20]}...: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            info = self.client.info("stats")
            return {
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": self.client.info("memory").get("used_memory_human", "unknown")
            }
        except RedisError as e:
            logger.error(f"Redis stats error: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health.
        
        Returns:
            Health status dictionary
        """
        try:
            # Ping test
            latency_ms = self.client.ping()
            
            # Get basic info
            info = self.client.info("server")
            
            return {
                "status": "healthy",
                "backend": "redis",
                "version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected": True
            }
        except RedisError as e:
            return {
                "status": "unhealthy",
                "backend": "redis",
                "error": str(e),
                "connected": False
            }
    
    def generate_cache_key(
        self,
        goal: str,
        intent: str,
        proficiency: str,
        tech_stack: Optional[List[str]],
        user_id: Optional[str] = None,
    ) -> str:
        """
        Generate a stable cache key for roadmap requests.
        
        Args:
            goal: User's learning goal
            intent: Intent (learn/build)
            proficiency: Proficiency level
            tech_stack: List of technologies
            user_id: Optional user ID for user-specific caching
            
        Returns:
            MD5 hash of normalized parameters
        """
        normalized_stack = sorted(
            (item or "").strip().lower() for item in (tech_stack or [])
        )
        payload = {
            "goal": (goal or "").strip(),
            "intent": (intent or "").strip().lower(),
            "proficiency": (proficiency or "").strip().lower(),
            "tech_stack": normalized_stack,
            "user_id": (user_id or "").strip(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"roadmap:{hashlib.md5(raw.encode('utf-8')).hexdigest()}"


# Global cache instance
_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """
    Get global Redis cache instance.
    
    Returns:
        RedisCache instance
        
    Raises:
        RuntimeError: If Redis is not configured or connection fails
    """
    global _redis_cache
    
    if _redis_cache is None:
        # Import here to avoid circular dependency
        from app.core.config import get_settings
        
        settings = get_settings()
        
        if not settings.REDIS_ENABLED:
            raise RuntimeError(
                "Redis is not enabled. Set REDIS_ENABLED=true in .env"
            )
        
        _redis_cache = RedisCache(
            redis_url=settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            password=settings.REDIS_PASSWORD
        )
    
    return _redis_cache
