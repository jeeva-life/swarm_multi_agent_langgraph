"""
Memory backend implementations for different storage systems.
Supports Redis, in-memory, and other backends.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import json
import asyncio
import logging


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the backend connection."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory backend for development and testing."""
    
    def __init__(self):
        self._storage: dict = {}
        self._ttl: dict = {}
        self.logger = logging.getLogger("in_memory_backend")
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL."""
        try:
            self._storage[key] = value
            if ttl:
                self._ttl[key] = ttl
            return True
        except Exception as e:
            self.logger.error(f"Error setting key {key}: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        try:
            if key in self._storage:
                # Check TTL if it exists
                if key in self._ttl:
                    # Simple TTL implementation - in production, you'd want proper time tracking
                    # For now, we'll just return the value
                    pass
                return self._storage[key]
            return None
        except Exception as e:
            self.logger.error(f"Error getting key {key}: {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            if key in self._storage:
                del self._storage[key]
            if key in self._ttl:
                del self._ttl[key]
            return True
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {str(e)}")
            return False
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern."""
        try:
            import fnmatch
            return [key for key in self._storage.keys() if fnmatch.fnmatch(key, pattern)]
        except Exception as e:
            self.logger.error(f"Error getting keys with pattern {pattern}: {str(e)}")
            return []
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            return key in self._storage
        except Exception as e:
            self.logger.error(f"Error checking existence of key {key}: {str(e)}")
            return False
    
    async def close(self):
        """Close the backend connection."""
        self._storage.clear()
        self._ttl.clear()
        self.logger.info("In-memory backend closed")


class RedisBackend(MemoryBackend):
    """Redis backend for production use."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._redis = None
        self.logger = logging.getLogger("redis_backend")
    
    async def _get_redis(self):
        """Get Redis connection, creating it if necessary."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                self.logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except ImportError:
                self.logger.error("Redis package not installed. Install with: pip install redis")
                raise
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
        return self._redis
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL."""
        try:
            redis = await self._get_redis()
            if ttl:
                await redis.setex(key, ttl, value)
            else:
                await redis.set(key, value)
            return True
        except Exception as e:
            self.logger.error(f"Error setting key {key}: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        try:
            redis = await self._get_redis()
            return await redis.get(key)
        except Exception as e:
            self.logger.error(f"Error getting key {key}: {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            redis = await self._get_redis()
            result = await redis.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {str(e)}")
            return False
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern."""
        try:
            redis = await self._get_redis()
            return await redis.keys(pattern)
        except Exception as e:
            self.logger.error(f"Error getting keys with pattern {pattern}: {str(e)}")
            return []
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            redis = await self._get_redis()
            result = await redis.exists(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error checking existence of key {key}: {str(e)}")
            return False
    
    async def close(self):
        """Close the backend connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self.logger.info("Redis backend closed")
    
    def get_stats(self) -> dict:
        """Get Redis connection statistics."""
        if self._redis:
            return {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "connected": True
            }
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "connected": False
        }
