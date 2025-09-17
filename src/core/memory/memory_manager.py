"""
Central memory management system for the multi-agent swarm.
Handles both short-term and long-term memory with multiple backend support.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import logging

from core.memory.short_term_memory import ShortTermMemory
from core.memory.long_term_memory import LongTermMemory
from core.memory.memory_backends import MemoryBackend, RedisBackend, InMemoryBackend
from core.config.config import config


class MemoryManager:
    """
    Central memory management system for the multi-agent swarm.
    Provides unified interface for short-term and long-term memory operations.
    """
    
    def __init__(self, backend: Optional[MemoryBackend] = None, langchain_client: Optional[Any] = None):
        self.logger = logging.getLogger("memory_manager")
        self.langchain_client = langchain_client
        
        # Initialize backend
        if backend is None:
            if config.memory.memory_backend == "redis":
                self.backend = RedisBackend()
            else:
                self.backend = InMemoryBackend()
        else:
            self.backend = backend
        
        # Initialize memory systems
        self.short_term = ShortTermMemory(self.backend, config.memory.short_term_ttl, langchain_client)
        self.long_term = LongTermMemory(self.backend, config.memory.long_term_ttl)
        
        self.logger.info(f"Memory manager initialized with {config.memory.memory_backend} backend")
    
    async def search_memory(
        self, 
        query: str, 
        memory_type: str = "both",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search through memory for relevant information.
        
        Args:
            query: Search query
            memory_type: "short", "long", or "both"
            limit: Maximum number of results to return
        """
        results = []
        
        try:
            if memory_type in ["short", "both"]:
                short_results = await self.short_term.search(query, limit)
                results.extend(short_results)
            
            if memory_type in ["long", "both"]:
                long_results = await self.long_term.search(query, limit)
                results.extend(long_results)
            
            # Sort by relevance score if available
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search memory: {str(e)}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            short_stats = await self.short_term.get_stats()
            long_stats = await self.long_term.get_stats()
            
            return {
                "short_term": short_stats,
                "long_term": long_stats,
                "backend": self.backend.get_stats() if hasattr(self.backend, 'get_stats') else {}
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {str(e)}")
            return {}
    
    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired memory entries."""
        try:
            short_cleaned = await self.short_term.cleanup_expired()
            long_cleaned = await self.long_term.cleanup_expired()
            
            return {
                "short_term_cleaned": short_cleaned,
                "long_term_cleaned": long_cleaned
            }
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memory: {str(e)}")
            return {"short_term_cleaned": 0, "long_term_cleaned": 0}
    
    async def close(self):
        """Close memory manager and clean up resources."""
        try:
            if hasattr(self.backend, 'close'):
                await self.backend.close()
            self.logger.info("Memory manager closed")
        except Exception as e:
            self.logger.error(f"Error closing memory manager: {str(e)}")