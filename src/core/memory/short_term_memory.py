"""
Short-term memory implementation for agent conversations and sessions.
Stores temporary data that expires after a configurable TTL.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import logging

from core.memory.memory_backends import MemoryBackend


class ShortTermMemory:
    """
    Short-term memory for storing temporary conversation data.
    Data expires after a configurable TTL (Time To Live).
    """
    
    def __init__(self, backend: MemoryBackend, default_ttl: int = 3600, langchain_client: Optional[Any] = None):
        self.backend = backend
        self.default_ttl = default_ttl
        self.langchain_client = langchain_client
        self.logger = logging.getLogger("short_term_memory")
        self.prefix = "stm:"  # Short-term memory prefix
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store data in short-term memory.
        
        Args:
            key: Unique key for the data
            value: Data to store (must be JSON serializable)
            ttl: Time to live in seconds (uses default if not provided)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            ttl = ttl or self.default_ttl
            full_key = f"{self.prefix}{key}"
            
            # Serialize value with metadata
            data = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "ttl": ttl
            }
            
            serialized_data = json.dumps(data)
            
            # Store in backend
            success = await self.backend.set(full_key, serialized_data, ttl)
            
            if success:
                self.logger.debug(f"Stored short-term memory: {key}")
            else:
                self.logger.warning(f"Failed to store short-term memory: {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing short-term memory: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from short-term memory.
        
        Args:
            key: Key to retrieve
        
        Returns:
            Stored value or None if not found/expired
        """
        try:
            full_key = f"{self.prefix}{key}"
            serialized_data = await self.backend.get(full_key)
            
            if not serialized_data:
                return None
            
            data = json.loads(serialized_data)
            
            # Check if data has expired
            timestamp = datetime.fromisoformat(data["timestamp"])
            ttl = data["ttl"]
            
            if datetime.now() - timestamp > timedelta(seconds=ttl):
                # Data has expired, remove it
                await self.backend.delete(full_key)
                return None
            
            return data["value"]
            
        except Exception as e:
            self.logger.error(f"Error retrieving short-term memory: {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete data from short-term memory.
        
        Args:
            key: Key to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            full_key = f"{self.prefix}{key}"
            success = await self.backend.delete(full_key)
            
            if success:
                self.logger.debug(f"Deleted short-term memory: {key}")
            else:
                self.logger.warning(f"Failed to delete short-term memory: {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting short-term memory: {str(e)}")
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search through short-term memory using embeddings if available.
        """
        try:
            if self.langchain_client and self.langchain_client.is_initialized():
                return await self._semantic_search(query, limit)
            else:
                return await self._keyword_search(query, limit)
                
        except Exception as e:
            self.logger.error(f"Error searching short-term memory: {str(e)}")
            return []
    
    async def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        try:
            # Get all conversation keys
            pattern = f"{self.prefix}conversation:*"
            keys = await self.backend.keys(pattern)
            
            if not keys:
                return []
            
            # Retrieve stored conversations
            stored_data = []
            for key in keys[:50]:  # Limit to prevent memory issues
                try:
                    data = await self.get(key.replace(self.prefix, ""))
                    if data:
                        stored_data.append({
                            "key": key,
                            "data": data,
                            "text": str(data)
                        })
                except Exception:
                    continue
            
            if not stored_data:
                return []
            
            # Generate query embedding
            query_embedding = await self.langchain_client.generate_query_embedding(query)
            
            # Generate document embeddings
            texts = [item["text"] for item in stored_data]
            doc_embeddings = await self.langchain_client.generate_embeddings(texts)
            
            # Calculate similarities
            results = []
            for i, item in enumerate(stored_data):
                similarity = self._cosine_similarity(query_embedding, doc_embeddings[i])
                results.append({
                    "key": item["key"],
                    "data": item["data"],
                    "score": similarity
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    async def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            results = []
            query_lower = query.lower()
            
            for key in keys[:limit * 2]:  # Get more keys than needed for filtering
                try:
                    data = await self.get(key.replace(self.prefix, ""))
                    if data and self._matches_query(data, query_lower):
                        results.append({
                            "key": key,
                            "data": data,
                            "score": 1.0  # Simple keyword match
                        })
                except Exception:
                    continue
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _matches_query(self, data: Any, query_lower: str) -> bool:
        """Check if data matches the query."""
        try:
            if isinstance(data, str):
                return query_lower in data.lower()
            elif isinstance(data, dict):
                return any(query_lower in str(v).lower() for v in data.values())
            elif isinstance(data, list):
                return any(query_lower in str(item).lower() for item in data)
            else:
                return query_lower in str(data).lower()
        except Exception:
            return False
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(a) != len(b):
                return 0.0
            
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get short-term memory statistics."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            return {
                "total_entries": len(keys),
                "prefix": self.prefix,
                "default_ttl": self.default_ttl
            }
        except Exception as e:
            self.logger.error(f"Error getting short-term memory stats: {str(e)}")
            return {"total_entries": 0, "prefix": self.prefix, "default_ttl": self.default_ttl}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from short-term memory."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                try:
                    # Try to get the data to check if it's expired
                    data = await self.get(key.replace(self.prefix, ""))
                    if data is None:  # Data was expired and removed
                        cleaned_count += 1
                except Exception:
                    continue
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired short-term memory: {str(e)}")
            return 0