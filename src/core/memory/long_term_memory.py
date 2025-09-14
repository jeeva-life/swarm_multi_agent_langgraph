"""
Long-term memory implementation for persistent agent knowledge and learnings.
Stores data that persists across sessions and has longer TTL.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import logging

from core.memory.memory_backends import MemoryBackend


class LongTermMemory:
    """
    Long-term memory for storing persistent agent knowledge and learnings.
    Data persists across sessions with configurable TTL.
    """
    
    def __init__(self, backend: MemoryBackend, default_ttl: int = 2592000):  # 30 days
        self.backend = backend
        self.default_ttl = default_ttl
        self.logger = logging.getLogger("long_term_memory")
        self.prefix = "ltm:"  # Long-term memory prefix
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store data in long-term memory.
        
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
                "ttl": ttl,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat()
            }
            
            serialized_data = json.dumps(data, default=str)
            success = await self.backend.set(full_key, serialized_data, ttl)
            
            if success:
                self.logger.debug(f"Stored long-term memory: {key}")
            else:
                self.logger.warning(f"Failed to store long-term memory: {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing long-term memory: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from long-term memory.
        
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
            
            # Update access statistics
            data["access_count"] = data.get("access_count", 0) + 1
            data["last_accessed"] = datetime.now().isoformat()
            
            # Store updated data back
            updated_data = json.dumps(data, default=str)
            await self.backend.set(full_key, updated_data, ttl)
            
            return data["value"]
            
        except Exception as e:
            self.logger.error(f"Error retrieving long-term memory: {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete data from long-term memory.
        
        Args:
            key: Key to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            full_key = f"{self.prefix}{key}"
            success = await self.backend.delete(full_key)
            
            if success:
                self.logger.debug(f"Deleted long-term memory: {key}")
            else:
                self.logger.warning(f"Failed to delete long-term memory: {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting long-term memory: {str(e)}")
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search through long-term memory for relevant data.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching results with metadata
        """
        try:
            # Get all keys with long-term memory prefix
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            results = []
            query_lower = query.lower()
            
            for key in keys[:limit * 2]:  # Get more keys than needed for filtering
                try:
                    data = await self.get(key.replace(self.prefix, ""))
                    if data and self._matches_query(data, query_lower):
                        # Get access statistics
                        full_key = f"{self.prefix}{key.replace(self.prefix, '')}"
                        serialized_data = await self.backend.get(full_key)
                        if serialized_data:
                            metadata = json.loads(serialized_data)
                            access_count = metadata.get("access_count", 0)
                        else:
                            access_count = 0
                        
                        results.append({
                            "key": key.replace(self.prefix, ""),
                            "value": data,
                            "score": self._calculate_relevance_score(data, query_lower),
                            "access_count": access_count
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing key {key}: {str(e)}")
                    continue
            
            # Sort by relevance score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching long-term memory: {str(e)}")
            return []
    
    async def get_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently accessed long-term memory entries.
        
        Args:
            limit: Maximum number of results
        
        Returns:
            List of most accessed entries with metadata
        """
        try:
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            entries = []
            
            for key in keys:
                try:
                    full_key = f"{self.prefix}{key.replace(self.prefix, '')}"
                    serialized_data = await self.backend.get(full_key)
                    
                    if serialized_data:
                        data = json.loads(serialized_data)
                        entries.append({
                            "key": key.replace(self.prefix, ""),
                            "value": data["value"],
                            "access_count": data.get("access_count", 0),
                            "last_accessed": data.get("last_accessed", ""),
                            "timestamp": data.get("timestamp", "")
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing key {key}: {str(e)}")
                    continue
            
            # Sort by access count and return top results
            entries.sort(key=lambda x: x["access_count"], reverse=True)
            return entries[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting most accessed entries: {str(e)}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get long-term memory statistics."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.backend.keys(pattern)
            
            total_access_count = 0
            for key in keys:
                try:
                    full_key = f"{self.prefix}{key.replace(self.prefix, '')}"
                    serialized_data = await self.backend.get(full_key)
                    if serialized_data:
                        data = json.loads(serialized_data)
                        total_access_count += data.get("access_count", 0)
                except Exception:
                    continue
            
            return {
                "total_entries": len(keys),
                "total_access_count": total_access_count,
                "prefix": self.prefix,
                "default_ttl": self.default_ttl
            }
        except Exception as e:
            self.logger.error(f"Error getting long-term memory stats: {str(e)}")
            return {"total_entries": 0, "total_access_count": 0, "prefix": self.prefix, "default_ttl": self.default_ttl}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from long-term memory."""
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
                    # If there's an error, assume it's expired and remove it
                    if await self.backend.delete(key):
                        cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired long-term memory entries")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired long-term memory: {str(e)}")
            return 0
    
    def _matches_query(self, data: Any, query: str) -> bool:
        """Check if data matches the search query."""
        if isinstance(data, str):
            return query in data.lower()
        elif isinstance(data, dict):
            return any(query in str(v).lower() for v in data.values())
        elif isinstance(data, list):
            return any(query in str(item).lower() for item in data)
        else:
            return query in str(data).lower()
    
    def _calculate_relevance_score(self, data: Any, query: str) -> float:
        """Calculate relevance score for search results."""
        if isinstance(data, str):
            return data.lower().count(query) / len(data) if data else 0.0
        elif isinstance(data, dict):
            total_matches = sum(str(v).lower().count(query) for v in data.values())
            total_length = sum(len(str(v)) for v in data.values())
            return total_matches / total_length if total_length > 0 else 0.0
        else:
            data_str = str(data).lower()
            return data_str.count(query) / len(data_str) if data_str else 0.0
