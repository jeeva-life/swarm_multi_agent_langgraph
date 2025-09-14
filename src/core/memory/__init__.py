"""
Memory management system for short-term and long-term agent memory.
Supports multiple backends and provides unified interface.
"""

from core.memory.memory_manager import MemoryManager
from core.memory.short_term_memory import ShortTermMemory
from core.memory.long_term_memory import LongTermMemory
from core.memory.memory_backends import MemoryBackend, RedisBackend, InMemoryBackend

__all__ = [
    "MemoryManager",
    "ShortTermMemory", 
    "LongTermMemory",
    "MemoryBackend",
    "RedisBackend",
    "InMemoryBackend"
]
