"""
Core modules for the multi-agent swarm system.
Contains shared utilities, configurations, and base classes.
"""

# Core modules only - avoid circular imports
from core.config.config import config as Config
from core.state.state import State
from core.memory import MemoryManager
from core.monitoring import MetricsCollector, AlertManager, DriftDetector, Dashboard

__all__ = [
    "Config",
    "State",
    "MemoryManager",
    "MetricsCollector",
    "AlertManager", 
    "DriftDetector",
    "Dashboard"
]