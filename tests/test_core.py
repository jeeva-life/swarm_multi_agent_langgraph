"""
Tests for core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.core.memory import MemoryManager, InMemoryBackend
from src.core.monitoring import MetricsCollector, AlertManager, DriftDetector, AlertLevel
from src.core.config import Config


class TestMemoryManager:
    """Test Memory Manager functionality."""
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        backend = InMemoryBackend()
        manager = MemoryManager(backend)
        
        assert manager.short_term is not None
        assert manager.long_term is not None
    
    @pytest.mark.asyncio
    async def test_short_term_memory_operations(self):
        """Test short-term memory operations."""
        backend = InMemoryBackend()
        manager = MemoryManager(backend)
        
        # Test store and retrieve
        test_data = {"key": "value", "number": 42}
        success = await manager.short_term.store("test_key", test_data)
        assert success
        
        retrieved = await manager.short_term.get("test_key")
        assert retrieved == test_data
    
    @pytest.mark.asyncio
    async def test_long_term_memory_operations(self):
        """Test long-term memory operations."""
        backend = InMemoryBackend()
        manager = MemoryManager(backend)
        
        # Test store and retrieve
        test_data = {"key": "value", "number": 42}
        success = await manager.long_term.store("test_key", test_data)
        assert success
        
        retrieved = await manager.long_term.get("test_key")
        assert retrieved == test_data


class TestMetricsCollector:
    """Test Metrics Collector functionality."""
    
    @pytest.mark.asyncio
    async def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert collector._metrics is not None
        assert "agent_performance" in collector._metrics
    
    @pytest.mark.asyncio
    async def test_record_agent_metrics(self):
        """Test recording agent metrics."""
        collector = MetricsCollector()
        
        await collector.record_agent_metrics(
            agent_name="test_agent",
            execution_time=1.5,
            success=True,
            confidence=0.8
        )
        
        # Check that metrics were recorded
        assert len(collector._metrics["agent_performance"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_agent_performance(self):
        """Test getting agent performance metrics."""
        collector = MetricsCollector()
        
        # Record some test metrics
        await collector.record_agent_metrics(
            agent_name="test_agent",
            execution_time=1.0,
            success=True,
            confidence=0.9
        )
        
        performance = await collector.get_agent_performance("test_agent")
        
        assert performance["total_requests"] > 0
        assert performance["success_rate"] > 0


class TestAlertManager:
    """Test Alert Manager functionality."""
    
    @pytest.mark.asyncio
    async def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert manager._alerts is not None
        assert manager._alert_handlers is not None
    
    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alerts."""
        manager = AlertManager()
        
        alert_id = await manager.send_alert(
            level=AlertLevel.INFO,
            message="Test alert",
            details={"test": "data"}
        )
        
        assert alert_id is not None
        assert len(manager._alerts) > 0
    
    @pytest.mark.asyncio
    async def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        manager = AlertManager()
        
        # Send some test alerts
        await manager.send_alert(AlertLevel.INFO, "Test alert 1")
        await manager.send_alert(AlertLevel.WARNING, "Test alert 2")
        
        recent = await manager.get_recent_alerts()
        
        assert len(recent) >= 2


class TestDriftDetector:
    """Test Drift Detector functionality."""
    
    @pytest.mark.asyncio
    async def test_drift_detector_initialization(self):
        """Test drift detector initialization."""
        detector = DriftDetector()
        
        assert detector.window_size > 0
        assert detector.sensitivity > 0
        assert detector._baselines is not None
    
    @pytest.mark.asyncio
    async def test_update_metrics(self):
        """Test updating drift metrics."""
        detector = DriftDetector()
        
        await detector.update_metrics(
            agent_name="test_agent",
            response_time=1.0,
            confidence=0.8,
            success=True
        )
        
        assert "test_agent" in detector._recent_metrics
        assert len(detector._recent_metrics["test_agent"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_drift_summary(self):
        """Test getting drift summary."""
        detector = DriftDetector()
        
        summary = await detector.get_drift_summary()
        
        assert "total_drifts" in summary
        assert "agents" in summary


class TestConfig:
    """Test Configuration functionality."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        
        assert config.environment is not None
        assert config.aws is not None
        assert config.database is not None
        assert config.memory is not None
        assert config.monitoring is not None
        assert config.agent is not None
    
    def test_aws_config(self):
        """Test AWS configuration."""
        config = Config()
        
        assert config.aws.region is not None
        assert config.aws.bedrock_model_id is not None
    
    def test_database_config(self):
        """Test database configuration."""
        config = Config()
        
        assert config.database.host is not None
        assert config.database.port > 0
        assert config.database.database is not None
    
    def test_memory_config(self):
        """Test memory configuration."""
        config = Config()
        
        assert config.memory.short_term_ttl > 0
        assert config.memory.long_term_ttl > 0
        assert config.memory.memory_backend is not None