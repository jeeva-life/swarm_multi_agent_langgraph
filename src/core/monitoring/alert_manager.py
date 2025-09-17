"""
Alert management system for monitoring and notifying about system issues.
Handles alerting for performance degradation, errors, and drift detection.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json

from core.config.config import config


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """Represents an alert with metadata."""
    
    def __init__(
        self,
        level: AlertLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.level = level
        self.message = message
        self.details = details or {}
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.id = f"{self.timestamp.isoformat()}_{self.level.value}_{hash(message)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.message}"


class AlertManager:
    """
    Manages alerts and notifications for the multi-agent swarm system.
    Handles alert routing, deduplication, and escalation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("alert_manager")
        self._alerts: List[Alert] = []
        self._alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        self._suppressed_alerts: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, int] = {}
        
        # Initialize default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default alert handlers."""
        # Log all alerts
        self.add_handler(AlertLevel.INFO, self._log_alert)
        self.add_handler(AlertLevel.WARNING, self._log_alert)
        self.add_handler(AlertLevel.ERROR, self._log_alert)
        self.add_handler(AlertLevel.CRITICAL, self._log_alert)
        
        # Console output for critical alerts
        self.add_handler(AlertLevel.CRITICAL, self._console_alert)
    
    def add_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """Add an alert handler for a specific level."""
        self._alert_handlers[level].append(handler)
        self.logger.debug(f"Added alert handler for {level.value}")
    
    async def send_alert(
        self,
        level: AlertLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        suppress_duplicates: bool = True,
        suppress_duration: int = 300  # 5 minutes
    ) -> str:
        """
        Send an alert.
        
        Args:
            level: Alert severity level
            message: Alert message
            details: Additional details
            source: Source of the alert
            suppress_duplicates: Whether to suppress duplicate alerts
            suppress_duration: Duration to suppress duplicates (seconds)
        
        Returns:
            Alert ID
        """
        try:
            alert = Alert(level, message, details, source)
            
            # Check for duplicate suppression
            if suppress_duplicates:
                alert_key = f"{level.value}:{message}:{source}"
                if alert_key in self._suppressed_alerts:
                    last_sent = self._suppressed_alerts[alert_key]
                    if datetime.now() - last_sent < timedelta(seconds=suppress_duration):
                        self.logger.debug(f"Suppressed duplicate alert: {message}")
                        return alert.id
                
                self._suppressed_alerts[alert_key] = datetime.now()
            
            # Store alert
            self._alerts.append(alert)
            
            # Update alert counts
            self._alert_counts[level.value] = self._alert_counts.get(level.value, 0) + 1
            
            # Send to handlers
            await self._send_to_handlers(alert)
            
            # Clean up old suppressed alerts
            await self._cleanup_suppressed_alerts()
            
            self.logger.info(f"Alert sent: {alert}")
            return alert.id
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            return ""
    
    async def _send_to_handlers(self, alert: Alert):
        """Send alert to all registered handlers."""
        try:
            handlers = self._alert_handlers.get(alert.level, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error sending to handlers: {str(e)}")
    
    def _log_alert(self, alert: Alert):
        """Default log handler for alerts."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT: {alert}")
        if alert.details:
            self.logger.log(log_level, f"Details: {json.dumps(alert.details, indent=2)}")
    
    def _console_alert(self, alert: Alert):
        """Console output handler for critical alerts."""
        print(f"\n CRITICAL ALERT: {alert.message}")
        if alert.details:
            print(f"Details: {json.dumps(alert.details, indent=2)}")
        print(f"Time: {alert.timestamp}")
        print("=" * 50)
    
    async def check_performance_thresholds(
        self,
        agent_name: str,
        response_time: float,
        success_rate: float,
        confidence: float
    ):
        """
        Check performance thresholds and send alerts if needed.
        
        Args:
            agent_name: Name of the agent
            response_time: Response time in seconds
            success_rate: Success rate (0.0 to 1.0)
            confidence: Confidence score (0.0 to 1.0)
        """
        try:
            # Check response time threshold
            if response_time > config.monitoring.alert_threshold_response_time:
                await self.send_alert(
                    AlertLevel.WARNING,
                    f"High response time for agent {agent_name}",
                    {
                        "agent_name": agent_name,
                        "response_time": response_time,
                        "threshold": config.monitoring.alert_threshold_response_time
                    },
                    source="performance_monitor"
                )
            
            # Check error rate threshold
            error_rate = 1.0 - success_rate
            if error_rate > config.monitoring.alert_threshold_error_rate:
                await self.send_alert(
                    AlertLevel.ERROR,
                    f"High error rate for agent {agent_name}",
                    {
                        "agent_name": agent_name,
                        "error_rate": error_rate,
                        "success_rate": success_rate,
                        "threshold": config.monitoring.alert_threshold_error_rate
                    },
                    source="performance_monitor"
                )
            
            # Check confidence threshold
            if confidence < 0.5:
                await self.send_alert(
                    AlertLevel.WARNING,
                    f"Low confidence for agent {agent_name}",
                    {
                        "agent_name": agent_name,
                        "confidence": confidence,
                        "threshold": 0.5
                    },
                    source="performance_monitor"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking performance thresholds: {str(e)}")
    
    async def check_system_health(
        self,
        cpu_usage: float,
        memory_usage: float,
        active_agents: int,
        active_sessions: int
    ):
        """
        Check system health and send alerts if needed.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            active_agents: Number of active agents
            active_sessions: Number of active sessions
        """
        try:
            # Check CPU usage
            if cpu_usage > 90:
                await self.send_alert(
                    AlertLevel.CRITICAL,
                    "High CPU usage detected",
                    {
                        "cpu_usage": cpu_usage,
                        "active_agents": active_agents,
                        "active_sessions": active_sessions
                    },
                    source="system_monitor"
                )
            elif cpu_usage > 80:
                await self.send_alert(
                    AlertLevel.WARNING,
                    "Elevated CPU usage",
                    {
                        "cpu_usage": cpu_usage,
                        "active_agents": active_agents,
                        "active_sessions": active_sessions
                    },
                    source="system_monitor"
                )
            
            # Check memory usage
            if memory_usage > 90:
                await self.send_alert(
                    AlertLevel.CRITICAL,
                    "High memory usage detected",
                    {
                        "memory_usage": memory_usage,
                        "active_agents": active_agents,
                        "active_sessions": active_sessions
                    },
                    source="system_monitor"
                )
            elif memory_usage > 80:
                await self.send_alert(
                    AlertLevel.WARNING,
                    "Elevated memory usage",
                    {
                        "memory_usage": memory_usage,
                        "active_agents": active_agents,
                        "active_sessions": active_sessions
                    },
                    source="system_monitor"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")
    
    async def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            level: Filter by alert level (None for all levels)
            hours: Number of hours to look back
        
        Returns:
            List of alert dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            alerts = [
                alert.to_dict() for alert in self._alerts
                if alert.timestamp >= cutoff_time
                and (level is None or alert.level == level)
            ]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {str(e)}")
            return []
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            total_alerts = len(self._alerts)
            alerts_by_level = {}
            
            for level in AlertLevel:
                count = self._alert_counts.get(level.value, 0)
                alerts_by_level[level.value] = count
            
            # Recent alerts (last 24 hours)
            recent_alerts = await self.get_recent_alerts(hours=24)
            recent_count = len(recent_alerts)
            
            return {
                "total_alerts": total_alerts,
                "alerts_by_level": alerts_by_level,
                "recent_alerts_24h": recent_count,
                "suppressed_alerts": len(self._suppressed_alerts)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {str(e)}")
            return {}
    
    async def _cleanup_suppressed_alerts(self):
        """Clean up old suppressed alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            expired_keys = [
                key for key, timestamp in self._suppressed_alerts.items()
                if timestamp < cutoff_time
            ]
            
            for key in expired_keys:
                del self._suppressed_alerts[key]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up suppressed alerts: {str(e)}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of current alert status."""
        try:
            return {
                "total_alerts": len(self._alerts),
                "alert_counts": self._alert_counts,
                "suppressed_count": len(self._suppressed_alerts),
                "handlers_count": sum(len(handlers) for handlers in self._alert_handlers.values())
            }
        except Exception as e:
            self.logger.error(f"Error getting alert summary: {str(e)}")
            return {}
