"""
Dashboard system for visualizing agent performance and system health.
Provides real-time monitoring and visualization capabilities.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging

from core.monitoring.metrics_collector import MetricsCollector
from core.monitoring.alert_manager import AlertManager, AlertLevel
from core.monitoring.drift_detector import DriftDetector


class Dashboard:
    """
    Dashboard for visualizing multi-agent swarm system performance.
    Provides real-time monitoring and system health visualization.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        drift_detector: DriftDetector
    ):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.drift_detector = drift_detector
        self.logger = logging.getLogger("dashboard")
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system overview."""
        try:
            # Get system health
            system_health = await self.metrics_collector.get_system_health()
            
            # Get alert statistics
            alert_stats = await self.alert_manager.get_alert_statistics()
            
            # Get drift summary
            drift_summary = await self.drift_detector.get_drift_summary()
            
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_health": system_health,
                "alerts": alert_stats,
                "drift": drift_summary,
                "metrics": metrics_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system overview: {str(e)}")
            return {}
    
    async def get_agent_performance_dashboard(self) -> Dict[str, Any]:
        """Get agent performance dashboard data."""
        try:
            # Get performance metrics for all agents
            # This would need to be implemented based on your agent naming convention
            agents = ["rag_agent", "nl2sql_agent", "invoice_agent"]  # Example agent names
            
            agent_performance = {}
            for agent in agents:
                performance = await self.metrics_collector.get_agent_performance(agent)
                agent_performance[agent] = performance
            
            # Get handoff statistics
            handoff_stats = await self.metrics_collector.get_handoff_statistics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "agents": agent_performance,
                "handoffs": handoff_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent performance dashboard: {str(e)}")
            return {}
    
    async def get_alerts_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Get alerts dashboard data."""
        try:
            # Get recent alerts by level
            alerts_by_level = {}
            for level in AlertLevel:
                alerts = await self.alert_manager.get_recent_alerts(level, hours)
                alerts_by_level[level.value] = alerts
            
            # Get alert trends (hourly counts)
            alert_trends = await self._calculate_alert_trends(hours)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "alerts_by_level": alerts_by_level,
                "trends": alert_trends,
                "total_alerts": sum(len(alerts) for alerts in alerts_by_level.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting alerts dashboard: {str(e)}")
            return {}
    
    async def get_drift_dashboard(self) -> Dict[str, Any]:
        """Get drift detection dashboard data."""
        try:
            # Get drift summary
            drift_summary = await self.drift_detector.get_drift_summary()
            
            # Get recent drifts
            recent_drifts = await self.drift_detector.get_recent_drifts(24)
            
            # Get detector stats
            detector_stats = self.drift_detector.get_detector_stats()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": drift_summary,
                "recent_drifts": recent_drifts,
                "detector_stats": detector_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting drift dashboard: {str(e)}")
            return {}
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live monitoring."""
        try:
            # Get recent performance data
            recent_performance = await self.metrics_collector.get_agent_performance(
                time_window=timedelta(minutes=5)
            )
            
            # Get recent alerts
            recent_alerts = await self.alert_manager.get_recent_alerts(hours=1)
            
            # Get system health
            system_health = await self.metrics_collector.get_system_health()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "performance": recent_performance,
                "alerts": recent_alerts,
                "system_health": system_health
            }
            
        except Exception as e:
            self.logger.error(f"Error getting real-time metrics: {str(e)}")
            return {}
    
    async def _calculate_alert_trends(self, hours: int) -> List[Dict[str, Any]]:
        """Calculate alert trends over time."""
        try:
            trends = []
            current_time = datetime.now()
            
            for i in range(hours):
                hour_start = current_time - timedelta(hours=i+1)
                hour_end = current_time - timedelta(hours=i)
                
                # Get alerts for this hour
                hour_alerts = await self.alert_manager.get_recent_alerts(hours=1)
                
                # Filter alerts within this hour
                hour_alerts = [
                    alert for alert in hour_alerts
                    if hour_start <= datetime.fromisoformat(alert["timestamp"]) < hour_end
                ]
                
                trends.append({
                    "hour": hour_start.isoformat(),
                    "alert_count": len(hour_alerts),
                    "alerts_by_level": {
                        level.value: len([a for a in hour_alerts if a["level"] == level.value])
                        for level in AlertLevel
                    }
                })
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating alert trends: {str(e)}")
            return []
    
    async def export_dashboard_data(self, filepath: str) -> bool:
        """Export dashboard data to JSON file."""
        try:
            dashboard_data = {
                "system_overview": await self.get_system_overview(),
                "agent_performance": await self.get_agent_performance_dashboard(),
                "alerts": await self.get_alerts_dashboard(),
                "drift": await self.get_drift_dashboard(),
                "memory": await self.get_memory_dashboard(),
                "real_time": await self.get_real_time_metrics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            self.logger.info(f"Dashboard data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {str(e)}")
            return False
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return {
            "refresh_interval": 30,  # seconds
            "chart_types": {
                "performance": "line",
                "alerts": "bar",
                "drift": "scatter",
                "memory": "area"
            },
            "time_ranges": {
                "real_time": "5m",
                "short_term": "1h",
                "medium_term": "24h",
                "long_term": "7d"
            }
        }
