"""
Metrics collection system for monitoring agent performance and system health.
Collects and stores metrics for analysis and alerting.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import json

from core.config import config


class MetricsCollector:
    """
    Collects and stores metrics for the multi-agent swarm system.
    Tracks agent performance, response times, success rates, and more.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("metrics_collector")
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._start_time = datetime.now()
        
        # Initialize metrics storage
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metrics storage structure."""
        self._metrics = {
            "agent_performance": [],
            "response_times": [],
            "error_rates": [],
            "handoff_metrics": [],
            "memory_usage": [],
            "system_health": []
        }
    
    async def record_agent_metrics(
        self,
        agent_name: str,
        execution_time: float,
        success: bool,
        confidence: float,
        query_length: Optional[int] = None,
        response_length: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record metrics for agent performance.
        
        Args:
            agent_name: Name of the agent
            execution_time: Time taken to process the query
            success: Whether the operation was successful
            confidence: Confidence score of the response
            query_length: Length of the input query
            response_length: Length of the response
            metadata: Additional metadata
        """
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "execution_time": execution_time,
                "success": success,
                "confidence": confidence,
                "query_length": query_length,
                "response_length": response_length,
                "metadata": metadata or {}
            }
            
            self._metrics["agent_performance"].append(metric)
            
            # Also record response time separately
            self._metrics["response_times"].append({
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "response_time": execution_time
            })
            
            # Record error rate
            self._metrics["error_rates"].append({
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "is_error": not success
            })
            
            # Keep only recent metrics to prevent memory issues
            await self._cleanup_old_metrics()
            
            self.logger.debug(f"Recorded metrics for agent {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Error recording agent metrics: {str(e)}")
    
    async def record_handoff_metrics(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        execution_time: float,
        success: bool
    ):
        """
        Record metrics for agent handoffs.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            reason: Reason for handoff
            execution_time: Time taken for handoff
            success: Whether handoff was successful
        """
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "execution_time": execution_time,
                "success": success
            }
            
            self._metrics["handoff_metrics"].append(metric)
            await self._cleanup_old_metrics()
            
            self.logger.debug(f"Recorded handoff metrics: {from_agent} -> {to_agent}")
            
        except Exception as e:
            self.logger.error(f"Error recording handoff metrics: {str(e)}")
    
    async def record_memory_usage(
        self,
        memory_type: str,
        usage_bytes: int,
        entry_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record memory usage metrics.
        
        Args:
            memory_type: Type of memory (short_term, long_term)
            usage_bytes: Memory usage in bytes
            entry_count: Number of entries
            metadata: Additional metadata
        """
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "memory_type": memory_type,
                "usage_bytes": usage_bytes,
                "entry_count": entry_count,
                "metadata": metadata or {}
            }
            
            self._metrics["memory_usage"].append(metric)
            await self._cleanup_old_metrics()
            
            self.logger.debug(f"Recorded memory usage for {memory_type}")
            
        except Exception as e:
            self.logger.error(f"Error recording memory usage: {str(e)}")
    
    async def record_system_health(
        self,
        cpu_usage: float,
        memory_usage: float,
        active_agents: int,
        active_sessions: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record system health metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            active_agents: Number of active agents
            active_sessions: Number of active sessions
            metadata: Additional metadata
        """
        try:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "active_agents": active_agents,
                "active_sessions": active_sessions,
                "metadata": metadata or {}
            }
            
            self._metrics["system_health"].append(metric)
            await self._cleanup_old_metrics()
            
            self.logger.debug("Recorded system health metrics")
            
        except Exception as e:
            self.logger.error(f"Error recording system health: {str(e)}")
    
    async def get_agent_performance(
        self,
        agent_name: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Args:
            agent_name: Specific agent name (None for all agents)
            time_window: Time window for metrics (None for all time)
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = self._metrics["agent_performance"]
            
            # Filter by agent name if specified
            if agent_name:
                metrics = [m for m in metrics if m["agent_name"] == agent_name]
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [
                    m for m in metrics 
                    if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
                ]
            
            if not metrics:
                return {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "average_response_time": 0.0,
                    "average_confidence": 0.0
                }
            
            # Calculate metrics
            total_requests = len(metrics)
            successful_requests = sum(1 for m in metrics if m["success"])
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            
            response_times = [m["execution_time"] for m in metrics]
            average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            confidences = [m["confidence"] for m in metrics if m["confidence"] is not None]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "average_response_time": average_response_time,
                "average_confidence": average_confidence,
                "min_response_time": min(response_times) if response_times else 0.0,
                "max_response_time": max(response_times) if response_times else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent performance: {str(e)}")
            return {}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            if not self._metrics["system_health"]:
                return {"status": "unknown", "uptime": 0}
            
            latest_health = self._metrics["system_health"][-1]
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            return {
                "status": "healthy" if latest_health["cpu_usage"] < 80 and latest_health["memory_usage"] < 80 else "warning",
                "uptime": uptime,
                "cpu_usage": latest_health["cpu_usage"],
                "memory_usage": latest_health["memory_usage"],
                "active_agents": latest_health["active_agents"],
                "active_sessions": latest_health["active_sessions"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {str(e)}")
            return {"status": "error", "uptime": 0}
    
    async def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        try:
            handoff_metrics = self._metrics["handoff_metrics"]
            
            if not handoff_metrics:
                return {"total_handoffs": 0, "success_rate": 0.0}
            
            total_handoffs = len(handoff_metrics)
            successful_handoffs = sum(1 for m in handoff_metrics if m["success"])
            success_rate = successful_handoffs / total_handoffs if total_handoffs > 0 else 0.0
            
            # Group by handoff pairs
            handoff_pairs = {}
            for metric in handoff_metrics:
                pair = f"{metric['from_agent']} -> {metric['to_agent']}"
                if pair not in handoff_pairs:
                    handoff_pairs[pair] = {"count": 0, "successful": 0}
                handoff_pairs[pair]["count"] += 1
                if metric["success"]:
                    handoff_pairs[pair]["successful"] += 1
            
            return {
                "total_handoffs": total_handoffs,
                "successful_handoffs": successful_handoffs,
                "success_rate": success_rate,
                "handoff_pairs": handoff_pairs
            }
            
        except Exception as e:
            self.logger.error(f"Error getting handoff statistics: {str(e)}")
            return {"total_handoffs": 0, "success_rate": 0.0}
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues."""
        try:
            cutoff_time = datetime.now() - timedelta(days=config.monitoring.metrics_retention_days)
            
            for metric_type in self._metrics:
                self._metrics[metric_type] = [
                    m for m in self._metrics[metric_type]
                    if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
                ]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        try:
            return {
                "agent_performance_count": len(self._metrics["agent_performance"]),
                "response_times_count": len(self._metrics["response_times"]),
                "error_rates_count": len(self._metrics["error_rates"]),
                "handoff_metrics_count": len(self._metrics["handoff_metrics"]),
                "memory_usage_count": len(self._metrics["memory_usage"]),
                "system_health_count": len(self._metrics["system_health"]),
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {str(e)}")
            return {}
