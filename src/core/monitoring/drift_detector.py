"""
Drift detection system for monitoring agent behavior changes over time.
Detects when agents start behaving differently from their baseline.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import logging
import json

from core.config import config


class DriftDetector:
    """
    Detects drift in agent behavior and performance over time.
    Monitors for changes in response patterns, confidence scores, and success rates.
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.1):
        self.logger = logging.getLogger("drift_detector")
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # Store baseline metrics for each agent
        self._baselines: Dict[str, Dict[str, Any]] = {}
        self._recent_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._drift_alerts: List[Dict[str, Any]] = []
        
    async def update_metrics(
        self,
        agent_name: str,
        response_time: float,
        confidence: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update metrics for drift detection.
        
        Args:
            agent_name: Name of the agent
            response_time: Response time in seconds
            confidence: Confidence score (0.0 to 1.0)
            success: Whether the operation was successful
            metadata: Additional metadata
        """
        try:
            if agent_name not in self._recent_metrics:
                self._recent_metrics[agent_name] = []
            
            metric = {
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "confidence": confidence,
                "success": success,
                "metadata": metadata or {}
            }
            
            self._recent_metrics[agent_name].append(metric)
            
            # Keep only recent metrics within window
            if len(self._recent_metrics[agent_name]) > self.window_size:
                self._recent_metrics[agent_name] = self._recent_metrics[agent_name][-self.window_size:]
            
            # Check for drift if we have enough data
            if len(self._recent_metrics[agent_name]) >= self.window_size // 2:
                await self._check_drift(agent_name)
                
        except Exception as e:
            self.logger.error(f"Error updating drift metrics: {str(e)}")
    
    async def _check_drift(self, agent_name: str):
        """Check for drift in agent behavior."""
        try:
            if agent_name not in self._recent_metrics:
                return
            
            recent_metrics = self._recent_metrics[agent_name]
            
            # Calculate current metrics
            current_metrics = self._calculate_metrics(recent_metrics)
            
            # Get or create baseline
            if agent_name not in self._baselines:
                self._baselines[agent_name] = current_metrics
                self.logger.info(f"Created baseline for agent {agent_name}")
                return
            
            baseline = self._baselines[agent_name]
            
            # Check for drift in different metrics
            drift_detected = False
            drift_details = {}
            
            # Response time drift
            if self._detect_response_time_drift(baseline, current_metrics):
                drift_detected = True
                drift_details["response_time"] = {
                    "baseline": baseline["avg_response_time"],
                    "current": current_metrics["avg_response_time"],
                    "change": current_metrics["avg_response_time"] - baseline["avg_response_time"]
                }
            
            # Confidence drift
            if self._detect_confidence_drift(baseline, current_metrics):
                drift_detected = True
                drift_details["confidence"] = {
                    "baseline": baseline["avg_confidence"],
                    "current": current_metrics["avg_confidence"],
                    "change": current_metrics["avg_confidence"] - baseline["avg_confidence"]
                }
            
            # Success rate drift
            if self._detect_success_rate_drift(baseline, current_metrics):
                drift_detected = True
                drift_details["success_rate"] = {
                    "baseline": baseline["success_rate"],
                    "current": current_metrics["success_rate"],
                    "change": current_metrics["success_rate"] - baseline["success_rate"]
                }
            
            # Pattern drift (response time variance)
            if self._detect_pattern_drift(baseline, current_metrics):
                drift_detected = True
                drift_details["pattern"] = {
                    "baseline_variance": baseline["response_time_variance"],
                    "current_variance": current_metrics["response_time_variance"],
                    "change": current_metrics["response_time_variance"] - baseline["response_time_variance"]
                }
            
            if drift_detected:
                await self._handle_drift_detection(agent_name, drift_details, current_metrics)
            
            # Update baseline periodically (every 50 new metrics)
            if len(recent_metrics) % 50 == 0:
                self._baselines[agent_name] = current_metrics
                self.logger.info(f"Updated baseline for agent {agent_name}")
                
        except Exception as e:
            self.logger.error(f"Error checking drift for agent {agent_name}: {str(e)}")
    
    def _calculate_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from a list of metric data."""
        if not metrics:
            return {}
        
        response_times = [m["response_time"] for m in metrics]
        confidences = [m["confidence"] for m in metrics if m["confidence"] is not None]
        successes = [m["success"] for m in metrics]
        
        return {
            "avg_response_time": np.mean(response_times) if response_times else 0.0,
            "response_time_variance": np.var(response_times) if response_times else 0.0,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "confidence_variance": np.var(confidences) if confidences else 0.0,
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "sample_size": len(metrics)
        }
    
    def _detect_response_time_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Detect drift in response time."""
        try:
            baseline_avg = baseline.get("avg_response_time", 0)
            current_avg = current.get("avg_response_time", 0)
            
            if baseline_avg == 0:
                return False
            
            # Calculate percentage change
            change = abs(current_avg - baseline_avg) / baseline_avg
            
            return change > self.sensitivity
        except Exception:
            return False
    
    def _detect_confidence_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Detect drift in confidence scores."""
        try:
            baseline_avg = baseline.get("avg_confidence", 0)
            current_avg = current.get("avg_confidence", 0)
            
            if baseline_avg == 0:
                return False
            
            # Calculate percentage change
            change = abs(current_avg - baseline_avg) / baseline_avg
            
            return change > self.sensitivity
        except Exception:
            return False
    
    def _detect_success_rate_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Detect drift in success rate."""
        try:
            baseline_rate = baseline.get("success_rate", 0)
            current_rate = current.get("success_rate", 0)
            
            if baseline_rate == 0:
                return False
            
            # Calculate percentage change
            change = abs(current_rate - baseline_rate) / baseline_rate
            
            return change > self.sensitivity
        except Exception:
            return False
    
    def _detect_pattern_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Detect drift in response patterns (variance)."""
        try:
            baseline_var = baseline.get("response_time_variance", 0)
            current_var = current.get("response_time_variance", 0)
            
            if baseline_var == 0:
                return False
            
            # Calculate percentage change in variance
            change = abs(current_var - baseline_var) / baseline_var
            
            return change > (self.sensitivity * 2)  # More sensitive to pattern changes
        except Exception:
            return False
    
    async def _handle_drift_detection(
        self,
        agent_name: str,
        drift_details: Dict[str, Any],
        current_metrics: Dict[str, Any]
    ):
        """Handle detected drift."""
        try:
            drift_alert = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "drift_details": drift_details,
                "current_metrics": current_metrics,
                "severity": self._calculate_drift_severity(drift_details)
            }
            
            self._drift_alerts.append(drift_alert)
            
            # Log drift detection
            self.logger.warning(f"Drift detected for agent {agent_name}: {drift_details}")
            
            # Store drift information for analysis
            await self._store_drift_info(agent_name, drift_alert)
            
        except Exception as e:
            self.logger.error(f"Error handling drift detection: {str(e)}")
    
    def _calculate_drift_severity(self, drift_details: Dict[str, Any]) -> str:
        """Calculate severity of drift detection."""
        try:
            severity_score = 0
            
            for metric, details in drift_details.items():
                change = abs(details.get("change", 0))
                
                if change > 0.5:  # 50% change
                    severity_score += 3
                elif change > 0.2:  # 20% change
                    severity_score += 2
                elif change > 0.1:  # 10% change
                    severity_score += 1
            
            if severity_score >= 6:
                return "critical"
            elif severity_score >= 4:
                return "high"
            elif severity_score >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    async def _store_drift_info(self, agent_name: str, drift_alert: Dict[str, Any]):
        """Store drift information for analysis."""
        try:
            # This could be extended to store in a database or external system
            self.logger.info(f"Stored drift info for agent {agent_name}")
        except Exception as e:
            self.logger.error(f"Error storing drift info: {str(e)}")
    
    async def get_drift_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get drift detection summary."""
        try:
            if agent_name:
                alerts = [alert for alert in self._drift_alerts if alert["agent_name"] == agent_name]
            else:
                alerts = self._drift_alerts
            
            if not alerts:
                return {"total_drifts": 0, "agents": []}
            
            # Group by agent
            agent_drifts = {}
            for alert in alerts:
                agent = alert["agent_name"]
                if agent not in agent_drifts:
                    agent_drifts[agent] = {
                        "total_drifts": 0,
                        "severity_counts": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                        "last_drift": None
                    }
                
                agent_drifts[agent]["total_drifts"] += 1
                severity = alert.get("severity", "unknown")
                if severity in agent_drifts[agent]["severity_counts"]:
                    agent_drifts[agent]["severity_counts"][severity] += 1
                
                if not agent_drifts[agent]["last_drift"] or alert["timestamp"] > agent_drifts[agent]["last_drift"]:
                    agent_drifts[agent]["last_drift"] = alert["timestamp"]
            
            return {
                "total_drifts": len(alerts),
                "agents": agent_drifts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting drift summary: {str(e)}")
            return {"total_drifts": 0, "agents": []}
    
    async def get_recent_drifts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent drift alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_drifts = [
                alert for alert in self._drift_alerts
                if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
            ]
            
            # Sort by timestamp (newest first)
            recent_drifts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return recent_drifts
            
        except Exception as e:
            self.logger.error(f"Error getting recent drifts: {str(e)}")
            return []
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get drift detector statistics."""
        try:
            return {
                "window_size": self.window_size,
                "sensitivity": self.sensitivity,
                "monitored_agents": list(self._recent_metrics.keys()),
                "total_drift_alerts": len(self._drift_alerts),
                "baselines_established": len(self._baselines)
            }
        except Exception as e:
            self.logger.error(f"Error getting detector stats: {str(e)}")
            return {}
