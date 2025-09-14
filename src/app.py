"""
Main application entry point for the Multi-Agent Swarm system.
Initializes and orchestrates all components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core.config import config
from core.memory import MemoryManager
from core.monitoring import MetricsCollector, AlertManager, DriftDetector, Dashboard
from infrastructure import DatabaseClient, setup_logging, LangChainBedrockClient
from agents import HandoffSwarm


class MultiAgentSwarmApp:
    """
    Main application class for the Multi-Agent Swarm system.
    Orchestrates all components and provides the main interface.
    """
    
    def __init__(self):
        # Initialize logging
        self.logger = setup_logging()
        
        # Core components
        self.memory_manager: Optional[MemoryManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.alert_manager: Optional[AlertManager] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.dashboard: Optional[Dashboard] = None
        
        # Infrastructure components
        self.database_client: Optional[DatabaseClient] = None
        self.langchain_client: Optional[LangChainBedrockClient] = None
        
        # Agent swarm (Handoff-based)
        self.agent_swarm: Optional[HandoffSwarm] = None
        
        # Application state
        self.initialized = False
        self.running = False
    
    async def initialize(self) -> bool:
        """
        Initialize all components of the multi-agent swarm system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing Multi-Agent Swarm System...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize infrastructure
            await self._initialize_infrastructure()
            
            # Initialize agent swarm
            await self._initialize_agent_swarm()
            
            # Initialize dashboard
            await self._initialize_dashboard()
            
            self.initialized = True
            self.logger.info("Multi-Agent Swarm System initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {str(e)}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core system components."""
        try:
            # Initialize memory manager
            self.memory_manager = MemoryManager(langchain_client=self.langchain_client)
            self.logger.info("Memory manager initialized")
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            self.logger.info("Metrics collector initialized")
            
            # Initialize alert manager
            self.alert_manager = AlertManager()
            self.logger.info("Alert manager initialized")
            
            # Initialize drift detector
            self.drift_detector = DriftDetector()
            self.logger.info("Drift detector initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing core components: {str(e)}")
            raise
    
    async def _initialize_infrastructure(self):
        """Initialize infrastructure components."""
        try:
            # Initialize database client
            self.database_client = DatabaseClient()
            if not await self.database_client.initialize():
                self.logger.warning("Database client initialization failed")
            else:
                self.logger.info("Database client initialized")
            
            # Initialize LangChain Bedrock client
            self.langchain_client = LangChainBedrockClient()
            if await self.langchain_client.initialize():
                self.logger.info("LangChain Bedrock client initialized")
            else:
                self.logger.warning("LangChain Bedrock client initialization failed")
            
        except Exception as e:
            self.logger.error(f"Error initializing infrastructure: {str(e)}")
            raise
    
    async def _initialize_agent_swarm(self):
        """Initialize the agent swarm with LangGraph."""
        try:
            if not all([self.memory_manager, self.metrics_collector, self.alert_manager, self.drift_detector]):
                raise Exception("Core components not initialized")
            
            # Try to initialize Handoff-based swarm first
            try:
                self.agent_swarm = HandoffSwarm(
                    memory_manager=self.memory_manager,
                    metrics_collector=self.metrics_collector,
                    alert_manager=self.alert_manager,
                    drift_detector=self.drift_detector,
                    langchain_client=self.langchain_client,
                    database_client=self.database_client
                )
                self.logger.info("Handoff-based swarm initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Handoff-based swarm initialization failed: {str(e)}")
                raise
            
        except Exception as e:
            self.logger.error(f"Error initializing agent swarm: {str(e)}")
            raise
    
    async def _initialize_dashboard(self):
        """Initialize the monitoring dashboard."""
        try:
            if not all([self.metrics_collector, self.alert_manager, self.drift_detector]):
                raise Exception("Monitoring components not initialized")
            
            self.dashboard = Dashboard(
                metrics_collector=self.metrics_collector,
                alert_manager=self.alert_manager,
                drift_detector=self.drift_detector
            )
            
            self.logger.info("Dashboard initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing dashboard: {str(e)}")
            raise
    
    async def process_query(
        self,
        query: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the agent swarm.
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier (generated if not provided)
        
        Returns:
            Response dictionary with agent response and metadata
        """
        try:
            if not self.initialized:
                return {
                    "error": "System not initialized",
                    "success": False
                }
            
            if not self.agent_swarm:
                return {
                    "error": "No agent swarm available",
                    "success": False
                }
            
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Process query through LangGraph swarm
            response = await self.agent_swarm.process_query(
                query=query,
                user_id=user_id,
                session_id=session_id
            )
            
            # Convert response to dictionary
            result = {
                "success": True,
                "response": response.content,
                "agent_name": response.agent_name,
                "confidence": response.confidence,
                "metadata": response.metadata,
                "execution_time": response.execution_time,
                "timestamp": response.timestamp.isoformat() if response.timestamp else None
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            if not self.initialized:
                return {
                    "status": "not_initialized",
                    "initialized": False,
                    "running": False
                }
            
            status = {
                "status": "initialized",
                "initialized": self.initialized,
                "running": self.running,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add component status
            if self.langchain_client:
                status["aws_status"] = await self.langchain_client.health_check()
            
            if self.database_client:
                status["database_status"] = await self.database_client.health_check()
            
            if self.agent_swarm:
                status["agent_swarm_status"] = await self.agent_swarm.get_swarm_status()
                status["swarm_type"] = "LangGraph"
            
            if self.metrics_collector:
                status["metrics_summary"] = self.metrics_collector.get_metrics_summary()
            
            if self.alert_manager:
                status["alert_summary"] = await self.alert_manager.get_alert_statistics()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "initialized": self.initialized,
                "running": self.running
            }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring."""
        try:
            if not self.dashboard:
                return {"error": "Dashboard not initialized"}
            
            return {
                "system_overview": await self.dashboard.get_system_overview(),
                "agent_performance": await self.dashboard.get_agent_performance_dashboard(),
                "alerts": await self.dashboard.get_alerts_dashboard(),
                "drift": await self.dashboard.get_drift_dashboard(),
                "real_time_metrics": await self.dashboard.get_real_time_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {str(e)}")
            return {"error": str(e)}
    
    async def start(self):
        """Start the multi-agent swarm system."""
        try:
            if not self.initialized:
                self.logger.error("System not initialized. Call initialize() first.")
                return False
            
            self.running = True
            self.logger.info("Multi-Agent Swarm System started")
            
            # Start background tasks
            asyncio.create_task(self._background_monitoring())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}")
            return False
    
    async def stop(self):
        """Stop the multi-agent swarm system."""
        try:
            self.running = False
            
            # Shutdown components
            if self.agent_swarm:
                await self.agent_swarm.shutdown()
            
            if self.langchain_client:
                await self.langchain_client.close()
            
            if self.database_client:
                await self.database_client.close()
            
            if self.memory_manager:
                await self.memory_manager.close()
            
            self.logger.info("Multi-Agent Swarm System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}")
    
    async def _background_monitoring(self):
        """Background monitoring task."""
        try:
            while self.running:
                # Update system health metrics
                if self.metrics_collector:
                    # This would typically get real system metrics
                    await self.metrics_collector.record_system_health(
                        cpu_usage=50.0,  # Placeholder
                        memory_usage=60.0,  # Placeholder
                        active_agents=3,
                        active_sessions=1
                    )
                
                # Check for drift
                if self.drift_detector:
                    # This would typically check for drift in agent behavior
                    pass
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except Exception as e:
            self.logger.error(f"Error in background monitoring: {str(e)}")
    
    async def export_system_data(self, filepath: str) -> bool:
        """Export system data for analysis."""
        try:
            if not self.dashboard:
                return False
            
            return await self.dashboard.export_dashboard_data(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting system data: {str(e)}")
            return False


# Global application instance
app_instance: Optional[MultiAgentSwarmApp] = None


async def get_app() -> MultiAgentSwarmApp:
    """Get the global application instance."""
    global app_instance
    if app_instance is None:
        app_instance = MultiAgentSwarmApp()
        await app_instance.initialize()
    return app_instance


async def main():
    """Main entry point for the application."""
    try:
        # Create and initialize application
        app = MultiAgentSwarmApp()
        
        if not await app.initialize():
            print("Failed to initialize application")
            return
        
        # Start the application
        if not await app.start():
            print("Failed to start application")
            return
        
        print("Multi-Agent Swarm System is running...")
        print("Press Ctrl+C to stop")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            await app.stop()
            
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
