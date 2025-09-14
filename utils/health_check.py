#!/usr/bin/env python3
"""
Health check utility for the Multi-Agent Swarm system.
Checks the status of all components.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import MultiAgentSwarmApp

async def check_health():
    """Check the health of all system components."""
    print("Multi-Agent Swarm Health Check")
    print("=" * 50)
    
    try:
        # Reload config to ensure latest environment variables
        from core.config import config
        config.reload()
        
        # Initialize the application
        app = MultiAgentSwarmApp()
        
        if not await app.initialize():
            print("[ERROR] Application initialization failed")
            return False
        
        print("[SUCCESS] Application initialized successfully")
        
        # Get system status
        status = await app.get_system_status()
        
        print(f"\nSystem Status: {status.get('status', 'Unknown')}")
        print(f"Initialized: {status.get('initialized', False)}")
        print(f"Running: {status.get('running', False)}")
        
        # Check individual components
        print("\nComponent Status:")
        print("-" * 30)
        
        # AWS Status
        aws_status = status.get('aws_status', {})
        if aws_status:
            print(f"[SUCCESS] AWS Bedrock: {aws_status.get('status', 'Unknown')}")
        else:
            print("[ERROR] AWS Bedrock: Not available")
        
        # Database Status
        db_status = status.get('database_status', {})
        if db_status:
            print(f"[SUCCESS] Database: {db_status.get('status', 'Unknown')}")
        else:
            print("[ERROR] Database: Not available")
        
        # Agent Swarm Status
        swarm_status = status.get('agent_swarm_status', {})
        if swarm_status:
            print(f"[SUCCESS] Agent Swarm: {swarm_status.get('status', 'Unknown')}")
            print(f"   Swarm Type: {status.get('swarm_type', 'Unknown')}")
        else:
            print("[ERROR] Agent Swarm: Not available")
        
        # Memory Status
        memory_status = status.get('memory_status', {})
        if memory_status:
            print(f"[SUCCESS] Memory: {memory_status.get('status', 'Unknown')}")
        else:
            print("[ERROR] Memory: Not available")
        
        # Metrics Status
        metrics_status = status.get('metrics_summary', {})
        if metrics_status:
            print(f"[SUCCESS] Metrics: Available")
        else:
            print("[ERROR] Metrics: Not available")
        
        # Cleanup (if available)
        if hasattr(app, 'cleanup'):
            await app.cleanup()
        
        print(f"\n{'='*50}")
        print("Health check completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Health check failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main health check function."""
    success = await check_health()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
