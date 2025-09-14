#!/usr/bin/env python3
"""
Handoff-based Multi-Agent Swarm Example.
Demonstrates agent handoffs using create_handoff_tool and create_swarm.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from app import MultiAgentSwarmApp


async def main():
    """Demonstrate handoff-based multi-agent coordination."""
    print("Handoff-based Multi-Agent Swarm Example")
    print("=" * 60)
    
    try:
        # Initialize the application
        print("Initializing Multi-Agent Swarm Application...")
        app = MultiAgentSwarmApp()
        
        if not await app.initialize():
            print("[ERROR] Failed to initialize application")
            return
        
        print("[SUCCESS] Application initialized successfully")
        
        # Show system status
        status = await app.get_system_status()
        print(f"\nSwarm Type: {status.get('swarm_type', 'Unknown')}")
        print(f"Agents Available: {status.get('agent_swarm_status', {}).get('agents', [])}")
        
        # Show the swarm graph
        if app.agent_swarm:
            print("\nSwarm Workflow Graph:")
            app.agent_swarm.show_graph()
        
        # Test handoff scenarios
        print("\nTesting Agent Handoffs")
        print("=" * 40)
        
        handoff_queries = [
            "What are the customer service policies?",  # RAG
            "Show me all albums by The Rolling Stones",  # NL2SQL
            "What was my most recent invoice for customer ID 2?",  # Invoice
            "I need help with invoices and want to see our policies",  # Handoff
            "Show me albums by The Beatles and help me with billing"  # Handoff
        ]
        
        for i, query in enumerate(handoff_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 30)
            
            try:
                response = await app.process_query(
                    query=query,
                    user_id="test_user",
                    session_id=f"handoff_test_{i}"
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:100]}{'...' if len(response.get('content', '')) > 100 else ''}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
        
        # Cleanup
        await app.cleanup()
        print("\n[SUCCESS] Handoff-based Multi-Agent Swarm Example Completed!")
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
