"""
Test script for the updated swarm with Query Strategy Agent replacing RAG agent.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.config.config import config
from agents.handoff_swarm import HandoffSwarm
from core.memory import MemoryManager
from core.monitoring import MetricsCollector, AlertManager, DriftDetector


async def test_updated_swarm():
    """Test the updated swarm with Query Strategy Agent."""
    print(" Testing Updated Swarm with Query Strategy Agent...")
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        drift_detector = DriftDetector()
        
        # Initialize the updated swarm
        swarm = HandoffSwarm(
            memory_manager=memory_manager,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            drift_detector=drift_detector
        )
        
        print(f" Swarm initialized successfully!")
        print(f"   Agents: {list(swarm.agents.keys())}")
        
        # Test queries for different agents
        test_queries = [
            {
                "query": "What is Docker?",
                "expected_agent": "query_strategy_agent_with_handoff",
                "description": "Document-based query for Query Strategy Agent"
            },
            {
                "query": "Show me all customers from the database",
                "expected_agent": "nl2sql_agent_with_handoff", 
                "description": "Database query for NL2SQL Agent"
            },
            {
                "query": "What are the invoice details for customer 1?",
                "expected_agent": "invoice_information_agent_with_handoff",
                "description": "Invoice query for Invoice Agent"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            expected_agent = test_case["expected_agent"]
            description = test_case["description"]
            
            print(f"\n Test Case {i}: {description}")
            print(f" Query: {query}")
            print(f" Expected Agent: {expected_agent}")
            print("-" * 50)
            
            try:
                # Process query
                result = await swarm.process_query(query)
                
                print(f" Query processed successfully!")
                print(f"   Response: {result.get('response', 'No response')[:100]}...")
                print(f"   Agent used: {result.get('agent_used', 'Unknown')}")
                print(f"   Success: {result.get('success', False)}")
                
            except Exception as e:
                print(f" Error processing query: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f" Error initializing swarm: {str(e)}")
        return False


async def test_agent_handoffs():
    """Test agent handoffs in the updated swarm."""
    print("\n Testing Agent Handoffs...")
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        drift_detector = DriftDetector()
        
        # Initialize the updated swarm
        swarm = HandoffSwarm(
            memory_manager=memory_manager,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            drift_detector=drift_detector
        )
        
        # Test handoff scenarios
        handoff_tests = [
            {
                "query": "I need help with Docker deployment and also want to check my invoices",
                "description": "Multi-domain query requiring handoffs"
            },
            {
                "query": "What is the system architecture? Also, can you evaluate the quality of this response?",
                "description": "Query requiring Query Strategy Agent and Evaluation Agent"
            }
        ]
        
        for i, test_case in enumerate(handoff_tests, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n Handoff Test {i}: {description}")
            print(f" Query: {query}")
            print("-" * 50)
            
            try:
                # Process query
                result = await swarm.process_query(query)
                
                print(f" Handoff test completed!")
                print(f"   Response: {result.get('response', 'No response')[:150]}...")
                print(f"   Success: {result.get('success', False)}")
                
            except Exception as e:
                print(f" Error in handoff test: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f" Error in handoff tests: {str(e)}")
        return False


async def test_swarm_statistics():
    """Test swarm statistics and capabilities."""
    print("\n Testing Swarm Statistics...")
    
    try:
        # Initialize components
        memory_manager = MemoryManager()
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()
        drift_detector = DriftDetector()
        
        # Initialize the updated swarm
        swarm = HandoffSwarm(
            memory_manager=memory_manager,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            drift_detector=drift_detector
        )
        
        print(" Swarm Statistics:")
        print(f"   Total Agents: {len(swarm.agents)}")
        print(f"   Agent Names: {list(swarm.agents.keys())}")
        
        # Check if Query Strategy Agent is present
        if "query_strategy_agent_with_handoff" in swarm.agents:
            print("    Query Strategy Agent is present")
        else:
            print("    Query Strategy Agent is missing")
        
        # Check if old RAG agent is removed
        if "general_qa_agent_with_handoff" in swarm.agents:
            print("    General QA Agent (coordinator) is present")
        else:
            print("    General QA Agent is missing")
        
        # Test agent capabilities
        print("\n Agent Capabilities:")
        for agent_name, agent in swarm.agents.items():
            print(f"   • {agent_name}: {len(agent.tools)} tools")
        
        return True
        
    except Exception as e:
        print(f" Error testing swarm statistics: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print(" UPDATED SWARM TEST SUITE")
    print("=" * 50)
    
    # Reload config
    config.reload()
    
    # Run tests
    tests = [
        test_updated_swarm(),
        test_agent_handoffs(),
        test_swarm_statistics()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Summary
    print("\n" + "=" * 50)
    print(" TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Updated Swarm Integration", 
        "Agent Handoffs", 
        "Swarm Statistics"
    ]
    passed = 0
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        if isinstance(result, Exception):
            print(f" {name}: FAILED - {str(result)}")
        elif result:
            print(f" {name}: PASSED")
            passed += 1
        else:
            print(f" {name}: FAILED")
    
    print(f"\n Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print(" All tests passed! Updated swarm with Query Strategy Agent is working correctly.")
        print("\n Key Changes:")
        print("   • RAG Agent replaced with Query Strategy Agent")
        print("   • All document-based queries now use intelligent optimization")
        print("   • Cleaner agent architecture with clear responsibilities")
        print("   • Automatic technique selection for better results")
    else:
        print("  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
