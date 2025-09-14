#!/usr/bin/env python3
"""
Complete System Test for Handoff-based Multi-Agent Swarm.
Tests the exact scenarios from the reference with proper handoff mechanisms.
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app import MultiAgentSwarmApp


async def test_case_1_rolling_stones():
    """Test Case #1: Rolling Stones albums query - demonstrates General QA -> NL2SQL handoff."""
    print("=" * 80)
    print("TEST CASE #1: Rolling Stones Albums Query")
    print("=" * 80)
    print("Testing: General QA Agent -> NL2SQL Agent handoff")
    print()
    
    try:
        # Initialize the application
        app = MultiAgentSwarmApp()
        if not await app.initialize():
            print(" Failed to initialize application")
            return
        
        # Create a new thread
        thread_id = uuid.uuid4()
        question = "Do you have any albums by the Rolling Stones?"
        
        print(f"Thread ID: {thread_id}")
        print(f"Question: {question}")
        print()
        
        # Process the query
        response = await app.process_query(
            query=question,
            user_id="test_user",
            session_id=str(thread_id)
        )
        
        print("=" * 50)
        print("HUMAN MESSAGE")
        print("=" * 50)
        print(question)
        print()
        
        print("=" * 50)
        print("AI MESSAGE")
        print("=" * 50)
        print(f"Agent Used: {response.get('agent_used', 'Unknown')}")
        print(f"Response: {response.get('content', 'No response')}")
        print()
        
        # Check if handoff occurred
        metadata = response.get('metadata', {})
        if 'swarm_type' in metadata:
            print(f"Swarm Type: {metadata['swarm_type']}")
        
        await app.cleanup()
        
    except Exception as e:
        print(f" Error in Test Case #1: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_case_2_invoice_query():
    """Test Case #2: Invoice query - demonstrates General QA -> Invoice handoff."""
    print("=" * 80)
    print("TEST CASE #2: Invoice Query")
    print("=" * 80)
    print("Testing: General QA Agent -> Invoice Agent handoff")
    print()
    
    try:
        # Initialize the application
        app = MultiAgentSwarmApp()
        if not await app.initialize():
            print(" Failed to initialize application")
            return
        
        # Create a new thread
        thread_id = uuid.uuid4()
        question = "My customer id is 2. What was my most recent invoice, and who was the employee that helped me with it?"
        
        print(f"Thread ID: {thread_id}")
        print(f"Question: {question}")
        print()
        
        # Process the query
        response = await app.process_query(
            query=question,
            user_id="test_user",
            session_id=str(thread_id)
        )
        
        print("=" * 50)
        print("HUMAN MESSAGE")
        print("=" * 50)
        print(question)
        print()
        
        print("=" * 50)
        print("AI MESSAGE")
        print("=" * 50)
        print(f"Agent Used: {response.get('agent_used', 'Unknown')}")
        print(f"Response: {response.get('content', 'No response')}")
        print()
        
        # Check if handoff occurred
        metadata = response.get('metadata', {})
        if 'swarm_type' in metadata:
            print(f"Swarm Type: {metadata['swarm_type']}")
        
        await app.cleanup()
        
    except Exception as e:
        print(f" Error in Test Case #2: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_case_3_company_location():
    """Test Case #3: Company location query - demonstrates General QA staying with RAG."""
    print("=" * 80)
    print("TEST CASE #3: Company Location Query")
    print("=" * 80)
    print("Testing: General QA Agent stays with RAG (no handoff needed)")
    print()
    
    try:
        # Initialize the application
        app = MultiAgentSwarmApp()
        if not await app.initialize():
            print(" Failed to initialize application")
            return
        
        # Create a new thread
        thread_id = uuid.uuid4()
        question = "Tell me about the company location."
        
        print(f"Thread ID: {thread_id}")
        print(f"Question: {question}")
        print()
        
        # Process the query
        response = await app.process_query(
            query=question,
            user_id="test_user",
            session_id=str(thread_id)
        )
        
        print("=" * 50)
        print("HUMAN MESSAGE")
        print("=" * 50)
        print(question)
        print()
        
        print("=" * 50)
        print("AI MESSAGE")
        print("=" * 50)
        print(f"Agent Used: {response.get('agent_used', 'Unknown')}")
        print(f"Response: {response.get('content', 'No response')}")
        print()
        
        # Check if handoff occurred
        metadata = response.get('metadata', {})
        if 'swarm_type' in metadata:
            print(f"Swarm Type: {metadata['swarm_type']}")
        
        await app.cleanup()
        
    except Exception as e:
        print(f" Error in Test Case #3: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_handoff_workflow():
    """Test the complete handoff workflow with detailed logging."""
    print("=" * 80)
    print("COMPLETE HANDOFF WORKFLOW TEST")
    print("=" * 80)
    print("Testing the complete handoff mechanism with detailed analysis")
    print()
    
    try:
        # Initialize the application
        app = MultiAgentSwarmApp()
        if not await app.initialize():
            print(" Failed to initialize application")
            return
        
        # Test queries that should trigger different handoffs
        test_scenarios = [
            {
                "query": "Do you have any albums by The Beatles?",
                "expected_handoff": "General QA -> NL2SQL",
                "description": "Music catalog query should hand off to NL2SQL agent"
            },
            {
                "query": "What was my most recent invoice for customer ID 1?",
                "expected_handoff": "General QA -> Invoice",
                "description": "Invoice query should hand off to Invoice agent"
            },
            {
                "query": "What are the customer service policies?",
                "expected_handoff": "General QA stays (RAG)",
                "description": "Document query should stay with General QA agent"
            },
            {
                "query": "Show me all customers in the database",
                "expected_handoff": "General QA -> NL2SQL",
                "description": "Database query should hand off to NL2SQL agent"
            },
            {
                "query": "Who was the employee that helped me with invoice 293?",
                "expected_handoff": "General QA -> Invoice",
                "description": "Employee query should hand off to Invoice agent"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['description']}")
            print(f"   Query: {scenario['query']}")
            print(f"   Expected Handoff: {scenario['expected_handoff']}")
            print("-" * 60)
            
            try:
                # Process the query
                response = await app.process_query(
                    query=scenario['query'],
                    user_id="test_user",
                    session_id=f"workflow_test_{i}"
                )
                
                print(f"   Agent Used: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:150]}{'...' if len(response.get('content', '')) > 150 else ''}")
                
                # Analyze handoff
                agent_used = response.get('agent_used', 'Unknown')
                if 'nl2sql' in agent_used.lower():
                    print("  Handoff to NL2SQL agent successful")
                elif 'invoice' in agent_used.lower():
                    print("  Handoff to Invoice agent successful")
                elif 'rag' in agent_used.lower() or 'general' in agent_used.lower():
                    print("  General QA agent handled query (no handoff needed)")
                else:
                    print(f"   Unexpected agent: {agent_used}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
        
        await app.cleanup()
        
    except Exception as e:
        print(f" Error in workflow test: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_multi_turn_conversation():
    """Test multi-turn conversation with handoffs."""
    print("=" * 80)
    print("MULTI-TURN CONVERSATION TEST")
    print("=" * 80)
    print("Testing conversation flow with multiple handoffs")
    print()
    
    try:
        # Initialize the application
        app = MultiAgentSwarmApp()
        if not await app.initialize():
            print(" Failed to initialize application")
            return
        
        # Multi-turn conversation
        conversation = [
            "I need help with my account",
            "What was my most recent invoice?",
            "Who helped me with that invoice?",
            "Can you show me all my invoices?",
            "What albums do you have by The Rolling Stones?",
            "Tell me about the company policies"
        ]
        
        session_id = str(uuid.uuid4())
        print(f"Session ID: {session_id}")
        print()
        
        for i, query in enumerate(conversation, 1):
            print(f"Turn {i}: {query}")
            print("-" * 40)
            
            try:
                response = await app.process_query(
                    query=query,
                    user_id="test_user",
                    session_id=session_id
                )
                
                print(f"Agent: {response.get('agent_used', 'Unknown')}")
                print(f"Response: {response.get('content', 'No response')[:100]}{'...' if len(response.get('content', '')) > 100 else ''}")
                print()
                
            except Exception as e:
                print(f"Error: {str(e)}")
                print()
        
        await app.cleanup()
        
    except Exception as e:
        print(f" Error in multi-turn test: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("COMPLETE SYSTEM TEST")
    print("=" * 80)
    print("Testing the Handoff-based Multi-Agent Swarm System")
    print("=" * 80)
    print()
    
    try:
        # Run individual test cases
        await test_case_1_rolling_stones()
        print("\n" + "=" * 80 + "\n")
        
        await test_case_2_invoice_query()
        print("\n" + "=" * 80 + "\n")
        
        await test_case_3_company_location()
        print("\n" + "=" * 80 + "\n")
        
        # Run comprehensive tests
        await test_handoff_workflow()
        print("\n" + "=" * 80 + "\n")
        
        await test_multi_turn_conversation()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED!")
        print("=" * 80)
        
    except Exception as e:
        print(f" Error in main: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
