#!/usr/bin/env python3
"""
Comprehensive Systems Demo - demonstrates all core systems.
Shows RAG, NL2SQL, Invoice, and State management in one unified example.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from app import MultiAgentSwarmApp


class ComprehensiveSystemsDemo:
    """Demonstrates all core systems in a unified example."""
    
    def __init__(self):
        self.logger = logging.getLogger("comprehensive_demo")
        self.setup_logging()
        
        # Initialize the main application
        self.app = None
    
    def setup_logging(self):
        """Setup logging for the demo."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize(self):
        """Initialize the application."""
        print("Comprehensive Systems Demo")
        print("=" * 60)
        print("Demonstrating RAG, NL2SQL, Invoice, and State Management")
        print("=" * 60)
        
        try:
            # Reload config to ensure latest environment variables
            from core.config.config import config
            config.reload()
            
            # Initialize the application
            print("Initializing Multi-Agent Swarm Application...")
            self.app = MultiAgentSwarmApp()
            
            if not await self.app.initialize():
                print("[ERROR] Failed to initialize application")
                return False
            
            print("[SUCCESS] Application initialized successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error initializing application: {str(e)}")
            return False
    
    async def demonstrate_rag_system(self):
        """Demonstrate RAG system capabilities."""
        print("\n" + "=" * 60)
        print("RAG SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        rag_queries = [
            "What are the customer service policies?",
            "Tell me about the company's data privacy policy",
            "How does the refund process work?",
            "What are the terms and conditions?",
            "Can you explain the company's mission statement?"
        ]
        
        for i, query in enumerate(rag_queries, 1):
            print(f"\n{i}. RAG Query: {query}")
            print("-" * 40)
            
            try:
                response = await self.app.process_query(
                    query=query,
                    user_id="demo_user",
                    session_id=f"rag_demo_{i}"
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:150]}{'...' if len(response.get('content', '')) > 150 else ''}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
    
    async def demonstrate_nl2sql_system(self):
        """Demonstrate NL2SQL system capabilities."""
        print("\n" + "=" * 60)
        print("NL2SQL SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        nl2sql_queries = [
            "Show me all albums by The Rolling Stones",
            "What is the total revenue from all invoices?",
            "List all customers from the USA",
            "Show me the top 5 best-selling albums",
            "What's the average invoice amount?"
        ]
        
        for i, query in enumerate(nl2sql_queries, 1):
            print(f"\n{i}. NL2SQL Query: {query}")
            print("-" * 40)
            
            try:
                response = await self.app.process_query(
                    query=query,
                    user_id="demo_user",
                    session_id=f"nl2sql_demo_{i}"
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:150]}{'...' if len(response.get('content', '')) > 150 else ''}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
    
    async def demonstrate_invoice_system(self):
        """Demonstrate Invoice system capabilities."""
        print("\n" + "=" * 60)
        print("INVOICE SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        invoice_queries = [
            "What was my most recent invoice for customer ID 2?",
            "Who was the employee that helped me with invoice 293?",
            "Show me all invoices for customer ID 1",
            "What's the total amount for invoice 293?",
            "Which customer has the highest invoice total?"
        ]
        
        for i, query in enumerate(invoice_queries, 1):
            print(f"\n{i}. Invoice Query: {query}")
            print("-" * 40)
            
            try:
                response = await self.app.process_query(
                    query=query,
                    user_id="demo_user",
                    session_id=f"invoice_demo_{i}"
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:150]}{'...' if len(response.get('content', '')) > 150 else ''}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
    
    async def demonstrate_agent_handoffs(self):
        """Demonstrate agent handoff capabilities."""
        print("\n" + "=" * 60)
        print("AGENT HANDOFF DEMONSTRATION")
        print("=" * 60)
        
        handoff_queries = [
            "I need help with invoices and want to see our company policies",
            "Show me albums by The Beatles and help me with my recent invoice",
            "What are the customer service policies and show me the revenue data?",
            "I want to query the database and need help with billing",
            "Tell me about our refund policy and show me customer data"
        ]
        
        for i, query in enumerate(handoff_queries, 1):
            print(f"\n{i}. Handoff Query: {query}")
            print("-" * 40)
            
            try:
                response = await self.app.process_query(
                    query=query,
                    user_id="demo_user",
                    session_id=f"handoff_demo_{i}"
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:150]}{'...' if len(response.get('content', '')) > 150 else ''}")
                
                # Show handoff information
                metadata = response.get('metadata', {})
                if 'swarm_type' in metadata:
                    print(f"   Swarm Type: {metadata['swarm_type']}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
    
    async def demonstrate_state_management(self):
        """Demonstrate state management across conversations."""
        print("\n" + "=" * 60)
        print("STATE MANAGEMENT DEMONSTRATION")
        print("=" * 60)
        
        # Multi-turn conversation to show state persistence
        conversation = [
            "I need help with my account",
            "What was my most recent invoice?",
            "Who helped me with that invoice?",
            "Can you show me all my invoices?",
            "What albums do you have by The Rolling Stones?",
            "Tell me about the company policies"
        ]
        
        session_id = "state_demo_session"
        print(f"Multi-turn conversation (Session: {session_id})")
        print("-" * 40)
        
        for i, query in enumerate(conversation, 1):
            print(f"\nTurn {i}: {query}")
            
            try:
                response = await self.app.process_query(
                    query=query,
                    user_id="demo_user",
                    session_id=session_id
                )
                
                print(f"   Agent: {response.get('agent_used', 'Unknown')}")
                print(f"   Response: {response.get('content', 'No response')[:100]}{'...' if len(response.get('content', '')) > 100 else ''}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
    
    async def show_system_status(self):
        """Show comprehensive system status."""
        print("\n" + "=" * 60)
        print("SYSTEM STATUS")
        print("=" * 60)
        
        try:
            status = await self.app.get_system_status()
            
            print(f"Status: {status.get('status', 'Unknown')}")
            print(f"Initialized: {status.get('initialized', False)}")
            print(f"Running: {status.get('running', False)}")
            
            # Agent Swarm Status
            swarm_status = status.get('agent_swarm_status', {})
            if swarm_status:
                print(f"\nAgent Swarm Status: {swarm_status.get('status', 'Unknown')}")
                print(f"Swarm Type: {status.get('swarm_type', 'Unknown')}")
                print(f"Agents Available: {swarm_status.get('agents', [])}")
                print(f"Tools Available: {swarm_status.get('tools_available', 0)}")
                print(f"RAG Tools: {swarm_status.get('rag_tools', 0)}")
                print(f"NL2SQL Tools: {swarm_status.get('nl2sql_tools', 0)}")
                print(f"Invoice Tools: {swarm_status.get('invoice_tools', 0)}")
            
        except Exception as e:
            print(f"Error getting system status: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        
        try:
            if self.app:
                await self.app.cleanup()
                print("[SUCCESS] Cleanup completed successfully")
        except Exception as e:
            print(f"[ERROR] Error during cleanup: {str(e)}")
    
    async def run_demo(self):
        """Run the comprehensive demo."""
        try:
            # Initialize
            if not await self.initialize():
                return
            
            # Demonstrate each system
            await self.demonstrate_rag_system()
            await self.demonstrate_nl2sql_system()
            await self.demonstrate_invoice_system()
            await self.demonstrate_agent_handoffs()
            await self.demonstrate_state_management()
            
            # Show system status
            await self.show_system_status()
            
            # Cleanup
            await self.cleanup()
            
            print(f"\n{'='*60}")
            print("COMPREHENSIVE SYSTEMS DEMO COMPLETED!")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"[ERROR] Error in demo: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main function."""
    demo = ComprehensiveSystemsDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
