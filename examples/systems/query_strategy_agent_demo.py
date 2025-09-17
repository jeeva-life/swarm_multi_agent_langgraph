"""
Demonstration of the Query Strategy Agent with RAG Optimization Tools.
Shows how the agent automatically selects and uses the best RAG optimization technique.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.config.config import config
from agents.query_strategy_agent_with_tools import get_query_strategy_agent_with_tools
from src.rag.optimization.rag_optimization_tools import RAG_OPTIMIZATION_TOOLS
from langchain_core.documents import Document


async def demonstrate_query_strategy_agent():
    """Demonstrate the Query Strategy Agent with tools."""
    
    print("=" * 80)
    print("QUERY STRATEGY AGENT WITH RAG OPTIMIZATION TOOLS")
    print("=" * 80)
    print()
    
    # Initialize the agent
    query_strategy_agent = get_query_strategy_agent_with_tools()
    
    # Create a mock retriever function for demonstration
    def mock_retriever_func(query: str, k: int = 3) -> list[Document]:
        """Mock retriever function for demonstration."""
        mock_docs = []
        for i in range(min(k, 3)):
            doc = Document(
                page_content=f"Mock document {i+1} for query: {query}. This contains relevant information about the topic.",
                metadata={
                    "source": f"mock_source_{i+1}.md",
                    "topic": "demonstration",
                    "relevance_score": 0.9 - (i * 0.1)
                }
            )
            mock_docs.append(doc)
        return mock_docs
    
    def mock_response_generator(query: str, context: str) -> str:
        """Mock response generator for demonstration."""
        return f"Based on the optimized retrieval for '{query}', here is a comprehensive answer using the selected RAG optimization technique. The system has successfully analyzed and processed your query."
    
    # Test queries representing different complexity levels
    test_queries = [
        {
            "query": "What is Docker?",
            "description": "Simple, direct question",
            "expected_tool": "standard_rag_tool"
        },
        {
            "query": "How does the multi-agent system coordinate and handle complex queries with different domains?",
            "description": "Complex, multi-faceted question requiring reasoning",
            "expected_tool": "multi_hop_rag_tool"
        },
        {
            "query": "Compare Docker vs Kubernetes for deployment strategies",
            "description": "Comparison question requiring multiple perspectives",
            "expected_tool": "multi_query_rag_tool"
        },
        {
            "query": "What are the security implications of using Docker in production environments?",
            "description": "Domain-specific question requiring specialized knowledge",
            "expected_tool": "query_routing_tool"
        },
        {
            "query": "How to implement monitoring and alerting for a distributed system?",
            "description": "Multi-step implementation question",
            "expected_tool": "hybrid_rag_tool"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected_tool = test_case["expected_tool"]
        
        print(f" Test Case {i}: {description}")
        print(f" Query: {query}")
        print(f" Expected Tool: {expected_tool}")
        print("-" * 60)
        
        try:
            # Process query with the agent
            result = await query_strategy_agent.analyze_and_process_query(
                query=query,
                retriever_func=mock_retriever_func,
                generate_response=True,
                response_generator_func=mock_response_generator
            )
            
            # Display results
            print(f" Agent Output:")
            print(f"   {result['agent_output'][:200]}...")
            
            print(f"\n Parsed Result:")
            parsed = result['parsed_result']
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    print(f"   {key}: {value}")
            
            print(f"\n Final Response:")
            print(f"   {result['final_response'][:150]}...")
            
            print(f"\n Success: {result['success']}")
            
        except Exception as e:
            print(f" Error processing query: {str(e)}")
        
        print("\n" + "=" * 80 + "\n")


async def demonstrate_individual_tools():
    """Demonstrate individual RAG optimization tools."""
    
    print("=" * 80)
    print("INDIVIDUAL RAG OPTIMIZATION TOOLS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the agent
    query_strategy_agent = get_query_strategy_agent_with_tools()
    
    # Create a mock retriever function
    def mock_retriever_func(query: str, k: int = 3) -> list[Document]:
        """Mock retriever function for demonstration."""
        mock_docs = []
        for i in range(min(k, 3)):
            doc = Document(
                page_content=f"Document {i+1}: {query} - This document contains information about the topic.",
                metadata={
                    "source": f"source_{i+1}.md",
                    "topic": "test",
                    "relevance_score": 0.9 - (i * 0.1)
                }
            )
            mock_docs.append(doc)
        return mock_docs
    
    # Test individual tools
    test_query = "How to deploy and monitor a multi-agent system with Docker and Kubernetes?"
    
    print(f" Test Query: {test_query}")
    print()
    
    # Test each tool
    tools_to_test = [
        ("standard_rag_tool", {"k": 3}),
        ("query_rewriting_tool", {"expansion_factor": 2.0, "rewrite_count": 3}),
        ("multi_query_rag_tool", {"query_count": 5, "final_k": 8}),
        ("query_routing_tool", {"max_routes": 3, "k_per_route": 2}),
        ("multi_hop_rag_tool", {"max_hops": 2, "k_per_hop": 2}),
        ("hybrid_rag_tool", {"primary_technique": "multi_query_rag", "secondary_technique": "query_rewriting"})
    ]
    
    for tool_name, parameters in tools_to_test:
        print(f" Testing Tool: {tool_name}")
        print(f"   Parameters: {parameters}")
        print("-" * 50)
        
        try:
            result = await query_strategy_agent.process_query_with_tool(
                query=test_query,
                tool_name=tool_name,
                retriever_func=mock_retriever_func,
                tool_parameters=parameters
            )
            
            print(f"   Success: {result['success']}")
            if result['success'] and result['tool_result']:
                tool_result = result['tool_result']
                print(f"   Technique: {tool_result.technique}")
                print(f"   Documents Retrieved: {len(tool_result.documents)}")
                print(f"   Success: {tool_result.success}")
                if tool_result.metadata:
                    print(f"   Metadata: {tool_result.metadata}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
        
        print()


async def demonstrate_agent_capabilities():
    """Demonstrate agent capabilities and statistics."""
    
    print("=" * 80)
    print("AGENT CAPABILITIES AND STATISTICS")
    print("=" * 80)
    print()
    
    # Initialize the agent
    query_strategy_agent = get_query_strategy_agent_with_tools()
    
    # Get available tools
    available_tools = query_strategy_agent.get_available_tools()
    
    print(" Available RAG Optimization Tools:")
    for tool in available_tools:
        print(f"   • {tool['name']}: {tool['description']}")
        print(f"     Best for: {tool['best_for']}")
        print(f"     Parameters: {', '.join(tool['parameters'])}")
        print()
    
    # Get agent statistics
    agent_stats = query_strategy_agent.get_agent_statistics()
    
    print(" Agent Statistics:")
    for key, value in agent_stats.items():
        if isinstance(value, list):
            print(f"   {key}: {', '.join(value)}")
        else:
            print(f"   {key}: {value}")
    
    print()
    
    # Show RAG optimization tools
    print("  RAG Optimization Tools Available:")
    for tool in RAG_OPTIMIZATION_TOOLS:
        print(f"   • {tool.name}: {tool.description}")
    
    print()


async def main():
    """Main demonstration function."""
    
    print(" QUERY STRATEGY AGENT WITH TOOLS DEMONSTRATION")
    print("=" * 80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Reload config to ensure fresh environment variables
        config.reload()
        
        # Run demonstrations
        await demonstrate_query_strategy_agent()
        await demonstrate_individual_tools()
        await demonstrate_agent_capabilities()
        
        print("=" * 80)
        print(" QUERY STRATEGY AGENT DEMONSTRATION COMPLETED")
        print("=" * 80)
        print()
        print(" Key Features Demonstrated:")
        print("   • Intelligent query analysis and tool selection")
        print("   • RAG optimization tools as agent capabilities")
        print("   • Automatic technique selection based on query characteristics")
        print("   • Integration with multi-agent swarm system")
        print("   • Comprehensive tool parameter optimization")
        print()
        print(" Benefits:")
        print("   • Tools-based architecture for modularity")
        print("   • Agent can reason about which tool to use")
        print("   • Easy to add new optimization techniques")
        print("   • Transparent decision-making process")
        print("   • Seamless integration with existing swarm")
        
    except Exception as e:
        print(f" Error in demonstration: {str(e)}")
        logging.error(f"Demonstration error: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    asyncio.run(main())
