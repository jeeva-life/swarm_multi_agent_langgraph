"""
Demonstration of the Intelligent RAG System with automatic technique selection.
Shows how the system automatically chooses the best retrieval technique for different queries.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.config.config import config
from src.rag.systems.intelligent_rag_system import get_intelligent_rag_system
from src.rag.strategies.query_strategy_agent import RetrievalTechnique
from src.rag.systems.rag_system import get_rag_system
from langchain_core.documents import Document


async def demonstrate_technique_selection():
    """Demonstrate automatic technique selection for different query types."""
    
    print("=" * 80)
    print("INTELLIGENT RAG SYSTEM - AUTOMATIC TECHNIQUE SELECTION")
    print("=" * 80)
    print()
    
    # Initialize systems
    intelligent_rag = get_intelligent_rag_system()
    rag_system = get_rag_system()
    
    # Create a mock retriever function for demonstration
    def mock_retriever_func(query: str, k: int = 3) -> list[Document]:
        """Mock retriever function for demonstration."""
        # Create mock documents based on query content
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
        return f"Based on the retrieved information about '{query}', here is a comprehensive answer using the provided context. The system has successfully analyzed and processed your query using the selected technique."
    
    # Test queries representing different complexity levels and intents
    test_queries = [
        {
            "query": "What is Docker?",
            "expected_complexity": "simple",
            "description": "Simple, direct question"
        },
        {
            "query": "How does the multi-agent system coordinate and handle complex queries with different domains?",
            "expected_complexity": "complex",
            "description": "Complex, multi-faceted question requiring reasoning"
        },
        {
            "query": "Compare Docker vs Kubernetes for deployment strategies",
            "expected_complexity": "moderate",
            "description": "Comparison question requiring multiple perspectives"
        },
        {
            "query": "What are the security implications of using Docker in production environments?",
            "expected_complexity": "moderate",
            "description": "Domain-specific question requiring specialized knowledge"
        },
        {
            "query": "How to implement monitoring and alerting for a distributed system with multiple components?",
            "expected_complexity": "complex",
            "description": "Multi-step implementation question"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_complexity = test_case["expected_complexity"]
        description = test_case["description"]
        
        print(f" Test Case {i}: {description}")
        print(f" Query: {query}")
        print(f" Expected Complexity: {expected_complexity}")
        print("-" * 60)
        
        try:
            # Process query with intelligent RAG
            result = await intelligent_rag.process_query(
                query=query,
                retriever_func=mock_retriever_func,
                generate_response=True,
                response_generator_func=mock_response_generator
            )
            
            # Display results
            print(f" Query Analysis:")
            print(f"   Complexity: {result.query_analysis.complexity}")
            print(f"   Intent: {result.query_analysis.intent}")
            print(f"   Domain: {result.query_analysis.domain}")
            print(f"   Requires reasoning: {result.query_analysis.requires_reasoning}")
            print(f"   Requires multiple sources: {result.query_analysis.requires_multiple_sources}")
            print(f"   Requires specialized knowledge: {result.query_analysis.requires_specialized_knowledge}")
            print(f"   Confidence: {result.query_analysis.confidence_score:.2f}")
            
            print(f"\n Strategy Decision:")
            print(f"   Selected Technique: {result.selected_technique.value}")
            print(f"   Reasoning: {result.strategy_decision.reasoning}")
            print(f"   Confidence: {result.strategy_decision.confidence:.2f}")
            print(f"   Parameters: {result.strategy_decision.parameters}")
            
            print(f"\n Processing Results:")
            print(f"   Documents Retrieved: {len(result.retrieved_documents)}")
            print(f"   Response Generated: {'Yes' if result.final_response else 'No'}")
            print(f"   Technique Metadata: {result.technique_metadata}")
            
            if result.final_response:
                print(f"\n💬 Generated Response:")
                print(f"   {result.final_response[:200]}...")
            
        except Exception as e:
            print(f" Error processing query: {str(e)}")
        
        print("\n" + "=" * 80 + "\n")


async def demonstrate_technique_comparison():
    """Demonstrate how different techniques handle the same query."""
    
    print("=" * 80)
    print("TECHNIQUE COMPARISON - SAME QUERY, DIFFERENT APPROACHES")
    print("=" * 80)
    print()
    
    # Initialize systems
    intelligent_rag = get_intelligent_rag_system()
    
    # Test query
    test_query = "How to deploy and monitor a multi-agent system with Docker and Kubernetes?"
    
    print(f" Test Query: {test_query}")
    print()
    
    # Create mock retriever function
    def mock_retriever_func(query: str, k: int = 3) -> list[Document]:
        """Mock retriever function for demonstration."""
        mock_docs = []
        for i in range(min(k, 3)):
            doc = Document(
                page_content=f"Document {i+1}: {query} - This document contains information about deployment, monitoring, and container orchestration.",
                metadata={
                    "source": f"deployment_guide_{i+1}.md",
                    "topic": "deployment",
                    "relevance_score": 0.9 - (i * 0.1)
                }
            )
            mock_docs.append(doc)
        return mock_docs
    
    def mock_response_generator(query: str, context: str) -> str:
        """Mock response generator for demonstration."""
        return f"Comprehensive answer for '{query}' using the selected technique and retrieved context."
    
    # Test with different techniques
    techniques_to_test = [
        RetrievalTechnique.STANDARD_RAG,
        RetrievalTechnique.QUERY_REWRITING,
        RetrievalTechnique.MULTI_QUERY_RAG,
        RetrievalTechnique.QUERY_ROUTING,
        RetrievalTechnique.MULTI_HOP_RAG
    ]
    
    for technique in techniques_to_test:
        print(f" Testing Technique: {technique.value}")
        print("-" * 50)
        
        try:
            # Override strategy decision to test specific technique
            original_process_query = intelligent_rag.strategy_agent.process_query
            
            async def mock_process_query(query: str):
                from src.rag.strategies.query_strategy_agent import QueryAnalysis, StrategyDecision
                return (
                    QueryAnalysis(
                        query=query,
                        complexity="moderate",
                        intent="problem solving",
                        domain="technical",
                        requires_reasoning=True,
                        requires_multiple_sources=True,
                        requires_specialized_knowledge=True,
                        confidence_score=0.8
                    ),
                    StrategyDecision(
                        technique=technique,
                        reasoning=f"Testing {technique.value} technique",
                        confidence=0.9,
                        parameters={"k": 3}
                    )
                )
            
            intelligent_rag.strategy_agent.process_query = mock_process_query
            
            # Process query
            result = await intelligent_rag.process_query(
                query=test_query,
                retriever_func=mock_retriever_func,
                generate_response=True,
                response_generator_func=mock_response_generator
            )
            
            # Display results
            print(f"   Documents Retrieved: {len(result.retrieved_documents)}")
            print(f"   Processing Metadata: {result.processing_metadata}")
            print(f"   Technique Metadata: {result.technique_metadata}")
            
            if result.final_response:
                print(f"   Response: {result.final_response[:100]}...")
            
            # Restore original method
            intelligent_rag.strategy_agent.process_query = original_process_query
            
        except Exception as e:
            print(f"    Error: {str(e)}")
        
        print()


async def demonstrate_system_capabilities():
    """Demonstrate system capabilities and statistics."""
    
    print("=" * 80)
    print("SYSTEM CAPABILITIES AND STATISTICS")
    print("=" * 80)
    print()
    
    # Initialize system
    intelligent_rag = get_intelligent_rag_system()
    
    # Get available techniques
    available_techniques = intelligent_rag.get_available_techniques()
    
    print(" Available Retrieval Techniques:")
    for technique in available_techniques:
        print(f"   • {technique['technique']}: {technique['description']}")
        print(f"     Best for: {technique['best_for']}")
        print(f"     Parameters: {', '.join(technique['parameters'])}")
        print()
    
    # Get system statistics
    system_stats = intelligent_rag.get_system_statistics()
    
    print(" System Statistics:")
    for key, value in system_stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    print()


async def main():
    """Main demonstration function."""
    
    print(" INTELLIGENT RAG SYSTEM DEMONSTRATION")
    print("=" * 80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Reload config to ensure fresh environment variables
        config.reload()
        
        # Run demonstrations
        await demonstrate_technique_selection()
        await demonstrate_technique_comparison()
        await demonstrate_system_capabilities()
        
        print("=" * 80)
        print(" INTELLIGENT RAG SYSTEM DEMONSTRATION COMPLETED")
        print("=" * 80)
        print()
        print(" Key Features Demonstrated:")
        print("   • Automatic query analysis and complexity assessment")
        print("   • Intelligent technique selection based on query characteristics")
        print("   • Multiple retrieval techniques: Query Rewriting, Multi-Query RAG, Query Routing, Multi-Hop RAG")
        print("   • Comprehensive processing metadata and statistics")
        print("   • Fallback mechanisms for robustness")
        print()
        print(" Benefits:")
        print("   • Optimal technique selection for each query type")
        print("   • Improved retrieval quality through specialized approaches")
        print("   • Transparent decision-making with reasoning")
        print("   • Configurable parameters for fine-tuning")
        print("   • Comprehensive monitoring and statistics")
        
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
