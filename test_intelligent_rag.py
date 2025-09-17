"""
Simple test script for the Intelligent RAG System.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.config.config import config
from src.rag.systems.intelligent_rag_system import get_intelligent_rag_system
from src.rag.strategies.query_strategy_agent import RetrievalTechnique
from langchain_core.documents import Document


async def test_strategy_agent():
    """Test the query strategy agent."""
    print(" Testing Query Strategy Agent...")
    
    try:
        from src.rag.strategies.query_strategy_agent import get_query_strategy_agent
        
        strategy_agent = get_query_strategy_agent()
        
        test_query = "How does the multi-agent system work?"
        print(f"Query: {test_query}")
        
        query_analysis, strategy_decision = await strategy_agent.process_query(test_query)
        
        print(f" Strategy agent test successful!")
        print(f"   Complexity: {query_analysis.complexity}")
        print(f"   Intent: {query_analysis.intent}")
        print(f"   Domain: {query_analysis.domain}")
        print(f"   Selected technique: {strategy_decision.technique}")
        print(f"   Confidence: {strategy_decision.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Strategy agent test failed: {str(e)}")
        return False


async def test_multi_query_rag():
    """Test the multi-query RAG system."""
    print("\n Testing Multi-Query RAG...")
    
    try:
        from src.rag.strategies.multi_query_rag import get_multi_query_rag
        
        multi_query_rag = get_multi_query_rag()
        
        # Mock retriever function
        def mock_retriever(query: str, k: int = 3) -> list[Document]:
            return [
                Document(
                    page_content=f"Mock document for query: {query}",
                    metadata={"source": "mock.md", "topic": "test"}
                )
                for _ in range(min(k, 2))
            ]
        
        test_query = "What are the benefits of using Docker?"
        print(f"Query: {test_query}")
        
        result = await multi_query_rag.process_query(
            query=test_query,
            retriever_func=mock_retriever,
            query_count=3,
            k_per_query=2,
            final_k=5
        )
        
        print(f" Multi-query RAG test successful!")
        print(f"   Generated queries: {len(result.generated_queries)}")
        print(f"   Retrieved documents: {len(result.retrieved_documents)}")
        print(f"   Fused documents: {len(result.fused_documents)}")
        
        return True
        
    except Exception as e:
        print(f" Multi-query RAG test failed: {str(e)}")
        return False


async def test_query_routing():
    """Test the query routing system."""
    print("\n Testing Query Routing...")
    
    try:
        from src.rag.strategies.query_routing import get_query_routing_system
        
        query_routing = get_query_routing_system()
        
        # Mock retriever function
        def mock_retriever(query: str, k: int = 3) -> list[Document]:
            return [
                Document(
                    page_content=f"Mock document for query: {query}",
                    metadata={"source": "mock.md", "topic": "test"}
                )
                for _ in range(min(k, 2))
            ]
        
        test_query = "How to deploy the system with Docker?"
        print(f"Query: {test_query}")
        
        result = await query_routing.process_query(
            query=test_query,
            retriever_func=mock_retriever,
            max_routes=2,
            k_per_route=2
        )
        
        print(f" Query routing test successful!")
        print(f"   Selected routes: {len(result.selected_routes)}")
        print(f"   Retrieved documents: {len(result.retrieved_documents)}")
        print(f"   Route scores: {result.route_scores}")
        
        return True
        
    except Exception as e:
        print(f" Query routing test failed: {str(e)}")
        return False


async def test_multi_hop_rag():
    """Test the multi-hop RAG system."""
    print("\n Testing Multi-Hop RAG...")
    
    try:
        from src.rag.strategies.multi_hop_rag import get_multi_hop_rag
        
        multi_hop_rag = get_multi_hop_rag()
        
        # Mock retriever function
        def mock_retriever(query: str, k: int = 3) -> list[Document]:
            return [
                Document(
                    page_content=f"Mock document for query: {query}",
                    metadata={"source": "mock.md", "topic": "test"}
                )
                for _ in range(min(k, 2))
            ]
        
        test_query = "How to implement monitoring for a distributed system?"
        print(f"Query: {test_query}")
        
        result = await multi_hop_rag.process_query(
            query=test_query,
            retriever_func=mock_retriever,
            max_hops=2,
            k_per_hop=2
        )
        
        print(f" Multi-hop RAG test successful!")
        print(f"   Hops executed: {len(result.hops)}")
        print(f"   Final documents: {len(result.final_documents)}")
        print(f"   Reasoning chain: {len(result.reasoning_chain)} steps")
        
        return True
        
    except Exception as e:
        print(f" Multi-hop RAG test failed: {str(e)}")
        return False


async def test_intelligent_rag_system():
    """Test the complete intelligent RAG system."""
    print("\n Testing Intelligent RAG System...")
    
    try:
        intelligent_rag = get_intelligent_rag_system()
        
        # Mock retriever function
        def mock_retriever(query: str, k: int = 3) -> list[Document]:
            return [
                Document(
                    page_content=f"Mock document for query: {query}",
                    metadata={"source": "mock.md", "topic": "test"}
                )
                for _ in range(min(k, 2))
            ]
        
        def mock_response_generator(query: str, context: str) -> str:
            return f"Mock response for query: {query}"
        
        test_query = "What is the architecture of the multi-agent system?"
        print(f"Query: {test_query}")
        
        result = await intelligent_rag.process_query(
            query=test_query,
            retriever_func=mock_retriever,
            generate_response=True,
            response_generator_func=mock_response_generator
        )
        
        print(f" Intelligent RAG system test successful!")
        print(f"   Selected technique: {result.selected_technique}")
        print(f"   Query complexity: {result.query_analysis.complexity}")
        print(f"   Retrieved documents: {len(result.retrieved_documents)}")
        print(f"   Response generated: {'Yes' if result.final_response else 'No'}")
        print(f"   Confidence: {result.strategy_decision.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Intelligent RAG system test failed: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print(" INTELLIGENT RAG SYSTEM TEST SUITE")
    print("=" * 50)
    
    # Reload config
    config.reload()
    
    # Run tests
    tests = [
        test_strategy_agent(),
        test_multi_query_rag(),
        test_query_routing(),
        test_multi_hop_rag(),
        test_intelligent_rag_system()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Summary
    print("\n" + "=" * 50)
    print(" TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Query Strategy Agent", 
        "Multi-Query RAG", 
        "Query Routing", 
        "Multi-Hop RAG", 
        "Intelligent RAG System"
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
        print(" All tests passed! Intelligent RAG system is working correctly.")
    else:
        print("  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
