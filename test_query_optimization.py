"""
Simple test script for query optimization features.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.config.config import config
from src.rag.optimization.query_optimizer import QueryRewriter, MMRReranker, QueryOptimizer
from langchain_core.documents import Document


async def test_query_rewriting():
    """Test query rewriting functionality."""
    print(" Testing Query Rewriting...")
    
    try:
        query_rewriter = QueryRewriter()
        
        test_query = "How does the system work?"
        print(f"Original query: {test_query}")
        
        result = await query_rewriter.rewrite_query(test_query)
        
        print(f" Query rewriting successful!")
        print(f"   Rewritten queries: {len(result.rewritten_queries)}")
        print(f"   Expansion terms: {len(result.expansion_terms)}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Query rewriting failed: {str(e)}")
        return False


def test_mmr_reranking():
    """Test MMR reranking functionality."""
    print("\n Testing MMR Reranking...")
    
    try:
        mmr_reranker = MMRReranker()
        
        # Create sample documents
        sample_docs = [
            Document(
                page_content="Docker is a containerization platform for packaging applications.",
                metadata={"source": "docker.md", "topic": "containerization"}
            ),
            Document(
                page_content="Docker containers provide isolation and consistency across environments.",
                metadata={"source": "containers.md", "topic": "isolation"}
            ),
            Document(
                page_content="Docker Compose manages multi-container applications with YAML files.",
                metadata={"source": "compose.md", "topic": "orchestration"}
            )
        ]
        
        test_query = "How to use Docker for deployment?"
        print(f"Query: {test_query}")
        print(f"Input documents: {len(sample_docs)}")
        
        mmr_result = mmr_reranker.calculate_mmr_scores(
            documents=sample_docs,
            query=test_query,
            lambda_param=0.7,
            k=2
        )
        
        print(f" MMR reranking successful!")
        print(f"   Output documents: {len(mmr_result.documents)}")
        print(f"   MMR scores: {[f'{score:.3f}' for score in mmr_result.scores]}")
        print(f"   Relevance scores: {[f'{score:.3f}' for score in mmr_result.relevance_scores]}")
        print(f"   Diversity scores: {[f'{score:.3f}' for score in mmr_result.diversity_scores]}")
        
        return True
        
    except Exception as e:
        print(f" MMR reranking failed: {str(e)}")
        return False


async def test_query_optimizer():
    """Test complete query optimizer."""
    print("\n Testing Complete Query Optimizer...")
    
    try:
        query_optimizer = QueryOptimizer()
        
        # Create sample documents
        sample_docs = [
            Document(
                page_content="The system uses LangGraph for multi-agent coordination.",
                metadata={"source": "langgraph.md", "topic": "coordination"}
            ),
            Document(
                page_content="LangGraph enables dynamic agent handoffs and intelligent routing.",
                metadata={"source": "handoffs.md", "topic": "handoffs"}
            ),
            Document(
                page_content="The system includes RAG, NL2SQL, and Invoice processing capabilities.",
                metadata={"source": "systems.md", "topic": "capabilities"}
            )
        ]
        
        test_query = "How does the multi-agent system work?"
        print(f"Query: {test_query}")
        print(f"Input documents: {len(sample_docs)}")
        
        optimized_docs, metadata = await query_optimizer.optimize_retrieval(
            query=test_query,
            documents=sample_docs,
            lambda_param=0.7,
            k=2,
            use_query_rewriting=True
        )
        
        print(f" Query optimization successful!")
        print(f"   Output documents: {len(optimized_docs)}")
        print(f"   Final doc count: {metadata['mmr_result']['final_doc_count']}")
        print(f"   Avg relevance: {metadata['mmr_result']['avg_relevance_score']:.3f}")
        print(f"   Avg diversity: {metadata['mmr_result']['avg_diversity_score']:.3f}")
        print(f"   Avg MMR score: {metadata['mmr_result']['avg_mmr_score']:.3f}")
        
        if 'rewrite_result' in metadata:
            rewrite_info = metadata['rewrite_result']
            print(f"   Rewritten queries: {len(rewrite_info['rewritten_queries'])}")
            print(f"   Expansion terms: {len(rewrite_info['expansion_terms'])}")
        
        return True
        
    except Exception as e:
        print(f" Query optimization failed: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print(" QUERY OPTIMIZATION TEST SUITE")
    print("=" * 50)
    
    # Reload config
    config.reload()
    
    # Run tests
    tests = [
        test_query_rewriting(),
        test_mmr_reranking(),
        test_query_optimizer()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Summary
    print("\n" + "=" * 50)
    print(" TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = ["Query Rewriting", "MMR Reranking", "Query Optimizer"]
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
        print(" All tests passed! Query optimization is working correctly.")
    else:
        print("  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
