"""
Demonstration of Query Optimization features in RAG system.
Shows query rewriting and Maximum Marginal Relevance (MMR) reranking.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.config.config import config
from src.rag.systems.rag_system import RAGSystem
from src.rag.optimization.query_optimizer import QueryRewriter, MMRReranker, QueryOptimizer


async def demonstrate_query_rewriting():
    """Demonstrate query rewriting capabilities."""
    
    print("=" * 80)
    print("QUERY REWRITING DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize query rewriter
    query_rewriter = QueryRewriter()
    
    # Test queries
    test_queries = [
        "How does the system work?",
        "What are the benefits of using Docker?",
        "How to deploy the application?",
        "What is the architecture of the system?"
    ]
    
    for query in test_queries:
        print(f" Original Query: {query}")
        print("-" * 50)
        
        try:
            # Rewrite the query
            rewrite_result = await query_rewriter.rewrite_query(query)
            
            print(f" Rewritten Queries ({len(rewrite_result.rewritten_queries)}):")
            for i, rewritten_query in enumerate(rewrite_result.rewritten_queries, 1):
                print(f"   {i}. {rewritten_query}")
            
            print(f" Expansion Terms ({len(rewrite_result.expansion_terms)}):")
            print(f"   {', '.join(rewrite_result.expansion_terms)}")
            
            print(f" Confidence Score: {rewrite_result.confidence_score:.2f}")
            
        except Exception as e:
            print(f" Error rewriting query: {str(e)}")
        
        print()


async def demonstrate_mmr_reranking():
    """Demonstrate MMR reranking capabilities."""
    
    print("=" * 80)
    print("MMR RERANKING DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize MMR reranker
    mmr_reranker = MMRReranker()
    
    # Create sample documents (simulating retrieved documents)
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
            metadata={"source": "docker_guide.md", "topic": "containerization"}
        ),
        Document(
            page_content="Docker containers provide isolation and consistency across different environments, making deployment easier and more reliable.",
            metadata={"source": "deployment.md", "topic": "deployment"}
        ),
        Document(
            page_content="Docker Compose is a tool for defining and running multi-container Docker applications using YAML files.",
            metadata={"source": "docker_compose.md", "topic": "orchestration"}
        ),
        Document(
            page_content="Docker images are read-only templates used to create containers. They contain the application code, runtime, and dependencies.",
            metadata={"source": "docker_basics.md", "topic": "fundamentals"}
        ),
        Document(
            page_content="Docker Hub is a cloud-based registry service where you can store and share Docker images with others.",
            metadata={"source": "docker_hub.md", "topic": "registry"}
        )
    ]
    
    test_query = "How to use Docker for application deployment?"
    
    print(f" Query: {test_query}")
    print(f" Original Documents: {len(sample_docs)}")
    print()
    
    # Test different lambda values
    lambda_values = [0.3, 0.5, 0.7, 0.9]
    
    for lambda_val in lambda_values:
        print(f"  Lambda = {lambda_val} (Relevance: {lambda_val:.1f}, Diversity: {1-lambda_val:.1f})")
        print("-" * 50)
        
        try:
            mmr_result = mmr_reranker.calculate_mmr_scores(
                documents=sample_docs,
                query=test_query,
                lambda_param=lambda_val,
                k=3
            )
            
            print(f" MMR Scores: {[f'{score:.3f}' for score in mmr_result.scores]}")
            print(f" Relevance Scores: {[f'{score:.3f}' for score in mmr_result.relevance_scores]}")
            print(f" Diversity Scores: {[f'{score:.3f}' for score in mmr_result.diversity_scores]}")
            
            print(" Reranked Documents:")
            for i, doc in enumerate(mmr_result.documents, 1):
                print(f"   {i}. {doc.metadata['source']} - {doc.metadata['topic']}")
                print(f"      {doc.page_content[:100]}...")
            
        except Exception as e:
            print(f" Error in MMR reranking: {str(e)}")
        
        print()


async def demonstrate_rag_optimization():
    """Demonstrate RAG system with query optimization."""
    
    print("=" * 80)
    print("RAG SYSTEM WITH QUERY OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Initialize RAG system with optimization enabled
    rag_system = RAGSystem(
        folder_path="docs",
        enable_query_optimization=True,
        mmr_lambda=0.7
    )
    
    # Get optimization stats
    optimization_stats = rag_system.get_optimization_stats()
    print(" Optimization Configuration:")
    for key, value in optimization_stats.items():
        print(f"   {key}: {value}")
    print()
    
    # Test queries
    test_queries = [
        "What is the system architecture?",
        "How to deploy with Docker?",
        "What are the monitoring capabilities?",
        "How does the multi-agent system work?"
    ]
    
    for query in test_queries:
        print(f" Query: {query}")
        print("-" * 50)
        
        try:
            # Test with optimization
            print(" With Query Optimization:")
            optimized_docs = rag_system.search_documents(query, k=3, use_optimization=True)
            print(f"   Retrieved {len(optimized_docs)} optimized documents")
            
            if optimized_docs:
                for i, doc in enumerate(optimized_docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"   {i}. {source}")
            
            # Test without optimization
            print(" Without Query Optimization:")
            standard_docs = rag_system.search_documents(query, k=3, use_optimization=False)
            print(f"   Retrieved {len(standard_docs)} standard documents")
            
            if standard_docs:
                for i, doc in enumerate(standard_docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"   {i}. {source}")
            
            # Generate response with optimization
            print(" Generated Response (with optimization):")
            response = rag_system.rag_response_generator(query)
            print(f"   {response[:200]}...")
            
        except Exception as e:
            print(f" Error in RAG optimization: {str(e)}")
        
        print()


async def demonstrate_query_optimizer():
    """Demonstrate the complete query optimizer."""
    
    print("=" * 80)
    print("COMPLETE QUERY OPTIMIZER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize query optimizer
    query_optimizer = QueryOptimizer()
    
    # Create sample documents
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="The system uses LangGraph for multi-agent coordination and state management.",
            metadata={"source": "architecture.md", "topic": "coordination"}
        ),
        Document(
            page_content="LangGraph enables dynamic agent handoffs and intelligent query routing.",
            metadata={"source": "langgraph.md", "topic": "handoffs"}
        ),
        Document(
            page_content="The system includes RAG, NL2SQL, and Invoice processing capabilities.",
            metadata={"source": "systems.md", "topic": "capabilities"}
        ),
        Document(
            page_content="Monitoring and alerting provide real-time system health visibility.",
            metadata={"source": "monitoring.md", "topic": "observability"}
        ),
        Document(
            page_content="Docker containerization enables easy deployment and scaling.",
            metadata={"source": "deployment.md", "topic": "deployment"}
        )
    ]
    
    test_query = "How does the multi-agent system coordinate and handle queries?"
    
    print(f" Query: {test_query}")
    print(f" Input Documents: {len(sample_docs)}")
    print()
    
    # Test optimization with different parameters
    test_configs = [
        {"lambda_param": 0.3, "k": 3, "use_query_rewriting": True},
        {"lambda_param": 0.7, "k": 3, "use_query_rewriting": True},
        {"lambda_param": 0.9, "k": 3, "use_query_rewriting": True},
        {"lambda_param": 0.7, "k": 3, "use_query_rewriting": False}
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f" Test Configuration {i}:")
        print(f"   Lambda: {config['lambda_param']}, K: {config['k']}, Query Rewriting: {config['use_query_rewriting']}")
        print("-" * 50)
        
        try:
            optimized_docs, metadata = await query_optimizer.optimize_retrieval(
                query=test_query,
                documents=sample_docs,
                **config
            )
            
            print(f" Optimization Results:")
            print(f"   Final documents: {metadata['mmr_result']['final_doc_count']}")
            print(f"   Avg relevance: {metadata['mmr_result']['avg_relevance_score']:.3f}")
            print(f"   Avg diversity: {metadata['mmr_result']['avg_diversity_score']:.3f}")
            print(f"   Avg MMR score: {metadata['mmr_result']['avg_mmr_score']:.3f}")
            
            if config['use_query_rewriting'] and 'rewrite_result' in metadata:
                rewrite_info = metadata['rewrite_result']
                print(f"   Rewritten queries: {len(rewrite_info['rewritten_queries'])}")
                print(f"   Expansion terms: {len(rewrite_info['expansion_terms'])}")
                print(f"   Confidence: {rewrite_info['confidence_score']:.3f}")
            
            print(" Selected Documents:")
            for j, doc in enumerate(optimized_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                topic = doc.metadata.get('topic', 'Unknown')
                print(f"   {j}. {source} ({topic})")
                print(f"      {doc.page_content[:80]}...")
            
        except Exception as e:
            print(f" Error in optimization: {str(e)}")
        
        print()


async def main():
    """Main demonstration function."""
    
    print(" QUERY OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Reload config to ensure fresh environment variables
        config.reload()
        
        # Run demonstrations
        await demonstrate_query_rewriting()
        await demonstrate_mmr_reranking()
        await demonstrate_rag_optimization()
        await demonstrate_query_optimizer()
        
        print("=" * 80)
        print(" QUERY OPTIMIZATION DEMONSTRATION COMPLETED")
        print("=" * 80)
        print()
        print(" Key Features Demonstrated:")
        print("   • Query rewriting with LLM-generated alternatives")
        print("   • Maximum Marginal Relevance (MMR) reranking")
        print("   • Balance between relevance and diversity")
        print("   • Integration with RAG system")
        print("   • Configurable optimization parameters")
        print()
        print(" Benefits:")
        print("   • Improved retrieval quality through query expansion")
        print("   • Better document diversity with MMR")
        print("   • Configurable relevance vs diversity balance")
        print("   • Fallback mechanisms for robustness")
        print("   • Comprehensive optimization metadata")
        
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
