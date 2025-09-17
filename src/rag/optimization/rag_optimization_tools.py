"""
RAG Optimization Tools for the Query Strategy Agent.
These tools implement different query optimization techniques that the agent can use.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic

from core.config.config import config
from .query_optimizer import get_query_optimizer
from ..strategies.multi_query_rag import get_multi_query_rag
from ..strategies.query_routing import get_query_routing_system
from ..strategies.multi_hop_rag import get_multi_hop_rag


@dataclass
class RAGToolResult:
    """Result from a RAG optimization tool."""
    technique: str
    documents: List[Document]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class RAGOptimizationTools:
    """
    Collection of RAG optimization tools for the Query Strategy Agent.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG optimization tools.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("rag_optimization_tools")
        self.embedding_model = embedding_model
        
        # Initialize LLM for tool operations
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Initialize optimization systems
        self.query_optimizer = get_query_optimizer(embedding_model)
        self.multi_query_rag = get_multi_query_rag(embedding_model)
        self.query_routing = get_query_routing_system(embedding_model)
        self.multi_hop_rag = get_multi_hop_rag(embedding_model)
        
        self.logger.info("RAG optimization tools initialized successfully")
    
    def _create_retriever_function(self, base_retriever_func):
        """Create a retriever function wrapper."""
        def retriever_func(query: str, k: int = 3) -> List[Document]:
            try:
                return base_retriever_func(query, k)
            except Exception as e:
                self.logger.error(f"Error in retriever function: {str(e)}")
                return []
        return retriever_func


# Create tool instances
_rag_tools_instance = None

def get_rag_optimization_tools(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> RAGOptimizationTools:
    """Get or create the global RAG optimization tools instance."""
    global _rag_tools_instance
    if _rag_tools_instance is None:
        _rag_tools_instance = RAGOptimizationTools(embedding_model)
    return _rag_tools_instance


# Tool 1: Query Rewriting Tool
@tool
async def query_rewriting_tool(
    query: str, 
    retriever_func, 
    expansion_factor: float = 2.0,
    rewrite_count: int = 3,
    similarity_threshold: float = 0.8
) -> RAGToolResult:
    """
    Tool for query rewriting and expansion to improve retrieval coverage.
    
    Args:
        query: Original user query
        retriever_func: Function to retrieve documents
        expansion_factor: Factor to multiply k for initial retrieval
        rewrite_count: Number of query variations to generate
        similarity_threshold: Minimum similarity threshold for optimization
        
    Returns:
        RAGToolResult with optimized documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing query rewriting tool for: {query}")
        
        # Create retriever wrapper
        wrapped_retriever = tools._create_retriever_function(retriever_func)
        
        # Retrieve initial documents
        initial_k = int(3 * expansion_factor)
        initial_docs = wrapped_retriever(query, initial_k)
        
        if not initial_docs:
            return RAGToolResult(
                technique="query_rewriting",
                documents=[],
                metadata={"error": "No initial documents retrieved"},
                success=False,
                error_message="No documents found for initial retrieval"
            )
        
        # Apply query optimization
        optimized_docs, optimization_metadata = await tools.query_optimizer.optimize_retrieval(
            query=query,
            documents=initial_docs,
            lambda_param=0.7,
            k=3,
            use_query_rewriting=True
        )
        
        return RAGToolResult(
            technique="query_rewriting",
            documents=optimized_docs,
            metadata=optimization_metadata,
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in query rewriting tool: {str(e)}")
        return RAGToolResult(
            technique="query_rewriting",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Tool 2: Multi-Query RAG Tool
@tool
async def multi_query_rag_tool(
    query: str,
    retriever_func,
    query_count: int = 5,
    k_per_query: int = 3,
    fusion_method: str = "reciprocal_rank",
    final_k: int = 10
) -> RAGToolResult:
    """
    Tool for multi-query RAG that generates multiple queries and fuses results.
    
    Args:
        query: Original user query
        retriever_func: Function to retrieve documents
        query_count: Number of queries to generate
        k_per_query: Number of documents to retrieve per query
        fusion_method: Method for fusing results ("reciprocal_rank" or "similarity")
        final_k: Number of final documents to return
        
    Returns:
        RAGToolResult with fused documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing multi-query RAG tool for: {query}")
        
        # Create retriever wrapper
        wrapped_retriever = tools._create_retriever_function(retriever_func)
        
        # Process with multi-query RAG
        multi_query_result = await tools.multi_query_rag.process_query(
            query=query,
            retriever_func=wrapped_retriever,
            query_count=query_count,
            k_per_query=k_per_query,
            fusion_method=fusion_method,
            final_k=final_k
        )
        
        return RAGToolResult(
            technique="multi_query_rag",
            documents=multi_query_result.fused_documents,
            metadata=multi_query_result.metadata,
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in multi-query RAG tool: {str(e)}")
        return RAGToolResult(
            technique="multi_query_rag",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Tool 3: Query Routing Tool
@tool
async def query_routing_tool(
    query: str,
    retriever_func,
    max_routes: int = 3,
    routing_threshold: float = 0.7,
    k_per_route: int = 3
) -> RAGToolResult:
    """
    Tool for query routing to specialized knowledge domains.
    
    Args:
        query: Original user query
        retriever_func: Function to retrieve documents
        max_routes: Maximum number of routes to use
        routing_threshold: Minimum score threshold for routing
        k_per_route: Number of documents to retrieve per route
        
    Returns:
        RAGToolResult with routed documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing query routing tool for: {query}")
        
        # Create retriever wrapper
        wrapped_retriever = tools._create_retriever_function(retriever_func)
        
        # Process with query routing
        routing_result = await tools.query_routing.process_query(
            query=query,
            retriever_func=wrapped_retriever,
            max_routes=max_routes,
            routing_threshold=routing_threshold,
            k_per_route=k_per_route
        )
        
        return RAGToolResult(
            technique="query_routing",
            documents=routing_result.retrieved_documents,
            metadata=routing_result.routing_metadata,
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in query routing tool: {str(e)}")
        return RAGToolResult(
            technique="query_routing",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Tool 4: Multi-Hop RAG Tool
@tool
async def multi_hop_rag_tool(
    query: str,
    retriever_func,
    max_hops: int = 3,
    k_per_hop: int = 3,
    hop_strategy: str = "sequential",
    reasoning_threshold: float = 0.7
) -> RAGToolResult:
    """
    Tool for multi-hop RAG that performs iterative reasoning.
    
    Args:
        query: Original user query
        retriever_func: Function to retrieve documents
        max_hops: Maximum number of reasoning hops
        k_per_hop: Number of documents to retrieve per hop
        hop_strategy: Strategy for hop execution ("sequential" or "parallel")
        reasoning_threshold: Minimum confidence threshold for continuing
        
    Returns:
        RAGToolResult with final documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing multi-hop RAG tool for: {query}")
        
        # Create retriever wrapper
        wrapped_retriever = tools._create_retriever_function(retriever_func)
        
        # Process with multi-hop RAG
        multi_hop_result = await tools.multi_hop_rag.process_query(
            query=query,
            retriever_func=wrapped_retriever,
            max_hops=max_hops,
            k_per_hop=k_per_hop,
            hop_strategy=hop_strategy,
            reasoning_threshold=reasoning_threshold
        )
        
        return RAGToolResult(
            technique="multi_hop_rag",
            documents=multi_hop_result.final_documents,
            metadata=multi_hop_result.metadata,
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in multi-hop RAG tool: {str(e)}")
        return RAGToolResult(
            technique="multi_hop_rag",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Tool 5: Standard RAG Tool
@tool
async def standard_rag_tool(
    query: str,
    retriever_func,
    k: int = 3,
    similarity_threshold: float = 0.8
) -> RAGToolResult:
    """
    Tool for standard RAG retrieval without optimization.
    
    Args:
        query: User query
        retriever_func: Function to retrieve documents
        k: Number of documents to retrieve
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        RAGToolResult with retrieved documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing standard RAG tool for: {query}")
        
        # Create retriever wrapper
        wrapped_retriever = tools._create_retriever_function(retriever_func)
        
        # Standard retrieval
        documents = wrapped_retriever(query, k)
        
        return RAGToolResult(
            technique="standard_rag",
            documents=documents,
            metadata={
                "technique": "standard_rag",
                "k": len(documents),
                "similarity_threshold": similarity_threshold
            },
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in standard RAG tool: {str(e)}")
        return RAGToolResult(
            technique="standard_rag",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Tool 6: Hybrid RAG Tool (combines multiple techniques)
@tool
async def hybrid_rag_tool(
    query: str,
    retriever_func,
    primary_technique: str = "multi_query_rag",
    secondary_technique: str = "query_rewriting",
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3
) -> RAGToolResult:
    """
    Tool that combines multiple RAG techniques for optimal results.
    
    Args:
        query: User query
        retriever_func: Function to retrieve documents
        primary_technique: Primary technique to use
        secondary_technique: Secondary technique to use
        primary_weight: Weight for primary technique results
        secondary_weight: Weight for secondary technique results
        
    Returns:
        RAGToolResult with combined documents and metadata
    """
    try:
        tools = get_rag_optimization_tools()
        tools.logger.info(f"Executing hybrid RAG tool for: {query}")
        
        # Execute primary technique
        if primary_technique == "multi_query_rag":
            primary_result = await multi_query_rag_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "query_count": 5,
                "final_k": 5
            })
        elif primary_technique == "query_routing":
            primary_result = await query_routing_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "max_routes": 3,
                "k_per_route": 2
            })
        elif primary_technique == "multi_hop_rag":
            primary_result = await multi_hop_rag_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "max_hops": 2,
                "k_per_hop": 2
            })
        else:
            primary_result = await standard_rag_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "k": 3
            })
        
        # Execute secondary technique
        if secondary_technique == "query_rewriting":
            secondary_result = await query_rewriting_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "expansion_factor": 1.5,
                "rewrite_count": 2
            })
        else:
            secondary_result = await standard_rag_tool.ainvoke({
                "query": query,
                "retriever_func": retriever_func,
                "k": 2
            })
        
        # Combine results
        all_documents = []
        if primary_result.success:
            all_documents.extend(primary_result.documents)
        if secondary_result.success:
            all_documents.extend(secondary_result.documents)
        
        # Remove duplicates (simple approach)
        unique_documents = []
        seen_content = set()
        for doc in all_documents:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
        
        # Limit to reasonable number
        final_documents = unique_documents[:8]
        
        return RAGToolResult(
            technique="hybrid_rag",
            documents=final_documents,
            metadata={
                "primary_technique": primary_technique,
                "secondary_technique": secondary_technique,
                "primary_weight": primary_weight,
                "secondary_weight": secondary_weight,
                "primary_success": primary_result.success,
                "secondary_success": secondary_result.success,
                "total_combined": len(all_documents),
                "final_documents": len(final_documents)
            },
            success=True
        )
        
    except Exception as e:
        tools.logger.error(f"Error in hybrid RAG tool: {str(e)}")
        return RAGToolResult(
            technique="hybrid_rag",
            documents=[],
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )


# Export all tools
RAG_OPTIMIZATION_TOOLS = [
    query_rewriting_tool,
    multi_query_rag_tool,
    query_routing_tool,
    multi_hop_rag_tool,
    standard_rag_tool,
    hybrid_rag_tool
]
