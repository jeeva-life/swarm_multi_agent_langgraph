"""
Intelligent RAG System that integrates all retrieval techniques.
Uses the Query Strategy Agent to automatically select the best technique.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from langchain_core.documents import Document

from ..strategies.query_strategy_agent import (
    QueryStrategyAgent, 
    QueryAnalysis, 
    StrategyDecision, 
    RetrievalTechnique,
    get_query_strategy_agent
)
from ..optimization.query_optimizer import QueryOptimizer, get_query_optimizer
from ..strategies.multi_query_rag import MultiQueryRAG, get_multi_query_rag
from ..strategies.query_routing import QueryRoutingSystem, get_query_routing_system
from ..strategies.multi_hop_rag import MultiHopRAG, get_multi_hop_rag


@dataclass
class IntelligentRAGResult:
    """Result of intelligent RAG processing."""
    original_query: str
    selected_technique: RetrievalTechnique
    query_analysis: QueryAnalysis
    strategy_decision: StrategyDecision
    retrieved_documents: List[Document]
    final_response: str
    processing_metadata: Dict[str, Any]
    technique_metadata: Dict[str, Any]


class IntelligentRAGSystem:
    """
    Intelligent RAG system that automatically selects the best retrieval technique.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the intelligent RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("intelligent_rag_system")
        self.embedding_model = embedding_model
        
        # Initialize strategy agent
        self.strategy_agent = get_query_strategy_agent()
        
        # Initialize all retrieval techniques
        self.query_optimizer = get_query_optimizer(embedding_model)
        self.multi_query_rag = get_multi_query_rag(embedding_model)
        self.query_routing = get_query_routing_system(embedding_model)
        self.multi_hop_rag = get_multi_hop_rag(embedding_model)
        
        self.logger.info("Intelligent RAG system initialized successfully")
    
    def _create_retriever_function(self, base_retriever_func):
        """
        Create a retriever function that works with all techniques.
        
        Args:
            base_retriever_func: Base retriever function from RAG system
            
        Returns:
            Wrapped retriever function
        """
        def retriever_func(query: str, k: int = 3) -> List[Document]:
            """Retriever function for use with different techniques."""
            try:
                return base_retriever_func(query, k)
            except Exception as e:
                self.logger.error(f"Error in retriever function: {str(e)}")
                return []
        
        return retriever_func
    
    async def _execute_technique(
        self, 
        technique: RetrievalTechnique, 
        query: str, 
        parameters: Dict[str, Any],
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Execute the selected retrieval technique.
        
        Args:
            technique: Selected retrieval technique
            query: User query
            parameters: Technique-specific parameters
            retriever_func: Function to retrieve documents
            
        Returns:
            Tuple of (retrieved_documents, technique_metadata)
        """
        try:
            self.logger.info(f"Executing technique: {technique}")
            
            if technique == RetrievalTechnique.QUERY_REWRITING:
                return await self._execute_query_rewriting(query, parameters, retriever_func)
            
            elif technique == RetrievalTechnique.MULTI_QUERY_RAG:
                return await self._execute_multi_query_rag(query, parameters, retriever_func)
            
            elif technique == RetrievalTechnique.QUERY_ROUTING:
                return await self._execute_query_routing(query, parameters, retriever_func)
            
            elif technique == RetrievalTechnique.MULTI_HOP_RAG:
                return await self._execute_multi_hop_rag(query, parameters, retriever_func)
            
            elif technique == RetrievalTechnique.STANDARD_RAG:
                return await self._execute_standard_rag(query, parameters, retriever_func)
            
            else:
                self.logger.warning(f"Unknown technique: {technique}, falling back to standard RAG")
                return await self._execute_standard_rag(query, parameters, retriever_func)
                
        except Exception as e:
            self.logger.error(f"Error executing technique {technique}: {str(e)}")
            # Fallback to standard retrieval
            return await self._execute_standard_rag(query, parameters, retriever_func)
    
    async def _execute_query_rewriting(
        self, 
        query: str, 
        parameters: Dict[str, Any], 
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute query rewriting technique."""
        try:
            # Use query optimizer with rewriting enabled
            initial_docs = retriever_func(query, parameters.get("k", 3) * 2)
            
            optimized_docs, optimization_metadata = await self.query_optimizer.optimize_retrieval(
                query=query,
                documents=initial_docs,
                lambda_param=parameters.get("lambda_param", 0.7),
                k=parameters.get("k", 3),
                use_query_rewriting=True
            )
            
            return optimized_docs, optimization_metadata
            
        except Exception as e:
            self.logger.error(f"Error in query rewriting: {str(e)}")
            return retriever_func(query, parameters.get("k", 3)), {"error": str(e)}
    
    async def _execute_multi_query_rag(
        self, 
        query: str, 
        parameters: Dict[str, Any], 
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute multi-query RAG technique."""
        try:
            multi_query_result = await self.multi_query_rag.process_query(
                query=query,
                retriever_func=retriever_func,
                query_count=parameters.get("query_count", 5),
                k_per_query=parameters.get("k_per_query", 3),
                fusion_method=parameters.get("fusion_method", "reciprocal_rank"),
                final_k=parameters.get("final_k", 10)
            )
            
            return multi_query_result.fused_documents, multi_query_result.metadata
            
        except Exception as e:
            self.logger.error(f"Error in multi-query RAG: {str(e)}")
            return retriever_func(query, parameters.get("k", 3)), {"error": str(e)}
    
    async def _execute_query_routing(
        self, 
        query: str, 
        parameters: Dict[str, Any], 
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute query routing technique."""
        try:
            routing_result = await self.query_routing.process_query(
                query=query,
                retriever_func=retriever_func,
                max_routes=parameters.get("max_routes", 3),
                routing_threshold=parameters.get("routing_threshold", 0.7),
                k_per_route=parameters.get("k_per_route", 3)
            )
            
            return routing_result.retrieved_documents, routing_result.routing_metadata
            
        except Exception as e:
            self.logger.error(f"Error in query routing: {str(e)}")
            return retriever_func(query, parameters.get("k", 3)), {"error": str(e)}
    
    async def _execute_multi_hop_rag(
        self, 
        query: str, 
        parameters: Dict[str, Any], 
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute multi-hop RAG technique."""
        try:
            multi_hop_result = await self.multi_hop_rag.process_query(
                query=query,
                retriever_func=retriever_func,
                max_hops=parameters.get("max_hops", 3),
                k_per_hop=parameters.get("k_per_hop", 3),
                hop_strategy=parameters.get("hop_strategy", "sequential"),
                reasoning_threshold=parameters.get("reasoning_threshold", 0.7)
            )
            
            return multi_hop_result.final_documents, multi_hop_result.metadata
            
        except Exception as e:
            self.logger.error(f"Error in multi-hop RAG: {str(e)}")
            return retriever_func(query, parameters.get("k", 3)), {"error": str(e)}
    
    async def _execute_standard_rag(
        self, 
        query: str, 
        parameters: Dict[str, Any], 
        retriever_func
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute standard RAG technique."""
        try:
            docs = retriever_func(query, parameters.get("k", 3))
            return docs, {"technique": "standard_rag", "k": len(docs)}
            
        except Exception as e:
            self.logger.error(f"Error in standard RAG: {str(e)}")
            return [], {"error": str(e)}
    
    async def process_query(
        self, 
        query: str, 
        retriever_func,
        generate_response: bool = True,
        response_generator_func = None
    ) -> IntelligentRAGResult:
        """
        Process a query using intelligent technique selection.
        
        Args:
            query: User query to process
            retriever_func: Function to retrieve documents
            generate_response: Whether to generate a final response
            response_generator_func: Function to generate response from documents
            
        Returns:
            IntelligentRAGResult with complete processing information
        """
        try:
            self.logger.info(f"Processing query with intelligent RAG: {query}")
            
            # Step 1: Analyze query and decide strategy
            query_analysis, strategy_decision = await self.strategy_agent.process_query(query)
            
            # Step 2: Create retriever function
            wrapped_retriever = self._create_retriever_function(retriever_func)
            
            # Step 3: Execute selected technique
            retrieved_documents, technique_metadata = await self._execute_technique(
                technique=strategy_decision.technique,
                query=query,
                parameters=strategy_decision.parameters,
                retriever_func=wrapped_retriever
            )
            
            # Step 4: Generate response if requested
            final_response = ""
            if generate_response and response_generator_func and retrieved_documents:
                try:
                    # Prepare context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in retrieved_documents])
                    final_response = response_generator_func(query, context)
                except Exception as e:
                    self.logger.error(f"Error generating response: {str(e)}")
                    final_response = f"Error generating response: {str(e)}"
            
            # Step 5: Create result
            result = IntelligentRAGResult(
                original_query=query,
                selected_technique=strategy_decision.technique,
                query_analysis=query_analysis,
                strategy_decision=strategy_decision,
                retrieved_documents=retrieved_documents,
                final_response=final_response,
                processing_metadata={
                    "technique_used": strategy_decision.technique.value,
                    "confidence": strategy_decision.confidence,
                    "reasoning": strategy_decision.reasoning,
                    "parameters_used": strategy_decision.parameters,
                    "documents_retrieved": len(retrieved_documents),
                    "response_generated": bool(final_response)
                },
                technique_metadata=technique_metadata
            )
            
            self.logger.info(f"Intelligent RAG processing completed: {strategy_decision.technique}, {len(retrieved_documents)} documents")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in intelligent RAG processing: {str(e)}")
            # Return fallback result
            return IntelligentRAGResult(
                original_query=query,
                selected_technique=RetrievalTechnique.STANDARD_RAG,
                query_analysis=QueryAnalysis(
                    query=query,
                    complexity="simple",
                    intent="information seeking",
                    domain="general",
                    requires_reasoning=False,
                    requires_multiple_sources=False,
                    requires_specialized_knowledge=False,
                    confidence_score=0.5
                ),
                strategy_decision=StrategyDecision(
                    technique=RetrievalTechnique.STANDARD_RAG,
                    reasoning="Fallback due to error",
                    confidence=0.5,
                    parameters={"k": 3}
                ),
                retrieved_documents=[],
                final_response=f"Error processing query: {str(e)}",
                processing_metadata={"error": str(e), "fallback": True},
                technique_metadata={"error": str(e)}
            )
    
    def get_available_techniques(self) -> List[Dict[str, Any]]:
        """
        Get information about available retrieval techniques.
        
        Returns:
            List of technique information dictionaries
        """
        return [
            {
                "technique": RetrievalTechnique.QUERY_REWRITING.value,
                "description": "Expands and rewrites queries to improve retrieval coverage",
                "best_for": "Ambiguous queries, queries that need expansion",
                "parameters": ["expansion_factor", "rewrite_count", "similarity_threshold"]
            },
            {
                "technique": RetrievalTechnique.MULTI_QUERY_RAG.value,
                "description": "Generates multiple queries and fuses results for comprehensive answers",
                "best_for": "Complex queries requiring multiple perspectives",
                "parameters": ["query_count", "fusion_method", "k_per_query"]
            },
            {
                "technique": RetrievalTechnique.QUERY_ROUTING.value,
                "description": "Routes queries to specialized knowledge domains or systems",
                "best_for": "Domain-specific queries, specialized knowledge required",
                "parameters": ["routing_threshold", "max_routes", "domain_weights"]
            },
            {
                "technique": RetrievalTechnique.MULTI_HOP_RAG.value,
                "description": "Performs multi-step reasoning by iteratively gathering information",
                "best_for": "Complex reasoning requiring multiple information gathering steps",
                "parameters": ["max_hops", "hop_strategy", "reasoning_threshold"]
            },
            {
                "technique": RetrievalTechnique.STANDARD_RAG.value,
                "description": "Standard retrieval augmented generation with single query",
                "best_for": "Simple, direct queries",
                "parameters": ["k", "similarity_threshold"]
            }
        ]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "system_type": "Intelligent RAG System",
            "strategy_agent_available": self.strategy_agent is not None,
            "available_techniques": len(self.get_available_techniques()),
            "embedding_model": self.embedding_model,
            "technique_statistics": {
                "query_optimizer": self.query_optimizer.get_statistics() if hasattr(self.query_optimizer, 'get_statistics') else {},
                "multi_query_rag": self.multi_query_rag.get_statistics(),
                "query_routing": self.query_routing.get_statistics(),
                "multi_hop_rag": self.multi_hop_rag.get_statistics()
            }
        }


# Global intelligent RAG system instance
_intelligent_rag_system_instance = None


def get_intelligent_rag_system(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> IntelligentRAGSystem:
    """
    Get or create the global intelligent RAG system instance.
    
    Args:
        embedding_model: Name of the sentence transformer model
        
    Returns:
        IntelligentRAGSystem instance
    """
    global _intelligent_rag_system_instance
    
    if _intelligent_rag_system_instance is None:
        _intelligent_rag_system_instance = IntelligentRAGSystem(embedding_model)
    
    return _intelligent_rag_system_instance
