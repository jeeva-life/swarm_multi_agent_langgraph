"""
Multi-Query RAG implementation.
Generates multiple queries from a single user query and fuses the results.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config.config import config


@dataclass
class MultiQueryResult:
    """Result of multi-query RAG processing."""
    original_query: str
    generated_queries: List[str]
    retrieved_documents: List[Document]
    fused_documents: List[Document]
    fusion_scores: List[float]
    metadata: Dict[str, Any]


class MultiQueryGenerator:
    """
    Generates multiple queries from a single user query.
    """
    
    def __init__(self):
        """Initialize the multi-query generator."""
        self.logger = logging.getLogger("multi_query_generator")
        
        # Initialize LLM for query generation
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.7,  # Higher temperature for diversity
            max_tokens=500
        )
        
        # Query generation prompt
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating multiple diverse queries from a single user query.

Your task is to generate 3-7 alternative queries that would help retrieve comprehensive information for the original query.

Guidelines:
- Generate queries that approach the topic from different angles
- Include both specific and general variations
- Consider different user intents and perspectives
- Ensure queries are semantically diverse but related
- Use natural language that would appear in documents
- Avoid generating queries that are too similar to each other

Format your response as a JSON array of query strings:
["query1", "query2", "query3", ...]

Example:
Original: "How does Docker work?"
Generated: ["What is Docker containerization?", "How do Docker containers isolate applications?", "Docker architecture and components", "Benefits of using Docker for deployment", "Docker vs virtual machines"]"""),
            ("human", "Original query: {query}\n\nGenerate diverse alternative queries:")
        ])
    
    async def generate_queries(self, query: str, count: int = 5) -> List[str]:
        """
        Generate multiple queries from a single query.
        
        Args:
            query: Original user query
            count: Number of queries to generate
            
        Returns:
            List of generated queries
        """
        try:
            self.logger.info(f"Generating {count} queries for: {query}")
            
            # Create generation chain
            generation_chain = self.generation_prompt | self.llm | StrOutputParser()
            
            # Generate queries
            generated_text = await generation_chain.ainvoke({"query": query})
            
            # Parse the JSON response
            import json
            try:
                generated_queries = json.loads(generated_text)
                if not isinstance(generated_queries, list):
                    generated_queries = [generated_text]
            except json.JSONDecodeError:
                # Fallback: split by newlines and clean
                generated_queries = [
                    q.strip().strip('"').strip("'") 
                    for q in generated_text.split('\n') 
                    if q.strip() and not q.strip().startswith('#')
                ]
            
            # Limit to requested count and ensure we have the original query
            generated_queries = generated_queries[:count]
            if query not in generated_queries:
                generated_queries.insert(0, query)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in generated_queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            
            self.logger.info(f"Generated {len(unique_queries)} unique queries")
            return unique_queries
            
        except Exception as e:
            self.logger.error(f"Error generating queries: {str(e)}")
            # Fallback to original query
            return [query]


class DocumentFusion:
    """
    Fuses documents retrieved from multiple queries.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the document fusion system.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("document_fusion")
        self.embedding_model = embedding_model
        
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_transformer = SentenceTransformer(embedding_model)
            self.logger.info(f"Initialized document fusion with model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
    
    def reciprocal_rank_fusion(
        self, 
        query_results: List[List[Document]], 
        k: int = 10,
        rrf_constant: int = 60
    ) -> List[Document]:
        """
        Fuse documents using Reciprocal Rank Fusion (RRF).
        
        Args:
            query_results: List of document lists from each query
            k: Number of top documents to return
            rrf_constant: RRF constant (typically 60)
            
        Returns:
            List of fused documents sorted by RRF score
        """
        try:
            self.logger.info(f"Performing RRF fusion on {len(query_results)} query results")
            
            # Calculate RRF scores for each document
            doc_scores = {}
            doc_metadata = {}
            
            for query_idx, docs in enumerate(query_results):
                for rank, doc in enumerate(docs):
                    # Create a unique identifier for the document
                    doc_id = f"{doc.page_content[:100]}_{doc.metadata.get('source', 'unknown')}"
                    
                    # Calculate RRF score
                    rrf_score = 1.0 / (rrf_constant + rank + 1)
                    
                    if doc_id in doc_scores:
                        doc_scores[doc_id] += rrf_score
                    else:
                        doc_scores[doc_id] = rrf_score
                        doc_metadata[doc_id] = doc
            
            # Sort documents by RRF score
            sorted_docs = sorted(
                doc_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Return top k documents
            fused_docs = [doc_metadata[doc_id] for doc_id, score in sorted_docs[:k]]
            
            self.logger.info(f"RRF fusion completed. Selected {len(fused_docs)} documents")
            return fused_docs
            
        except Exception as e:
            self.logger.error(f"Error in RRF fusion: {str(e)}")
            # Fallback: return documents from first query
            return query_results[0] if query_results else []
    
    def similarity_fusion(
        self, 
        query_results: List[List[Document]], 
        k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Document]:
        """
        Fuse documents using similarity-based clustering.
        
        Args:
            query_results: List of document lists from each query
            k: Number of top documents to return
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of fused documents
        """
        try:
            self.logger.info(f"Performing similarity fusion on {len(query_results)} query results")
            
            # Flatten all documents
            all_docs = []
            for docs in query_results:
                all_docs.extend(docs)
            
            if not all_docs:
                return []
            
            # Get document embeddings
            doc_texts = [doc.page_content for doc in all_docs]
            doc_embeddings = self.sentence_transformer.encode(doc_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(doc_embeddings)
            
            # Group similar documents
            doc_groups = []
            used_indices = set()
            
            for i, doc in enumerate(all_docs):
                if i in used_indices:
                    continue
                
                # Find similar documents
                similar_indices = [
                    j for j in range(len(all_docs))
                    if j not in used_indices and similarity_matrix[i][j] >= similarity_threshold
                ]
                
                if similar_indices:
                    # Create group with the most relevant document first
                    group_docs = [all_docs[j] for j in similar_indices]
                    doc_groups.append(group_docs)
                    used_indices.update(similar_indices)
                else:
                    # Single document group
                    doc_groups.append([doc])
                    used_indices.add(i)
            
            # Select representative document from each group
            fused_docs = []
            for group in doc_groups:
                # Select the first document as representative
                # (could be improved by selecting based on relevance score)
                fused_docs.append(group[0])
            
            # Limit to k documents
            fused_docs = fused_docs[:k]
            
            self.logger.info(f"Similarity fusion completed. Selected {len(fused_docs)} documents from {len(doc_groups)} groups")
            return fused_docs
            
        except Exception as e:
            self.logger.error(f"Error in similarity fusion: {str(e)}")
            # Fallback: return documents from first query
            return query_results[0] if query_results else []


class MultiQueryRAG:
    """
    Multi-Query RAG system that generates multiple queries and fuses results.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Multi-Query RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("multi_query_rag")
        
        # Initialize components
        self.query_generator = MultiQueryGenerator()
        self.document_fusion = DocumentFusion(embedding_model)
        
        self.logger.info("Multi-Query RAG system initialized successfully")
    
    async def process_query(
        self, 
        query: str, 
        retriever_func,
        query_count: int = 5,
        k_per_query: int = 3,
        fusion_method: str = "reciprocal_rank",
        final_k: int = 10
    ) -> MultiQueryResult:
        """
        Process a query using multi-query RAG.
        
        Args:
            query: Original user query
            retriever_func: Function to retrieve documents for a query
            query_count: Number of queries to generate
            k_per_query: Number of documents to retrieve per query
            fusion_method: Method for fusing results ("reciprocal_rank" or "similarity")
            final_k: Number of final documents to return
            
        Returns:
            MultiQueryResult with all processing information
        """
        try:
            self.logger.info(f"Processing multi-query RAG for: {query}")
            
            # Step 1: Generate multiple queries
            generated_queries = await self.query_generator.generate_queries(query, query_count)
            
            # Step 2: Retrieve documents for each query
            query_results = []
            all_retrieved_docs = []
            
            for gen_query in generated_queries:
                try:
                    # Retrieve documents for this query
                    docs = retriever_func(gen_query, k_per_query)
                    query_results.append(docs)
                    all_retrieved_docs.extend(docs)
                    
                    self.logger.info(f"Retrieved {len(docs)} documents for query: {gen_query}")
                    
                except Exception as e:
                    self.logger.warning(f"Error retrieving documents for query '{gen_query}': {str(e)}")
                    query_results.append([])
            
            # Step 3: Fuse documents
            if fusion_method == "reciprocal_rank":
                fused_docs = self.document_fusion.reciprocal_rank_fusion(
                    query_results, k=final_k
                )
            elif fusion_method == "similarity":
                fused_docs = self.document_fusion.similarity_fusion(
                    query_results, k=final_k
                )
            else:
                self.logger.warning(f"Unknown fusion method: {fusion_method}, using reciprocal_rank")
                fused_docs = self.document_fusion.reciprocal_rank_fusion(
                    query_results, k=final_k
                )
            
            # Step 4: Calculate fusion scores (simplified)
            fusion_scores = [1.0 / (i + 1) for i in range(len(fused_docs))]
            
            # Create result
            result = MultiQueryResult(
                original_query=query,
                generated_queries=generated_queries,
                retrieved_documents=all_retrieved_docs,
                fused_documents=fused_docs,
                fusion_scores=fusion_scores,
                metadata={
                    "query_count": len(generated_queries),
                    "total_retrieved": len(all_retrieved_docs),
                    "final_documents": len(fused_docs),
                    "fusion_method": fusion_method,
                    "k_per_query": k_per_query,
                    "final_k": final_k
                }
            )
            
            self.logger.info(f"Multi-query RAG completed. Generated {len(generated_queries)} queries, fused to {len(fused_docs)} documents")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-query RAG processing: {str(e)}")
            # Return fallback result
            return MultiQueryResult(
                original_query=query,
                generated_queries=[query],
                retrieved_documents=[],
                fused_documents=[],
                fusion_scores=[],
                metadata={"error": str(e), "fallback": True}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Multi-Query RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "system_type": "Multi-Query RAG",
            "query_generator_available": self.query_generator is not None,
            "document_fusion_available": self.document_fusion is not None,
            "embedding_model": self.document_fusion.embedding_model,
            "supported_fusion_methods": ["reciprocal_rank", "similarity"]
        }


# Global multi-query RAG instance
_multi_query_rag_instance = None


def get_multi_query_rag(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> MultiQueryRAG:
    """
    Get or create the global multi-query RAG instance.
    
    Args:
        embedding_model: Name of the sentence transformer model
        
    Returns:
        MultiQueryRAG instance
    """
    global _multi_query_rag_instance
    
    if _multi_query_rag_instance is None:
        _multi_query_rag_instance = MultiQueryRAG(embedding_model)
    
    return _multi_query_rag_instance
