"""
Query optimization module for RAG system.
Implements query rewriting and Maximum Marginal Relevance (MMR) reranking.
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
class QueryRewriteResult:
    """Result of query rewriting process."""
    original_query: str
    rewritten_queries: List[str]
    expansion_terms: List[str]
    confidence_score: float


@dataclass
class MMRResult:
    """Result of Maximum Marginal Relevance reranking."""
    documents: List[Document]
    scores: List[float]
    diversity_scores: List[float]
    relevance_scores: List[float]


class QueryRewriter:
    """
    Query rewriting system to expand and improve search queries.
    Uses LLM to generate alternative phrasings and related terms.
    """
    
    def __init__(self):
        """Initialize the query rewriter."""
        self.logger = logging.getLogger("query_rewriter")
        
        # Initialize LLM for query rewriting
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.3,  # Lower temperature for more consistent rewrites
            max_tokens=500
        )
        
        # Query rewriting prompts
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at query rewriting and expansion for information retrieval systems.

Your task is to rewrite and expand the given query to improve search results. Generate:
1. Alternative phrasings of the original query
2. Related terms and synonyms
3. Broader and narrower concepts
4. Context-specific variations

Guidelines:
- Keep the core meaning intact
- Generate 3-5 alternative queries
- Include both specific and general variations
- Consider different user intents
- Use natural language that would appear in documents
- Separate each query with a newline

Format your response as:
QUERY_1
QUERY_2
QUERY_3
etc."""),
            ("human", "Original query: {query}\n\nGenerate alternative queries:")
        ])
        
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting key terms and concepts from queries.

Extract important terms, synonyms, and related concepts that would help find relevant documents.

Guidelines:
- Extract key nouns, verbs, and concepts
- Include synonyms and related terms
- Consider technical terms and jargon
- Include both specific and general terms
- Separate terms with commas

Format your response as a comma-separated list of terms."""),
            ("human", "Query: {query}\n\nExtract key terms and concepts:")
        ])
    
    async def rewrite_query(self, query: str) -> QueryRewriteResult:
        """
        Rewrite and expand a query to improve retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            QueryRewriteResult with rewritten queries and expansion terms
        """
        try:
            self.logger.info(f"Rewriting query: {query}")
            
            # Generate alternative queries
            rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
            rewritten_text = await rewrite_chain.ainvoke({"query": query})
            
            # Parse rewritten queries
            rewritten_queries = [
                q.strip() for q in rewritten_text.split('\n') 
                if q.strip() and not q.strip().startswith('#')
            ]
            
            # Generate expansion terms
            expansion_chain = self.expansion_prompt | self.llm | StrOutputParser()
            expansion_text = await expansion_chain.ainvoke({"query": query})
            
            # Parse expansion terms
            expansion_terms = [
                term.strip() for term in expansion_text.split(',')
                if term.strip()
            ]
            
            # Calculate confidence score based on number of variations
            confidence_score = min(1.0, len(rewritten_queries) / 5.0)
            
            result = QueryRewriteResult(
                original_query=query,
                rewritten_queries=rewritten_queries,
                expansion_terms=expansion_terms,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Generated {len(rewritten_queries)} rewritten queries and {len(expansion_terms)} expansion terms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error rewriting query: {str(e)}")
            # Fallback to original query
            return QueryRewriteResult(
                original_query=query,
                rewritten_queries=[query],
                expansion_terms=query.split(),
                confidence_score=0.5
            )
    
    def rewrite_query_sync(self, query: str) -> QueryRewriteResult:
        """
        Synchronous version of query rewriting.
        
        Args:
            query: Original user query
            
        Returns:
            QueryRewriteResult with rewritten queries and expansion terms
        """
        try:
            self.logger.info(f"Rewriting query (sync): {query}")
            
            # Generate alternative queries
            rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
            rewritten_text = rewrite_chain.invoke({"query": query})
            
            # Parse rewritten queries
            rewritten_queries = [
                q.strip() for q in rewritten_text.split('\n') 
                if q.strip() and not q.strip().startswith('#')
            ]
            
            # Generate expansion terms
            expansion_chain = self.expansion_prompt | self.llm | StrOutputParser()
            expansion_text = expansion_chain.invoke({"query": query})
            
            # Parse expansion terms
            expansion_terms = [
                term.strip() for term in expansion_text.split(',')
                if term.strip()
            ]
            
            # Calculate confidence score
            confidence_score = min(1.0, len(rewritten_queries) / 5.0)
            
            result = QueryRewriteResult(
                original_query=query,
                rewritten_queries=rewritten_queries,
                expansion_terms=expansion_terms,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Generated {len(rewritten_queries)} rewritten queries and {len(expansion_terms)} expansion terms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error rewriting query: {str(e)}")
            # Fallback to original query
            return QueryRewriteResult(
                original_query=query,
                rewritten_queries=[query],
                expansion_terms=query.split(),
                confidence_score=0.5
            )


class MMRReranker:
    """
    Maximum Marginal Relevance (MMR) reranker for document diversity.
    Balances relevance and diversity in retrieved documents.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the MMR reranker.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("mmr_reranker")
        self.embedding_model = embedding_model
        
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_transformer = SentenceTransformer(embedding_model)
            self.logger.info(f"Initialized MMR reranker with model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
    
    def calculate_mmr_scores(
        self, 
        documents: List[Document], 
        query: str, 
        lambda_param: float = 0.7,
        k: int = None
    ) -> MMRResult:
        """
        Calculate MMR scores for document reranking.
        
        Args:
            documents: List of documents to rerank
            query: Original query
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            k: Number of documents to return (None for all)
            
        Returns:
            MMRResult with reranked documents and scores
        """
        try:
            if not documents:
                return MMRResult(documents=[], scores=[], diversity_scores=[], relevance_scores=[])
            
            self.logger.info(f"Calculating MMR scores for {len(documents)} documents")
            
            # Set k to all documents if not specified
            if k is None:
                k = len(documents)
            k = min(k, len(documents))
            
            # Get query embedding
            query_embedding = self.sentence_transformer.encode([query])
            
            # Get document embeddings
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = self.sentence_transformer.encode(doc_texts)
            
            # Calculate relevance scores (cosine similarity with query)
            relevance_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Initialize MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            mmr_scores = []
            diversity_scores = []
            
            # Select first document (highest relevance)
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            mmr_scores.append(relevance_scores[first_idx])
            diversity_scores.append(0.0)  # First document has no diversity score
            
            # Select remaining documents using MMR
            for _ in range(k - 1):
                if not remaining_indices:
                    break
                
                best_mmr_score = -float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # Calculate maximum similarity with already selected documents
                    if selected_indices:
                        selected_embeddings = doc_embeddings[selected_indices]
                        current_embedding = doc_embeddings[idx].reshape(1, -1)
                        max_similarity = np.max(cosine_similarity(current_embedding, selected_embeddings))
                    else:
                        max_similarity = 0.0
                    
                    # Calculate MMR score
                    relevance = relevance_scores[idx]
                    diversity = 1.0 - max_similarity  # Diversity is inverse of similarity
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    mmr_scores.append(best_mmr_score)
                    
                    # Calculate diversity score for this document
                    if len(selected_indices) > 1:
                        selected_embeddings = doc_embeddings[selected_indices[:-1]]
                        current_embedding = doc_embeddings[best_idx].reshape(1, -1)
                        max_similarity = np.max(cosine_similarity(current_embedding, selected_embeddings))
                        diversity_scores.append(1.0 - max_similarity)
                    else:
                        diversity_scores.append(0.0)
            
            # Create reranked documents and scores
            reranked_docs = [documents[i] for i in selected_indices]
            final_relevance_scores = [relevance_scores[i] for i in selected_indices]
            
            result = MMRResult(
                documents=reranked_docs,
                scores=mmr_scores,
                diversity_scores=diversity_scores,
                relevance_scores=final_relevance_scores
            )
            
            self.logger.info(f"MMR reranking completed. Selected {len(reranked_docs)} documents")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating MMR scores: {str(e)}")
            # Fallback to original documents
            return MMRResult(
                documents=documents[:k] if k else documents,
                scores=[1.0] * min(k or len(documents), len(documents)),
                diversity_scores=[0.0] * min(k or len(documents), len(documents)),
                relevance_scores=[1.0] * min(k or len(documents), len(documents))
            )
    
    def rerank_documents(
        self, 
        documents: List[Document], 
        query: str, 
        lambda_param: float = 0.7,
        k: int = None
    ) -> List[Document]:
        """
        Rerank documents using MMR algorithm.
        
        Args:
            documents: List of documents to rerank
            query: Original query
            lambda_param: Balance between relevance and diversity
            k: Number of documents to return
            
        Returns:
            List of reranked documents
        """
        mmr_result = self.calculate_mmr_scores(documents, query, lambda_param, k)
        return mmr_result.documents


class QueryOptimizer:
    """
    Main query optimization system that combines query rewriting and MMR reranking.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the query optimizer.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("query_optimizer")
        
        # Initialize components
        self.query_rewriter = QueryRewriter()
        self.mmr_reranker = MMRReranker(embedding_model)
        
        self.logger.info("Query optimizer initialized successfully")
    
    async def optimize_retrieval(
        self, 
        query: str, 
        documents: List[Document], 
        lambda_param: float = 0.7,
        k: int = None,
        use_query_rewriting: bool = True
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Optimize retrieval using query rewriting and MMR reranking.
        
        Args:
            query: Original user query
            documents: List of documents to optimize
            lambda_param: MMR balance parameter (0.0 = diversity, 1.0 = relevance)
            k: Number of documents to return
            use_query_rewriting: Whether to use query rewriting
            
        Returns:
            Tuple of (optimized_documents, optimization_metadata)
        """
        try:
            self.logger.info(f"Optimizing retrieval for query: {query}")
            
            optimization_metadata = {
                "original_query": query,
                "original_doc_count": len(documents),
                "lambda_param": lambda_param,
                "k": k,
                "use_query_rewriting": use_query_rewriting
            }
            
            # Step 1: Query rewriting (optional)
            if use_query_rewriting and documents:
                rewrite_result = await self.query_rewriter.rewrite_query(query)
                optimization_metadata["rewrite_result"] = {
                    "rewritten_queries": rewrite_result.rewritten_queries,
                    "expansion_terms": rewrite_result.expansion_terms,
                    "confidence_score": rewrite_result.confidence_score
                }
                self.logger.info(f"Query rewriting generated {len(rewrite_result.rewritten_queries)} variations")
            
            # Step 2: MMR reranking
            if documents:
                mmr_result = self.mmr_reranker.calculate_mmr_scores(
                    documents, query, lambda_param, k
                )
                
                optimization_metadata["mmr_result"] = {
                    "final_doc_count": len(mmr_result.documents),
                    "avg_relevance_score": np.mean(mmr_result.relevance_scores) if mmr_result.relevance_scores else 0.0,
                    "avg_diversity_score": np.mean(mmr_result.diversity_scores) if mmr_result.diversity_scores else 0.0,
                    "avg_mmr_score": np.mean(mmr_result.scores) if mmr_result.scores else 0.0
                }
                
                self.logger.info(f"MMR reranking completed. Final documents: {len(mmr_result.documents)}")
                return mmr_result.documents, optimization_metadata
            else:
                optimization_metadata["mmr_result"] = {
                    "final_doc_count": 0,
                    "avg_relevance_score": 0.0,
                    "avg_diversity_score": 0.0,
                    "avg_mmr_score": 0.0
                }
                return [], optimization_metadata
                
        except Exception as e:
            self.logger.error(f"Error optimizing retrieval: {str(e)}")
            # Fallback to original documents
            fallback_docs = documents[:k] if k and documents else documents
            optimization_metadata["error"] = str(e)
            optimization_metadata["fallback"] = True
            return fallback_docs, optimization_metadata
    
    def optimize_retrieval_sync(
        self, 
        query: str, 
        documents: List[Document], 
        lambda_param: float = 0.7,
        k: int = None,
        use_query_rewriting: bool = True
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Synchronous version of retrieval optimization.
        
        Args:
            query: Original user query
            documents: List of documents to optimize
            lambda_param: MMR balance parameter
            k: Number of documents to return
            use_query_rewriting: Whether to use query rewriting
            
        Returns:
            Tuple of (optimized_documents, optimization_metadata)
        """
        try:
            self.logger.info(f"Optimizing retrieval (sync) for query: {query}")
            
            optimization_metadata = {
                "original_query": query,
                "original_doc_count": len(documents),
                "lambda_param": lambda_param,
                "k": k,
                "use_query_rewriting": use_query_rewriting
            }
            
            # Step 1: Query rewriting (optional)
            if use_query_rewriting and documents:
                rewrite_result = self.query_rewriter.rewrite_query_sync(query)
                optimization_metadata["rewrite_result"] = {
                    "rewritten_queries": rewrite_result.rewritten_queries,
                    "expansion_terms": rewrite_result.expansion_terms,
                    "confidence_score": rewrite_result.confidence_score
                }
                self.logger.info(f"Query rewriting generated {len(rewrite_result.rewritten_queries)} variations")
            
            # Step 2: MMR reranking
            if documents:
                mmr_result = self.mmr_reranker.calculate_mmr_scores(
                    documents, query, lambda_param, k
                )
                
                optimization_metadata["mmr_result"] = {
                    "final_doc_count": len(mmr_result.documents),
                    "avg_relevance_score": np.mean(mmr_result.relevance_scores) if mmr_result.relevance_scores else 0.0,
                    "avg_diversity_score": np.mean(mmr_result.diversity_scores) if mmr_result.diversity_scores else 0.0,
                    "avg_mmr_score": np.mean(mmr_result.scores) if mmr_result.scores else 0.0
                }
                
                self.logger.info(f"MMR reranking completed. Final documents: {len(mmr_result.documents)}")
                return mmr_result.documents, optimization_metadata
            else:
                optimization_metadata["mmr_result"] = {
                    "final_doc_count": 0,
                    "avg_relevance_score": 0.0,
                    "avg_diversity_score": 0.0,
                    "avg_mmr_score": 0.0
                }
                return [], optimization_metadata
                
        except Exception as e:
            self.logger.error(f"Error optimizing retrieval: {str(e)}")
            # Fallback to original documents
            fallback_docs = documents[:k] if k and documents else documents
            optimization_metadata["error"] = str(e)
            optimization_metadata["fallback"] = True
            return fallback_docs, optimization_metadata


# Global query optimizer instance
_query_optimizer_instance = None


def get_query_optimizer(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> QueryOptimizer:
    """
    Get or create the global query optimizer instance.
    
    Args:
        embedding_model: Name of the sentence transformer model
        
    Returns:
        QueryOptimizer instance
    """
    global _query_optimizer_instance
    
    if _query_optimizer_instance is None:
        _query_optimizer_instance = QueryOptimizer(embedding_model)
    
    return _query_optimizer_instance
