"""
Multi-Hop RAG implementation.
Performs multi-step reasoning by iteratively gathering information and building context.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config.config import config


@dataclass
class HopResult:
    """Result of a single reasoning hop."""
    hop_number: int
    query: str
    retrieved_documents: List[Document]
    reasoning: str
    next_queries: List[str]
    confidence: float
    is_final: bool


@dataclass
class MultiHopResult:
    """Result of multi-hop RAG processing."""
    original_query: str
    hops: List[HopResult]
    final_documents: List[Document]
    final_answer: str
    reasoning_chain: List[str]
    metadata: Dict[str, Any]


class MultiHopReasoner:
    """
    Performs multi-step reasoning by iteratively gathering information.
    """
    
    def __init__(self):
        """Initialize the multi-hop reasoner."""
        self.logger = logging.getLogger("multi_hop_reasoner")
        
        # Initialize LLM for reasoning
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.3,  # Moderate temperature for reasoning
            max_tokens=1000
        )
        
        # Reasoning prompt for each hop
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at multi-step reasoning and information gathering.

Your task is to analyze the current context and determine:
1. What information you have gathered so far
2. What additional information is needed to answer the original question
3. What specific queries should be made next
4. Whether you have enough information to provide a final answer

Guidelines:
- Be specific about what information is still needed
- Generate focused queries that will gather missing information
- Consider different angles and perspectives
- Stop when you have sufficient information to answer the original question
- Provide clear reasoning for your decisions

Respond with a JSON object containing:
- reasoning: Your analysis of the current situation
- next_queries: List of specific queries to make next (empty if done)
- is_final: Boolean indicating if you have enough information
- confidence: Confidence score (0.0-1.0) in your current understanding"""),
            ("human", """Original Question: {original_query}

Current Context:
{context}

Retrieved Information:
{retrieved_info}

Analyze the situation and determine next steps:""")
        ])
        
        # Final answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing information from multiple sources to provide comprehensive answers.

Based on all the information gathered through multiple reasoning steps, provide a complete and accurate answer to the original question.

Guidelines:
- Synthesize information from all sources
- Provide a comprehensive answer that addresses all aspects of the question
- Cite specific information when possible
- Be clear about any limitations or uncertainties
- Structure your answer logically

Provide a well-structured, comprehensive answer."""),
            ("human", """Original Question: {original_query}

Information Gathered:
{all_information}

Reasoning Chain:
{reasoning_chain}

Provide a comprehensive answer:""")
        ])
    
    async def analyze_hop(
        self, 
        original_query: str, 
        context: str, 
        retrieved_info: str,
        hop_number: int
    ) -> Tuple[str, List[str], bool, float]:
        """
        Analyze the current hop and determine next steps.
        
        Args:
            original_query: Original user query
            context: Current context from previous hops
            retrieved_info: Information retrieved in current hop
            hop_number: Current hop number
            
        Returns:
            Tuple of (reasoning, next_queries, is_final, confidence)
        """
        try:
            self.logger.info(f"Analyzing hop {hop_number}")
            
            # Create reasoning chain
            reasoning_chain = self.reasoning_prompt | self.llm | JsonOutputParser()
            
            # Analyze current situation
            analysis_result = await reasoning_chain.ainvoke({
                "original_query": original_query,
                "context": context,
                "retrieved_info": retrieved_info
            })
            
            # Extract results
            reasoning = analysis_result.get("reasoning", "No reasoning provided")
            next_queries = analysis_result.get("next_queries", [])
            is_final = analysis_result.get("is_final", False)
            confidence = analysis_result.get("confidence", 0.5)
            
            # Ensure next_queries is a list
            if isinstance(next_queries, str):
                next_queries = [next_queries]
            
            self.logger.info(f"Hop {hop_number} analysis: {len(next_queries)} next queries, final: {is_final}")
            return reasoning, next_queries, is_final, confidence
            
        except Exception as e:
            self.logger.error(f"Error analyzing hop {hop_number}: {str(e)}")
            # Return fallback
            return f"Error in hop {hop_number} analysis", [], True, 0.3
    
    async def generate_final_answer(
        self, 
        original_query: str, 
        all_information: str, 
        reasoning_chain: List[str]
    ) -> str:
        """
        Generate the final answer based on all gathered information.
        
        Args:
            original_query: Original user query
            all_information: All information gathered across hops
            reasoning_chain: Chain of reasoning from all hops
            
        Returns:
            Final comprehensive answer
        """
        try:
            self.logger.info("Generating final answer")
            
            # Create answer generation chain
            answer_chain = self.answer_prompt | self.llm | StrOutputParser()
            
            # Generate final answer
            final_answer = await answer_chain.ainvoke({
                "original_query": original_query,
                "all_information": all_information,
                "reasoning_chain": "\n".join(reasoning_chain)
            })
            
            self.logger.info("Final answer generated successfully")
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Error generating final answer: {str(e)}")
            return f"Error generating final answer: {str(e)}"


class MultiHopRAG:
    """
    Multi-Hop RAG system that performs iterative reasoning and information gathering.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Multi-Hop RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("multi_hop_rag")
        
        # Initialize components
        self.reasoner = MultiHopReasoner()
        
        # Initialize sentence transformer for document similarity
        try:
            self.sentence_transformer = SentenceTransformer(embedding_model)
            self.logger.info(f"Initialized Multi-Hop RAG with model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
        
        self.logger.info("Multi-Hop RAG system initialized successfully")
    
    def _calculate_document_similarity(self, docs1: List[Document], docs2: List[Document]) -> float:
        """
        Calculate similarity between two sets of documents.
        
        Args:
            docs1: First set of documents
            docs2: Second set of documents
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            if not docs1 or not docs2:
                return 0.0
            
            # Get embeddings for both sets
            texts1 = [doc.page_content for doc in docs1]
            texts2 = [doc.page_content for doc in docs2]
            
            embeddings1 = self.sentence_transformer.encode(texts1)
            embeddings2 = self.sentence_transformer.encode(texts2)
            
            # Calculate average similarity
            similarities = cosine_similarity(embeddings1, embeddings2)
            avg_similarity = np.mean(similarities)
            
            return float(avg_similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating document similarity: {str(e)}")
            return 0.0
    
    def _deduplicate_documents(self, documents: List[Document], threshold: float = 0.8) -> List[Document]:
        """
        Remove duplicate or highly similar documents.
        
        Args:
            documents: List of documents to deduplicate
            threshold: Similarity threshold for considering documents as duplicates
            
        Returns:
            List of deduplicated documents
        """
        try:
            if not documents:
                return []
            
            unique_docs = []
            seen_embeddings = []
            
            for doc in documents:
                # Get embedding for current document
                doc_embedding = self.sentence_transformer.encode([doc.page_content])
                
                # Check similarity with already seen documents
                is_duplicate = False
                for seen_embedding in seen_embeddings:
                    similarity = cosine_similarity(doc_embedding, seen_embedding.reshape(1, -1))[0][0]
                    if similarity >= threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_docs.append(doc)
                    seen_embeddings.append(doc_embedding[0])
            
            self.logger.info(f"Deduplicated {len(documents)} documents to {len(unique_docs)} unique documents")
            return unique_docs
            
        except Exception as e:
            self.logger.error(f"Error deduplicating documents: {str(e)}")
            return documents
    
    async def process_query(
        self, 
        query: str, 
        retriever_func,
        max_hops: int = 3,
        k_per_hop: int = 3,
        hop_strategy: Literal["sequential", "parallel"] = "sequential",
        reasoning_threshold: float = 0.7
    ) -> MultiHopResult:
        """
        Process a query using multi-hop RAG.
        
        Args:
            query: Original user query
            retriever_func: Function to retrieve documents for a query
            max_hops: Maximum number of reasoning hops
            k_per_hop: Number of documents to retrieve per hop
            hop_strategy: Strategy for hop execution ("sequential" or "parallel")
            reasoning_threshold: Minimum confidence threshold for continuing
            
        Returns:
            MultiHopResult with complete reasoning chain and final answer
        """
        try:
            self.logger.info(f"Processing multi-hop query: {query}")
            
            hops = []
            all_documents = []
            reasoning_chain = []
            current_context = ""
            
            # Initial hop with original query
            current_queries = [query]
            
            for hop_number in range(max_hops):
                self.logger.info(f"Executing hop {hop_number + 1}")
                
                # Retrieve documents for current queries
                hop_documents = []
                for current_query in current_queries:
                    try:
                        docs = retriever_func(current_query, k_per_hop)
                        hop_documents.extend(docs)
                    except Exception as e:
                        self.logger.warning(f"Error retrieving documents for query '{current_query}': {str(e)}")
                
                # Deduplicate documents
                hop_documents = self._deduplicate_documents(hop_documents)
                all_documents.extend(hop_documents)
                
                # Prepare context for reasoning
                retrieved_info = "\n\n".join([doc.page_content for doc in hop_documents])
                
                # Analyze current hop
                reasoning, next_queries, is_final, confidence = await self.reasoner.analyze_hop(
                    original_query=query,
                    context=current_context,
                    retrieved_info=retrieved_info,
                    hop_number=hop_number + 1
                )
                
                # Create hop result
                hop_result = HopResult(
                    hop_number=hop_number + 1,
                    query=current_queries[0] if current_queries else query,
                    retrieved_documents=hop_documents,
                    reasoning=reasoning,
                    next_queries=next_queries,
                    confidence=confidence,
                    is_final=is_final
                )
                
                hops.append(hop_result)
                reasoning_chain.append(f"Hop {hop_number + 1}: {reasoning}")
                
                # Update context
                current_context += f"\n\nHop {hop_number + 1} Information:\n{retrieved_info}"
                
                # Check if we should stop
                if is_final or confidence < reasoning_threshold or not next_queries:
                    self.logger.info(f"Stopping at hop {hop_number + 1}: final={is_final}, confidence={confidence:.3f}")
                    break
                
                # Prepare for next hop
                current_queries = next_queries[:3]  # Limit to 3 queries per hop
            
            # Deduplicate all documents
            final_documents = self._deduplicate_documents(all_documents)
            
            # Generate final answer
            all_information = "\n\n".join([doc.page_content for doc in final_documents])
            final_answer = await self.reasoner.generate_final_answer(
                original_query=query,
                all_information=all_information,
                reasoning_chain=reasoning_chain
            )
            
            # Create result
            result = MultiHopResult(
                original_query=query,
                hops=hops,
                final_documents=final_documents,
                final_answer=final_answer,
                reasoning_chain=reasoning_chain,
                metadata={
                    "total_hops": len(hops),
                    "total_retrieved": len(all_documents),
                    "final_documents": len(final_documents),
                    "hop_strategy": hop_strategy,
                    "reasoning_threshold": reasoning_threshold,
                    "k_per_hop": k_per_hop
                }
            )
            
            self.logger.info(f"Multi-hop RAG completed. Executed {len(hops)} hops, generated final answer")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-hop RAG processing: {str(e)}")
            # Return fallback result
            return MultiHopResult(
                original_query=query,
                hops=[],
                final_documents=[],
                final_answer=f"Error processing multi-hop query: {str(e)}",
                reasoning_chain=[f"Error: {str(e)}"],
                metadata={"error": str(e), "fallback": True}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Multi-Hop RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "system_type": "Multi-Hop RAG",
            "reasoner_available": self.reasoner is not None,
            "embedding_model": self.sentence_transformer.model_name if hasattr(self.sentence_transformer, 'model_name') else "unknown",
            "supported_strategies": ["sequential", "parallel"],
            "max_hops_recommended": 5
        }


# Global multi-hop RAG instance
_multi_hop_rag_instance = None


def get_multi_hop_rag(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> MultiHopRAG:
    """
    Get or create the global multi-hop RAG instance.
    
    Args:
        embedding_model: Name of the sentence transformer model
        
    Returns:
        MultiHopRAG instance
    """
    global _multi_hop_rag_instance
    
    if _multi_hop_rag_instance is None:
        _multi_hop_rag_instance = MultiHopRAG(embedding_model)
    
    return _multi_hop_rag_instance
