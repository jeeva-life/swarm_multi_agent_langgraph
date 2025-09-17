"""
Query Strategy Agent for intelligent retrieval technique selection.
Analyzes user queries and decides which retrieval technique to use:
- Query Rewriting
- Multi-Query RAG
- Query Routing
- Multi-Hop RAG
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic
# from pydantic import BaseModel, Field  # Not used

from core.config.config import config


class RetrievalTechnique(str, Enum):
    """Available retrieval techniques."""
    QUERY_REWRITING = "query_rewriting"
    MULTI_QUERY_RAG = "multi_query_rag"
    QUERY_ROUTING = "query_routing"
    MULTI_HOP_RAG = "multi_hop_rag"
    STANDARD_RAG = "standard_rag"


@dataclass
class QueryAnalysis:
    """Analysis result of a user query."""
    query: str
    complexity: Literal["simple", "moderate", "complex"]
    intent: str
    domain: str
    requires_reasoning: bool
    requires_multiple_sources: bool
    requires_specialized_knowledge: bool
    confidence_score: float


@dataclass
class StrategyDecision:
    """Decision made by the strategy agent."""
    technique: RetrievalTechnique
    reasoning: str
    confidence: float
    parameters: Dict[str, Any]
    fallback_technique: Optional[RetrievalTechnique] = None


class QueryStrategyAgent:
    """
    Intelligent agent that analyzes queries and selects the best retrieval technique.
    """
    
    def __init__(self):
        """Initialize the query strategy agent."""
        self.logger = logging.getLogger("query_strategy_agent")
        
        # Initialize LLM for strategy decisions
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.1,  # Low temperature for consistent decisions
            max_tokens=1000
        )
        
        # Query analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing user queries to determine the best retrieval strategy.

Analyze the given query and provide a structured assessment of its characteristics.

Consider these factors:
1. **Complexity**: How complex is the query?
   - Simple: Direct, single-concept questions
   - Moderate: Multi-part questions or comparisons
   - Complex: Multi-step reasoning, synthesis, or analysis required

2. **Intent**: What is the user trying to accomplish?
   - Information seeking: Looking for facts or explanations
   - Problem solving: Need step-by-step solutions
   - Comparison: Comparing multiple options or concepts
   - Analysis: Deep understanding or synthesis required

3. **Domain**: What knowledge domain does this belong to?
   - Technical: System architecture, implementation details
   - Business: Processes, workflows, organizational aspects
   - General: Broad knowledge or common topics

4. **Requirements**:
   - Requires reasoning: Needs logical thinking or inference
   - Requires multiple sources: Needs information from different documents/sections
   - Requires specialized knowledge: Needs domain-specific expertise

Respond with a JSON object containing your analysis."""),
            ("human", "Query: {query}\n\nAnalyze this query and provide your assessment:")
        ])
        
        # Strategy decision prompt
        self.strategy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at selecting the optimal retrieval strategy for different types of queries.

Based on the query analysis, select the best retrieval technique:

**Available Techniques:**

1. **Query Rewriting** (query_rewriting)
   - Best for: Ambiguous queries, queries that need expansion
   - When to use: Simple to moderate complexity, unclear intent
   - Parameters: expansion_factor (1.0-3.0), rewrite_count (2-5)

2. **Multi-Query RAG** (multi_query_rag)
   - Best for: Complex queries requiring multiple perspectives
   - When to use: Moderate complexity, requires multiple sources
   - Parameters: query_count (3-7), fusion_method ("reciprocal_rank", "rrf")

3. **Query Routing** (query_routing)
   - Best for: Queries that need specialized knowledge or specific domains
   - When to use: Domain-specific queries, specialized knowledge required
   - Parameters: routing_threshold (0.7-0.9), max_routes (2-4)

4. **Multi-Hop RAG** (multi_hop_rag)
   - Best for: Complex reasoning requiring multiple information gathering steps
   - When to use: Complex queries, requires reasoning, multi-step process
   - Parameters: max_hops (2-5), hop_strategy ("sequential", "parallel")

5. **Standard RAG** (standard_rag)
   - Best for: Simple, direct queries
   - When to use: Simple complexity, clear intent, single source sufficient
   - Parameters: k (3-5), similarity_threshold (0.7-0.9)

**Decision Guidelines:**
- Start with query complexity and intent
- Consider if multiple sources or reasoning is needed
- Choose technique that best matches query characteristics
- Always provide a fallback technique
- Include confidence score (0.0-1.0)

Respond with a JSON object containing your strategy decision."""),
            ("human", """Query Analysis:
- Query: {query}
- Complexity: {complexity}
- Intent: {intent}
- Domain: {domain}
- Requires reasoning: {requires_reasoning}
- Requires multiple sources: {requires_multiple_sources}
- Requires specialized knowledge: {requires_specialized_knowledge}
- Confidence: {confidence_score}

Select the optimal retrieval strategy:""")
        ])
        
        # Initialize parsers
        self.analysis_parser = JsonOutputParser()
        self.strategy_parser = JsonOutputParser()
        
        self.logger.info("Query Strategy Agent initialized successfully")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query to understand its characteristics.
        
        Args:
            query: User query to analyze
            
        Returns:
            QueryAnalysis with query characteristics
        """
        try:
            self.logger.info(f"Analyzing query: {query}")
            
            # Create analysis chain
            analysis_chain = self.analysis_prompt | self.llm | self.analysis_parser
            
            # Analyze the query
            analysis_result = await analysis_chain.ainvoke({"query": query})
            
            # Create QueryAnalysis object
            query_analysis = QueryAnalysis(
                query=query,
                complexity=analysis_result.get("complexity", "simple"),
                intent=analysis_result.get("intent", "information seeking"),
                domain=analysis_result.get("domain", "general"),
                requires_reasoning=analysis_result.get("requires_reasoning", False),
                requires_multiple_sources=analysis_result.get("requires_multiple_sources", False),
                requires_specialized_knowledge=analysis_result.get("requires_specialized_knowledge", False),
                confidence_score=analysis_result.get("confidence_score", 0.8)
            )
            
            self.logger.info(f"Query analysis completed: {query_analysis.complexity} complexity, {query_analysis.intent} intent")
            return query_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            # Return default analysis
            return QueryAnalysis(
                query=query,
                complexity="simple",
                intent="information seeking",
                domain="general",
                requires_reasoning=False,
                requires_multiple_sources=False,
                requires_specialized_knowledge=False,
                confidence_score=0.5
            )
    
    async def decide_strategy(self, query_analysis: QueryAnalysis) -> StrategyDecision:
        """
        Decide on the best retrieval strategy based on query analysis.
        
        Args:
            query_analysis: Analysis of the user query
            
        Returns:
            StrategyDecision with selected technique and parameters
        """
        try:
            self.logger.info(f"Deciding strategy for {query_analysis.complexity} complexity query")
            
            # Create strategy decision chain
            strategy_chain = self.strategy_prompt | self.llm | self.strategy_parser
            
            # Make strategy decision
            strategy_result = await strategy_chain.ainvoke({
                "query": query_analysis.query,
                "complexity": query_analysis.complexity,
                "intent": query_analysis.intent,
                "domain": query_analysis.domain,
                "requires_reasoning": query_analysis.requires_reasoning,
                "requires_multiple_sources": query_analysis.requires_multiple_sources,
                "requires_specialized_knowledge": query_analysis.requires_specialized_knowledge,
                "confidence_score": query_analysis.confidence_score
            })
            
            # Create StrategyDecision object
            strategy_decision = StrategyDecision(
                technique=RetrievalTechnique(strategy_result.get("technique", "standard_rag")),
                reasoning=strategy_result.get("reasoning", "Standard retrieval selected"),
                confidence=strategy_result.get("confidence", 0.7),
                parameters=strategy_result.get("parameters", {}),
                fallback_technique=RetrievalTechnique(strategy_result.get("fallback_technique", "standard_rag")) if strategy_result.get("fallback_technique") else None
            )
            
            self.logger.info(f"Strategy decided: {strategy_decision.technique} (confidence: {strategy_decision.confidence:.2f})")
            return strategy_decision
            
        except Exception as e:
            self.logger.error(f"Error deciding strategy: {str(e)}")
            # Return default strategy
            return StrategyDecision(
                technique=RetrievalTechnique.STANDARD_RAG,
                reasoning="Fallback to standard retrieval due to error",
                confidence=0.5,
                parameters={"k": 3, "similarity_threshold": 0.8},
                fallback_technique=RetrievalTechnique.STANDARD_RAG
            )
    
    async def process_query(self, query: str) -> Tuple[QueryAnalysis, StrategyDecision]:
        """
        Complete processing of a query: analysis + strategy decision.
        
        Args:
            query: User query to process
            
        Returns:
            Tuple of (QueryAnalysis, StrategyDecision)
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Analyze the query
            query_analysis = await self.analyze_query(query)
            
            # Step 2: Decide on strategy
            strategy_decision = await self.decide_strategy(query_analysis)
            
            self.logger.info(f"Query processing completed: {strategy_decision.technique}")
            return query_analysis, strategy_decision
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            # Return default analysis and strategy
            default_analysis = QueryAnalysis(
                query=query,
                complexity="simple",
                intent="information seeking",
                domain="general",
                requires_reasoning=False,
                requires_multiple_sources=False,
                requires_specialized_knowledge=False,
                confidence_score=0.5
            )
            
            default_strategy = StrategyDecision(
                technique=RetrievalTechnique.STANDARD_RAG,
                reasoning="Fallback due to processing error",
                confidence=0.5,
                parameters={"k": 3, "similarity_threshold": 0.8}
            )
            
            return default_analysis, default_strategy
    
    def get_technique_description(self, technique: RetrievalTechnique) -> str:
        """
        Get a description of a retrieval technique.
        
        Args:
            technique: Retrieval technique
            
        Returns:
            Description of the technique
        """
        descriptions = {
            RetrievalTechnique.QUERY_REWRITING: "Expands and rewrites queries to improve retrieval coverage",
            RetrievalTechnique.MULTI_QUERY_RAG: "Generates multiple queries and fuses results for comprehensive answers",
            RetrievalTechnique.QUERY_ROUTING: "Routes queries to specialized knowledge domains or systems",
            RetrievalTechnique.MULTI_HOP_RAG: "Performs multi-step reasoning by iteratively gathering information",
            RetrievalTechnique.STANDARD_RAG: "Standard retrieval augmented generation with single query"
        }
        
        return descriptions.get(technique, "Unknown technique")
    
    def get_technique_parameters(self, technique: RetrievalTechnique) -> Dict[str, Any]:
        """
        Get default parameters for a retrieval technique.
        
        Args:
            technique: Retrieval technique
            
        Returns:
            Default parameters for the technique
        """
        default_params = {
            RetrievalTechnique.QUERY_REWRITING: {
                "expansion_factor": 2.0,
                "rewrite_count": 3,
                "similarity_threshold": 0.8
            },
            RetrievalTechnique.MULTI_QUERY_RAG: {
                "query_count": 5,
                "fusion_method": "reciprocal_rank",
                "k_per_query": 3
            },
            RetrievalTechnique.QUERY_ROUTING: {
                "routing_threshold": 0.8,
                "max_routes": 3,
                "domain_weights": {}
            },
            RetrievalTechnique.MULTI_HOP_RAG: {
                "max_hops": 3,
                "hop_strategy": "sequential",
                "reasoning_threshold": 0.7
            },
            RetrievalTechnique.STANDARD_RAG: {
                "k": 3,
                "similarity_threshold": 0.8
            }
        }
        
        return default_params.get(technique, {})


# Global query strategy agent instance
_query_strategy_agent_instance = None


def get_query_strategy_agent() -> QueryStrategyAgent:
    """
    Get or create the global query strategy agent instance.
    
    Returns:
        QueryStrategyAgent instance
    """
    global _query_strategy_agent_instance
    
    if _query_strategy_agent_instance is None:
        _query_strategy_agent_instance = QueryStrategyAgent()
    
    return _query_strategy_agent_instance
