"""
Query Routing implementation.
Routes queries to specialized knowledge domains or systems based on query analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config.config import config


class KnowledgeDomain(str, Enum):
    """Available knowledge domains for routing."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    DATABASE = "database"
    API = "api"
    SECURITY = "security"
    GENERAL = "general"


@dataclass
class Route:
    """A routing destination with metadata."""
    domain: KnowledgeDomain
    name: str
    description: str
    keywords: List[str]
    retriever_func: Optional[Callable] = None
    weight: float = 1.0
    threshold: float = 0.7


@dataclass
class RoutingResult:
    """Result of query routing."""
    original_query: str
    selected_routes: List[Route]
    route_scores: List[float]
    retrieved_documents: List[Document]
    routing_metadata: Dict[str, Any]


class QueryRouter:
    """
    Routes queries to appropriate knowledge domains or specialized systems.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the query router.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("query_router")
        self.embedding_model = embedding_model
        
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_transformer = SentenceTransformer(embedding_model)
            self.logger.info(f"Initialized query router with model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
        
        # Initialize LLM for routing decisions
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.1,  # Low temperature for consistent routing
            max_tokens=500
        )
        
        # Routing decision prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at routing queries to appropriate knowledge domains.

Analyze the given query and determine which knowledge domains are most relevant.

Available domains:
- technical: Technical implementation details, code, APIs, system internals
- business: Business processes, workflows, organizational aspects
- architecture: System design, components, relationships, patterns
- deployment: Deployment strategies, infrastructure, DevOps
- monitoring: Observability, metrics, logging, alerting
- database: Data storage, queries, schemas, data management
- api: API design, endpoints, integration, web services
- security: Security practices, authentication, authorization, compliance
- general: General knowledge, broad topics, common information

For each relevant domain, provide:
1. Domain name
2. Relevance score (0.0-1.0)
3. Reasoning for why this domain is relevant

Respond with a JSON object containing domain scores and reasoning."""),
            ("human", "Query: {query}\n\nAnalyze and route this query to appropriate domains:")
        ])
        
        # Initialize default routes
        self.routes = self._initialize_default_routes()
        
        self.logger.info("Query router initialized successfully")
    
    def _initialize_default_routes(self) -> List[Route]:
        """Initialize default routing destinations."""
        return [
            Route(
                domain=KnowledgeDomain.TECHNICAL,
                name="Technical Documentation",
                description="Technical implementation details, code examples, APIs",
                keywords=["code", "implementation", "api", "technical", "programming", "development"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.BUSINESS,
                name="Business Processes",
                description="Business workflows, processes, organizational aspects",
                keywords=["business", "process", "workflow", "organization", "management"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.ARCHITECTURE,
                name="System Architecture",
                description="System design, components, relationships, patterns",
                keywords=["architecture", "design", "components", "system", "structure"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.DEPLOYMENT,
                name="Deployment & Infrastructure",
                description="Deployment strategies, infrastructure, DevOps",
                keywords=["deployment", "infrastructure", "devops", "docker", "kubernetes"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.MONITORING,
                name="Monitoring & Observability",
                description="Observability, metrics, logging, alerting",
                keywords=["monitoring", "metrics", "logging", "alerting", "observability"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.DATABASE,
                name="Database & Data",
                description="Data storage, queries, schemas, data management",
                keywords=["database", "data", "sql", "storage", "schema"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.API,
                name="API & Integration",
                description="API design, endpoints, integration, web services",
                keywords=["api", "endpoint", "integration", "web service", "rest"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.SECURITY,
                name="Security & Compliance",
                description="Security practices, authentication, authorization",
                keywords=["security", "authentication", "authorization", "compliance"],
                weight=1.0,
                threshold=0.7
            ),
            Route(
                domain=KnowledgeDomain.GENERAL,
                name="General Knowledge",
                description="General information, broad topics, common knowledge",
                keywords=["general", "overview", "introduction", "basics"],
                weight=0.5,
                threshold=0.5
            )
        ]
    
    def add_route(self, route: Route) -> bool:
        """
        Add a new routing destination.
        
        Args:
            route: Route to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.routes.append(route)
            self.logger.info(f"Added route: {route.name} ({route.domain})")
            return True
        except Exception as e:
            self.logger.error(f"Error adding route: {str(e)}")
            return False
    
    def remove_route(self, domain: KnowledgeDomain) -> bool:
        """
        Remove a routing destination.
        
        Args:
            domain: Domain to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.routes = [route for route in self.routes if route.domain != domain]
            self.logger.info(f"Removed route: {domain}")
            return True
        except Exception as e:
            self.logger.error(f"Error removing route: {str(e)}")
            return False
    
    async def analyze_query_domains(self, query: str) -> Dict[str, float]:
        """
        Analyze which domains are relevant for a query.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary mapping domain names to relevance scores
        """
        try:
            self.logger.info(f"Analyzing query domains for: {query}")
            
            # Create routing chain
            routing_chain = self.routing_prompt | self.llm | JsonOutputParser()
            
            # Analyze domains
            analysis_result = await routing_chain.ainvoke({"query": query})
            
            # Extract domain scores
            domain_scores = {}
            if isinstance(analysis_result, dict):
                for domain_name, score in analysis_result.items():
                    if isinstance(score, (int, float)):
                        domain_scores[domain_name] = float(score)
                    elif isinstance(score, dict) and "score" in score:
                        domain_scores[domain_name] = float(score["score"])
            
            # Ensure we have scores for all domains
            for route in self.routes:
                if route.domain.value not in domain_scores:
                    domain_scores[route.domain.value] = 0.0
            
            self.logger.info(f"Domain analysis completed: {domain_scores}")
            return domain_scores
            
        except Exception as e:
            self.logger.error(f"Error analyzing query domains: {str(e)}")
            # Return default scores
            return {route.domain.value: 0.5 for route in self.routes}
    
    def calculate_route_scores(
        self, 
        query: str, 
        domain_scores: Dict[str, float]
    ) -> List[Tuple[Route, float]]:
        """
        Calculate final route scores combining domain analysis and keyword matching.
        
        Args:
            query: User query
            domain_scores: Domain relevance scores from LLM analysis
            
        Returns:
            List of (Route, score) tuples sorted by score
        """
        try:
            self.logger.info("Calculating route scores")
            
            route_scores = []
            query_lower = query.lower()
            
            for route in self.routes:
                # Get domain score from LLM analysis
                domain_score = domain_scores.get(route.domain.value, 0.0)
                
                # Calculate keyword match score
                keyword_matches = sum(
                    1 for keyword in route.keywords 
                    if keyword.lower() in query_lower
                )
                keyword_score = min(1.0, keyword_matches / len(route.keywords)) if route.keywords else 0.0
                
                # Combine scores with weights
                combined_score = (
                    domain_score * 0.7 +  # LLM analysis weight
                    keyword_score * 0.3   # Keyword matching weight
                ) * route.weight
                
                route_scores.append((route, combined_score))
            
            # Sort by score (descending)
            route_scores.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Route scores calculated: {[(r.name, f'{s:.3f}') for r, s in route_scores[:3]]}")
            return route_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating route scores: {str(e)}")
            return [(route, 0.5) for route in self.routes]
    
    async def route_query(
        self, 
        query: str, 
        max_routes: int = 3,
        routing_threshold: float = 0.7
    ) -> List[Tuple[Route, float]]:
        """
        Route a query to appropriate destinations.
        
        Args:
            query: User query to route
            max_routes: Maximum number of routes to return
            routing_threshold: Minimum score threshold for routing
            
        Returns:
            List of (Route, score) tuples for selected routes
        """
        try:
            self.logger.info(f"Routing query: {query}")
            
            # Step 1: Analyze query domains
            domain_scores = await self.analyze_query_domains(query)
            
            # Step 2: Calculate route scores
            route_scores = self.calculate_route_scores(query, domain_scores)
            
            # Step 3: Filter by threshold and limit
            selected_routes = [
                (route, score) for route, score in route_scores
                if score >= routing_threshold
            ][:max_routes]
            
            # If no routes meet threshold, return top route anyway
            if not selected_routes and route_scores:
                selected_routes = [route_scores[0]]
            
            self.logger.info(f"Query routed to {len(selected_routes)} destinations")
            return selected_routes
            
        except Exception as e:
            self.logger.error(f"Error routing query: {str(e)}")
            # Return general route as fallback
            general_route = next((route for route in self.routes if route.domain == KnowledgeDomain.GENERAL), None)
            if general_route:
                return [(general_route, 0.5)]
            return []


class QueryRoutingSystem:
    """
    Complete query routing system that routes queries and retrieves documents.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the query routing system.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.logger = logging.getLogger("query_routing_system")
        
        # Initialize router
        self.router = QueryRouter(embedding_model)
        
        self.logger.info("Query routing system initialized successfully")
    
    async def process_query(
        self, 
        query: str, 
        retriever_func,
        max_routes: int = 3,
        routing_threshold: float = 0.7,
        k_per_route: int = 3
    ) -> RoutingResult:
        """
        Process a query using routing system.
        
        Args:
            query: User query to process
            retriever_func: Function to retrieve documents for a query
            max_routes: Maximum number of routes to use
            routing_threshold: Minimum score threshold for routing
            k_per_route: Number of documents to retrieve per route
            
        Returns:
            RoutingResult with routing information and retrieved documents
        """
        try:
            self.logger.info(f"Processing query with routing: {query}")
            
            # Step 1: Route the query
            selected_routes = await self.router.route_query(
                query, max_routes, routing_threshold
            )
            
            # Step 2: Retrieve documents from each route
            all_documents = []
            route_scores = []
            
            for route, score in selected_routes:
                try:
                    # Create route-specific query (could be enhanced)
                    route_query = f"{query} {route.description}"
                    
                    # Retrieve documents
                    docs = retriever_func(route_query, k_per_route)
                    all_documents.extend(docs)
                    route_scores.append(score)
                    
                    self.logger.info(f"Retrieved {len(docs)} documents from route: {route.name}")
                    
                except Exception as e:
                    self.logger.warning(f"Error retrieving from route {route.name}: {str(e)}")
                    route_scores.append(0.0)
            
            # Step 3: Remove duplicate documents
            unique_docs = []
            seen_content = set()
            
            for doc in all_documents:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            # Create result
            result = RoutingResult(
                original_query=query,
                selected_routes=[route for route, _ in selected_routes],
                route_scores=route_scores,
                retrieved_documents=unique_docs,
                routing_metadata={
                    "total_routes": len(selected_routes),
                    "total_retrieved": len(all_documents),
                    "unique_documents": len(unique_docs),
                    "routing_threshold": routing_threshold,
                    "k_per_route": k_per_route
                }
            )
            
            self.logger.info(f"Query routing completed. Used {len(selected_routes)} routes, retrieved {len(unique_docs)} unique documents")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in query routing processing: {str(e)}")
            # Return fallback result
            return RoutingResult(
                original_query=query,
                selected_routes=[],
                route_scores=[],
                retrieved_documents=[],
                routing_metadata={"error": str(e), "fallback": True}
            )
    
    def get_available_routes(self) -> List[Dict[str, Any]]:
        """
        Get information about available routes.
        
        Returns:
            List of route information dictionaries
        """
        return [
            {
                "domain": route.domain.value,
                "name": route.name,
                "description": route.description,
                "keywords": route.keywords,
                "weight": route.weight,
                "threshold": route.threshold
            }
            for route in self.router.routes
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the query routing system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "system_type": "Query Routing",
            "router_available": self.router is not None,
            "embedding_model": self.router.embedding_model,
            "available_routes": len(self.router.routes),
            "route_domains": [route.domain.value for route in self.router.routes]
        }


# Global query routing system instance
_query_routing_system_instance = None


def get_query_routing_system(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> QueryRoutingSystem:
    """
    Get or create the global query routing system instance.
    
    Args:
        embedding_model: Name of the sentence transformer model
        
    Returns:
        QueryRoutingSystem instance
    """
    global _query_routing_system_instance
    
    if _query_routing_system_instance is None:
        _query_routing_system_instance = QueryRoutingSystem(embedding_model)
    
    return _query_routing_system_instance
