"""
Query Strategy Agent with RAG Optimization Tools.
This agent analyzes queries and uses appropriate RAG optimization tools.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

from core.config.config import config
from rag.strategies.query_strategy_agent import QueryAnalysis, StrategyDecision, RetrievalTechnique
from rag.optimization.rag_optimization_tools import (
    RAG_OPTIMIZATION_TOOLS,
    query_rewriting_tool,
    multi_query_rag_tool,
    query_routing_tool,
    multi_hop_rag_tool,
    standard_rag_tool,
    hybrid_rag_tool
)


class QueryStrategyAgentWithTools:
    """
    Query Strategy Agent that uses RAG optimization tools to process queries.
    """
    
    def __init__(self):
        """Initialize the Query Strategy Agent with tools."""
        self.logger = logging.getLogger("query_strategy_agent_with_tools")
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=0.1,  # Low temperature for consistent decisions
            max_tokens=2000
        )
        
        # Create tools mapping
        self.tools_mapping = {
            "query_rewriting": query_rewriting_tool,
            "multi_query_rag": multi_query_rag_tool,
            "query_routing": query_routing_tool,
            "multi_hop_rag": multi_hop_rag_tool,
            "standard_rag": standard_rag_tool,
            "hybrid_rag": hybrid_rag_tool
        }
        
        # Create agent prompt
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Query Strategy Agent that analyzes user queries and selects the best RAG optimization technique.

Your task is to:
1. Analyze the user query to understand its characteristics
2. Select the most appropriate RAG optimization tool
3. Execute the tool with optimal parameters
4. Return the results

Available RAG Optimization Tools:
- query_rewriting_tool: Expands and rewrites queries to improve retrieval coverage
  Best for: Ambiguous queries, queries that need expansion
  Parameters: expansion_factor, rewrite_count, similarity_threshold

- multi_query_rag_tool: Generates multiple queries and fuses results
  Best for: Complex queries requiring multiple perspectives
  Parameters: query_count, k_per_query, fusion_method, final_k

- query_routing_tool: Routes queries to specialized knowledge domains
  Best for: Domain-specific queries, specialized knowledge required
  Parameters: max_routes, routing_threshold, k_per_route

- multi_hop_rag_tool: Performs multi-step reasoning by iteratively gathering information
  Best for: Complex reasoning requiring multiple information gathering steps
  Parameters: max_hops, k_per_hop, hop_strategy, reasoning_threshold

- standard_rag_tool: Standard retrieval without optimization
  Best for: Simple, direct queries
  Parameters: k, similarity_threshold

- hybrid_rag_tool: Combines multiple techniques for optimal results
  Best for: Complex queries that benefit from multiple approaches
  Parameters: primary_technique, secondary_technique, primary_weight, secondary_weight

Guidelines:
- Analyze query complexity, intent, and domain
- Choose the tool that best matches the query characteristics
- Use appropriate parameters for the selected tool
- If unsure, start with standard_rag_tool and then try more advanced techniques
- Always provide reasoning for your tool selection

Format your response as a JSON object with:
- analysis: Your analysis of the query
- selected_tool: The tool you selected
- parameters: Parameters for the tool
- reasoning: Why you selected this tool and parameters"""),
            ("human", "Query: {query}\n\nAnalyze this query and select the best RAG optimization tool:")
        ])
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=RAG_OPTIMIZATION_TOOLS,
            prompt=self.agent_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=RAG_OPTIMIZATION_TOOLS,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        self.logger.info("Query Strategy Agent with tools initialized successfully")
    
    async def analyze_and_process_query(
        self, 
        query: str, 
        retriever_func,
        generate_response: bool = True,
        response_generator_func = None
    ) -> Dict[str, Any]:
        """
        Analyze a query and process it using the appropriate RAG optimization tool.
        
        Args:
            query: User query to process
            retriever_func: Function to retrieve documents
            generate_response: Whether to generate a final response
            response_generator_func: Function to generate response from documents
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Processing query with tools: {query}")
            
            # Prepare input for the agent
            agent_input = {
                "input": f"Query: {query}\n\nRetriever function available: {retriever_func.__name__ if hasattr(retriever_func, '__name__') else 'custom_function'}"
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Extract results
            agent_output = result.get("output", "")
            
            # Try to parse JSON response if available
            try:
                import json
                parsed_result = json.loads(agent_output)
            except:
                parsed_result = {"analysis": agent_output, "selected_tool": "standard_rag", "reasoning": "Fallback parsing"}
            
            # Generate response if requested
            final_response = ""
            if generate_response and response_generator_func:
                try:
                    # For now, use a simple approach to get documents
                    # In a real implementation, you'd extract documents from the tool results
                    context = f"Query: {query}\nAgent Analysis: {parsed_result.get('analysis', '')}"
                    final_response = response_generator_func(query, context)
                except Exception as e:
                    self.logger.error(f"Error generating response: {str(e)}")
                    final_response = f"Error generating response: {str(e)}"
            
            return {
                "query": query,
                "agent_output": agent_output,
                "parsed_result": parsed_result,
                "final_response": final_response,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query with tools: {str(e)}")
            return {
                "query": query,
                "agent_output": "",
                "parsed_result": {"error": str(e)},
                "final_response": f"Error processing query: {str(e)}",
                "success": False
            }
    
    async def process_query_with_tool(
        self, 
        query: str, 
        tool_name: str, 
        retriever_func,
        tool_parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a query using a specific RAG optimization tool.
        
        Args:
            query: User query to process
            tool_name: Name of the tool to use
            retriever_func: Function to retrieve documents
            tool_parameters: Parameters for the tool
            
        Returns:
            Dictionary with tool execution results
        """
        try:
            self.logger.info(f"Processing query with specific tool: {tool_name}")
            
            # Get the tool
            tool = self.tools_mapping.get(tool_name)
            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Set default parameters
            if tool_parameters is None:
                tool_parameters = {}
            
            # Add retriever function to parameters
            tool_parameters["retriever_func"] = retriever_func
            tool_parameters["query"] = query
            
            # Execute the tool
            tool_result = await tool.ainvoke(tool_parameters)
            
            return {
                "query": query,
                "tool_name": tool_name,
                "tool_parameters": tool_parameters,
                "tool_result": tool_result,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "query": query,
                "tool_name": tool_name,
                "tool_parameters": tool_parameters,
                "tool_result": None,
                "success": False,
                "error": str(e)
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available RAG optimization tools.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": "query_rewriting_tool",
                "description": "Expands and rewrites queries to improve retrieval coverage",
                "best_for": "Ambiguous queries, queries that need expansion",
                "parameters": ["expansion_factor", "rewrite_count", "similarity_threshold"]
            },
            {
                "name": "multi_query_rag_tool",
                "description": "Generates multiple queries and fuses results",
                "best_for": "Complex queries requiring multiple perspectives",
                "parameters": ["query_count", "k_per_query", "fusion_method", "final_k"]
            },
            {
                "name": "query_routing_tool",
                "description": "Routes queries to specialized knowledge domains",
                "best_for": "Domain-specific queries, specialized knowledge required",
                "parameters": ["max_routes", "routing_threshold", "k_per_route"]
            },
            {
                "name": "multi_hop_rag_tool",
                "description": "Performs multi-step reasoning by iteratively gathering information",
                "best_for": "Complex reasoning requiring multiple information gathering steps",
                "parameters": ["max_hops", "k_per_hop", "hop_strategy", "reasoning_threshold"]
            },
            {
                "name": "standard_rag_tool",
                "description": "Standard retrieval without optimization",
                "best_for": "Simple, direct queries",
                "parameters": ["k", "similarity_threshold"]
            },
            {
                "name": "hybrid_rag_tool",
                "description": "Combines multiple techniques for optimal results",
                "best_for": "Complex queries that benefit from multiple approaches",
                "parameters": ["primary_technique", "secondary_technique", "primary_weight", "secondary_weight"]
            }
        ]
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Query Strategy Agent.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "agent_type": "Query Strategy Agent with Tools",
            "available_tools": len(self.tools_mapping),
            "llm_model": config.anthropic.model,
            "max_iterations": 3,
            "tools": list(self.tools_mapping.keys())
        }


# Global query strategy agent with tools instance
_query_strategy_agent_with_tools_instance = None


def get_query_strategy_agent_with_tools() -> QueryStrategyAgentWithTools:
    """
    Get or create the global Query Strategy Agent with tools instance.
    
    Returns:
        QueryStrategyAgentWithTools instance
    """
    global _query_strategy_agent_with_tools_instance
    
    if _query_strategy_agent_with_tools_instance is None:
        _query_strategy_agent_with_tools_instance = QueryStrategyAgentWithTools()
    
    return _query_strategy_agent_with_tools_instance
