"""
LangGraph Swarm with handoff tools for proper multi-agent coordination.
Uses create_handoff_tool and create_swarm for seamless agent handoffs.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

from core.config import Config
from core.memory import MemoryManager
from core.monitoring import MetricsCollector, AlertManager, DriftDetector
from core.agent_tools import get_agent_tools
from agents.evaluation_agent_with_handoff import EvaluationAgentWithHandoff


class HandoffSwarm:
    """
    LangGraph Swarm with handoff tools for proper multi-agent coordination.
    Uses create_handoff_tool and create_swarm for seamless agent handoffs.
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        drift_detector: DriftDetector,
        langchain_client: Optional[Any] = None,
        database_client: Optional[Any] = None
    ):
        self.logger = logging.getLogger("handoff_swarm")
        self.memory_manager = memory_manager
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.drift_detector = drift_detector
        self.langchain_client = langchain_client
        self.database_client = database_client
        
        # Initialize configuration
        self.config = Config()
        
        # Initialize agent tools
        self.agent_tools = get_agent_tools(database_client)
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=self.config.anthropic.model,
            api_key=self.config.anthropic.api_key,
            temperature=self.config.anthropic.temperature,
            max_tokens=self.config.anthropic.max_tokens
        )
        
        # Initialize evaluation agent
        self.evaluation_agent = EvaluationAgentWithHandoff(
            memory_manager=memory_manager,
            metrics_collector=metrics_collector
        )
        
        # Initialize agents and swarm
        self.agents = {}
        self.swarm_workflow = None
        
        # Initialize the swarm
        self._initialize_agents()
        self._create_swarm_workflow()
    
    def _initialize_agents(self):
        """Initialize all agents with handoff tools."""
        try:
            # Get tool lists
            rag_tools = self.agent_tools.get_rag_tools()
            nl2sql_tools = self.agent_tools.get_nl2sql_tools()
            invoice_tools = self.agent_tools.get_invoice_tools()
            
            # General QA Agent (acts as router/coordinator)
            self.agents["general_qa_agent_with_handoff"] = create_react_agent(
                model=self.llm,
                tools=[
                    *rag_tools,  # Unpack RAG tools
                    create_handoff_tool(
                        agent_name="invoice_information_agent_with_handoff",
                        description="Transfer user to the invoice information agent that can help with invoice information, billing, payments, and customer data"
                    ),
                    create_handoff_tool(
                        agent_name="nl2sql_agent_with_handoff",
                        description="Transfer user to the nl2sql agent that can help with database related queries, SQL operations, and data retrieval"
                    ),
                    create_handoff_tool(
                        agent_name="evaluation_agent_with_handoff",
                        description="Transfer to evaluation agent to assess response quality using RAGAS and DEEPEVAL metrics"
                    ),
                ],
                name="general_qa_agent_with_handoff",
                prompt=self._get_general_qa_prompt()
            )
            
            # NL2SQL Agent
            self.agents["nl2sql_agent_with_handoff"] = create_react_agent(
                model=self.llm,
                tools=[
                    *nl2sql_tools,  # Unpack NL2SQL tools
                    create_handoff_tool(
                        agent_name="general_qa_agent_with_handoff",
                        description="Transfer user to the general qa agent that can help with general information queries about companies, documents, or knowledge base content"
                    ),
                    create_handoff_tool(
                        agent_name="invoice_information_agent_with_handoff",
                        description="Transfer user to the invoice information agent that can help with invoice information, billing, and payment data"
                    ),
                    create_handoff_tool(
                        agent_name="evaluation_agent_with_handoff",
                        description="Transfer to evaluation agent to assess response quality using RAGAS and DEEPEVAL metrics"
                    ),
                ],
                name="nl2sql_agent_with_handoff",
                prompt=self._get_nl2sql_prompt()
            )
            
            # Invoice Information Agent
            self.agents["invoice_information_agent_with_handoff"] = create_react_agent(
                model=self.llm,
                tools=[
                    *invoice_tools,  # Unpack invoice tools
                    create_handoff_tool(
                        agent_name="general_qa_agent_with_handoff",
                        description="Transfer user to the general qa agent that can help with general information queries about companies, documents, or knowledge base content"
                    ),
                    create_handoff_tool(
                        agent_name="nl2sql_agent_with_handoff",
                        description="Transfer user to the nl2sql agent that can help with database related queries and data operations"
                    ),
                    create_handoff_tool(
                        agent_name="evaluation_agent_with_handoff",
                        description="Transfer to evaluation agent to assess response quality using RAGAS and DEEPEVAL metrics"
                    ),
                ],
                name="invoice_information_agent_with_handoff",
                prompt=self._get_invoice_prompt()
            )
            
            # Evaluation Agent
            self.agents["evaluation_agent_with_handoff"] = create_react_agent(
                model=self.llm,
                tools=[
                    *self.evaluation_agent.handoff_tools,  # Unpack evaluation tools
                    create_handoff_tool(
                        agent_name="general_qa_agent_with_handoff",
                        description="Transfer user back to the general qa agent after evaluation"
                    ),
                    create_handoff_tool(
                        agent_name="nl2sql_agent_with_handoff",
                        description="Transfer user back to the nl2sql agent after evaluation"
                    ),
                    create_handoff_tool(
                        agent_name="invoice_information_agent_with_handoff",
                        description="Transfer user back to the invoice agent after evaluation"
                    ),
                ],
                name="evaluation_agent_with_handoff",
                prompt=self._get_evaluation_prompt()
            )
            
            self.logger.info(f"Initialized {len(self.agents)} agents with handoff tools: {list(self.agents.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _create_swarm_workflow(self):
        """Create the swarm workflow using create_swarm."""
        try:
            # Create the swarm workflow
            self.swarm_workflow = create_swarm(
                agents=list(self.agents.values()),
                default_active_agent="general_qa_agent_with_handoff"
            )
            
            # Compile with checkpointer (if available)
            # Note: In production, you would use a proper checkpointer
            self.swarm_agents = self.swarm_workflow.compile()
            
            self.logger.info("Swarm workflow created and compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating swarm workflow: {str(e)}")
            raise
    
    def _get_general_qa_prompt(self) -> str:
        """Get the prompt for the General QA agent."""
        return """You are a specialized assistant for document-based queries and general knowledge questions.

You have access to tools for:
- rag_general_qa_tool: Answer questions using retrieved documents
- rag_search_documents_tool: Search for relevant documents
- rag_load_documents_tool: Load documents into the knowledge base

IMPORTANT RULES:
- Use rag_general_qa_tool for most document-based questions
- Use rag_search_documents_tool when users want to see what documents are available
- Use rag_load_documents_tool when users want to add new documents
- If the query is about databases, invoices, or structured data, use the appropriate handoff tool
- You act as the main coordinator and can hand off to specialized agents when needed

CORE RESPONSIBILITIES:
- Answer questions using retrieved document context
- Provide accurate information from the knowledge base
- Help users find relevant documents
- Coordinate with other agents when specialized knowledge is needed
- Maintain a professional and helpful tone

Always use the appropriate tool to answer the user's question, and hand off to specialists when their expertise is needed."""

    def _get_nl2sql_prompt(self) -> str:
        """Get the prompt for the NL2SQL agent."""
        return """You are a specialized assistant for database queries and SQL operations.

You have access to tools for:
- nl2sql_query_tool: Convert natural language to SQL and execute queries
- nl2sql_schema_tool: Get database schema information
- nl2sql_validate_tool: Validate SQL queries

IMPORTANT RULES:
- Use nl2sql_query_tool for most database-related questions
- Use nl2sql_schema_tool when users need to understand the database structure
- Use nl2sql_validate_tool to check if SQL queries are safe
- If the query is about documents or general knowledge, hand off to the general QA agent
- If the query is about invoices or billing, hand off to the invoice agent

CORE RESPONSIBILITIES:
- Convert natural language questions to SQL queries
- Execute database queries safely
- Provide clear explanations of query results
- Help users understand database structure
- Hand off to other agents when their expertise is needed
- Maintain a professional and helpful tone

Always use the appropriate tool to answer the user's question, and hand off to other specialists when their expertise is needed."""

    def _get_invoice_prompt(self) -> str:
        """Get the prompt for the Invoice agent."""
        return """You are a specialized assistant for invoice and billing-related queries.

You have access to tools for:
- invoice_customer_tool: Get customer invoice information
- invoice_employee_tool: Get employee information for invoices
- invoice_summary_tool: Get customer invoice summaries
- invoice_details_tool: Get detailed invoice information
- invoice_sorted_tool: Get invoices sorted by various criteria

IMPORTANT RULES:
- Use the appropriate tool based on what the user is asking for
- invoice_customer_tool for general customer invoice queries
- invoice_employee_tool when users ask about who helped them
- invoice_summary_tool for customer statistics
- invoice_details_tool for specific invoice information
- invoice_sorted_tool for sorted invoice lists
- If the query is about documents or general knowledge, hand off to the general QA agent
- If the query is about database operations, hand off to the NL2SQL agent

CORE RESPONSIBILITIES:
- Help users with invoice-related questions
- Provide customer invoice information
- Find employee information for invoices
- Generate invoice summaries and reports
- Hand off to other agents when their expertise is needed
- Maintain a professional and helpful tone

Always use the appropriate tool to answer the user's question, and hand off to other specialists when their expertise is needed."""

    def _get_evaluation_prompt(self) -> str:
        """Get the prompt for the Evaluation agent."""
        return """You are a specialized evaluation agent that assesses response quality using RAGAS and DEEPEVAL metrics.

You have access to tools for:
- evaluate_response_quality: Comprehensive quality evaluation
- evaluate_rag_performance: RAG-specific metrics (faithfulness, relevancy, context precision)
- evaluate_llm_safety: Safety metrics (hallucination, bias, toxicity)
- get_quality_feedback: Generate human-readable feedback
- suggest_response_improvements: Provide specific improvement suggestions

IMPORTANT RULES:
- Use evaluate_response_quality for comprehensive assessment
- Use evaluate_rag_performance when context documents are available
- Use evaluate_llm_safety for safety and bias assessment
- Always provide constructive feedback and actionable suggestions
- Hand off back to the original agent after evaluation

CORE RESPONSIBILITIES:
- Evaluate response quality using multiple metrics
- Provide detailed feedback on response strengths and weaknesses
- Suggest specific improvements for better responses
- Assess RAG performance when context is available
- Check for safety issues like hallucination, bias, and toxicity
- Maintain objective and constructive evaluation approach

Always provide thorough evaluation results and hand off back to the appropriate agent after assessment."""

    async def process_query(self, query: str, user_id: str = "default", session_id: str = None) -> Dict[str, Any]:
        """
        Process a query using the handoff swarm.
        
        Args:
            query (str): User query
            user_id (str): User identifier
            session_id (str): Session identifier
        
        Returns:
            Dict[str, Any]: Response with agent output and metadata
        """
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Create initial message
            messages = [HumanMessage(content=query)]
            
            # Process through swarm
            result = self.swarm_agents.invoke({
                "messages": messages
            })
            
            # Extract response
            if result and "messages" in result:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = "I couldn't process your query."
            
            # Determine which agent was used (from the last message)
            agent_used = "general_qa_agent"  # Default
            if "invoice" in response_content.lower() or "billing" in response_content.lower():
                agent_used = "invoice_agent"
            elif "sql" in response_content.lower() or "database" in response_content.lower():
                agent_used = "nl2sql_agent"
            
            # Prepare response
            response = {
                "content": response_content,
                "agent_used": agent_used,
                "session_id": session_id,
                "metadata": {
                    "swarm_type": "handoff_swarm",
                    "agents_available": list(self.agents.keys()),
                    "tools_available": len(self.agent_tools.get_all_tools()),
                    "processing_time": datetime.now().isoformat()
                }
            }
            
            # Store in memory
            await self.memory_manager.short_term.store(
                f"session:{session_id}:last_query",
                {"query": query, "response": response_content},
                ttl=3600  # 1 hour
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "content": f"I encountered an error processing your query: {str(e)}",
                "agent_used": "error",
                "session_id": session_id or "unknown",
                "metadata": {"error": str(e)}
            }
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get the current status of the swarm."""
        try:
            return {
                "status": "running",
                "swarm_type": "handoff_swarm",
                "agents": list(self.agents.keys()),
                "default_agent": "general_qa_agent_with_handoff",
                "tools_available": len(self.agent_tools.get_all_tools()),
                "rag_tools": len(self.agent_tools.get_rag_tools()),
                "nl2sql_tools": len(self.agent_tools.get_nl2sql_tools()),
                "invoice_tools": len(self.agent_tools.get_invoice_tools()),
                "workflow_initialized": self.swarm_workflow is not None,
                "database_connected": self.database_client is not None
            }
        except Exception as e:
            self.logger.error(f"Error getting swarm status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def show_graph(self):
        """Display the swarm workflow graph."""
        try:
            if self.swarm_workflow:
                # This would show the graph similar to the image you provided
                print("Swarm Workflow Graph:")
                print("=" * 50)
                print("__start__ -> general_qa_agent_with_handoff")
                print("__start__ -> invoice_information_agent_with_handoff") 
                print("__start__ -> nl2sql_agent_with_handoff")
                print()
                print("general_qa_agent_with_handoff <-> invoice_information_agent_with_handoff")
                print("general_qa_agent_with_handoff <-> nl2sql_agent_with_handoff")
                print("invoice_information_agent_with_handoff <-> nl2sql_agent_with_handoff")
                print("=" * 50)
            else:
                print("Swarm workflow not initialized")
        except Exception as e:
            self.logger.error(f"Error showing graph: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the swarm and cleanup resources."""
        try:
            self.logger.info("Shutting down handoff swarm...")
            # Cleanup if needed
            self.logger.info("Handoff swarm shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
