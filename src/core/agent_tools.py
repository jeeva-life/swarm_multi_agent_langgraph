"""
LangChain tools for all agents in the multi-agent swarm system.
Provides tool-based architecture instead of class-based agents.
"""

from typing import Dict, List, Any, Optional
import logging
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from core.config.config import config as Config
from rag.systems.rag_system import get_rag_system
from systems.nl2sql_system import get_nl2sql_system
from systems.invoice_system import get_invoice_system


class AgentTools:
    """
    Centralized tool management for all agents.
    Provides LangChain tools for RAG, NL2SQL, and Invoice processing.
    """
    
    def __init__(self, database_client: Optional[Any] = None):
        """
        Initialize agent tools.
        
        Args:
            database_client: Database client for NL2SQL and Invoice operations
        """
        self.config = Config()
        self.database_client = database_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize systems
        self.rag_system = get_rag_system()
        self.nl2sql_system = get_nl2sql_system()
        self.invoice_system = get_invoice_system()
        
        # Set database client for systems that need it
        if database_client:
            self.nl2sql_system.database_client = database_client
            self.invoice_system.database_client = database_client
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=self.config.anthropic.model,
            api_key=self.config.anthropic.api_key,
            temperature=self.config.anthropic.temperature,
            max_tokens=self.config.anthropic.max_tokens
        )
    
    def get_rag_tools(self) -> List[Any]:
        """Get RAG-related tools."""
        return [
            self.rag_general_qa_tool,
            self.rag_search_documents_tool,
            self.rag_load_documents_tool
        ]
    
    def get_nl2sql_tools(self) -> List[Any]:
        """Get NL2SQL-related tools."""
        return [
            self.nl2sql_query_tool,
            self.nl2sql_schema_tool,
            self.nl2sql_validate_tool
        ]
    
    def get_invoice_tools(self) -> List[Any]:
        """Get Invoice-related tools."""
        return [
            self.invoice_customer_tool,
            self.invoice_employee_tool,
            self.invoice_summary_tool,
            self.invoice_details_tool,
            self.invoice_sorted_tool
        ]
    
    def get_all_tools(self) -> List[Any]:
        """Get all available tools."""
        return (
            self.get_rag_tools() + 
            self.get_nl2sql_tools() + 
            self.get_invoice_tools()
        )
    
    # RAG Tools
    @tool
    def rag_general_qa_tool(self, query: str) -> str:
        """
        Tool for handling general information queries using RAG system.
        Use this for questions about documents, policies, or knowledge base content.
        
        Args:
            query (str): The user's question about documents or knowledge.
        
        Returns:
            str: Answer based on retrieved documents.
        """
        try:
            return self.rag_system.rag_response_generator_async(query)
        except Exception as e:
            self.logger.error(f"Error in RAG general QA tool: {str(e)}")
            return f"I encountered an error retrieving information: {str(e)}"
    
    @tool
    def rag_search_documents_tool(self, query: str, k: int = 3) -> str:
        """
        Tool for searching relevant documents in the knowledge base.
        
        Args:
            query (str): Search query for documents.
            k (int): Number of documents to retrieve (default: 3).
        
        Returns:
            str: List of relevant documents.
        """
        try:
            results = self.rag_system.search_documents(query, k)
            if results:
                return f"Found {len(results)} relevant documents:\n" + "\n".join([f"- {doc}" for doc in results])
            else:
                return "No relevant documents found for your query."
        except Exception as e:
            self.logger.error(f"Error in RAG search tool: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    @tool
    def rag_load_documents_tool(self, folder_path: str) -> str:
        """
        Tool for loading documents into the knowledge base.
        
        Args:
            folder_path (str): Path to folder containing documents.
        
        Returns:
            str: Status of document loading operation.
        """
        try:
            self.rag_system.load_documents(folder_path)
            return f"Successfully loaded documents from {folder_path}"
        except Exception as e:
            self.logger.error(f"Error in RAG load documents tool: {str(e)}")
            return f"Error loading documents: {str(e)}"
    
    # NL2SQL Tools
    @tool
    def nl2sql_query_tool(self, query: str) -> str:
        """
        Tool for converting natural language to SQL and executing queries.
        Use this for database-related questions that require SQL queries.
        
        Args:
            query (str): Natural language question about database data.
        
        Returns:
            str: SQL query results or error message.
        """
        try:
            return self.nl2sql_system.sql_response_generator_async(query)
        except Exception as e:
            self.logger.error(f"Error in NL2SQL query tool: {str(e)}")
            return f"I encountered an error processing your database query: {str(e)}"
    
    @tool
    def nl2sql_schema_tool(self) -> str:
        """
        Tool for getting database schema information.
        
        Returns:
            str: Database schema information.
        """
        try:
            return self.nl2sql_system.get_database_info()
        except Exception as e:
            self.logger.error(f"Error in NL2SQL schema tool: {str(e)}")
            return f"Error retrieving database schema: {str(e)}"
    
    @tool
    def nl2sql_validate_tool(self, sql_query: str) -> str:
        """
        Tool for validating SQL queries.
        
        Args:
            sql_query (str): SQL query to validate.
        
        Returns:
            str: Validation result.
        """
        try:
            is_valid = self.nl2sql_system.validate_sql_query(sql_query)
            if is_valid:
                return "SQL query is valid and safe to execute."
            else:
                return "SQL query is invalid or contains unsafe operations."
        except Exception as e:
            self.logger.error(f"Error in NL2SQL validate tool: {str(e)}")
            return f"Error validating SQL query: {str(e)}"
    
    # Invoice Tools
    @tool
    def invoice_customer_tool(self, customer_id: str, sort_by: str = "date") -> str:
        """
        Tool for getting customer invoice information.
        
        Args:
            customer_id (str): Customer ID to look up.
            sort_by (str): Sort by "date" or "price" (default: "date").
        
        Returns:
            str: Customer invoice information.
        """
        try:
            if sort_by == "price":
                results = self.invoice_system.get_invoices_sorted_by_unit_price(customer_id)
            else:
                results = self.invoice_system.get_invoices_by_customer_sorted_by_date(customer_id)
            
            if results:
                return f"Found {len(results)} invoice records for customer {customer_id}:\n" + str(results)
            else:
                return f"No invoices found for customer {customer_id}"
        except Exception as e:
            self.logger.error(f"Error in invoice customer tool: {str(e)}")
            return f"Error retrieving customer invoices: {str(e)}"
    
    @tool
    def invoice_employee_tool(self, invoice_id: str, customer_id: str) -> str:
        """
        Tool for getting employee information for a specific invoice.
        
        Args:
            invoice_id (str): Invoice ID.
            customer_id (str): Customer ID.
        
        Returns:
            str: Employee information.
        """
        try:
            result = self.invoice_system.get_employee_by_invoice_and_customer(invoice_id, customer_id)
            if "error" not in result:
                return f"Employee information for invoice {invoice_id}:\n{result}"
            else:
                return result["error"]
        except Exception as e:
            self.logger.error(f"Error in invoice employee tool: {str(e)}")
            return f"Error retrieving employee information: {str(e)}"
    
    @tool
    def invoice_summary_tool(self, customer_id: str) -> str:
        """
        Tool for getting customer invoice summary.
        
        Args:
            customer_id (str): Customer ID to get summary for.
        
        Returns:
            str: Customer invoice summary.
        """
        try:
            result = self.invoice_system.get_customer_invoice_summary(customer_id)
            if "error" not in result:
                return f"Customer {customer_id} invoice summary:\n{result}"
            else:
                return result["error"]
        except Exception as e:
            self.logger.error(f"Error in invoice summary tool: {str(e)}")
            return f"Error retrieving customer summary: {str(e)}"
    
    @tool
    def invoice_details_tool(self, invoice_id: str) -> str:
        """
        Tool for getting detailed invoice information.
        
        Args:
            invoice_id (str): Invoice ID to get details for.
        
        Returns:
            str: Detailed invoice information.
        """
        try:
            result = self.invoice_system.get_invoice_details_by_id(invoice_id)
            if "error" not in result:
                return f"Invoice {invoice_id} details:\n{result}"
            else:
                return result["error"]
        except Exception as e:
            self.logger.error(f"Error in invoice details tool: {str(e)}")
            return f"Error retrieving invoice details: {str(e)}"
    
    @tool
    def invoice_sorted_tool(self, customer_id: str, sort_by: str = "unit_price") -> str:
        """
        Tool for getting invoices sorted by specific criteria.
        
        Args:
            customer_id (str): Customer ID.
            sort_by (str): Sort by "unit_price" or other criteria.
        
        Returns:
            str: Sorted invoice information.
        """
        try:
            if sort_by == "unit_price":
                results = self.invoice_system.get_invoices_sorted_by_unit_price(customer_id)
            else:
                results = self.invoice_system.get_invoices_by_customer_sorted_by_date(customer_id)
            
            if results:
                return f"Found {len(results)} invoice items for customer {customer_id}:\n{results}"
            else:
                return f"No invoice items found for customer {customer_id}"
        except Exception as e:
            self.logger.error(f"Error in invoice sorted tool: {str(e)}")
            return f"Error retrieving sorted invoices: {str(e)}"


# Global instance for easy access
_agent_tools_instance = None

def get_agent_tools(database_client: Optional[Any] = None) -> AgentTools:
    """Get the global agent tools instance."""
    global _agent_tools_instance
    if _agent_tools_instance is None or database_client is not None:
        _agent_tools_instance = AgentTools(database_client)
    return _agent_tools_instance
