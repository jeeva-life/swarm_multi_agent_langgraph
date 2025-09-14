"""
Comprehensive NL2SQL (Natural Language to SQL) system.
Handles natural language to SQL conversion and database query execution.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime

# LangChain imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from operator import itemgetter

from core.config import config


class NL2SQLSystem:
    """
    Natural Language to SQL conversion system using LangChain and AWS Bedrock.
    Converts natural language questions into SQL queries and executes them.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        engine: Optional[Any] = None
    ):
        """
        Initialize the NL2SQL system.
        
        Args:
            database_url: Database connection URL
            engine: SQLAlchemy engine (if provided, database_url is ignored)
        """
        self.logger = logging.getLogger("nl2sql_system")
        
        # Initialize database connection
        if engine:
            self.engine = engine
        elif database_url:
            from sqlalchemy import create_engine
            self.engine = create_engine(database_url)
        else:
            # Use default PostgreSQL connection from config
            database_url = self._build_database_url()
            from sqlalchemy import create_engine
            self.engine = create_engine(database_url)
        
        # Create SQLDatabase wrapper for LangChain integration
        self.db = SQLDatabase(self.engine)
        
        # Initialize Claude 3.5 Sonnet model via AWS Bedrock
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=config.anthropic.temperature,
            max_tokens=config.anthropic.max_tokens
        )
        
        # Create tool for executing SQL queries on the database
        self.sql_tool = QuerySQLDataBaseTool(db=self.db)
        
        # Initialize schema information
        self._schema_info = None
        self._table_info = None
        
        self.logger.info("NL2SQL system initialized successfully")
    
    def _build_database_url(self) -> str:
        """Build database URL from configuration."""
        return f"postgresql://{config.database.username}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.database}"
    
    def get_database_schema(self) -> str:
        """
        Get the database schema information for LLM context.
        
        Returns:
            Database schema as string
        """
        try:
            if self._schema_info is None:
                self._schema_info = self.db.get_table_info()
            return self._schema_info
        except Exception as e:
            self.logger.error(f"Error getting database schema: {str(e)}")
            return "Database schema not available"
    
    def get_table_names(self) -> List[str]:
        """
        Get list of table names in the database.
        
        Returns:
            List of table names
        """
        try:
            return self.db.get_usable_table_names()
        except Exception as e:
            self.logger.error(f"Error getting table names: {str(e)}")
            return []
    
    def clean_sql_query(self, markdown_query: str) -> str:
        """
        Clean LLM-generated SQL by removing markdown formatting and extra characters.
        
        Args:
            markdown_query: Raw SQL query response from LLM (may contain ```sql blocks)
            
        Returns:
            Clean SQL query string ready for execution
        """
        try:
            # Split query into individual lines for processing
            lines = markdown_query.strip().split('\n')
            cleaned_lines = []
            
            # Remove markdown code fence lines (```sql, ```, etc.)
            for line in lines:
                if line.strip().startswith('```') or line.strip().lower() == 'sql':
                    continue  # Skip markdown formatting lines
                cleaned_lines.append(line)
            
            # Join cleaned lines and remove backticks
            cleaned_query = ' '.join(cleaned_lines).strip()
            cleaned_query = cleaned_query.replace('`', '')  # Remove any remaining backticks
            
            # Remove trailing semicolon if present
            if cleaned_query.endswith(';'):
                cleaned_query = cleaned_query[:-1]
            
            return cleaned_query.strip()
            
        except Exception as e:
            self.logger.error(f"Error cleaning SQL query: {str(e)}")
            return markdown_query
    
    def validate_sql_query(self, query: str) -> bool:
        """
        Basic validation of SQL query.
        
        Args:
            query: SQL query to validate
            
        Returns:
            True if query appears valid, False otherwise
        """
        try:
            # Basic checks
            query_lower = query.lower().strip()
            
            # Must start with SELECT, INSERT, UPDATE, DELETE
            if not query_lower.startswith(('select', 'insert', 'update', 'delete')):
                return False
            
            # Check for dangerous operations
            dangerous_keywords = ['drop', 'truncate', 'alter', 'create', 'grant', 'revoke']
            for keyword in dangerous_keywords:
                if keyword in query_lower:
                    self.logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating SQL query: {str(e)}")
            return False
    
    def sql_response_generator(self, question: str) -> str:
        """
        Main method: Convert natural language question to SQL and execute it.
        
        Args:
            question: User's natural language query
            
        Returns:
            Database query results as formatted string
        """
        try:
            self.logger.info(f"Processing NL2SQL query: {question}")
            
            # Get database schema information for context
            schema = self.get_database_schema()
            
            # Create prompt template with system instructions and schema context
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 """You are a PostgreSQL expert. Given an input question, create a syntactically correct SQL query.
                 
                 Use the exact table names and column names from the schema below:
                 
                 {table_info}
                 
                 IMPORTANT RULES:
                 - Only use SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
                 - Use proper PostgreSQL syntax
                 - Include appropriate JOINs when needed
                 - Use LIMIT to avoid returning too many results (max 100 rows)
                 - Do not add explanations or semicolons
                 - Only return the SQL query
                 
                 If the question cannot be answered with a SELECT query, return "QUERY_NOT_SUPPORTED"
                 """),
                ("human", "{input}")  # User's natural language question
            ])
            
            # Create processing chain: prompt → LLM → string output
            sql_chain = sql_prompt | self.llm | StrOutputParser()
            
            # Generate SQL query by invoking the chain with question and schema
            raw_query = sql_chain.invoke({
                "input": question,
                "table_info": schema
            })
            
            self.logger.debug(f"Raw query from LLM: {raw_query}")
            
            # Check if query is not supported
            if "QUERY_NOT_SUPPORTED" in raw_query.upper():
                return "I can only answer questions that can be resolved with SELECT queries. Please rephrase your question to ask for data retrieval."
            
            # Clean the generated query (remove markdown formatting)
            cleaned_query = self.clean_sql_query(raw_query)
            
            self.logger.debug(f"Cleaned query: {cleaned_query}")
            
            # Validate the query
            if not self.validate_sql_query(cleaned_query):
                return "I cannot generate a valid SQL query for your question. Please try rephrasing it."
            
            # Execute the cleaned SQL query on the database
            result = self.sql_tool.run(cleaned_query)
            
            self.logger.debug(f"Query result: {result}")
            
            # Format the result for better readability
            formatted_result = self._format_query_result(result, question)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error in SQL response generation: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    async def sql_response_generator_async(self, question: str) -> str:
        """
        Async version of sql_response_generator.
        
        Args:
            question: User's natural language query
            
        Returns:
            Database query results as formatted string
        """
        try:
            # Run the synchronous method in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.sql_response_generator, question)
            return response
        except Exception as e:
            self.logger.error(f"Error in async SQL response generation: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def _format_query_result(self, result: Any, question: str) -> str:
        """
        Format query result for better readability.
        
        Args:
            result: Raw query result
            question: Original question
            
        Returns:
            Formatted result string
        """
        try:
            if not result:
                return "No results found for your query."
            
            # Handle different result types
            if isinstance(result, str):
                return result
            
            if isinstance(result, list):
                if not result:
                    return "No results found for your query."
                
                # Format list of tuples/rows
                if isinstance(result[0], (tuple, list)):
                    # Multiple columns
                    formatted_rows = []
                    for i, row in enumerate(result[:20]):  # Limit to 20 rows
                        if len(row) == 1:
                            formatted_rows.append(f"{i+1}. {row[0]}")
                        else:
                            formatted_rows.append(f"{i+1}. {' | '.join(str(cell) for cell in row)}")
                    
                    result_text = "\n".join(formatted_rows)
                    
                    if len(result) > 20:
                        result_text += f"\n... and {len(result) - 20} more results"
                    
                    return f"Found {len(result)} result(s):\n{result_text}"
                else:
                    # Single column
                    formatted_items = [f"{i+1}. {item}" for i, item in enumerate(result[:20])]
                    result_text = "\n".join(formatted_items)
                    
                    if len(result) > 20:
                        result_text += f"\n... and {len(result) - 20} more results"
                    
                    return f"Found {len(result)} result(s):\n{result_text}"
            
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Error formatting query result: {str(e)}")
            return str(result)
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.
        
        Returns:
            Dictionary with database information
        """
        try:
            table_names = self.get_table_names()
            schema = self.get_database_schema()
            
            return {
                "database_type": "PostgreSQL",
                "table_count": len(table_names),
                "table_names": table_names,
                "schema_available": bool(schema),
                "connection_status": "connected"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            return {
                "database_type": "PostgreSQL",
                "table_count": 0,
                "table_names": [],
                "schema_available": False,
                "connection_status": "error",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to execute a simple query
            result = self.db.run("SELECT 1 as test")
            return result is not None
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            return False


# Global NL2SQL system instance
_nl2sql_system_instance = None


def get_nl2sql_system(database_url: Optional[str] = None, engine: Optional[Any] = None) -> NL2SQLSystem:
    """
    Get or create the global NL2SQL system instance.
    
    Args:
        database_url: Database connection URL
        engine: SQLAlchemy engine
        
    Returns:
        NL2SQLSystem instance
    """
    global _nl2sql_system_instance
    
    if _nl2sql_system_instance is None:
        _nl2sql_system_instance = NL2SQLSystem(database_url=database_url, engine=engine)
    
    return _nl2sql_system_instance


# LangChain tool for NL2SQL functionality
@tool
def sql_tool(query: str) -> str:
    """
    Tool for handling database queries using NL2SQL system.
    Converts natural language questions into SQL queries and executes them.
    
    Args:
        query: The natural language question about the database
        
    Returns:
        Formatted query results
    """
    try:
        nl2sql_system = get_nl2sql_system()
        return nl2sql_system.sql_response_generator(query)
    except Exception as e:
        return f"Error processing database query: {str(e)}"


# Export the tool for use in agents
nl2sql_tool = sql_tool
