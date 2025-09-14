"""
Database client for PostgreSQL and other database operations.
Handles database connections, queries, and data management.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
from datetime import datetime
import json

from core.config import config


class DatabaseClient:
    """
    Database client for PostgreSQL and other database operations.
    Handles connections, queries, and data management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("database_client")
        self.host = config.database.host
        self.port = config.database.port
        self.database = config.database.database
        self.username = config.database.username
        self.password = config.database.password
        self.pool_size = config.database.connection_pool_size
        
        # Connection pool
        self._pool = None
        self._connected = False
        self._last_health_check = None
    
    async def initialize(self) -> bool:
        """
        Initialize database connection pool.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize connection pool
            await self._initialize_pool()
            
            # Test connectivity
            if await self._test_connectivity():
                self._connected = True
                self._last_health_check = datetime.now()
                self.logger.info("Database client initialized successfully")
                return True
            else:
                self.logger.error("Failed to connect to database")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing database client: {str(e)}")
            return False
    
    async def _initialize_pool(self):
        """Initialize database connection pool."""
        try:
            import asyncpg
            
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=30
            )
            
            self.logger.info(f"Database pool initialized with {self.pool_size} connections")
            
        except ImportError:
            self.logger.error("asyncpg not installed. Install with: pip install asyncpg")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing database pool: {str(e)}")
            raise
    
    async def _test_connectivity(self) -> bool:
        """Test database connectivity."""
        try:
            if not self._pool:
                return False
            
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    self.logger.info("Database connectivity test successful")
                    return True
                else:
                    self.logger.error("Database connectivity test failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error testing database connectivity: {str(e)}")
            return False
    
    async def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            List of result dictionaries
        """
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            async with self._pool.acquire() as conn:
                if params:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)
                
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    results.append(dict(row))
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    async def execute_command(
        self,
        command: str,
        params: Optional[List[Any]] = None
    ) -> str:
        """
        Execute an INSERT, UPDATE, or DELETE command.
        
        Args:
            command: SQL command
            params: Command parameters
        
        Returns:
            Command result message
        """
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            async with self._pool.acquire() as conn:
                if params:
                    result = await conn.execute(command, *params)
                else:
                    result = await conn.execute(command)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            raise
    
    async def execute_transaction(
        self,
        commands: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Execute multiple commands in a transaction.
        
        Args:
            commands: List of command dictionaries with 'query' and 'params' keys
        
        Returns:
            List of command results
        """
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    results = []
                    for cmd in commands:
                        query = cmd.get('query', '')
                        params = cmd.get('params', [])
                        
                        if params:
                            result = await conn.execute(query, *params)
                        else:
                            result = await conn.execute(query)
                        
                        results.append(result)
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"Error executing transaction: {str(e)}")
            raise
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Table schema information
        """
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = $1
                ORDER BY ordinal_position
            """
            
            columns = await self.execute_query(query, [table_name])
            
            return {
                "table_name": table_name,
                "columns": columns
            }
            
        except Exception as e:
            self.logger.error(f"Error getting table schema: {str(e)}")
            return {"table_name": table_name, "columns": []}
    
    async def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            
            results = await self.execute_query(query)
            return [row['table_name'] for row in results]
            
        except Exception as e:
            self.logger.error(f"Error listing tables: {str(e)}")
            return []
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            if not self._connected or not self._pool:
                raise Exception("Database client not initialized")
            
            # Get database size
            size_query = "SELECT pg_size_pretty(pg_database_size($1)) as size"
            size_result = await self.execute_query(size_query, [self.database])
            
            # Get table count
            table_count_query = """
                SELECT COUNT(*) as table_count 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            table_count_result = await self.execute_query(table_count_query)
            
            # Get connection count
            conn_count_query = """
                SELECT COUNT(*) as connection_count 
                FROM pg_stat_activity 
                WHERE datname = $1
            """
            conn_count_result = await self.execute_query(conn_count_query, [self.database])
            
            return {
                "database_name": self.database,
                "host": self.host,
                "port": self.port,
                "size": size_result[0]['size'] if size_result else "Unknown",
                "table_count": table_count_result[0]['table_count'] if table_count_result else 0,
                "connection_count": conn_count_result[0]['connection_count'] if conn_count_result else 0,
                "pool_size": self.pool_size
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection."""
        try:
            if not self._connected:
                return {
                    "status": "disconnected",
                    "last_check": self._last_health_check.isoformat() if self._last_health_check else None,
                    "error": "Not connected to database"
                }
            
            # Test connection
            start_time = datetime.now()
            result = await self.execute_query("SELECT 1 as test")
            response_time = (datetime.now() - start_time).total_seconds()
            
            if result and result[0]['test'] == 1:
                return {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                    "response_time": response_time,
                    "database": self.database,
                    "host": self.host
                }
            else:
                return {
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "response_time": response_time,
                    "error": "Health check query failed"
                }
                
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def close(self):
        """Close database connection pool."""
        try:
            if self._pool:
                await self._pool.close()
                self._pool = None
            
            self._connected = False
            self.logger.info("Database client closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database client: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if database client is connected."""
        return self._connected
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "pool_size": self.pool_size,
            "connected": self._connected
        }
