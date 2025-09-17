"""
FastAPI web server for the Multi-Agent Swarm system.
Provides HTTP endpoints for testing and integration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app import MultiAgentSwarmApp
from core.config.config import config

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Swarm API",
    description="API for the Multi-Agent Swarm system with RAG, NL2SQL, and Invoice processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application instance
swarm_app: Optional[MultiAgentSwarmApp] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    response: str
    agent: str
    session_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    running: bool
    components: Dict[str, str]
    timestamp: str

class SystemStatusResponse(BaseModel):
    status: str
    agents: List[str]
    tools: int
    rag_tools: int
    nl2sql_tools: int
    invoice_tools: int
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize the Multi-Agent Swarm system on startup."""
    global swarm_app
    
    # Reload config to ensure latest environment variables
    config.reload()
    
    # Initialize the application
    swarm_app = MultiAgentSwarmApp()
    
    if not await swarm_app.initialize():
        raise Exception("Failed to initialize Multi-Agent Swarm system")
    
    logging.info("Multi-Agent Swarm API server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global swarm_app
    if swarm_app:
        # Add any cleanup logic here if needed
        pass
    logging.info("Multi-Agent Swarm API server shutdown")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Agent Swarm API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not swarm_app:
        raise HTTPException(status_code=503, detail="Application not initialized")
    
    # Check component health
    components = {
        "database": "healthy" if swarm_app.database_client else "unhealthy",
        "memory": "healthy" if swarm_app.memory_manager else "unhealthy",
        "metrics": "healthy" if swarm_app.metrics_collector else "unhealthy",
        "agent_swarm": "healthy" if swarm_app.agent_swarm else "unhealthy",
        "langchain_client": "healthy" if swarm_app.langchain_client else "unhealthy"
    }
    
    return HealthResponse(
        status="healthy" if swarm_app.initialized else "unhealthy",
        initialized=swarm_app.initialized,
        running=swarm_app.running,
        components=components,
        timestamp=datetime.now().isoformat()
    )

@app.get("/status", response_model=SystemStatusResponse)
async def system_status():
    """Get detailed system status."""
    if not swarm_app or not swarm_app.agent_swarm:
        raise HTTPException(status_code=503, detail="Agent swarm not available")
    
    # Get agent information
    agents = list(swarm_app.agent_swarm.agents.keys()) if swarm_app.agent_swarm.agents else []
    
    # Get tool counts (approximate)
    rag_tools = 3  # RAG system tools
    nl2sql_tools = 3  # NL2SQL system tools
    invoice_tools = 5  # Invoice system tools
    total_tools = rag_tools + nl2sql_tools + invoice_tools
    
    return SystemStatusResponse(
        status="running" if swarm_app.running else "stopped",
        agents=agents,
        tools=total_tools,
        rag_tools=rag_tools,
        nl2sql_tools=nl2sql_tools,
        invoice_tools=invoice_tools,
        timestamp=datetime.now().isoformat()
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the Multi-Agent Swarm system."""
    if not swarm_app or not swarm_app.agent_swarm:
        raise HTTPException(status_code=503, detail="Agent swarm not available")
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process the query
        result = await swarm_app.agent_swarm.process_query(
            query=request.query,
            session_id=session_id
        )
        
        # Extract response information
        if isinstance(result, dict):
            response_text = result.get('response', 'No response generated')
            agent_name = result.get('agent', 'Unknown')
            metadata = result.get('metadata', {})
        else:
            response_text = str(result) if result else 'No response generated'
            agent_name = 'Unknown'
            metadata = {}
        
        return QueryResponse(
            response=response_text,
            agent=agent_name,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/rag")
async def rag_query(request: QueryRequest):
    """Process a RAG-specific query."""
    if not swarm_app or not swarm_app.agent_swarm:
        raise HTTPException(status_code=503, detail="Agent swarm not available")
    
    try:
        # Add RAG context to the query
        rag_query = f"RAG Query: {request.query}"
        
        result = await swarm_app.agent_swarm.process_query(
            query=rag_query,
            session_id=request.session_id or f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return QueryResponse(
            response=str(result) if result else 'No response generated',
            agent="RAG Agent",
            session_id=request.session_id or f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            metadata={"query_type": "rag"}
        )
        
    except Exception as e:
        logging.error(f"Error processing RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")

@app.post("/query/nl2sql")
async def nl2sql_query(request: QueryRequest):
    """Process a NL2SQL-specific query."""
    if not swarm_app or not swarm_app.agent_swarm:
        raise HTTPException(status_code=503, detail="Agent swarm not available")
    
    try:
        # Add NL2SQL context to the query
        nl2sql_query = f"NL2SQL Query: {request.query}"
        
        result = await swarm_app.agent_swarm.process_query(
            query=nl2sql_query,
            session_id=request.session_id or f"nl2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return QueryResponse(
            response=str(result) if result else 'No response generated',
            agent="NL2SQL Agent",
            session_id=request.session_id or f"nl2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            metadata={"query_type": "nl2sql"}
        )
        
    except Exception as e:
        logging.error(f"Error processing NL2SQL query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing NL2SQL query: {str(e)}")

@app.post("/query/invoice")
async def invoice_query(request: QueryRequest):
    """Process an Invoice-specific query."""
    if not swarm_app or not swarm_app.agent_swarm:
        raise HTTPException(status_code=503, detail="Agent swarm not available")
    
    try:
        # Add Invoice context to the query
        invoice_query = f"Invoice Query: {request.query}"
        
        result = await swarm_app.agent_swarm.process_query(
            query=invoice_query,
            session_id=request.session_id or f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return QueryResponse(
            response=str(result) if result else 'No response generated',
            agent="Invoice Agent",
            session_id=request.session_id or f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            metadata={"query_type": "invoice"}
        )
        
    except Exception as e:
        logging.error(f"Error processing Invoice query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing Invoice query: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
