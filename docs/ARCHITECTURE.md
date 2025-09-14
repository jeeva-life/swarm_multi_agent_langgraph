# Multi-Agent Swarm Architecture

## Overview
This project implements a production-level multi-agent system using LangGraph, AWS Bedrock, and Python with three specialized agents: RAG, NL2SQL, and Invoice Processing.

## Architecture Components

### Core System (`src/`)

#### 1. Agent Implementations (`src/agents/`)
- **`handoff_swarm.py`** - Primary LangGraph implementation using `create_handoff_tool` and `create_swarm`
- **`langgraph_swarm.py`** - Alternative implementation using `StateGraph` for workflow management
- **`tool_based_swarm.py`** - Tool-based approach using `create_react_agent`
- **`rag_agent.py`** - RAG (Retrieval-Augmented Generation) agent
- **`nl2sql_agent.py`** - Natural Language to SQL conversion agent
- **`invoice_agent.py`** - Invoice processing and billing agent

#### 2. Core Components (`src/core/`)
- **`config.py`** - Configuration management
- **`state.py`** - State schema and management
- **`agent_tools.py`** - LangChain tools for agents
- **`memory/`** - Memory management (short-term and long-term)
- **`monitoring/`** - Metrics, alerts, and drift detection

#### 3. Infrastructure (`src/infrastructure/`)
- **`aws_client.py`** - AWS Bedrock integration
- **`langchain_bedrock_client.py`** - LangChain Bedrock client
- **`database_client.py`** - Database connectivity
- **`logging_config.py`** - Logging configuration

#### 4. Main Application (`src/app.py`)
- **`MultiAgentSwarmApp`** - Main application orchestrating all components
- Uses `HandoffSwarm` as the primary agent coordination mechanism

### Examples and Tests (`examples/`)

#### 1. Integration Examples (`examples/integration/`)
- **`handoff_swarm_example.py`** - Basic handoff swarm example
- **`complete_system_test.py`** - Comprehensive system testing
- **`simple_handoff_test.py`** - Simple handoff test cases
- **`tool_based_swarm_example.py`** - Tool-based swarm example
- **`langgraph_swarm_example.py`** - LangGraph workflow example

#### 2. System Examples (`examples/systems/`)
- **`rag_system_example.py`** - RAG system demonstration
- **`nl2sql_system_example.py`** - NL2SQL system demonstration
- **`invoice_system_example.py`** - Invoice system demonstration
- **`state_schema_example.py`** - State management example

## Key Features

### 1. Dynamic Agent Handoffs
- Uses `create_handoff_tool` for seamless agent transitions
- Intelligent routing based on query type
- Context preservation across handoffs

### 2. Memory Management
- **Short-term Memory**: Redis-backed with windowing and summarization
- **Long-term Memory**: Persistent knowledge storage
- **Semantic Search**: Hugging Face embeddings for context retrieval

### 3. Monitoring & Observability
- Real-time metrics collection
- Performance monitoring
- Drift detection and alerting
- System health dashboards

### 4. Production Features
- AWS Bedrock integration
- Database connectivity (PostgreSQL)
- Vector database (Chroma)
- Comprehensive error handling
- Logging and debugging

## Database Architecture

### Long-Term Memory
- **Primary**: Redis (configurable via `MEMORY_BACKEND=redis`)
- **Fallback**: In-memory storage for development
- **TTL**: 30 days default for long-term memory
- **Features**: JSON serialization, TTL support, pattern matching

### Short-Term Memory
- **Backend**: Same as long-term memory (Redis/in-memory)
- **TTL**: 1 hour default
- **Features**: Windowing, summarization, semantic search

### Structured Data
- **Database**: PostgreSQL
- **Purpose**: NL2SQL queries, invoice data, structured information
- **Connection Pool**: Configurable pool size

## Usage

### Quick Start
```bash
# Install dependencies
pip install -e .

# Run simple handoff test
python examples/integration/simple_handoff_test.py

# Run comprehensive system test
python examples/integration/complete_system_test.py
```

### Main Application
```python
from src.app import MultiAgentSwarmApp

app = MultiAgentSwarmApp()
await app.initialize()

response = await app.process_query(
    query="Do you have any albums by The Rolling Stones?",
    user_id="test_user",
    session_id="test_session"
)
```

## Architecture Benefits

1. **Modular Design**: Clear separation of concerns
2. **Production Ready**: Comprehensive error handling and monitoring
3. **Scalable**: Easy to add new agents and capabilities
4. **Maintainable**: Well-organized code structure
5. **Testable**: Extensive test coverage and examples

## File Organization

```
src/
├── agents/           # Agent implementations
├── core/            # Core system components
├── infrastructure/  # External service integrations
└── app.py          # Main application

examples/
├── integration/     # Integration examples and tests
└── systems/        # Individual system examples

utils/
└── run_examples.py # Example runner utility
```

The architecture is properly organized with core implementations in `src/` and examples/tests in `examples/`, following production-level software engineering practices.
