# Multi-Agent Swarm System with LangGraph and AWS Bedrock

A production-ready multi-agent system that creates dynamic agent handoffs for complex query processing. This system features three specialized systems (RAG, NL2SQL, and Invoice Processing) with intelligent routing, comprehensive monitoring, and drift detection.

## Features

### Core Capabilities
- **LangGraph Multi-Agent Coordination**: State-based workflow management with automatic handoffs
- **Three Specialized Systems**: RAG, NL2SQL, and Invoice Processing systems
- **Dynamic Agent Handoffs**: Intelligent query routing powered by LangGraph
- **Short & Long-term Memory**: Persistent memory management with Redis support
- **Production Monitoring**: Real-time metrics, alerting, and drift detection
- **AWS Bedrock Integration**: Cloud-native LLM capabilities with LangChain
- **Database Support**: PostgreSQL with Chinook sample data

### Specialized Systems
1. **RAG System**: Document retrieval and knowledge-based Q&A
2. **NL2SQL System**: Natural language to SQL conversion
3. **Invoice System**: Financial data and billing operations

### Monitoring & Observability
- **Real-time Metrics**: Agent performance, response times, success rates
- **Drift Detection**: Behavioral change detection and alerting
- **Alert Management**: Multi-level alerting system with suppression
- **Dashboard**: Real-time visualization and monitoring
- **Health Checks**: System and component health monitoring

## Project Structure

```
swarm_multi_agent_langgraph/
├── main.py                    # Main application launcher
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # Project documentation
├── config/                   # Configuration files
│   └── env.example          # Environment variables template
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   └── DEPLOYMENT.md        # Deployment guide
├── examples/                 # Example scripts organized by category
│   ├── integration/         # Integration examples
│   │   ├── complete_system_test.py
│   │   └── handoff_swarm_example.py
│   └── systems/             # System examples
│       └── comprehensive_systems_demo.py
├── scripts/                  # Utility scripts organized by purpose
│   ├── database/            # Database scripts
│   ├── monitoring/          # Monitoring scripts
│   └── setup_complete.py    # Complete setup script
├── src/                      # Source code
│   ├── __init__.py
│   ├── app.py                # Main application entry point
│   ├── agents/               # Swarm coordination
│   │   ├── __init__.py       # Simplified exports
│   │   └── handoff_swarm.py  # Primary swarm implementation
│   ├── core/                 # Core system components
│   │   ├── __init__.py       # Core module exports
│   │   ├── agent_tools.py    # LangChain tools for agents
│   │   ├── config.py         # Configuration management
│   │   ├── invoice_system.py # Invoice processing system
│   │   ├── nl2sql_system.py  # NL2SQL system
│   │   ├── rag_system.py     # RAG system
│   │   ├── state.py          # State management
│   │   ├── memory/           # Memory management
│   │   │   ├── __init__.py
│   │   │   ├── long_term_memory.py
│   │   │   ├── memory_backends.py
│   │   │   ├── memory_manager.py
│   │   │   └── short_term_memory.py
│   │   └── monitoring/       # Monitoring system
│   │       ├── __init__.py
│   │       ├── alert_manager.py
│   │       ├── dashboard.py
│   │       ├── drift_detector.py
│   │       └── metrics_collector.py
│   └── infrastructure/       # Infrastructure components
│       ├── __init__.py
│       ├── aws_client.py
│       ├── database_client.py
│       ├── langchain_bedrock_client.py
│       └── logging_config.py
├── tests/                    # Test suite
└── utils/                    # Utility tools
    ├── health_check.py      # Health check utility
    └── run_examples.py      # Example runner
```

## Prerequisites

- Python 3.12+
- AWS Account with Bedrock access
- Docker Desktop (for PostgreSQL and Redis)
- Git

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd swarm_multi_agent_langgraph
```

### 2. Start Docker Services

```bash
# Start PostgreSQL and Redis with Docker
# Windows
scripts\docker_start.bat

# Linux/Mac
./scripts/docker_start.sh

# Or manually
docker-compose up -d
```

### 3. Install Dependencies

```bash
# Install with uv (recommended)
uv add -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 4. Set up AWS Credentials

The system requires AWS Bedrock access for LLM functionality. See [AWS_SETUP.md](AWS_SETUP.md) for detailed instructions.

**Quick Setup:**
1. Create an AWS IAM user with Bedrock permissions
2. Set environment variables:

```bash
# Windows (PowerShell)
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_REGION="us-east-1"

# Linux/Mac
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_REGION="us-east-1"
```

**Or create a `.env` file:**
```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.0

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chinook
DB_USER=your_username
DB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Memory Configuration
MEMORY_BACKEND=redis
SHORT_TERM_TTL=3600
LONG_TERM_TTL=86400

# Monitoring Configuration
METRICS_ENABLED=true
ALERT_ENABLED=true
DRIFT_DETECTION_ENABLED=true
```

### 4. Set up Redis for production

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

### 5. Set up PostgreSQL Database

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb chinook

# Import Chinook sample data (if available)
# psql -U postgres -d chinook -f chinook_data.sql
```

## Quick Start

### Run the Multi-Agent Swarm

```bash
python main.py
```

### Run Individual Examples

```bash
# Run all examples
python utils/run_examples.py

# Test Handoff-based Multi-Agent Swarm
python examples/integration/handoff_swarm_example.py

# Test Complete System with Handoff Scenarios
python examples/integration/complete_system_test.py

# Run Comprehensive Systems Demo
python examples/systems/comprehensive_systems_demo.py
```

### Utility Scripts

```bash
# Check system health
python utils/health_check.py

# Complete setup (recommended)
python scripts/setup_complete.py

# Setup database only
python scripts/database/setup_database.py

# Monitor memory usage
python scripts/monitoring/memory_monitor.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_REGION` | AWS region | us-east-1 |
| `BEDROCK_MODEL_ID` | Bedrock model ID | anthropic.claude-3-5-sonnet-20241022-v2:0 |
| `BEDROCK_TEMPERATURE` | LLM temperature | 0.0 |
| `DB_HOST` | Database host | localhost |
| `DB_PORT` | Database port | 5432 |
| `DB_NAME` | Database name | chinook |
| `REDIS_HOST` | Redis host | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `MEMORY_BACKEND` | Memory backend | redis |

### System Configuration

Each system can be configured through the `Config` class:

```python
from src.core.config import config

# Access system settings
print(config.agents.rag_agent.max_retries)
print(config.agents.nl2sql_agent.timeout)
print(config.agents.invoice_agent.confidence_threshold)
```

## Tool-based Multi-Agent Architecture

### LangChain Tools Integration

The system uses LangChain's tool-based approach with `create_react_agent`:

- **LangChain Tools**: All agents use LangChain tools instead of custom classes
- **Tool Integration**: Seamless integration with LangGraph workflow
- **Agent Specialization**: Each agent has specialized tools for their domain
- **Memory Integration**: Tools have access to memory and context
- **LLM Integration**: All tools use the same LLM for consistency
- **Error Handling**: Robust error recovery and fallback mechanisms

### Handoff-based Agent Coordination

The system implements proper LangGraph swarm architecture with handoff tools:

- **create_handoff_tool**: Enables seamless agent handoffs
- **create_swarm**: Creates fully connected swarm workflow
- **Agent Handoffs**: Each agent can hand off to any other agent
- **Context Preservation**: Full conversation context maintained during handoffs
- **Intelligent Routing**: Agents decide when to hand off based on query content
- **Fully Connected**: All agents can communicate with each other

### State Management

The system uses a centralized state schema that travels through the conversation:

```python
class State(TypedDict):
    customer_id: str
    messages: Annotated[list[Union[HumanMessage, AIMessage]], add_messages]
    remaining_steps: RemainingSteps
    active_agent: Literal["rag_agent", "nl2sql_agent", "invoice_agent", "router"]
    context: Dict[str, Any]
    agent_data: Dict[str, Any]
    session_metadata: Dict[str, Any]
    error_info: NotRequired[Optional[Dict[str, Any]]]
    performance_metrics: Dict[str, Any]
```

### Workflow Flow

1. **Query Reception**: User query received by router agent
2. **Agent Selection**: Router determines appropriate specialized agent
3. **State Handoff**: State passed to selected agent with context
4. **Processing**: Agent processes query using specialized capabilities
5. **Response Generation**: Agent generates response with confidence score
6. **State Update**: State updated with results and metadata
7. **Handoff Decision**: System decides if another agent is needed
8. **Response Delivery**: Final response delivered to user

### Memory Management

- **Short-term Memory**: Redis-backed conversation history with automatic windowing
- **Long-term Memory**: Persistent knowledge storage with semantic search
- **Memory Budgeting**: Automatic cleanup based on memory usage
- **Context Preservation**: Seamless context transfer between agents

### State Schema

- **Centralized State**: Single source of truth for conversation state
- **Context Preservation**: Maintains context across agent handoffs
- **Seamless Agent Handoffs**: State enables smooth transitions between agents
- **Error Handling**: Built-in error tracking and recovery mechanisms
- **Performance Metrics**: Automatic tracking of agent performance and system health
- **Memory Integration**: State works seamlessly with Redis-backed memory

### RAG System

- **Multi-Format Support**: TXT, PDF, DOCX, HTML, MD, CSV, JSON document loading
- **Intelligent Chunking**: Recursive text splitting with configurable overlap
- **Vector Embeddings**: Sentence Transformers for semantic similarity
- **Persistent Storage**: Chroma vector database with disk persistence
- **Semantic Search**: Vector similarity search for accurate retrieval
- **LangChain Integration**: Tool-based access for agent coordination
- **Async Support**: Full async/await compatibility
- **Metadata Preservation**: Source tracking and document metadata

### NL2SQL System

- **Natural Language Processing**: Converts natural language to SQL queries
- **PostgreSQL Integration**: Production-ready database connectivity
- **SQL Validation**: Prevents dangerous operations (DROP, INSERT, etc.)
- **Query Cleaning**: Removes markdown formatting from LLM responses
- **Schema Awareness**: Uses database schema for accurate query generation
- **Result Formatting**: User-friendly presentation of query results
- **Security First**: Only SELECT queries allowed for safety
- **LangChain Integration**: Tool-based access for agent coordination

### Invoice System

- **LangChain Tools**: Tool-based architecture for invoice operations
- **Database Integration**: Works with PostgreSQL for invoice data
- **Customer Management**: Retrieve customer invoice information
- **Employee Lookup**: Find support representatives for invoices
- **Invoice Sorting**: Sort by date, unit price, and other criteria
- **Summary Reports**: Generate customer invoice summaries
- **Mock Data Fallback**: Provides sample data when database unavailable
- **Tool Integration**: Seamless integration with LangGraph workflow

## Production Considerations

### Scalability

- **Horizontal Scaling**: Stateless agents enable horizontal scaling
- **Load Balancing**: Multiple agent instances can handle concurrent requests
- **Database Connection Pooling**: Efficient database resource management
- **Redis Clustering**: Distributed memory management for high availability

### Security

- **API Key Management**: Secure AWS credentials management
- **Input Validation**: Comprehensive input sanitization and validation
- **SQL Injection Prevention**: Parameterized queries and validation
- **Rate Limiting**: Request throttling and abuse prevention

### Monitoring

- **Health Checks**: Regular system and component health monitoring
- **Performance Metrics**: Real-time performance tracking and alerting
- **Error Tracking**: Comprehensive error logging and analysis
- **Drift Detection**: Behavioral change detection and alerting

### Deployment

- **Docker Support**: Containerized deployment with Docker
- **Environment Management**: Separate configurations for dev/staging/prod
- **Logging**: Structured logging with configurable levels
- **Backup Strategies**: Regular database and memory backups

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Adding New Systems

1. Create a new system class (e.g., `new_system.py`) in `src/core/`
2. Implement system functionality with LangChain tools
3. Add tools to `agent_tools.py` for the new system
4. Update `handoff_swarm.py` to include new system tools
5. Add handoff tools for the new system
6. Add tests for the new system

## Troubleshooting

### Common Issues

1. **AWS Bedrock Access**: Ensure proper IAM permissions and region configuration
2. **Redis Connection**: Check Redis server status and connection parameters
3. **Database Connection**: Verify PostgreSQL credentials and database existence
4. **Memory Issues**: Monitor Redis memory usage and adjust TTL settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check Redis
redis-cli ping

# Check PostgreSQL
psql -U postgres -d chinook -c "SELECT 1;"

# Check AWS Bedrock
aws bedrock list-foundation-models --region us-east-1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

- [Docker Setup](DOCKER_SETUP.md) - Complete Docker setup guide
- [AWS Setup](AWS_SETUP.md) - AWS credentials and Bedrock configuration
- [Architecture Overview](docs/ARCHITECTURE.md) - System architecture details
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment guide

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the example scripts in the `examples/` directory