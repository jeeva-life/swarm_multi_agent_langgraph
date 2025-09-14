# Multi-Agent Swarm Application Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy dependency files
COPY requirements.txt requirements-dev.txt requirements-minimal.txt pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY docs/ ./docs/
COPY examples/ ./examples/
COPY utils/ ./utils/
COPY scripts/ ./scripts/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "src/api_server.py"]
