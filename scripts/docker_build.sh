#!/bin/bash

# Docker Build and Test Script for Multi-Agent Swarm

set -e

echo " Building Multi-Agent Swarm Docker Image..."
echo "=============================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo " .env file not found. Creating template..."
    cat > .env << EOF
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Override default settings
ANTHROPIC_MODEL=claude-3-haiku-20240307
ENVIRONMENT=production
DEBUG=false
EOF
    echo " Please edit .env file with your actual API key"
    echo "   Then run this script again"
    exit 1
fi

# Check if API key is set
if grep -q "your_anthropic_api_key_here" .env; then
    echo " Please set your actual Anthropic API key in .env file"
    exit 1
fi

echo " .env file found and configured"

# Build the Docker image
echo " Building Docker image..."
docker build -t swarm-multi-agent:latest .

echo " Docker image built successfully!"

# Test the image
echo " Testing Docker image..."
docker run --rm --name swarm-test \
    --env-file .env \
    -p 8000:8000 \
    swarm-multi-agent:latest &
    
# Wait for the container to start
echo " Waiting for container to start..."
sleep 30

# Test health endpoint
echo "🔍 Testing health endpoint..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo " Health check passed!"
else
    echo " Health check failed!"
    docker logs swarm-test
    docker stop swarm-test
    exit 1
fi

# Test query endpoint
echo " Testing query endpoint..."
response=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Hello, can you help me?", "session_id": "test_1"}')

if echo "$response" | grep -q "response"; then
    echo " Query endpoint working!"
    echo " Response: $response"
else
    echo " Query endpoint failed!"
    echo " Response: $response"
fi

# Cleanup
echo " Cleaning up test container..."
docker stop swarm-test

echo ""
echo " Docker build and test completed successfully!"
echo ""
echo " To start the full stack:"
echo "   docker-compose up -d"
echo ""
echo " Access points:"
echo "   - API Server: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - pgAdmin: http://localhost:8080"
echo "   - Redis Commander: http://localhost:8081"
