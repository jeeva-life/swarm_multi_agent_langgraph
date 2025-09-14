#!/bin/bash
# Start Docker services for the Multi-Agent Swarm system

echo " Starting Multi-Agent Swarm Docker Services..."
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo " Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Start the services
echo " Starting PostgreSQL, Redis, and management UIs..."
docker-compose up -d

# Wait for services to be healthy
echo " Waiting for services to be ready..."
sleep 10

# Check service health
echo " Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U swarm_user -d swarm_db > /dev/null 2>&1; then
    echo " PostgreSQL is ready"
else
    echo "  PostgreSQL is starting up..."
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo " Redis is ready"
else
    echo " Redis is starting up..."
fi

echo ""
echo " Services started successfully!"
echo ""
echo " Management UIs:"
echo "   • pgAdmin (PostgreSQL): http://localhost:8080"
echo "   • Redis Commander: http://localhost:8081"
echo ""
echo " Connection Details:"
echo "   • PostgreSQL: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo " Useful commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
