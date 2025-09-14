#!/bin/bash
# Restart Docker services for the Multi-Agent Swarm system

echo " Restarting Multi-Agent Swarm Docker Services..."
echo "================================================="

# Stop services first
echo " Stopping services..."
docker-compose down

# Start services
echo " Starting services..."
docker-compose up -d

# Wait for services to be ready
echo " Waiting for services to be ready..."
sleep 10

echo " Services restarted successfully!"
echo ""
echo " Management UIs:"
echo "   • pgAdmin (PostgreSQL): http://localhost:8080"
echo "   • Redis Commander: http://localhost:8081"
