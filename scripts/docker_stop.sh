#!/bin/bash
# Stop Docker services for the Multi-Agent Swarm system

echo " Stopping Multi-Agent Swarm Docker Services..."
echo "==============================================="

# Stop the services
docker-compose down

echo " Services stopped successfully!"
echo ""
echo " To remove all data (volumes), run:"
echo "   docker-compose down -v"
