#  Docker Setup for Multi-Agent Swarm System

This guide explains how to use Docker to run PostgreSQL and Redis for the Multi-Agent Swarm system.

##  Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

##  Quick Start

### Option 1: Using Scripts (Recommended)

**Windows:**
```bash
# Start services
scripts\docker_start.bat

# Stop services
scripts\docker_stop.bat
```

**Linux/Mac:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Start services
./scripts/docker_start.sh

# Stop services
./scripts/docker_stop.sh
```

### Option 2: Using Docker Compose Directly

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

##  Services Included

### PostgreSQL Database
- **Container**: `swarm_postgres`
- **Port**: `5432`
- **Database**: `swarm_db`
- **Username**: `swarm_user`
- **Password**: `swarm_password`
- **Management UI**: pgAdmin at http://localhost:8080

### Redis Cache
- **Container**: `swarm_redis`
- **Port**: `6379`
- **Management UI**: Redis Commander at http://localhost:8081

##  Management UIs

### pgAdmin (PostgreSQL Management)
- **URL**: http://localhost:8080
- **Email**: admin@swarm.local
- **Password**: admin_password

**To connect to the database:**
1. Open pgAdmin
2. Right-click "Servers" → "Create" → "Server"
3. **Name**: Swarm Database
4. **Host**: postgres (or localhost)
5. **Port**: 5432
6. **Database**: swarm_db
7. **Username**: swarm_user
8. **Password**: swarm_password

### Redis Commander (Redis Management)
- **URL**: http://localhost:8081
- **Host**: redis (or localhost)
- **Port**: 6379

##  Database Schema

The PostgreSQL database includes the following tables:

- **conversations**: Stores conversation sessions
- **messages**: Stores individual messages
- **agent_interactions**: Tracks agent handoffs and actions
- **system_metrics**: Stores performance metrics
- **alerts**: Stores system alerts

##  Health Checks

All services include health checks:

```bash
# Check service status
docker-compose ps

# Check PostgreSQL health
docker-compose exec postgres pg_isready -U swarm_user -d swarm_db

# Check Redis health
docker-compose exec redis redis-cli ping
```

##  Useful Commands

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f redis

# Restart a specific service
docker-compose restart postgres

# Execute commands in containers
docker-compose exec postgres psql -U swarm_user -d swarm_db
docker-compose exec redis redis-cli

# Remove all data and start fresh
docker-compose down -v
docker-compose up -d
```

##  Troubleshooting

### Services Won't Start
1. Ensure Docker Desktop is running
2. Check if ports 5432, 6379, 8080, 8081 are available
3. Run `docker-compose logs` to see error messages

### Database Connection Issues
1. Wait for PostgreSQL to fully start (can take 30-60 seconds)
2. Check if the database is ready: `docker-compose exec postgres pg_isready -U swarm_user -d swarm_db`
3. Verify credentials in `src/core/config.py`

### Redis Connection Issues
1. Check if Redis is running: `docker-compose exec redis redis-cli ping`
2. Verify Redis configuration in your application

### Port Conflicts
If you have existing PostgreSQL or Redis running:
1. Stop the existing services
2. Or modify the ports in `docker-compose.yml`

##  Security Notes

 **Important**: The default credentials are for development only!

For production:
1. Change all passwords in `docker-compose.yml`
2. Update credentials in `src/core/config.py`
3. Use environment variables for sensitive data
4. Enable SSL/TLS for database connections

##  File Structure

```
├── docker-compose.yml          # Docker services configuration
├── scripts/
│   ├── init_db.sql            # Database initialization script
│   ├── docker_start.sh        # Start services (Linux/Mac)
│   ├── docker_start.bat       # Start services (Windows)
│   ├── docker_stop.sh         # Stop services (Linux/Mac)
│   └── docker_stop.bat        # Stop services (Windows)
└── DOCKER_SETUP.md            # This file
```

##  Next Steps

After starting the Docker services:

1. **Test the health check**:
   ```bash
   python utils/health_check.py
   ```

2. **Run examples**:
   ```bash
   python utils/run_examples.py
   ```

3. **Start the application**:
   ```bash
   python -m src.app
   ```

##  Tips

- Use `docker-compose up -d` to run services in the background
- Use `docker-compose logs -f` to follow logs in real-time
- Use `docker-compose down -v` to completely reset all data
- The database data persists between container restarts
- Redis data persists between container restarts
