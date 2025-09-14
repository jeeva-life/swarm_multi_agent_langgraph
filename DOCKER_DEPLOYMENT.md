# Docker Deployment Guide

This guide explains how to deploy the Multi-Agent Swarm system using Docker and Docker Compose.

## 🐳 **Docker Architecture**

The system consists of 5 services:

1. **swarm-app**: Main Multi-Agent Swarm application (FastAPI)
2. **postgres**: PostgreSQL database
3. **redis**: Redis for caching and memory management
4. **pgadmin**: PostgreSQL management UI (optional)
5. **redis-commander**: Redis management UI (optional)

## 📋 **Prerequisites**

- Docker Desktop installed
- Docker Compose v2.0+
- Anthropic API key

## 🚀 **Quick Start**

### 1. **Set Environment Variables**

Create a `.env` file in the project root:

```bash
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Override default settings
ANTHROPIC_MODEL=claude-3-haiku-20240307
ENVIRONMENT=production
DEBUG=false
```

### 2. **Start All Services**

```bash
# Start all services (including the app)
docker-compose up -d

# View logs
docker-compose logs -f swarm-app

# Check service status
docker-compose ps
```

### 3. **Access the Application**

- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **pgAdmin**: http://localhost:8080 (admin@swarm.local / admin_password)
- **Redis Commander**: http://localhost:8081

## 🔧 **Development vs Production**

### **Development Mode**
```bash
# Start only infrastructure services
docker-compose up -d postgres redis

# Run app locally for development
uv run python src/api_server.py
```

### **Production Mode**
```bash
# Start all services including the app
docker-compose up -d

# Scale the app (if needed)
docker-compose up -d --scale swarm-app=3
```

## 📊 **Service Management**

### **View Logs**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f swarm-app
docker-compose logs -f postgres
docker-compose logs -f redis
```

### **Restart Services**
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart swarm-app
```

### **Stop Services**
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## 🏗️ **Building the Application**

### **Build Only**
```bash
# Build the app image
docker build -t swarm-multi-agent .

# Run the built image
docker run -p 8000:8000 --env-file .env swarm-multi-agent
```

### **Build and Start**
```bash
# Build and start all services
docker-compose up --build -d
```

## 🔍 **Health Monitoring**

### **Application Health**
```bash
# Check app health
curl http://localhost:8000/health

# Check system status
curl http://localhost:8000/status
```

### **Service Health**
```bash
# Check all services
docker-compose ps

# Check specific service health
docker inspect swarm_app --format='{{.State.Health.Status}}'
```

## 🗄️ **Data Persistence**

### **Volumes**
- **PostgreSQL data**: `postgres_data` volume
- **Redis data**: `redis_data` volume
- **pgAdmin data**: `pgadmin_data` volume
- **Application logs**: `./logs` directory

### **Backup Database**
```bash
# Create backup
docker-compose exec postgres pg_dump -U swarm_user swarm_db > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U swarm_user swarm_db < backup.sql
```

## 🔧 **Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Required | Anthropic API key |
| `ANTHROPIC_MODEL` | claude-3-haiku-20240307 | Claude model to use |
| `DB_HOST` | postgres | Database host |
| `DB_PORT` | 5432 | Database port |
| `DB_NAME` | swarm_db | Database name |
| `DB_USER` | swarm_user | Database user |
| `DB_PASSWORD` | swarm_password | Database password |
| `REDIS_HOST` | redis | Redis host |
| `REDIS_PORT` | 6379 | Redis port |
| `ENVIRONMENT` | production | Environment mode |
| `DEBUG` | false | Debug mode |

### **Port Configuration**

| Service | Port | Description |
|---------|------|-------------|
| swarm-app | 8000 | Main API server |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| pgadmin | 8080 | PostgreSQL admin UI |
| redis-commander | 8081 | Redis admin UI |

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8000
   
   # Kill the process
   sudo kill -9 <PID>
   ```

2. **Database Connection Failed**
   ```bash
   # Check if postgres is running
   docker-compose ps postgres
   
   # Check postgres logs
   docker-compose logs postgres
   ```

3. **Redis Connection Failed**
   ```bash
   # Check if redis is running
   docker-compose ps redis
   
   # Test redis connection
   docker-compose exec redis redis-cli ping
   ```

4. **API Key Issues**
   ```bash
   # Check if API key is set
   docker-compose exec swarm-app env | grep ANTHROPIC
   
   # Update .env file and restart
   docker-compose restart swarm-app
   ```

### **Debug Mode**

```bash
# Run in debug mode
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up

# Or set DEBUG=true in .env
echo "DEBUG=true" >> .env
docker-compose restart swarm-app
```

## 📈 **Scaling**

### **Horizontal Scaling**
```bash
# Scale the application
docker-compose up -d --scale swarm-app=3

# Use a load balancer (nginx, traefik, etc.)
```

### **Resource Limits**
```yaml
# Add to docker-compose.yml
services:
  swarm-app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## 🔒 **Security**

### **Production Security**
1. Change default passwords
2. Use secrets for sensitive data
3. Enable SSL/TLS
4. Use a reverse proxy (nginx)
5. Implement authentication

### **Secrets Management**
```bash
# Create secrets
echo "your_api_key" | docker secret create anthropic_api_key -

# Use in docker-compose.yml
services:
  swarm-app:
    secrets:
      - anthropic_api_key
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key
```

## 📝 **API Testing**

### **Test Endpoints**
```bash
# Health check
curl http://localhost:8000/health

# General query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, can you help me?", "session_id": "test_1"}'

# RAG query
curl -X POST http://localhost:8000/query/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the company policies?", "session_id": "rag_1"}'
```

## 🎯 **Next Steps**

1. **Set up monitoring** (Prometheus, Grafana)
2. **Implement CI/CD** (GitHub Actions, GitLab CI)
3. **Add authentication** (JWT, OAuth)
4. **Set up logging** (ELK Stack, Fluentd)
5. **Configure backups** (automated database backups)
6. **Implement load balancing** (nginx, HAProxy)

---

**Happy Deploying! 🚀**
