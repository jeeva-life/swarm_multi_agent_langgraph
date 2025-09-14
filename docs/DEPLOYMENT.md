# Deployment Guide

## Prerequisites

- Python 3.11+
- AWS Account with Bedrock access
- Redis server
- PostgreSQL database
- Git

## Environment Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd swarm_multi_agent_langgraph
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Configure Environment

Copy the environment template:
```bash
cp config/env.example .env
```

Edit `.env` with your configuration:
```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

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
```

## Infrastructure Setup

### Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

### PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb chinook

# Create user
sudo -u postgres createuser --interactive
```

### AWS Bedrock Setup

1. Enable Bedrock in AWS Console
2. Request access to Claude 3.5 Sonnet
3. Configure IAM permissions
4. Test connection

## Running the Application

### Development

```bash
python main.py
```

### Production

```bash
# Using gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.app:app

# Using uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## Monitoring

### Health Checks

```bash
# Check Redis
redis-cli ping

# Check PostgreSQL
psql -U postgres -d chinook -c "SELECT 1;"

# Check AWS Bedrock
aws bedrock list-foundation-models --region us-east-1
```

### Logs

Logs are stored in the `logs/` directory:
- `application.log`: Application logs
- `error.log`: Error logs
- `access.log`: Access logs

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - DB_HOST=postgres
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: chinook
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

## Scaling

### Horizontal Scaling

- Multiple agent instances
- Load balancer configuration
- Redis clustering
- Database connection pooling

### Performance Tuning

- Adjust worker processes
- Configure memory limits
- Optimize database queries
- Cache frequently accessed data
