@echo off
REM Start Docker services for the Multi-Agent Swarm system

echo  Starting Multi-Agent Swarm Docker Services...
echo ================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Start the services
echo  Starting PostgreSQL, Redis, and management UIs...
docker-compose up -d

REM Wait for services to be healthy
echo  Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service health
echo  Checking service health...

REM Check PostgreSQL
docker-compose exec postgres pg_isready -U swarm_user -d swarm_db >nul 2>&1
if %errorlevel% equ 0 (
    echo  PostgreSQL is ready
) else (
    echo   PostgreSQL is starting up...
)

REM Check Redis
docker-compose exec redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo  Redis is ready
) else (
    echo  Redis is starting up...
)

echo.
echo  Services started successfully!
echo.
echo  Management UIs:
echo    • pgAdmin (PostgreSQL): http://localhost:8080
echo    • Redis Commander: http://localhost:8081
echo.
echo  Connection Details:
echo    • PostgreSQL: localhost:5432
echo    • Redis: localhost:6379
echo.
echo  Useful commands:
echo    • View logs: docker-compose logs -f
echo    • Stop services: docker-compose down
echo    • Restart services: docker-compose restart
echo.
pause
