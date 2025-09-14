@echo off
REM Docker Build and Test Script for Multi-Agent Swarm (Windows)

echo  Building Multi-Agent Swarm Docker Image...
echo ==============================================

REM Check if .env file exists
if not exist .env (
    echo  .env file not found. Creating template...
    (
        echo # Anthropic API Configuration
        echo ANTHROPIC_API_KEY=your_anthropic_api_key_here
        echo.
        echo # Optional: Override default settings
        echo ANTHROPIC_MODEL=claude-3-haiku-20240307
        echo ENVIRONMENT=production
        echo DEBUG=false
    ) > .env
    echo  Please edit .env file with your actual API key
    echo    Then run this script again
    pause
    exit /b 1
)

REM Check if API key is set
findstr /C:"your_anthropic_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo  Please set your actual Anthropic API key in .env file
    pause
    exit /b 1
)

echo  .env file found and configured

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t swarm-multi-agent:latest .

if %errorlevel% neq 0 (
    echo  Docker build failed!
    pause
    exit /b 1
)

echo  Docker image built successfully!

REM Test the image
echo  Testing Docker image...
docker run --rm --name swarm-test --env-file .env -p 8000:8000 swarm-multi-agent:latest &

REM Wait for the container to start
echo  Waiting for container to start...
timeout /t 30 /nobreak >nul

REM Test health endpoint
echo  Testing health endpoint...
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo  Health check passed!
) else (
    echo Health check failed!
    docker logs swarm-test
    docker stop swarm-test
    pause
    exit /b 1
)

REM Test query endpoint
echo  Testing query endpoint...
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"Hello, can you help me?\", \"session_id\": \"test_1\"}" > response.json
if exist response.json (
    echo  Query endpoint working!
    echo  Response:
    type response.json
    del response.json
) else (
    echo  Query endpoint failed!
)

REM Cleanup
echo  Cleaning up test container...
docker stop swarm-test

echo.
echo Docker build and test completed successfully!
echo.
echo To start the full stack:
echo    docker-compose up -d
echo.
echo Access points:
echo    - API Server: http://localhost:8000
echo    - API Docs: http://localhost:8000/docs
echo    - pgAdmin: http://localhost:8080
echo    - Redis Commander: http://localhost:8081

pause
