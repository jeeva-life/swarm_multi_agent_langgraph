#!/usr/bin/env python3
"""
Complete setup script for the Multi-Agent Swarm system.
Handles all setup tasks: dependencies, environment, database, and Redis.
"""

import subprocess
import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import config
from src.core.memory.memory_backends import RedisBackend
from src.infrastructure.database_client import DatabaseClient


class CompleteSetup:
    """Complete setup for the Multi-Agent Swarm system."""
    
    def __init__(self):
        self.logger = logging.getLogger("complete_setup")
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the setup process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and handle errors."""
        print(f"[SETUP] {description}...")
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"[SUCCESS] {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {description} failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("[CHECK] Checking Prerequisites")
        print("=" * 40)
        
        # Check Python version
        if sys.version_info < (3, 11):
            print("[ERROR] Python 3.11+ is required")
            return False
        print(f"[SUCCESS] Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Check if we're in the right directory
        if not Path("pyproject.toml").exists():
            print("[ERROR] Please run this script from the project root directory")
            return False
        print("[SUCCESS] Project root directory detected")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install all project dependencies."""
        print("\n[INSTALL] Installing Dependencies")
        print("=" * 40)
        
        # Install project in development mode
        if not self.run_command("pip install -e .", "Installing project in development mode"):
            print("[ERROR] Failed to install project dependencies")
            return False
        
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment configuration."""
        print("\n[CONFIG] Setting up Environment Configuration")
        print("=" * 40)
        
        # Create .env file if it doesn't exist
        if not Path(".env").exists():
            if Path("config/env.example").exists():
                self.run_command("cp config/env.example .env", "Creating .env file from template")
                print("[INFO] Please edit .env file with your API keys:")
                print("   - AWS_ACCESS_KEY_ID")
                print("   - AWS_SECRET_ACCESS_KEY")
                print("   - LANGCHAIN_API_KEY")
                print("   - HUGGINGFACE_API_KEY")
            else:
                print("[WARNING] config/env.example not found, please create .env manually")
                return False
        else:
            print("[SUCCESS] .env file already exists")
        
        return True
    
    async def test_redis_connection(self) -> bool:
        """Test Redis connection."""
        print("\n[TEST] Testing Redis Connection")
        print("=" * 40)
        
        try:
            redis_backend = RedisBackend()
            
            # Test basic operations
            test_key = "test:setup"
            test_value = {"test": "data", "timestamp": "2024-01-01T00:00:00"}
            
            # Test set operation
            success = await redis_backend.set(test_key, str(test_value), ttl=60)
            if not success:
                print("[ERROR] Failed to set test data in Redis")
                return False
            
            # Test get operation
            retrieved = await redis_backend.get(test_key)
            if not retrieved:
                print("[ERROR] Failed to retrieve test data from Redis")
                return False
            
            # Clean up test data
            await redis_backend.delete(test_key)
            
            print("[SUCCESS] Redis connection test successful")
            return True
            
        except Exception as e:
            print(f"[ERROR] Redis connection test failed: {str(e)}")
            print("[INFO] Make sure Redis is running: redis-server")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test database connection."""
        print("\n[TEST] Testing Database Connection")
        print("=" * 40)
        
        try:
            db_client = DatabaseClient()
            
            if not await db_client.initialize():
                print("[ERROR] Failed to connect to database")
                print("[INFO] Make sure PostgreSQL is running and configured")
                return False
            
            print("[SUCCESS] Database connection test successful")
            await db_client.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] Database connection test failed: {str(e)}")
            print("[INFO] Make sure PostgreSQL is running and configured")
            return False
    
    def test_imports(self) -> bool:
        """Test that all required modules can be imported."""
        print("\n[TEST] Testing Imports")
        print("=" * 40)
        
        test_imports = [
            "from langchain_aws import ChatBedrock",
            "from langchain_huggingface import HuggingFaceEmbeddings", 
            "from sentence_transformers import SentenceTransformer",
            "from src.app import MultiAgentSwarmApp"
        ]
        
        for import_stmt in test_imports:
            try:
                exec(import_stmt)
                print(f"[SUCCESS] {import_stmt}")
            except ImportError as e:
                print(f"[ERROR] {import_stmt} - {e}")
                return False
        
        return True
    
    def show_next_steps(self):
        """Show next steps after setup."""
        print("\n[COMPLETE] Setup Complete!")
        print("=" * 60)
        print("Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Start Redis: redis-server")
        print("3. Start PostgreSQL and create database")
        print("4. Run examples:")
        print("   - python examples/integration/handoff_swarm_example.py")
        print("   - python examples/systems/comprehensive_systems_demo.py")
        print("   - python main.py")
        print("\nFor detailed setup instructions, see README.md")
    
    async def run_setup(self):
        """Run the complete setup process."""
        print("[START] Multi-Agent Swarm Complete Setup")
        print("=" * 60)
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                sys.exit(1)
            
            # Install dependencies
            if not self.install_dependencies():
                sys.exit(1)
            
            # Setup environment
            if not self.setup_environment():
                sys.exit(1)
            
            # Test imports
            if not self.test_imports():
                print("[ERROR] Some imports failed. Please check your installation.")
                sys.exit(1)
            
            # Test Redis (optional)
            print("\n[TEST] Testing External Services (Optional)")
            print("=" * 40)
            redis_ok = await self.test_redis_connection()
            db_ok = await self.test_database_connection()
            
            if not redis_ok:
                print("[WARNING] Redis not available - memory features will use in-memory backend")
            if not db_ok:
                print("[WARNING] Database not available - NL2SQL features will be limited")
            
            # Show next steps
            self.show_next_steps()
            
        except Exception as e:
            print(f"[ERROR] Setup failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


async def main():
    """Main setup function."""
    setup = CompleteSetup()
    await setup.run_setup()


if __name__ == "__main__":
    asyncio.run(main())
