#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Swarm system.
This is the primary launcher for the application.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app import main

if __name__ == "__main__":
    print("Multi-Agent Swarm System")
    print("=" * 50)
    print("Starting application...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        sys.exit(1)
