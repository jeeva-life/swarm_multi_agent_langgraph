#!/usr/bin/env python3
"""
Utility script to run all examples.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_example(example_path: str, description: str):
    """Run a single example."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {example_path}")
    print(f"{'='*60}")
    
    try:
        # Import and run the example
        import importlib.util
        spec = importlib.util.spec_from_file_location("example", example_path)
        example_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_module)
        
        if hasattr(example_module, 'main'):
            if asyncio.iscoroutinefunction(example_module.main):
                asyncio.run(example_module.main())
            else:
                example_module.main()
        else:
            print(f"No main() function found in {example_path}")
            
    except Exception as e:
        print(f"Error running {example_path}: {str(e)}")

def main():
    """Run all examples."""
    print("Multi-Agent Swarm Examples")
    print("=" * 60)
    
    examples = [
        ("examples/integration/handoff_swarm_example.py", "Handoff Swarm Example"),
        ("examples/integration/complete_system_test.py", "Complete System Test"),
        ("examples/systems/comprehensive_systems_demo.py", "Comprehensive Systems Demo"),
    ]
    
    for example_path, description in examples:
        if Path(example_path).exists():
            run_example(example_path, description)
        else:
            print(f"Example not found: {example_path}")
    
    print(f"\n{'='*60}")
    print("All examples completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
