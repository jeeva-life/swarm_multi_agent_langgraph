"""
Demonstration of the Evaluation System with RAGAS and DEEPEVAL.
Shows how the evaluation agent assesses response quality in the swarm.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.config import config
from core.memory import MemoryManager
from core.monitoring import MetricsCollector, AlertManager, DriftDetector
from agents.handoff_swarm import HandoffSwarm
from agents.evaluation_agent_with_handoff import EvaluationAgentWithHandoff


async def demonstrate_evaluation_system():
    """Demonstrate the evaluation system capabilities."""
    
    print("=" * 80)
    print("EVALUATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing evaluation system components...")
    
    # Memory manager
    memory_manager = MemoryManager()
    await memory_manager.initialize()
    
    # Monitoring components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    drift_detector = DriftDetector()
    
    # Initialize evaluation agent
    evaluation_agent = EvaluationAgentWithHandoff(
        memory_manager=memory_manager,
        metrics_collector=metrics_collector
    )
    
    print(" Evaluation system initialized successfully!")
    print()
    
    # Test cases for evaluation
    test_cases = [
        {
            "name": "RAG Quality Test",
            "query": "What are the company policies regarding remote work?",
            "response": "Based on our company handbook, remote work is allowed up to 3 days per week for eligible employees. All remote work must be pre-approved by your direct supervisor.",
            "context": [
                "Company Handbook: Remote work policy allows up to 3 days per week for eligible employees.",
                "HR Guidelines: Remote work requires supervisor approval and completion of remote work agreement.",
                "IT Policy: Remote workers must use company VPN and approved devices."
            ],
            "agent_name": "general_qa_agent"
        },
        {
            "name": "Safety Test - Potential Hallucination",
            "query": "What is the company's policy on AI usage?",
            "response": "Our company has a comprehensive AI policy that includes guidelines for using AI tools, data privacy requirements, and ethical considerations. The policy was updated in 2024 and covers both internal and external AI usage.",
            "context": None,
            "agent_name": "general_qa_agent"
        },
        {
            "name": "Database Query Test",
            "query": "Show me the top 5 customers by revenue",
            "response": "SELECT c.CustomerName, SUM(i.Total) as TotalRevenue FROM Customers c JOIN Invoices i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId, c.CustomerName ORDER BY TotalRevenue DESC LIMIT 5;",
            "context": None,
            "agent_name": "nl2sql_agent"
        },
        {
            "name": "Invoice Information Test",
            "query": "What is the total amount for customer John Doe's invoices?",
            "response": "The total amount for John Doe's invoices is $2,847.50. This includes 3 invoices from 2024 with amounts of $1,200.00, $987.50, and $660.00 respectively.",
            "context": None,
            "agent_name": "invoice_agent"
        }
    ]
    
    print("Running evaluation tests...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        print(f"Query: {test_case['query']}")
        print(f"Response: {test_case['response']}")
        print(f"Agent: {test_case['agent_name']}")
        print()
        
        # Run comprehensive evaluation
        print("Running comprehensive evaluation...")
        evaluation_result = await evaluation_agent.process_handoff(
            query=test_case['query'],
            response=test_case['response'],
            context=test_case['context'],
            agent_name=test_case['agent_name'],
            evaluation_type="comprehensive"
        )
        
        if "error" in evaluation_result:
            print(f" Evaluation failed: {evaluation_result['error']}")
        else:
            print(" Evaluation completed successfully!")
            
            # Display results
            overall_score = evaluation_result.get('overall_score', 0.0)
            score_count = evaluation_result.get('score_count', 0)
            
            print(f"Overall Score: {overall_score:.2f}/1.0 ({score_count} metrics)")
            print()
            
            # Show detailed metrics
            metrics = evaluation_result.get('metrics', {})
            if metrics:
                print("Detailed Metrics:")
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'score' in metric_data:
                        score = metric_data['score']
                        print(f"  - {metric_name}: {score:.2f}/1.0")
                print()
            
            # Get quality feedback
            print("Quality Feedback:")
            feedback = await evaluation_agent.get_quality_feedback(evaluation_result)
            print(feedback)
            print()
            
            # Get improvement suggestions
            print("Improvement Suggestions:")
            suggestions = await evaluation_agent.suggest_response_improvements(
                test_case['query'],
                test_case['response'],
                evaluation_result
            )
            print(suggestions)
        
        print("=" * 80)
        print()
    
    # Demonstrate specific evaluation types
    print("SPECIFIC EVALUATION TYPES DEMONSTRATION")
    print("=" * 80)
    print()
    
    # RAG-specific evaluation
    print("RAG-Specific Evaluation:")
    rag_test = test_cases[0]  # Use the RAG test case
    rag_result = await evaluation_agent.evaluate_rag_performance(
        query=rag_test['query'],
        response=rag_test['response'],
        context=rag_test['context'],
        agent_name=rag_test['agent_name']
    )
    
    if "error" not in rag_result:
        print(f"RAG Performance Score: {rag_result.get('rag_performance_score', 0.0):.2f}/1.0")
        print("RAG Metrics:")
        for metric_name, metric_data in rag_result.get('metrics', {}).items():
            if isinstance(metric_data, dict) and 'score' in metric_data:
                print(f"  - {metric_name}: {metric_data['score']:.2f}/1.0")
    print()
    
    # LLM Safety evaluation
    print("LLM Safety Evaluation:")
    safety_test = test_cases[1]  # Use the safety test case
    safety_result = await evaluation_agent.evaluate_llm_safety(
        query=safety_test['query'],
        response=safety_test['response'],
        agent_name=safety_test['agent_name']
    )
    
    if "error" not in safety_result:
        print(f"Safety Score: {safety_result.get('safety_score', 0.0):.2f}/1.0")
        print("Safety Metrics:")
        for metric_name, metric_data in safety_result.get('metrics', {}).items():
            if isinstance(metric_data, dict) and 'score' in metric_data:
                print(f"  - {metric_name}: {metric_data['score']:.2f}/1.0")
    print()
    
    # Cleanup
    print("Cleaning up...")
    await memory_manager.cleanup()
    print(" Demo completed successfully!")
    
    print("\n" + "=" * 80)
    print("EVALUATION SYSTEM SUMMARY")
    print("=" * 80)
    print(" RAGAS integration for RAG quality metrics")
    print(" DEEPEVAL integration for LLM safety metrics")
    print(" Custom evaluation tools for completeness and coherence")
    print(" Comprehensive evaluation with overall scoring")
    print(" Quality feedback generation")
    print(" Improvement suggestions")
    print(" Integration with swarm handoff system")
    print("=" * 80)


async def demonstrate_swarm_integration():
    """Demonstrate evaluation agent integration with the swarm."""
    
    print("\n" + "=" * 80)
    print("SWARM INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the full swarm system
    print("Initializing full swarm system with evaluation agent...")
    
    # Memory manager
    memory_manager = MemoryManager()
    await memory_manager.initialize()
    
    # Monitoring components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    drift_detector = DriftDetector()
    
    # Initialize swarm
    swarm = HandoffSwarm(
        memory_manager=memory_manager,
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        drift_detector=drift_detector
    )
    
    print(" Swarm with evaluation agent initialized!")
    print(f"Available agents: {list(swarm.agents.keys())}")
    print()
    
    # Test queries that will trigger evaluation
    test_queries = [
        "What are the company policies? Please evaluate the quality of your response.",
        "Show me customer data and assess the response quality.",
        "I need invoice information. Can you evaluate how well you answered this?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        print("-" * 50)
        
        # Process query through swarm
        result = await swarm.process_query(query, session_id=f"eval_test_{i}")
        
        print(f"Response: {result.get('content', 'No response')}")
        print(f"Agent Used: {result.get('agent_used', 'Unknown')}")
        print(f"Metadata: {result.get('metadata', {})}")
        print()
    
    # Cleanup
    await memory_manager.cleanup()
    print(" Swarm integration demo completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstrations
    asyncio.run(demonstrate_evaluation_system())
    asyncio.run(demonstrate_swarm_integration())
