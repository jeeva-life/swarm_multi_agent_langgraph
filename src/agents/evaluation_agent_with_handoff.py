"""
Evaluation Agent with Handoff Capabilities for RAGAS and DEEPEVAL metrics.
This agent can be integrated into the swarm workflow and called by other agents.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from core.config import config
from core.evaluation_tools import (
    RAGASEvaluationTools,
    DEEPEVALEvaluationTools,
    CustomEvaluationTools,
    EvaluationResult
)


class EvaluationAgentWithHandoff:
    """
    Evaluation Agent with handoff capabilities for integration into the swarm.
    Can evaluate responses from other agents and provide feedback.
    """
    
    def __init__(self, memory_manager=None, metrics_collector=None):
        self.logger = logging.getLogger("evaluation_agent_with_handoff")
        self.memory_manager = memory_manager
        self.metrics_collector = metrics_collector
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=config.anthropic.temperature,
            max_tokens=config.anthropic.max_tokens
        )
        
        # Initialize evaluation tools
        self.ragas_tools = RAGASEvaluationTools()
        self.deepeval_tools = DEEPEVALEvaluationTools()
        self.custom_tools = CustomEvaluationTools()
        
        # Define handoff tools
        self.handoff_tools = [
            self.evaluate_response_quality,
            self.evaluate_rag_performance,
            self.evaluate_llm_safety,
            self.get_quality_feedback,
            self.suggest_response_improvements
        ]
        
        self.logger.info("Evaluation agent with handoff capabilities initialized")
    
    @tool
    async def evaluate_response_quality(
        self, 
        query: str, 
        response: str, 
        context: Optional[List[str]] = None,
        agent_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a response using comprehensive metrics.
        
        Args:
            query: The original query
            response: The response to evaluate
            context: Context documents used (optional)
            agent_name: Name of the agent that generated the response
        
        Returns:
            Dictionary with quality evaluation results
        """
        try:
            self.logger.info(f"Evaluating response quality for {agent_name}")
            
            evaluation_data = {
                "agent_name": agent_name,
                "query": query,
                "response": response,
                "context_count": len(context) if context else 0,
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_type": "comprehensive_quality",
                "metrics": {}
            }
            
            # RAG evaluation (if context provided)
            if context:
                rag_results = await self.ragas_tools.evaluate_comprehensive_rag(
                    query, response, context, agent_name
                )
                for result in rag_results:
                    evaluation_data["metrics"][f"rag_{result.metric_name}"] = {
                        "score": result.score,
                        "details": result.details
                    }
            
            # LLM evaluation
            llm_results = await self.deepeval_tools.evaluate_comprehensive_llm(
                query, response, agent_name
            )
            for result in llm_results:
                evaluation_data["metrics"][f"llm_{result.metric_name}"] = {
                    "score": result.score,
                    "details": result.details
                }
            
            # Custom evaluation
            custom_results = [
                await self.custom_tools.evaluate_response_completeness(query, response, agent_name),
                await self.custom_tools.evaluate_response_coherence(query, response, agent_name)
            ]
            for result in custom_results:
                evaluation_data["metrics"][f"custom_{result.metric_name}"] = {
                    "score": result.score,
                    "details": result.details
                }
            
            # Calculate overall quality score
            all_scores = []
            for metric_name, metric_data in evaluation_data["metrics"].items():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    score = metric_data["score"]
                    if score > 0:  # Only include valid scores
                        all_scores.append(score)
            
            if all_scores:
                evaluation_data["overall_quality_score"] = sum(all_scores) / len(all_scores)
                evaluation_data["score_count"] = len(all_scores)
            
            # Store evaluation in memory
            if self.memory_manager:
                await self.memory_manager.store_evaluation(
                    agent_name=agent_name,
                    query=query,
                    response=response,
                    evaluation_data=evaluation_data
                )
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_agent_metrics(
                    agent_name=f"evaluation_{agent_name}",
                    execution_time=0.0,  # Will be calculated by the system
                    success=True,
                    confidence=evaluation_data.get("overall_quality_score", 0.0)
                )
            
            return evaluation_data
            
        except Exception as e:
            self.logger.error(f"Error evaluating response quality: {str(e)}")
            return {
                "error": str(e),
                "agent_name": agent_name,
                "query": query,
                "response": response
            }
    
    @tool
    async def evaluate_rag_performance(
        self, 
        query: str, 
        response: str, 
        context: List[str],
        agent_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate RAG-specific performance metrics.
        
        Args:
            query: The original query
            response: The response to evaluate
            context: Context documents used
            agent_name: Name of the agent that generated the response
        
        Returns:
            Dictionary with RAG performance evaluation
        """
        try:
            self.logger.info(f"Evaluating RAG performance for {agent_name}")
            
            # Run RAG evaluation
            rag_results = await self.ragas_tools.evaluate_comprehensive_rag(
                query, response, context, agent_name
            )
            
            evaluation_data = {
                "agent_name": agent_name,
                "query": query,
                "response": response,
                "context_count": len(context),
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_type": "rag_performance",
                "metrics": {}
            }
            
            for result in rag_results:
                evaluation_data["metrics"][result.metric_name] = {
                    "score": result.score,
                    "details": result.details
                }
            
            # Calculate RAG-specific score
            rag_scores = [r.score for r in rag_results if r.score > 0]
            if rag_scores:
                evaluation_data["rag_performance_score"] = sum(rag_scores) / len(rag_scores)
            
            return evaluation_data
            
        except Exception as e:
            self.logger.error(f"Error evaluating RAG performance: {str(e)}")
            return {
                "error": str(e),
                "agent_name": agent_name,
                "query": query,
                "response": response
            }
    
    @tool
    async def evaluate_llm_safety(
        self, 
        query: str, 
        response: str,
        agent_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate LLM safety metrics (hallucination, bias, toxicity).
        
        Args:
            query: The original query
            response: The response to evaluate
            agent_name: Name of the agent that generated the response
        
        Returns:
            Dictionary with safety evaluation
        """
        try:
            self.logger.info(f"Evaluating LLM safety for {agent_name}")
            
            # Run LLM safety evaluation
            safety_results = await self.deepeval_tools.evaluate_comprehensive_llm(
                query, response, agent_name
            )
            
            evaluation_data = {
                "agent_name": agent_name,
                "query": query,
                "response": response,
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_type": "llm_safety",
                "metrics": {}
            }
            
            for result in safety_results:
                evaluation_data["metrics"][result.metric_name] = {
                    "score": result.score,
                    "details": result.details
                }
            
            # Calculate safety score
            safety_scores = [r.score for r in safety_results if r.score > 0]
            if safety_scores:
                evaluation_data["safety_score"] = sum(safety_scores) / len(safety_scores)
            
            return evaluation_data
            
        except Exception as e:
            self.logger.error(f"Error evaluating LLM safety: {str(e)}")
            return {
                "error": str(e),
                "agent_name": agent_name,
                "query": query,
                "response": response
            }
    
    @tool
    async def get_quality_feedback(
        self, 
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Generate quality feedback based on evaluation results.
        
        Args:
            evaluation_results: Results from evaluation methods
        
        Returns:
            Human-readable quality feedback
        """
        try:
            if "error" in evaluation_results:
                return f"Quality evaluation failed: {evaluation_results['error']}"
            
            agent_name = evaluation_results.get("agent_name", "Unknown")
            overall_score = evaluation_results.get("overall_quality_score", 0.0)
            metrics = evaluation_results.get("metrics", {})
            
            # Generate feedback using LLM
            feedback_prompt = f"""
            Based on the following evaluation results, provide constructive feedback for the agent "{agent_name}":
            
            Overall Quality Score: {overall_score:.2f}/1.0
            
            Detailed Metrics:
            {self._format_metrics_for_llm(metrics)}
            
            Please provide:
            1. A brief summary of the response quality
            2. Specific strengths identified
            3. Areas for improvement
            4. Actionable recommendations
            
            Keep the feedback constructive and specific.
            """
            
            messages = [
                SystemMessage(content="You are an expert AI quality assessor. Provide constructive feedback based on evaluation metrics."),
                HumanMessage(content=feedback_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            feedback = response.content if hasattr(response, 'content') else str(response)
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error generating quality feedback: {str(e)}")
            return f"Error generating feedback: {str(e)}"
    
    @tool
    async def suggest_response_improvements(
        self, 
        query: str, 
        response: str, 
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Suggest specific improvements for a response based on evaluation results.
        
        Args:
            query: The original query
            response: The response to improve
            evaluation_results: Results from evaluation methods
        
        Returns:
            Specific improvement suggestions
        """
        try:
            if "error" in evaluation_results:
                return f"Cannot suggest improvements due to evaluation error: {evaluation_results['error']}"
            
            metrics = evaluation_results.get("metrics", {})
            
            # Generate improvement suggestions using LLM
            improvement_prompt = f"""
            Based on the following evaluation results, suggest specific improvements for this response:
            
            Original Query: {query}
            Current Response: {response}
            
            Evaluation Metrics:
            {self._format_metrics_for_llm(metrics)}
            
            Please provide:
            1. Specific improvements for the response
            2. How to address low-scoring metrics
            3. Suggested revised response structure
            4. Best practices to follow
            
            Be specific and actionable in your suggestions.
            """
            
            messages = [
                SystemMessage(content="You are an expert AI response optimizer. Provide specific, actionable improvement suggestions."),
                HumanMessage(content=improvement_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            suggestions = response.content if hasattr(response, 'content') else str(response)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {str(e)}")
            return f"Error generating suggestions: {str(e)}"
    
    def _format_metrics_for_llm(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for LLM consumption."""
        formatted_metrics = []
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                formatted_metrics.append(f"- {metric_name}: {score:.2f}/1.0")
        return "\n".join(formatted_metrics)
    
    async def process_handoff(
        self, 
        query: str, 
        response: str, 
        context: Optional[List[str]] = None,
        agent_name: str = "unknown",
        evaluation_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Process a handoff request for evaluation.
        
        Args:
            query: The original query
            response: The response to evaluate
            context: Context documents (optional)
            agent_name: Name of the agent that generated the response
            evaluation_type: Type of evaluation to perform
        
        Returns:
            Evaluation results
        """
        try:
            self.logger.info(f"Processing evaluation handoff for {agent_name}")
            
            if evaluation_type == "rag":
                return await self.evaluate_rag_performance.ainvoke({
                    "query": query,
                    "response": response,
                    "context": context or [],
                    "agent_name": agent_name
                })
            elif evaluation_type == "safety":
                return await self.evaluate_llm_safety.ainvoke({
                    "query": query,
                    "response": response,
                    "agent_name": agent_name
                })
            else:  # comprehensive
                return await self.evaluate_response_quality.ainvoke({
                    "query": query,
                    "response": response,
                    "context": context,
                    "agent_name": agent_name
                })
                
        except Exception as e:
            self.logger.error(f"Error processing evaluation handoff: {str(e)}")
            return {
                "error": str(e),
                "agent_name": agent_name,
                "query": query,
                "response": response
            }
