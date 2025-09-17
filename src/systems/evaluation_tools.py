"""
Evaluation tools for RAGAS and DEEPEVAL metrics.
Provides tools for the evaluation agent to assess response quality.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
        AnswerSimilarity
    )
    from ragas.dataset import Dataset
    from ragas.testset import TestsetGenerator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import (
        HallucinationMetric,
        BiasMetric,
        ToxicityMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logging.warning("DEEPEVAL not available. Install with: pip install deepeval")

from core.config.config import config


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    agent_name: str
    query: str
    response: str
    context: Optional[List[str]] = None


class RAGASEvaluationTools:
    """Tools for RAGAS-based evaluation metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("ragas_evaluation")
        self.available = RAGAS_AVAILABLE
        
        if not self.available:
            self.logger.warning("RAGAS not available. Evaluation tools disabled.")
    
    async def evaluate_faithfulness(
        self, 
        query: str, 
        response: str, 
        context: List[str],
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate response faithfulness using RAGAS."""
        if not self.available:
            return self._create_unavailable_result("faithfulness", agent_name, query, response)
        
        try:
            # For now, use a simplified approach due to RAGAS API complexity
            # This is a placeholder that returns a basic score
            # In production, you would implement proper RAGAS integration
            
            # Simple heuristic: check if response contains context information
            context_words = set()
            for ctx in context:
                context_words.update(ctx.lower().split())
            
            response_words = set(response.lower().split())
            overlap = len(context_words.intersection(response_words))
            faithfulness_score = min(overlap / max(len(context_words), 1), 1.0)
            
            return EvaluationResult(
                metric_name="faithfulness",
                score=faithfulness_score,
                details={
                    "ragas_version": "0.3.4",
                    "context_count": len(context),
                    "response_length": len(response),
                    "method": "heuristic_placeholder"
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response,
                context=context
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating faithfulness: {str(e)}")
            return self._create_error_result("faithfulness", agent_name, query, response, str(e))
    
    async def evaluate_answer_relevancy(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate answer relevancy using RAGAS."""
        if not self.available:
            return self._create_unavailable_result("answer_relevancy", agent_name, query, response)
        
        try:
            # Simplified heuristic approach
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            relevancy_score = min(overlap / max(len(query_words), 1), 1.0)
            
            return EvaluationResult(
                metric_name="answer_relevancy",
                score=relevancy_score,
                details={
                    "ragas_version": "0.3.4",
                    "response_length": len(response),
                    "method": "heuristic_placeholder"
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer relevancy: {str(e)}")
            return self._create_error_result("answer_relevancy", agent_name, query, response, str(e))
    
    async def evaluate_context_precision(
        self, 
        query: str, 
        response: str, 
        context: List[str],
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate context precision using RAGAS."""
        if not self.available:
            return self._create_unavailable_result("context_precision", agent_name, query, response)
        
        try:
            # Simplified heuristic approach
            # Check how much of the response is covered by context
            context_text = " ".join(context).lower()
            response_lower = response.lower()
            
            # Simple word overlap calculation
            context_words = set(context_text.split())
            response_words = set(response_lower.split())
            overlap = len(context_words.intersection(response_words))
            precision_score = min(overlap / max(len(response_words), 1), 1.0)
            
            return EvaluationResult(
                metric_name="context_precision",
                score=precision_score,
                details={
                    "ragas_version": "0.3.4",
                    "context_count": len(context),
                    "method": "heuristic_placeholder"
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response,
                context=context
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating context precision: {str(e)}")
            return self._create_error_result("context_precision", agent_name, query, response, str(e))
    
    async def evaluate_comprehensive_rag(
        self, 
        query: str, 
        response: str, 
        context: List[str],
        agent_name: str
    ) -> List[EvaluationResult]:
        """Run comprehensive RAG evaluation with multiple metrics."""
        if not self.available:
            return [self._create_unavailable_result("comprehensive_rag", agent_name, query, response)]
        
        try:
            # For now, use individual metric evaluations instead of comprehensive
            # due to RAGAS API complexity
            results = []
            
            # Evaluate faithfulness
            try:
                faithfulness_result = await self.evaluate_faithfulness(query, response, context, agent_name)
                results.append(faithfulness_result)
            except Exception as e:
                self.logger.warning(f"Faithfulness evaluation failed: {str(e)}")
            
            # Evaluate answer relevancy
            try:
                relevancy_result = await self.evaluate_answer_relevancy(query, response, agent_name)
                results.append(relevancy_result)
            except Exception as e:
                self.logger.warning(f"Answer relevancy evaluation failed: {str(e)}")
            
            # Evaluate context precision
            try:
                precision_result = await self.evaluate_context_precision(query, response, context, agent_name)
                results.append(precision_result)
            except Exception as e:
                self.logger.warning(f"Context precision evaluation failed: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive RAG evaluation: {str(e)}")
            return [self._create_error_result("comprehensive_rag", agent_name, query, response, str(e))]
    
    def _create_unavailable_result(self, metric_name: str, agent_name: str, query: str, response: str) -> EvaluationResult:
        """Create result for unavailable metric."""
        return EvaluationResult(
            metric_name=metric_name,
            score=0.0,
            details={"error": "RAGAS not available"},
            timestamp=datetime.now(),
            agent_name=agent_name,
            query=query,
            response=response
        )
    
    def _create_error_result(self, metric_name: str, agent_name: str, query: str, response: str, error: str) -> EvaluationResult:
        """Create result for error case."""
        return EvaluationResult(
            metric_name=metric_name,
            score=0.0,
            details={"error": error},
            timestamp=datetime.now(),
            agent_name=agent_name,
            query=query,
            response=response
        )


class DEEPEVALEvaluationTools:
    """Tools for DEEPEVAL-based evaluation metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("deepeval_evaluation")
        self.available = DEEPEVAL_AVAILABLE
        
        if not self.available:
            self.logger.warning("DEEPEVAL not available. Evaluation tools disabled.")
    
    async def evaluate_hallucination(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate hallucination using DEEPEVAL."""
        if not self.available:
            return self._create_unavailable_result("hallucination", agent_name, query, response)
        
        try:
            # Create proper test case for DEEPEVAL 3.3.9+
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output="",  # Not required for hallucination
                context=[]  # Not required for hallucination
            )
            
            metric = HallucinationMetric()
            result = metric.measure(test_case)
            
            # Handle different result types
            if hasattr(result, 'score'):
                score = result.score
                reason = getattr(result, 'reason', 'No reason provided')
            else:
                # If result is just a number
                score = float(result) if isinstance(result, (int, float)) else 0.0
                reason = "Score returned directly"
            
            return EvaluationResult(
                metric_name="hallucination",
                score=score,
                details={
                    "deepeval_version": "3.3.9",
                    "reason": reason,
                    "response_length": len(response)
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating hallucination: {str(e)}")
            return self._create_error_result("hallucination", agent_name, query, response, str(e))
    
    async def evaluate_bias(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate bias using DEEPEVAL."""
        if not self.available:
            return self._create_unavailable_result("bias", agent_name, query, response)
        
        try:
            # Create proper test case for DEEPEVAL 3.3.9+
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output="",  # Not required for bias
                context=[]  # Not required for bias
            )
            
            metric = BiasMetric()
            result = metric.measure(test_case)
            
            # Handle different result types
            if hasattr(result, 'score'):
                score = result.score
                reason = getattr(result, 'reason', 'No reason provided')
            else:
                # If result is just a number
                score = float(result) if isinstance(result, (int, float)) else 0.0
                reason = "Score returned directly"
            
            return EvaluationResult(
                metric_name="bias",
                score=score,
                details={
                    "deepeval_version": "3.3.9",
                    "reason": reason,
                    "response_length": len(response)
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating bias: {str(e)}")
            return self._create_error_result("bias", agent_name, query, response, str(e))
    
    async def evaluate_toxicity(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate toxicity using DEEPEVAL."""
        if not self.available:
            return self._create_unavailable_result("toxicity", agent_name, query, response)
        
        try:
            # Create proper test case for DEEPEVAL 3.3.9+
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output="",  # Not required for toxicity
                context=[]  # Not required for toxicity
            )
            
            metric = ToxicityMetric()
            result = metric.measure(test_case)
            
            # Handle different result types
            if hasattr(result, 'score'):
                score = result.score
                reason = getattr(result, 'reason', 'No reason provided')
            else:
                # If result is just a number
                score = float(result) if isinstance(result, (int, float)) else 0.0
                reason = "Score returned directly"
            
            return EvaluationResult(
                metric_name="toxicity",
                score=score,
                details={
                    "deepeval_version": "3.3.9",
                    "reason": reason,
                    "response_length": len(response)
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating toxicity: {str(e)}")
            return self._create_error_result("toxicity", agent_name, query, response, str(e))
    
    async def evaluate_comprehensive_llm(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> List[EvaluationResult]:
        """Run comprehensive LLM evaluation with multiple DEEPEVAL metrics."""
        if not self.available:
            return [self._create_unavailable_result("comprehensive_llm", agent_name, query, response)]
        
        try:
            results = []
            
            # Evaluate hallucination
            hallucination_result = await self.evaluate_hallucination(query, response, agent_name)
            results.append(hallucination_result)
            
            # Evaluate bias
            bias_result = await self.evaluate_bias(query, response, agent_name)
            results.append(bias_result)
            
            # Evaluate toxicity
            toxicity_result = await self.evaluate_toxicity(query, response, agent_name)
            results.append(toxicity_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive LLM evaluation: {str(e)}")
            return [self._create_error_result("comprehensive_llm", agent_name, query, response, str(e))]
    
    def _create_unavailable_result(self, metric_name: str, agent_name: str, query: str, response: str) -> EvaluationResult:
        """Create result for unavailable metric."""
        return EvaluationResult(
            metric_name=metric_name,
            score=0.0,
            details={"error": "DEEPEVAL not available"},
            timestamp=datetime.now(),
            agent_name=agent_name,
            query=query,
            response=response
        )
    
    def _create_error_result(self, metric_name: str, agent_name: str, query: str, response: str, error: str) -> EvaluationResult:
        """Create result for error case."""
        return EvaluationResult(
            metric_name=metric_name,
            score=0.0,
            details={"error": error},
            timestamp=datetime.now(),
            agent_name=agent_name,
            query=query,
            response=response
        )


class CustomEvaluationTools:
    """Custom evaluation tools for additional metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("custom_evaluation")
    
    async def evaluate_response_completeness(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate response completeness based on query requirements."""
        try:
            # Simple heuristic: check if response addresses the query
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            # Calculate overlap
            overlap = len(query_words.intersection(response_words))
            completeness_score = min(overlap / len(query_words), 1.0) if query_words else 0.0
            
            return EvaluationResult(
                metric_name="response_completeness",
                score=completeness_score,
                details={
                    "query_word_count": len(query_words),
                    "response_word_count": len(response_words),
                    "overlap_count": overlap
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating response completeness: {str(e)}")
            return EvaluationResult(
                metric_name="response_completeness",
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
    
    async def evaluate_response_coherence(
        self, 
        query: str, 
        response: str,
        agent_name: str
    ) -> EvaluationResult:
        """Evaluate response coherence and readability."""
        try:
            # Simple heuristic: check sentence structure and length
            sentences = response.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Coherence score based on sentence length (not too short, not too long)
            if 5 <= avg_sentence_length <= 25:
                coherence_score = 1.0
            elif 3 <= avg_sentence_length <= 30:
                coherence_score = 0.8
            else:
                coherence_score = 0.6
            
            return EvaluationResult(
                metric_name="response_coherence",
                score=coherence_score,
                details={
                    "sentence_count": len(sentences),
                    "avg_sentence_length": avg_sentence_length,
                    "response_length": len(response)
                },
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating response coherence: {str(e)}")
            return EvaluationResult(
                metric_name="response_coherence",
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now(),
                agent_name=agent_name,
                query=query,
                response=response
            )
