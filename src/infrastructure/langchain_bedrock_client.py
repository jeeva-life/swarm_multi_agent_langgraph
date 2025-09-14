"""
LangChain Anthropic client for Claude integration.
Uses LangChain's ChatAnthropic for direct API access.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from core.config import config

# Load environment variables
load_dotenv()


class LangChainBedrockClient:
    """
    LangChain Bedrock client for Claude integration.
    Uses LangChain's ChatBedrock for consistent interface.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("langchain_bedrock_client")
        self.initialized = False
        self.model = None
        self.embeddings = None
        
    async def initialize(self) -> bool:
        """Initialize LangChain Anthropic client."""
        try:
            # Import LangChain components only when needed
            from langchain_anthropic import ChatAnthropic
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # Initialize ChatAnthropic model
            self.model = ChatAnthropic(
                model=config.anthropic.model,
                api_key=config.anthropic.api_key,
                temperature=config.anthropic.temperature,
                max_tokens=config.anthropic.max_tokens
            )
            
            # Initialize Hugging Face embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},  # Use CPU for local inference
                encode_kwargs={"normalize_embeddings": True}
            )
            
            self.initialized = True
            self.logger.info("LangChain Bedrock client initialized")
            return True
            
        except ImportError as e:
            self.logger.error(f"Required packages not installed: {str(e)}")
            self.logger.error("Install with: pip install langchain-aws langchain-huggingface")
            return False
        except Exception as e:
            self.logger.error(f"Error initializing LangChain Bedrock client: {str(e)}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate response using LangChain Bedrock."""
        try:
            if not self.initialized or not self.model:
                raise Exception("LangChain Bedrock client not initialized")
            
            # Create messages for the model
            messages = [{"role": "user", "content": prompt}]
            
            # Generate response
            response = await self.model.ainvoke(messages)
            
            return {
                "content": response.content,
                "usage": getattr(response, 'usage_metadata', {}),
                "model": config.aws.bedrock_model_id
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face model."""
        try:
            if not self.initialized or not self.embeddings:
                raise Exception("LangChain Bedrock client not initialized")
            
            # Generate embeddings
            embeddings = await self.embeddings.aembed_documents(texts)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            if not self.initialized or not self.embeddings:
                raise Exception("LangChain Bedrock client not initialized")
            
            # Generate query embedding
            embedding = await self.embeddings.aembed_query(query)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if LangChain Bedrock client is initialized."""
        return self.initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": config.aws.bedrock_model_id,
            "region": config.aws.region,
            "temperature": config.aws.temperature,
            "initialized": self.initialized
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the LangChain Bedrock client."""
        try:
            if not self.initialized:
                return {
                    "status": "unhealthy",
                    "message": "Client not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Test model connectivity with a simple prompt
            test_response = await self.generate_response("Hello, this is a health check.")
            
            return {
                "status": "healthy",
                "message": "Client is working properly",
                "timestamp": datetime.now().isoformat(),
                "model_info": self.get_model_info(),
                "test_response_length": len(test_response.get("content", ""))
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close the client and clean up resources."""
        try:
            # LangChain clients don't need explicit closing
            self.initialized = False
            self.model = None
            self.embeddings = None
            self.logger.info("LangChain Bedrock client closed")
        except Exception as e:
            self.logger.error(f"Error closing LangChain Bedrock client: {str(e)}")
