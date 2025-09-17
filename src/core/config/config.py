"""
Configuration management for the multi-agent swarm system.
Handles environment variables, AWS settings, and system configurations.
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AWSConfig:
    """AWS configuration settings."""
    region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    temperature: float = 0.0
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "swarm_db"
    username: str = "swarm_user"
    password: str = "swarm_password"
    connection_pool_size: int = 10


@dataclass
class MemoryConfig:
    """Memory configuration settings."""
    short_term_ttl: int = 3600  # 1 hour in seconds
    long_term_ttl: int = 86400 * 30  # 30 days in seconds
    max_memory_entries: int = 1000
    memory_backend: str = "redis"  # redis, memory, dynamodb


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_metrics: bool = True
    enable_alerting: bool = True
    metrics_retention_days: int = 30
    alert_threshold_response_time: float = 5.0  # seconds
    alert_threshold_error_rate: float = 0.05  # 5%
    drift_detection_enabled: bool = True


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    max_retries: int = 3
    timeout_seconds: int = 30
    temperature: float = 0.1
    max_tokens: int = 4000
    enable_streaming: bool = False


@dataclass
class LangChainConfig:
    """LangChain configuration settings."""
    api_key: Optional[str] = None
    tracing_v2: bool = False
    project: str = "multi-agent-swarm"
    endpoint: str = "https://api.smith.langchain.com"
    debug: bool = False


@dataclass
class AnthropicConfig:
    """Anthropic configuration settings."""
    api_key: Optional[str] = None
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4000
    temperature: float = 0.1


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration settings."""
    api_key: Optional[str] = None
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    endpoint: Optional[str] = None
    device: str = "cpu"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # AWS Configuration
        self.aws = AWSConfig(
            region=os.getenv("AWS_REGION", "us-east-1"),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            temperature=float(os.getenv("BEDROCK_TEMPERATURE", "0.0")),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN")
        )
        
        # Database Configuration
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "swarm_db"),
            username=os.getenv("DB_USER", "swarm_user"),
            password=os.getenv("DB_PASSWORD", "swarm_password"),
            connection_pool_size=int(os.getenv("DB_POOL_SIZE", "10"))
        )
        
        # Memory Configuration
        self.memory = MemoryConfig(
            short_term_ttl=int(os.getenv("MEMORY_SHORT_TTL", "3600")),
            long_term_ttl=int(os.getenv("MEMORY_LONG_TTL", "2592000")),
            max_memory_entries=int(os.getenv("MEMORY_MAX_ENTRIES", "1000")),
            memory_backend=os.getenv("MEMORY_BACKEND", "redis")
        )
        
        # Monitoring Configuration
        self.monitoring = MonitoringConfig(
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_alerting=os.getenv("ENABLE_ALERTING", "true").lower() == "true",
            metrics_retention_days=int(os.getenv("METRICS_RETENTION_DAYS", "30")),
            alert_threshold_response_time=float(os.getenv("ALERT_THRESHOLD_RESPONSE_TIME", "5.0")),
            alert_threshold_error_rate=float(os.getenv("ALERT_THRESHOLD_ERROR_RATE", "0.05")),
            drift_detection_enabled=os.getenv("DRIFT_DETECTION_ENABLED", "true").lower() == "true"
        )
        
        # Agent Configuration
        self.agent = AgentConfig(
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("AGENT_TIMEOUT", "30")),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4000")),
            enable_streaming=os.getenv("AGENT_ENABLE_STREAMING", "false").lower() == "true"
        )
        
        # LangChain Configuration
        self.langchain = LangChainConfig(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            project=os.getenv("LANGCHAIN_PROJECT", "multi-agent-swarm"),
            endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            debug=os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
        )
        
        # Anthropic Configuration
        self.anthropic = AnthropicConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.1"))
        )
        
        # Hugging Face Configuration
        self.huggingface = HuggingFaceConfig(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            model_name=os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            endpoint=os.getenv("HUGGINGFACE_ENDPOINT"),
            device=os.getenv("HUGGINGFACE_DEVICE", "cpu")
        )
    
    def reload(self):
        """Reload configuration from environment variables."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        self._load_config()


# Global configuration instance
config = Config()
