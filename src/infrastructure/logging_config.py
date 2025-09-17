"""
Logging configuration for the multi-agent swarm system.
Provides structured logging with different levels and outputs.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any

from core.config.config import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the multi-agent swarm system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (defaults to logs/swarm.log)
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger
    """
    try:
        # Determine log level
        if log_level is None:
            log_level = "DEBUG" if config.debug else "INFO"
        
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logs directory if it doesn't exist
        if enable_file:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            if log_file is None:
                log_file = os.path.join(log_dir, "swarm.log")
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if enable_file and log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set specific logger levels
        _configure_logger_levels()
        
        # Create main logger
        logger = logging.getLogger("swarm")
        logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        return logging.getLogger("swarm")


def _configure_logger_levels():
    """Configure specific logger levels for different components."""
    try:
        # Set specific levels for different components
        loggers = {
            "swarm": "INFO",
            "agent": "INFO",
            "memory": "INFO",
            "monitoring": "INFO",
            "aws_client": "WARNING",
            "database_client": "WARNING",
            "langchain_bedrock_client": "WARNING"
        }
        
        for logger_name, level in loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level, logging.INFO))
        
    except Exception as e:
        print(f"Error configuring logger levels: {str(e)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Logger name (usually the module name)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)