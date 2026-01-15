"""
Centralized logging configuration for Bato-AI.

Provides structured logging with:
- JSON formatting for production (easy parsing/searching)
- Console formatting for development (human-readable)
- Log rotation (prevents disk space issues)
- Request ID tracking (trace requests across services)
- Environment-specific log levels
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (e.g., request_id)
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format: [LEVEL] module.function:line - message
        formatted = (
            f"{color}[{record.levelname}]{self.RESET} "
            f"{record.name}:{record.lineno} - {record.getMessage()}"
        )
        
        # Add exception if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        environment: "development" or "production"
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None = no file logging)
    
    Example:
        # Development
        setup_logging("development", "DEBUG")
        
        # Production
        setup_logging("production", "INFO", Path("logs"))
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    
    if environment == "production":
        # Production: JSON format
        console_handler.setFormatter(JSONFormatter())
    else:
        # Development: Colored console format
        console_handler.setFormatter(ConsoleFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "bato-ai.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        
        # Always use JSON for file logs (easier to parse)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
        
        # Separate error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "bato-ai-errors.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: environment={environment}, "
        f"level={log_level}, file_logging={log_dir is not None}"
    )


class RequestIDFilter(logging.Filter):
    """Add request ID to log records."""
    
    def __init__(self, request_id: str):
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to record."""
        record.request_id = self.request_id
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Application started")
    """
    return logging.getLogger(name)


# Convenience function for adding request context
def add_request_context(logger: logging.Logger, request_id: str) -> None:
    """
    Add request ID to all logs from this logger.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
    
    Example:
        logger = get_logger(__name__)
        add_request_context(logger, "req-12345")
        logger.info("Processing request")  # Will include request_id
    """
    logger.addFilter(RequestIDFilter(request_id))
