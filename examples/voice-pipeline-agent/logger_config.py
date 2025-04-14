import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Configure logging with rotating file handler and console output.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (rotating, max 10MB per file, keep 30 days of logs)
    log_file = log_path / f"voice_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log system info
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    
    return logger

# Performance logging
def log_performance_metrics(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics for voice pipeline operations."""
    metrics = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        **kwargs
    }
    logger.debug(f"Performance: {metrics}")

# Error logging with context
def log_error_with_context(logger: logging.Logger, error: Exception, context: dict):
    """Log errors with additional context information."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context
    }
    logger.error(f"Error occurred: {error_info}")

# Connection state logging
def log_connection_state(logger: logging.Logger, state: str, details: dict):
    """Log connection state changes with details."""
    logger.info(f"Connection {state}: {details}")
