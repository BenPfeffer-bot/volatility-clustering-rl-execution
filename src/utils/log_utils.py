import logging
from pathlib import Path
from ..config.paths import LOGS_DIR

# Create all directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"

def setup_logging(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration for the given name.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if log_level == logging.DEBUG:
        formatter = logging.Formatter(DEBUG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger




