"""
Logging utility for consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from ..config.paths import LOGS_DIR

# Create all directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"


def get_default_log_file(name: str) -> Path:
    """
    Get default log file path based on logger name.

    Args:
        name: Logger name

    Returns:
        Path to log file
    """
    # Clean up name for filename
    filename = name.replace(".", "_").lower() + ".log"
    return LOGS_DIR / filename


def setup_logging(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format: str = FORMAT,
    debug: bool = False,
) -> logging.Logger:
    """
    Set up logging with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file. If None, will use default path from get_default_log_file()
        level: Logging level
        format: Log message format. Uses FORMAT by default, or DEBUG_FORMAT if debug=True
        debug: If True, uses DEBUG_FORMAT and sets level to DEBUG

    Returns:
        Configured logger instance
    """
    # Set debug options if enabled
    if debug:
        level = logging.DEBUG
        format = DEBUG_FORMAT

    # Get default log file if none specified
    if log_file is None:
        log_file = get_default_log_file(name)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
