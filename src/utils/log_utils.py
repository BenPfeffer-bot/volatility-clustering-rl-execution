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


def setup_logging(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Set up logging with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level
        format: Log message format

    Returns:
        Configured logger instance
    """
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

    # Add file handler if specified
    if log_file is not None:
        # Create directory if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
