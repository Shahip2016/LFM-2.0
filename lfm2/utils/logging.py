import logging
import sys
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates and configures a logger.
    
    Args:
        name: Name of the logger.
        level: Logging level.
        log_file: Optional path to a file to write logs to.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formatter for logs
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
