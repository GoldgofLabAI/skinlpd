"""
Logging utility for MIL training scripts.

This module provides a centralized logging setup that writes logs to both
console and file. Logs are saved to the output directory for each script run.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, output_dir: str = None, log_level: int = logging.INFO, 
                log_to_file: bool = True, log_to_console: bool = True) -> logging.Logger:
    """
    Set up a logger that writes to both console and file.
    
    Args:
        name: Logger name (typically __name__ or script name)
        output_dir: Directory where log file will be saved. If None, only console logging.
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to log to file (default: True)
        log_to_console: Whether to log to console (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        return logger
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger by name, or create a new one if it doesn't exist.
    
    Args:
        name: Logger name. If None, returns root logger.
    
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)


def log_args(logger: logging.Logger, args, script_name: str = None):
    """
    Log all command-line arguments in a readable format.
    
    Args:
        logger: Logger instance
        args: Parsed arguments (argparse.Namespace)
        script_name: Name of the script (optional)
    """
    if script_name:
        logger.info(f"=== {script_name} ===")
    logger.info("Command-line arguments:")
    for arg, value in sorted(vars(args).items()):
        # Skip None values for cleaner output
        if value is not None:
            logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO):
    """
    Log a section header for better readability.
    
    Args:
        logger: Logger instance
        title: Section title
        level: Logging level (default: INFO)
    """
    logger.log(level, "=" * 60)
    logger.log(level, f"  {title}")
    logger.log(level, "=" * 60)

