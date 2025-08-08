# common/logging.py
import logging
import logging.config
import os
import sys
import re
from datetime import datetime
from typing import Optional
from lpm_kernel.configs.logging import LOGGING_CONFIG, LOG_BASE_DIR, TRAIN_LOG_DIR, rename_existing_log_file


def setup_logging():
    try:
        # Ensure log directories exist
        os.makedirs(LOG_BASE_DIR, exist_ok=True)
        os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
        
        # Rename existing log file if needed
        rename_existing_log_file()
        # Ensure directory permissions are correct
        os.chmod(TRAIN_LOG_DIR, 0o755)
        os.chmod(LOG_BASE_DIR, 0o755)

        print(f"Log directory: {TRAIN_LOG_DIR}", file=sys.stderr)
        print(
            f"Log file: {LOGGING_CONFIG['handlers']['file']['filename']}",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"Error creating log directory: {e}", file=sys.stderr)
        # If unable to create directory, use standard output
        LOGGING_CONFIG["handlers"]["file"] = LOGGING_CONFIG["handlers"]["console"]

    try:
        # Configure logging
        logging.config.dictConfig(LOGGING_CONFIG)
        root_logger = logging.getLogger()
        root_logger.info("Logging system initialized successfully")
        print(f"Log level: {root_logger.getEffectiveLevel()}", file=sys.stderr)
        print(
            f"Log handlers: {[h.__class__.__name__ for h in root_logger.handlers]}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error configuring logging: {e}", file=sys.stderr)
        # If configuration fails, use basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Get module logger
    logger = logging.getLogger(__name__)
    logger.info("Logging module initialization complete")
    return logger


# Initialize global logger
logger = setup_logging()


def clean_ansi_sequences(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    
    Args:
        text: Text that may contain ANSI escape sequences
        
    Returns:
        Text with ANSI escape sequences removed
    """
    # ANSI escape sequence regex pattern
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)
