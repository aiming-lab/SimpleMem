"""
Logging configuration for Omni-Memory.
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
):
    """
    Set up logging configuration for Omni-Memory.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    # Create log directory
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    handlers = {}
    handler_list = []

    if log_to_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": log_level,
            "stream": "ext://sys.stdout",
        }
        handler_list.append("console")

    if log_to_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "level": log_level,
            "filename": os.path.join(log_dir, "omni_memory.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        handler_list.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(levelname)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": handlers,
        "loggers": {
            "omni_memory": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": handler_list,
            "level": "WARNING",
        },
    }

    logging.config.dictConfig(config)

    # Set specific loggers
    logging.getLogger("omni_memory").setLevel(getattr(logging, log_level))

    return logging.getLogger("omni_memory")
