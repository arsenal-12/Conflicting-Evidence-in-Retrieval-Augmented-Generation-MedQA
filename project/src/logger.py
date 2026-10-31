"""
===============================================================================
logger.py
===============================================================================

Purpose
-------
Provide a single logging system for the entire project.

Instead of every module creating its own logging configuration, the whole
pipeline shares one common setup. This keeps log messages consistent and
writes them both to the terminal and to a log file.

Pipeline Position
-----------------
config.yaml
      ↓
configure_logging()
      ↓
Every Module
      ↓
get_logger(__name__)
      ↓
logger.info(...)
logger.warning(...)
logger.error(...)

Responsibilities
----------------
• Configure logging once at application startup.
• Create log files automatically.
• Return a logger for each module.
• Prevent duplicate log messages.

Why this file exists
--------------------
Without this file, every module would need to configure logging
independently, leading to duplicated code and inconsistent log output.

This keeps logging centralized and easy to maintain.

When should I modify this file?
-------------------------------
Modify this file if you want to:

• Change the log format.
• Change where logs are stored.
• Add coloured logging.
• Add rotating log files.

Do NOT modify this file if you simply want to write log messages.

Other modules should only use:

    logger.info(...)
    logger.warning(...)
    logger.error(...)

Software Engineering
--------------------
✓ Single Responsibility Principle
    Handles only logging.

✓ Centralised Configuration
    Logging is configured once and shared everywhere.

✓ Reusability
    Every module uses the same logging system.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

# Keeps track of whether logging has already been configured.
# This prevents multiple handlers being added if configure_logging()
# is called more than once.
_LOGGING_CONFIGURED = False


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure the project's logging system.

    This function should be called only once when the application starts.

    After this, every module automatically uses the same logging settings.
    """
    global _LOGGING_CONFIGURED
    # Allow this function to update the module-level flag.
    if _LOGGING_CONFIGURED:

        return

    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    log_filename = log_cfg.get("log_filename", "pipeline.log")
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers first
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger using the project's central logging configuration.
    """
    return logging.getLogger(name)
