"""
Production Logging Configuration for ACME Model Evaluation System

This module provides enterprise-grade logging setup with environment-based configuration
for different deployment scenarios. Supports silent operation for production pipelines,
detailed debugging for development, and configurable file output for log aggregation.

The logging system is designed for operational monitoring, performance analysis,
and troubleshooting in production environments where model evaluation runs
as part of automated CI/CD pipelines or scheduled batch jobs.
"""

import logging
import os
from pathlib import Path


def setup_logging() -> None:
    """
    Configure application logging based on environment variables for flexible deployment.

    Supports three operational modes optimized for different use cases:
    - Silent (0): Production pipelines where only critical errors are logged
    - Info (1): Standard operation with key events and API interactions logged
    - Debug (2): Development mode with detailed tracing and performance metrics

    Environment Variables:
        LOG_LEVEL: "0" (silent), "1" (info), "2" (debug) - defaults to silent
        LOG_FILE: Output file path - defaults to "acmecli.log" in current directory

    The logging format includes timestamps for performance analysis and correlation
    with external systems in production environments.
    """
    # Environment-driven logging levels for operational flexibility
    level_map = {
        "0": logging.CRITICAL + 1,  # Silent mode for production automation
        "1": logging.INFO,  # Standard operational logging
        "2": logging.DEBUG,  # Detailed debugging and tracing
    }

    lvl = level_map.get(os.getenv("LOG_LEVEL", "0"), logging.CRITICAL + 1)
    path = os.getenv("LOG_FILE", "acmecli.log")

    # Ensure the directory for the log file exists (create if missing)
    try:
        log_path = Path(path)
        # If LOG_FILE points to an existing directory, reject it and fall back
        if log_path.exists() and log_path.is_dir():
            raise ValueError("LOG_FILE points to a directory, not a file")
        log_dir = log_path.parent if str(log_path.parent) != "" else Path(".")
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If path is invalid or directory creation fails, fall back to default file in CWD
        path = "acmecli.log"

    # Production-ready logging format for monitoring and analysis
    logging.basicConfig(
        filename=path,
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
