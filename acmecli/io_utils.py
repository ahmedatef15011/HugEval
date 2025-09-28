"""
I/O Utilities for ACME Model Evaluation System

This module provides essential data input/output operations optimized for high-performance
model evaluation workflows. Uses streaming approaches and efficient JSON serialization
to handle large batches of model URLs and evaluation results without memory bottlenecks.

The streaming design supports processing thousands of models while maintaining
low memory footprint and enabling real-time result output for pipeline integration.
"""

import sys
from typing import Any, Dict, Iterable

import orjson


def read_urls(path: str) -> Iterable[str]:
    """
    Stream model URLs from input file with memory-efficient processing.

    Reads newline-delimited URLs while filtering empty lines and whitespace,
    enabling processing of arbitrarily large URL lists without memory constraints.
    Essential for batch evaluation workflows and continuous integration pipelines.

    Args:
        path: File path containing newline-separated URLs or comma-separated URLs

    Yields:
        str: Clean, validated URLs ready for processing
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Handle comma-separated URLs on the same line
                for url in line.split(","):
                    url = url.strip()
                    if url:  # Skip empty URLs after splitting
                        yield url


def write_ndjson_line(d: Dict[str, Any]) -> None:
    """
    Write evaluation results in newline-delimited JSON format for streaming output.

    Uses high-performance orjson serialization for optimal throughput in production
    environments. NDJSON format enables real-time processing and easy integration
    with data pipeline tools, log aggregators, and analysis systems.

    Args:
        d: Dictionary containing model evaluation results and metadata
    """
    sys.stdout.write(orjson.dumps(d).decode() + "\n")
