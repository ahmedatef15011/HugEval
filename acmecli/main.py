"""
ACME Model Scoring CLI - Main Application Entry Point

This module provides the primary command-line interface for the ACME trustworthy model
evaluation system. It orchestrates the complete workflow from URL input to score generation,
integrating real-time API data with machine learning-based analysis for comprehensive
model assessment.

The application supports both technical JSON output for automated processing and
human-readable summary reports for business stakeholders, making it suitable
for both development pipelines and executive decision-making.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from typing import Any, Dict

from .io_utils import read_urls, write_ndjson_line
from .logging_cfg import setup_logging
from .metrics.hf_api import build_context_from_api
from .report import capture_and_summarize_results
from .scoring import compute_all_scores
from .urls import Category, classify


def parse_args() -> argparse.Namespace:
    """
    Configure and parse command-line arguments for the model evaluation system.

    Supports flexible output options including technical JSON data and business-friendly
    summary reports. The output parameter allows for timestamped batch processing.

    Returns:
        argparse.Namespace: Parsed command-line arguments with validation
    """
    ap = argparse.ArgumentParser(description="Evaluate Hugging Face models and generate scores")
    ap.add_argument("url_file", nargs="?", help="File with newline-delimited URLs")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Generate a human-readable summary report (saves files)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="evaluation",
        help="Base filename for output files (default: evaluation)",
    )
    return ap.parse_args()


def build_ctx_from_url(url: str) -> Dict[str, Any]:
    """
    Extract comprehensive model context from Hugging Face URL.

    This function serves as the primary data aggregation layer, combining real-time
    API data with repository analysis to create a complete picture of model quality.
    Includes automatic fallback mechanisms for resilient operation.

    Args:
        url: Hugging Face model URL for analysis

    Returns:
        Dict containing model metrics, documentation quality, performance indicators,
        and other assessment criteria needed for comprehensive scoring
    """
    return build_context_from_api(url)


def process_model(url: str) -> Dict[str, Any]:
    """
    Execute complete model evaluation pipeline for a single URL.

    Coordinates data collection, context building, and score computation to produce
    final assessment results. This is the core processing unit that handles individual
    models within the batch evaluation system.

    Args:
        url: Model URL to evaluate

    Returns:
        Dict containing model name, category classification, and all computed scores
    """
    ctx = build_ctx_from_url(url)
    fields = compute_all_scores(ctx)
    return {"name": url, "category": "MODEL", **fields}


def main() -> None:
    """
    Primary application entry point orchestrating the complete evaluation workflow.

    Manages command-line argument processing, logging setup, URL validation, and
    coordinates parallel model processing with optional summary report generation.
    Implements robust error handling and cross-platform compatibility.
    """
    args = parse_args()
    setup_logging()
    if not args.url_file:
        print("ERROR: missing URL_FILE. Usage: ./run URL_FILE [--summary]")
        raise SystemExit(1)

    urls = list(read_urls(args.url_file))
    models = [u for u in urls if classify(u) is Category.MODEL]

    # Collect results for potential summary generation
    results = []

    # Use ProcessPoolExecutor with cross-platform compatibility safeguards
    try:
        with cf.ProcessPoolExecutor() as ex:
            for rec in ex.map(process_model, models):
                write_ndjson_line(rec)
                if args.summary:
                    results.append(rec)
    except (RuntimeError, OSError):
        # Graceful fallback to sequential processing when parallel execution fails
        # This ensures compatibility across different operating systems and deployment environments
        for url in models:
            rec = process_model(url)
            write_ndjson_line(rec)
            if args.summary:
                results.append(rec)

    # Generate comprehensive business summary report when requested
    if args.summary and results:
        ndjson_file, summary_file = capture_and_summarize_results(results, args.output)
        print(f"\n📄 Results saved to: {ndjson_file}", flush=True)
        print(f"📊 Summary report: {summary_file}", flush=True)
        print(f"🔍 View summary: cat {summary_file}", flush=True)


if __name__ == "__main__":
    main()
