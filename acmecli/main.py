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
    """Build context from real Hugging Face API data."""
    return build_context_from_api(url)


def process_model(url: str) -> Dict[str, Any]:
    ctx = build_ctx_from_url(url)
    fields = compute_all_scores(ctx)
    return {"name": url, "category": "MODEL", **fields}


def main() -> None:
    args = parse_args()
    setup_logging()
    if not args.url_file:
        print("ERROR: missing URL_FILE. Usage: ./run URL_FILE [--summary]")
        raise SystemExit(1)

    urls = list(read_urls(args.url_file))
    models = [u for u in urls if classify(u) is Category.MODEL]

    # Collect results for potential summary generation
    results = []

    # Use ProcessPoolExecutor with proper __main__ guard for cross-platform compatibility
    try:
        with cf.ProcessPoolExecutor() as ex:
            for rec in ex.map(process_model, models):
                write_ndjson_line(rec)
                if args.summary:
                    results.append(rec)
    except (RuntimeError, OSError):
        # Fallback to sequential processing if ProcessPoolExecutor fails
        # This can happen on some systems or when running as a module
        for url in models:
            rec = process_model(url)
            write_ndjson_line(rec)
            if args.summary:
                results.append(rec)

    # Generate summary report if requested
    if args.summary and results:
        ndjson_file, summary_file = capture_and_summarize_results(results, args.output)
        print(f"\n📄 Results saved to: {ndjson_file}", flush=True)
        print(f"📊 Summary report: {summary_file}", flush=True)
        print(f"🔍 View summary: cat {summary_file}", flush=True)


if __name__ == "__main__":
    main()
