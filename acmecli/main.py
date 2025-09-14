from __future__ import annotations
import argparse, logging, concurrent.futures as cf
from .logging_cfg import setup_logging
from .io_utils import read_urls, write_ndjson_line
from .urls import classify, Category
from .scoring import compute_all_scores

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("url_file", nargs="?", help="File with newline-delimited URLs")
    return ap.parse_args()

def build_ctx_from_url(url: str) -> dict:
    # TODO: Step 3—replace with real HF API + repo scan.
    # For Milestone 2, return deterministic placeholder values:
    return {
        "total_bytes": 100_000_000,
        "license_text": "LGPL-2.1",
        "docs": {"readme": 1, "quickstart": 1, "tutorials": 1, "api_docs": 1, "reproducibility": 0.5},
        "contributors": 5,
        "dataset_present": True,
        "code_present": True,
        "dataset_doc": {"source": 1, "license": 1, "splits": 1, "ethics": 0.5},
        "flake8_errors": 5, "isort_sorted": True, "mypy_errors": 3,
        "perf": {"benchmarks": True, "citations": True},
    }

def process_model(url: str) -> dict:
    ctx = build_ctx_from_url(url)
    fields = compute_all_scores(ctx)
    return {"name": url, "category": "MODEL", **fields}

def main():
    args = parse_args()
    setup_logging()
    if not args.url_file:
        print("ERROR: missing URL_FILE. Usage: ./run URL_FILE")
        raise SystemExit(1)
    urls = list(read_urls(args.url_file))
    models = [u for u in urls if classify(u) is Category.MODEL]
    with cf.ProcessPoolExecutor() as ex:
        for rec in ex.map(process_model, models):
            write_ndjson_line(rec)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logging.exception("fatal")
        print(f"ERROR: {e}")
        raise SystemExit(1)
