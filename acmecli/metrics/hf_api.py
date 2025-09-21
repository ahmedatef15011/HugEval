from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List

import requests

from .base import timed

logger = logging.getLogger(__name__)

# Hugging Face API base URL
HF_API_BASE = "https://huggingface.co/api"


def extract_model_id(url: str) -> str:
    """Extract model ID from Hugging Face URL."""
    # https://huggingface.co/gpt2 -> gpt2
    # https://huggingface.co/microsoft/DialoGPT-medium -> microsoft/DialoGPT-medium
    if "huggingface.co/" in url:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 4:  # https://huggingface.co/model_name
            return "/".join(parts[3:])  # Handle org/model format
    raise ValueError(f"Invalid Hugging Face URL: {url}")


def fetch_model_info(model_id: str) -> Dict[str, Any]:
    """Fetch model information from Hugging Face API."""
    try:
        # Get model info
        response = requests.get(
            f"{HF_API_BASE}/models/{model_id}", timeout=10, headers={"User-Agent": "ACME-CLI/0.1.0"}
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch model info for {model_id}: {e}")
        return {}


def fetch_model_files(model_id: str) -> Dict[str, Any]:
    """Fetch model files information to calculate size."""
    try:
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}/tree/main",
            timeout=10,
            headers={"User-Agent": "ACME-CLI/0.1.0"},
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch model files for {model_id}: {e}")
        return {}


def calculate_model_size(files_data: List[Dict[str, Any]]) -> int:
    """Calculate total model size from files data."""
    total_size = 0
    for file_info in files_data:
        if isinstance(file_info, dict) and "size" in file_info:
            size_value = file_info.get("size", 0)
            if isinstance(size_value, int):
                total_size += size_value
    return total_size


def get_model_downloads(model_info: Dict[str, Any]) -> int:
    """Extract download count from model info."""
    downloads = model_info.get("downloads", 0)
    return downloads if isinstance(downloads, int) else 0


def get_model_likes(model_info: Dict[str, Any]) -> int:
    """Extract likes count from model info."""
    likes = model_info.get("likes", 0)
    return likes if isinstance(likes, int) else 0


def get_model_license(model_info: Dict[str, Any]) -> str:
    """Extract license information from model info."""
    # Try different places where license might be stored
    card_data = model_info.get("cardData", {})
    if isinstance(card_data, dict):
        license_info = card_data.get("license", "")
        if isinstance(license_info, str) and license_info:
            return license_info

    # Try direct license field
    license_direct = model_info.get("license", "")
    if isinstance(license_direct, str) and license_direct:
        return license_direct

    # Check in tags
    tags = model_info.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and ("license:" in tag.lower() or "lgpl" in tag.lower()):
                return tag

    return ""  # Return empty string if no license found


def get_days_since_update(model_info: Dict[str, Any]) -> int:
    """Calculate days since last update."""
    last_modified = model_info.get("lastModified")
    if not last_modified:
        return 365  # Default to 1 year if unknown

    try:
        # Parse ISO format datetime
        last_update = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        now = datetime.now(last_update.tzinfo)
        delta = now - last_update
        return delta.days
    except (ValueError, AttributeError):
        return 365  # Default if parsing fails


def build_context_from_api(url: str) -> Dict[str, Any]:
    """Build context dictionary from real Hugging Face API data."""
    try:
        model_id = extract_model_id(url)
        logger.info(f"Fetching data for model: {model_id}")

        # Fetch model information
        model_info = fetch_model_info(model_id)
        if not model_info:
            logger.warning(f"No model info found for {model_id}, using fallback data")
            return get_fallback_context()

        # Fetch file information for size calculation
        files_data = fetch_model_files(model_id)

        # Calculate model size
        if isinstance(files_data, list):
            total_bytes = calculate_model_size(files_data)
        else:
            total_bytes = 50_000_000  # Default 50MB if can't determine

        # Extract other information
        downloads = get_model_downloads(model_info)
        likes = get_model_likes(model_info)
        license_text = get_model_license(model_info)
        days_since_update = get_days_since_update(model_info)

        # Build context with real data
        context = {
            "total_bytes": total_bytes,
            "license_text": license_text,
            "downloads": downloads,
            "likes": likes,
            "days_since_update": days_since_update,
            # TODO: These would require additional API calls or repo scanning
            # For now, use reasonable defaults based on model popularity
            "docs": estimate_docs_quality(model_info),
            "contributors": estimate_contributors(model_info),
            "dataset_present": estimate_dataset_presence(model_info),
            "code_present": estimate_code_presence(model_info),
            "dataset_doc": estimate_dataset_docs(model_info),
            "flake8_errors": estimate_code_quality(model_info)["flake8_errors"],
            "isort_sorted": estimate_code_quality(model_info)["isort_sorted"],
            "mypy_errors": estimate_code_quality(model_info)["mypy_errors"],
            "perf": estimate_performance_claims(model_info),
        }

        logger.info(f"Successfully built context for {model_id}")
        return context

    except Exception as e:
        logger.error(f"Error building context for {url}: {e}")
        return get_fallback_context()


def get_fallback_context() -> Dict[str, Any]:
    """Return fallback context when API calls fail."""
    return {
        "total_bytes": 100_000_000,
        "license_text": "",
        "downloads": 0,
        "likes": 0,
        "days_since_update": 365,
        "docs": {
            "readme": 0.5,
            "quickstart": 0,
            "tutorials": 0,
            "api_docs": 0,
            "reproducibility": 0,
        },
        "contributors": 1,
        "dataset_present": False,
        "code_present": False,
        "dataset_doc": {"source": 0, "license": 0, "splits": 0, "ethics": 0},
        "flake8_errors": 20,
        "isort_sorted": False,
        "mypy_errors": 10,
        "perf": {"benchmarks": False, "citations": False},
    }


def estimate_docs_quality(model_info: Dict[str, Any]) -> Dict[str, float]:
    """Estimate documentation quality based on available info."""
    # Higher quality for models with more downloads/likes
    downloads = model_info.get("downloads", 0)
    likes = model_info.get("likes", 0)

    # Popular models likely have better docs
    popularity_score = min(1.0, (downloads / 10000) * 0.3 + (likes / 100) * 0.7)

    return {
        "readme": min(1.0, 0.3 + popularity_score * 0.7),  # Most models have some README
        "quickstart": popularity_score * 0.8,
        "tutorials": popularity_score * 0.6,
        "api_docs": popularity_score * 0.7,
        "reproducibility": popularity_score * 0.5,
    }


def estimate_contributors(model_info: Dict[str, Any]) -> int:
    """Estimate number of contributors."""
    # Popular models likely have more contributors
    downloads = model_info.get("downloads", 0)
    if downloads > 100000:
        return 8
    elif downloads > 10000:
        return 5
    elif downloads > 1000:
        return 3
    else:
        return 1


def estimate_dataset_presence(model_info: Dict[str, Any]) -> bool:
    """Estimate if model has associated dataset."""
    # Check tags for dataset mentions
    tags = model_info.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and any(
                word in tag.lower() for word in ["dataset", "data", "training"]
            ):
                return True

    # Popular models likely have datasets
    downloads = model_info.get("downloads", 0)
    return isinstance(downloads, int) and downloads > 1000


def estimate_code_presence(model_info: Dict[str, Any]) -> bool:
    """Estimate if model has associated code."""
    # Most HF models have some code (at minimum inference code)
    return True


def estimate_dataset_docs(model_info: Dict[str, Any]) -> Dict[str, float]:
    """Estimate dataset documentation quality."""
    popularity = min(1.0, model_info.get("downloads", 0) / 10000)
    return {
        "source": popularity * 0.8,
        "license": popularity * 0.9,
        "splits": popularity * 0.7,
        "ethics": popularity * 0.6,
    }


def estimate_code_quality(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate code quality metrics."""
    popularity = min(1.0, model_info.get("downloads", 0) / 10000)
    # Popular models likely have better code quality
    return {
        "flake8_errors": max(0, int(15 * (1 - popularity))),
        "isort_sorted": popularity > 0.3,
        "mypy_errors": max(0, int(10 * (1 - popularity))),
    }


def estimate_performance_claims(model_info: Dict[str, Any]) -> Dict[str, bool]:
    """Estimate if model has performance benchmarks/citations."""
    # Check model card for benchmark mentions
    card_data = model_info.get("cardData", {})
    model_card = str(card_data).lower()

    has_benchmarks = any(
        word in model_card for word in ["benchmark", "eval", "score", "accuracy", "bleu"]
    )
    has_citations = "citation" in model_card or model_info.get("downloads", 0) > 5000

    return {
        "benchmarks": has_benchmarks,
        "citations": has_citations,
    }


@timed
def popularity_downloads_likes(
    downloads: int, likes: int, d_cap: int = 100_000, l_cap: int = 1_000
) -> float:
    """Log-scaled normalization for popularity."""
    d_norm = min(1.0, math.log1p(max(0, downloads)) / math.log1p(d_cap))
    likes_norm = min(1.0, math.log1p(max(0, likes)) / math.log1p(l_cap))
    return 0.6 * d_norm + 0.4 * likes_norm


@timed
def freshness_days_since_update(days: int) -> float:
    """0 days → 1.0; 365+ days → 0.0 linearly."""
    return max(0.0, min(1.0, 1 - (max(0, days) / 365)))
