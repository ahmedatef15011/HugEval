"""
HuggingFace API Integration for Real-Time Model Analysis

(Edited to fail hard on 4xx and avoid silent fallbacks for missing/private models.)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .base import timed

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"


class ModelLookupError(RuntimeError):
    """Raised when a model cannot be fetched (not found, private, or other HTTP error)."""
    def __init__(self, model_id: str, status: int, msg: str):
        super().__init__(f"{model_id}: HTTP {status} - {msg}")
        self.model_id = model_id
        self.status = status
        self.msg = msg


def _headers(token: Optional[str] = None) -> Dict[str, str]:
    h = {"User-Agent": "ACME-CLI/0.1.0", "Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def extract_model_id(url: str) -> str:
    """
    Accepts:
      - https://huggingface.co/gpt2 -> "gpt2"
      - https://huggingface.co/org/model -> "org/model"
    """
    if "huggingface.co/" in url:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 4:
            return "/".join(parts[3:])
    raise ValueError(f"Invalid Hugging Face URL: {url}")


def fetch_readme_content(model_id: str, token: Optional[str] = None) -> str:
    """Retrieve README content (best-effort; never raises)."""
    try:
        r = requests.get(f"https://huggingface.co/{model_id}/raw/main/README.md",
                         timeout=10, headers=_headers(token))
        if r.status_code == 200:
            return r.text
        r = requests.get(f"https://huggingface.co/{model_id}/raw/main/README",
                         timeout=10, headers=_headers(token))
        if r.status_code == 200:
            return r.text
        logger.info(f"No README found for {model_id} (last status {r.status_code})")
        return ""
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch README for {model_id}: {e}")
        return ""


def fetch_model_info(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Authoritative existence check. Raises ModelLookupError on non-200.
    """
    url = f"{HF_API_BASE}/models/{model_id}"
    try:
        r = requests.get(url, timeout=10, headers=_headers(token))
        if r.status_code != 200:
            raise ModelLookupError(model_id, r.status_code, r.reason or "error")
        data = r.json()
        if not isinstance(data, dict):
            raise ModelLookupError(model_id, 500, "unexpected JSON payload")
        return data
    except requests.RequestException as e:
        raise RuntimeError(f"network error contacting HF for {model_id}: {e}") from e


def fetch_model_files(model_id: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Best-effort file listing. Returns [] on failure."""
    try:
        r = requests.get(f"{HF_API_BASE}/models/{model_id}/tree/main",
                         timeout=10, headers=_headers(token))
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        logger.info(f"model files listing not available for {model_id}: HTTP {r.status_code}")
        return []
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch model files for {model_id}: {e}")
        return []


def calculate_model_size(files_data: List[Dict[str, Any]]) -> int:
    total_size = 0
    for file_info in files_data:
        if isinstance(file_info, dict) and "size" in file_info:
            size_value = file_info.get("size", 0)
            if isinstance(size_value, int):
                total_size += size_value
    return total_size


def get_model_downloads(model_info: Dict[str, Any]) -> int:
    downloads = model_info.get("downloads", 0)
    return downloads if isinstance(downloads, int) else 0


def get_model_likes(model_info: Dict[str, Any]) -> int:
    likes = model_info.get("likes", 0)
    return likes if isinstance(likes, int) else 0


def get_model_license(model_info: Dict[str, Any]) -> str:
    card_data = model_info.get("cardData", {})
    if isinstance(card_data, dict):
        license_info = card_data.get("license", "")
        if isinstance(license_info, str) and license_info:
            return license_info
    license_direct = model_info.get("license", "")
    if isinstance(license_direct, str) and license_direct:
        return license_direct
    tags = model_info.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and ("license:" in tag.lower() or "lgpl" in tag.lower()):
                return tag
    return ""


def get_days_since_update(model_info: Dict[str, Any]) -> int:
    last_modified = model_info.get("lastModified")
    if not last_modified:
        return 365
    try:
        last_update = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        now = datetime.now(last_update.tzinfo)
        return (now - last_update).days
    except (ValueError, AttributeError):
        return 365


def build_context_from_api(url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Build context strictly from HF API data.
    Raises ModelLookupError on 401/403/404/etc. (no silent fallback).
    """
    model_id = extract_model_id(url)
    logger.info(f"Fetching data for model: {model_id}")

    model_info = fetch_model_info(model_id, token=token)  # may raise ModelLookupError
    files_data = fetch_model_files(model_id, token=token)
    total_bytes = calculate_model_size(files_data) if files_data else 50_000_000

    downloads = get_model_downloads(model_info)
    likes = get_model_likes(model_info)
    license_text = get_model_license(model_info)
    days_since_update = get_days_since_update(model_info)
    readme_content = fetch_readme_content(model_id, token=token)

    context = {
        "total_bytes": total_bytes,
        "license_text": license_text,
        "downloads": downloads,
        "likes": likes,
        "days_since_update": days_since_update,
        "docs": estimate_docs_quality(model_info, readme_content, model_id),
        "readme_content": readme_content,
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


# ---------- Heuristics ----------

def estimate_docs_quality(model_info: Dict[str, Any], readme_content: str = "", model_id: str = "") -> Dict[str, float]:
    from ..llm_analysis import analyze_readme_with_llm
    downloads = model_info.get("downloads", 0)
    likes = model_info.get("likes", 0)
    popularity_score = min(1.0, (downloads / 10000) * 0.3 + (likes / 100) * 0.7)
    base = {
        "readme": min(1.0, 0.3 + popularity_score * 0.7),
        "quickstart": popularity_score * 0.8,
        "tutorials": popularity_score * 0.6,
        "api_docs": popularity_score * 0.7,
        "reproducibility": popularity_score * 0.5,
    }
    if readme_content and model_id:
        try:
            llm = analyze_readme_with_llm(readme_content, model_id)
            base["readme"] = base["readme"] * 0.6 + llm.get("documentation_quality", 0.0) * 0.4
            if llm.get("installation_instructions", False):
                base["quickstart"] = min(1.0, base["quickstart"] + 0.2)
            if llm.get("usage_examples", False):
                base["tutorials"] = min(1.0, base["tutorials"] + 0.3)
            if llm.get("code_blocks_count", 0) >= 2:
                base["api_docs"] = min(1.0, base["api_docs"] + 0.2)
        except Exception as e:
            logger.warning(f"LLM enhancement failed for {model_id}: {e}")
    return base


def estimate_contributors(model_info: Dict[str, Any]) -> int:
    d = model_info.get("downloads", 0)
    return 8 if d > 100000 else 5 if d > 10000 else 3 if d > 1000 else 1


def estimate_dataset_presence(model_info: Dict[str, Any]) -> bool:
    tags = model_info.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and any(w in tag.lower() for w in ["dataset", "data", "training"]):
                return True
    d = model_info.get("downloads", 0)
    return isinstance(d, int) and d > 1000


def estimate_code_presence(model_info: Dict[str, Any]) -> bool:
    return True


def estimate_dataset_docs(model_info: Dict[str, Any]) -> Dict[str, float]:
    p = min(1.0, model_info.get("downloads", 0) / 10000)
    return {"source": p * 0.8, "license": p * 0.9, "splits": p * 0.7, "ethics": p * 0.6}


def estimate_code_quality(model_info: Dict[str, Any]) -> Dict[str, Any]:
    p = min(1.0, model_info.get("downloads", 0) / 10000)
    return {"flake8_errors": max(0, int(15 * (1 - p))), "isort_sorted": p > 0.3, "mypy_errors": max(0, int(10 * (1 - p)))}


def estimate_performance_claims(model_info: Dict[str, Any]) -> Dict[str, bool]:
    card = str(model_info.get("cardData", {})).lower()
    has_bench = any(w in card for w in ["benchmark", "eval", "score", "accuracy", "bleu"])
    has_cite = "citation" in card or model_info.get("downloads", 0) > 5000
    return {"benchmarks": has_bench, "citations": has_cite}


@timed
def popularity_downloads_likes(downloads: int, likes: int, d_cap: int = 100_000, l_cap: int = 1_000) -> float:
    d_norm = min(1.0, math.log1p(max(0, downloads)) / math.log1p(d_cap))
    l_norm = min(1.0, math.log1p(max(0, likes)) / math.log1p(l_cap))
    return 0.6 * d_norm + 0.4 * l_norm


@timed
def freshness_days_since_update(days: int) -> float:
    return max(0.0, min(1.0, 1 - (max(0, days) / 365)))
