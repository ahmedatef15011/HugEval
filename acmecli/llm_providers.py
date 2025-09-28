"""
LLM provider abstraction (Purdue GenAI Studio only).

Environment variables:
- LLM_PROVIDER: must be 'purdue' (default: purdue)
- PURDUE_GENAI_BASE_URL: Base URL for Purdue GenAI Studio REST API
- PURDUE_GENAI_API_KEY: API key/token for Purdue GenAI Studio
- PURDUE_GENAI_MODEL: Model identifier/name for Purdue GenAI Studio

Usage:
    from .llm_providers import get_llm_provider
    provider = get_llm_provider()
    result = provider.analyze_readme(model_name, readme_text)

Contract: analyze_readme returns dict with keys
- documentation_quality: float [0,1]
- ease_of_use: float [0,1]
- examples_present: bool
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def analyze_readme(self, model_name: str, readme: str) -> Dict[str, Any]:
        raise NotImplementedError


class PurdueGenAIProvider(LLMProvider):
    def __init__(
        self, base_url: str, api_key: str, model: str, path: str = "/v1/chat/completions"
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.path = path if path.startswith("/") else f"/{path}"

    def analyze_readme(self, model_name: str, readme: str) -> Dict[str, Any]:
        """Call Purdue GenAI Studio REST API.

            Assumed endpoint (example): POST {base_url}/v1/chat/completions
            Headers: Authorization: Bearer <API_KEY>, Content-Type: application/json
        Body: { model, messages: [{role:"user", content: prompt}], temperature: 0.0,
        max_tokens: 150 }
            Adjust as needed for the actual Purdue API.
        """
        prompt = (
            f"Analyze README for model '{model_name}'. Return JSON with keys: "
            "documentation_quality, ease_of_use, examples_present (bool).\n\n"
            f"README:\n{readme[:2000]}..."
        )
        url = f"{self.base_url}{self.path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 150,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"Purdue GenAI HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            # Expected OpenAI-compatible shape; adapt mapping if needed.
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {
                "documentation_quality": float(parsed.get("documentation_quality", 0.0)),
                "ease_of_use": float(parsed.get("ease_of_use", 0.0)),
                "examples_present": bool(parsed.get("examples_present", False)),
            }
        except Exception as e:
            logger.warning(f"Purdue GenAI analyze_readme failed: {e}")
            raise


def get_llm_provider() -> LLMProvider | None:
    # Purdue only; default to purdue and return None if not configured
    raw = os.getenv("LLM_PROVIDER", "purdue")
    provider = (raw or "purdue").strip().lower()
    if provider != "purdue":
        logger.info(f"Unsupported LLM_PROVIDER '{provider}', only 'purdue' is allowed")
        return None
    base_url = os.getenv("PURDUE_GENAI_BASE_URL")
    api_key = os.getenv("PURDUE_GENAI_API_KEY")
    model = os.getenv("PURDUE_GENAI_MODEL", "gpt-4o-mini")
    path = os.getenv("PURDUE_GENAI_PATH", "/v1/chat/completions")
    if not base_url or not api_key:
        logger.info("Purdue GenAI not configured (missing base URL or API key)")
        return None
    return PurdueGenAIProvider(base_url, api_key, model, path)
